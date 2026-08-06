"""Figures for the write-up. matplotlib only — no seaborn, no pandas.

    python -m analysis.plots --arm-b B_summary.json --arm-c C_summary.json \
        --out-dir figures/ [--trace-analysis armB_L3_trace.json]

One figure per file, emitted as **both PNG and SVG**.

EVERY FIGURE CARRIES ITS PROVENANCE
-----------------------------------
Each one gets a footnote with the GPU UUID, the short fixture sha256 and
``max_new_tokens``. A figure gets lifted out of a repository and dropped into a
deck, where it is separated from its manifest forever; without the footnote it
becomes a number with no way back to the run that produced it. That is the
failure this whole study is a reaction to, and a figure is the most likely
artefact to escape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.loader import RunSet, load_run  # noqa: E402
from analysis.raw_stats import _itls, _ok, _records_for_level  # noqa: E402

__all__ = ["FIGURES", "NoDataForFigure", "footnote_text", "render_all"]

ARM_B_COLOUR = "#B4501E"   # warm — the naive stack
ARM_C_COLOUR = "#1F6FB4"   # cool — the optimized stack
GRID_KW = dict(alpha=0.25, linewidth=0.6)

FIGURES = (
    "fig1_throughput_vs_concurrency",
    "fig2_latency_cdf",
    "fig3_gpu_utilisation_timeseries",
    "fig4_vram_slope",
    "fig5_itl_distribution",
    "fig6_launch_overhead",
)


def _plt():
    """Import matplotlib with a headless backend.

    Deferred so the module imports (and its helpers stay unit-testable) on a
    machine without matplotlib.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def footnote_text(runs: Sequence[RunSet], extra: str = "") -> str:
    """GPU UUID + fixture hash + max_new_tokens, for the figure footer."""
    m = next((r.manifest for r in runs if r.manifest), {})
    uuid = m.get("gpu_uuid") or "gpu-unknown"
    fixture = (m.get("fixture_sha256") or "fixture-unknown")[:12]
    mnt = m.get("max_new_tokens")
    parts = [
        f"GPU {uuid}",
        f"fixture {fixture}",
        f"max_new_tokens={mnt}",
    ]
    ids = ", ".join(r.run_id for r in runs)
    parts.append(f"runs: {ids}")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


class NoDataForFigure(RuntimeError):
    """Raised when a figure would be written with nothing plotted on it.

    An empty PNG sitting in the output directory looks like a rendered figure.
    render_all() records the reason as a skip instead, so a missing series is
    visible rather than something the reader has to notice for themselves.
    """


def _require_plotted(ax, name: str, detail: str = "") -> None:
    if ax.lines or ax.patches or ax.collections or ax.containers:
        return
    raise NoDataForFigure(
        f"{name}: nothing to plot"
        + (f" ({detail})" if detail else "")
        + ". The figure was NOT written."
    )


def _finish(fig, ax_or_axes, runs, out_dir, name, footnote_extra=""):
    plt = _plt()
    fig.text(
        0.005, 0.005, footnote_text(runs, footnote_extra),
        fontsize=5.5, color="#555555", ha="left", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# fig 1 — throughput vs concurrency
# --------------------------------------------------------------------------


def fig1_throughput_vs_concurrency(runs: Sequence[RunSet], out_dir: str) -> List[str]:
    """Output tok/s against offered load. Log y — the arms differ by ~100x."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for r, colour in zip(runs, (ARM_B_COLOUR, ARM_C_COLOUR)):
        xs = r.concurrencies()
        ys = [(r.level(c) or {}).get("output_tok_per_s") for c in xs]
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pairs:
            continue
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                marker="o", color=colour, linewidth=2, label=r.display)
        for x, y in pairs:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=colour)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Offered concurrency (requests outstanding)")
    ax.set_ylabel("Output throughput (tok/s, log scale)")
    ax.set_title("Decode throughput vs offered load")
    ax.grid(True, which="both", **GRID_KW)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    _require_plotted(ax, "fig1_throughput_vs_concurrency", "no output_tok_per_s in any level")
    return _finish(fig, ax, runs, out_dir, "fig1_throughput_vs_concurrency")


# --------------------------------------------------------------------------
# fig 2 — latency CDF
# --------------------------------------------------------------------------


def fig2_latency_cdf(runs: Sequence[RunSet], out_dir: str,
                     concurrency: int = 32) -> List[str]:
    """Empirical CDF of end-to-end latency.

    Percentiles compress the shape away. A CDF shows whether a distribution is
    a tight spike or a broad smear, which p50/p95/p99 alone cannot distinguish.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for r, colour in zip(runs, (ARM_B_COLOUR, ARM_C_COLOUR)):
        try:
            records = _records_for_level(r, concurrency)
        except FileNotFoundError:
            continue
        vals = sorted((rec["t_done"] - rec["t_submit"]) * 1000.0
                      for rec in records if _ok(rec))
        if not vals:
            continue
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, ys, where="post", color=colour, linewidth=2,
                label=f"{r.display}  (n={len(vals)})")

    ax.set_xscale("log")
    ax.set_xlabel(f"End-to-end latency at concurrency {concurrency} (ms, log scale)")
    ax.set_ylabel("Cumulative fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"End-to-end latency distribution (C={concurrency})")
    ax.grid(True, which="both", **GRID_KW)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    _require_plotted(ax, "fig2_latency_cdf", "no completed requests at this concurrency in _raw.json")
    return _finish(fig, ax, runs, out_dir, "fig2_latency_cdf",
                   f"C={concurrency}")


# --------------------------------------------------------------------------
# fig 3 — GPU utilisation timeseries
# --------------------------------------------------------------------------


def fig3_gpu_utilisation_timeseries(runs: Sequence[RunSet], out_dir: str,
                                    concurrency: int = 32) -> List[str]:
    """NVML utilisation over the run, both arms, time normalised to run start."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for r, colour in zip(runs, (ARM_B_COLOUR, ARM_C_COLOUR)):
        try:
            rows = r.monitor(concurrency)
        except FileNotFoundError:
            continue
        pts = [(row.get("t_rel_s"), row.get("util_gpu_pct")) for row in rows]
        pts = [(t, u) for t, u in pts if t is not None and u is not None]
        if not pts:
            continue
        t0 = pts[0][0]
        xs = [t - t0 for t, _ in pts]
        ys = [u for _, u in pts]
        mean = sum(ys) / len(ys)
        ax.plot(xs, ys, color=colour, linewidth=1.1, alpha=0.85,
                label=f"{r.display}  (mean {mean:.0f}%)")
        ax.axhline(mean, color=colour, linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Time since level start (s)")
    ax.set_ylabel("NVML GPU utilisation (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"GPU utilisation over the run (C={concurrency})")
    ax.grid(True, **GRID_KW)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="center right")
    # The one caveat that must not be lost when this figure is quoted. Placed
    # BELOW the axes rather than inside them: at these two utilisation bands an
    # in-axes annotation lands on top of the arm B trace.
    ax.text(0.5, -0.16,
            "NVML utilisation is the fraction of time at least one kernel was "
            "resident. It is NOT achieved occupancy —\na single small kernel "
            "pins it at 100%. Occupancy requires ncu (Layer 4).",
            transform=ax.transAxes, fontsize=7, color="#444444",
            ha="center", va="top")
    _require_plotted(ax, "fig3_gpu_utilisation_timeseries", "no monitor CSV rows at this concurrency")
    return _finish(fig, ax, runs, out_dir, "fig3_gpu_utilisation_timeseries",
                   f"C={concurrency}")


# --------------------------------------------------------------------------
# fig 4 — VRAM slope (arm B only)
# --------------------------------------------------------------------------


def _linfit(xs: Sequence[float], ys: Sequence[float]):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if n < 2 or denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return slope, intercept, r2


def fig4_vram_slope(arm_b: RunSet, out_dir: str) -> List[str]:
    """Arm B only. Arm C cannot appear here — see the in-figure caption."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    xs = arm_b.concurrencies()
    ys = [((arm_b.level(c) or {}).get("resources") or {}).get("vram_peak_mb")
          for c in xs]
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None]

    if pairs:
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs],
                   color=ARM_B_COLOUR, s=52, zorder=3, label="measured NVML peak")
        fit = _linfit([p[0] for p in pairs], [p[1] for p in pairs])
        if fit:
            slope, intercept, r2 = fit
            x_line = [0, max(p[0] for p in pairs) * 1.08]
            ax.plot(x_line, [slope * x + intercept for x in x_line],
                    color=ARM_B_COLOUR, linestyle="--", linewidth=1.4,
                    label="least-squares fit")
            label = f"slope = {slope:.1f} MB/request"
            if r2 is not None:
                label += f"\n$R^2$ = {r2:.3f}"
            label += f"\nintercept = {intercept:.0f} MB (weights + runtime)"
            ax.annotate(label, xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF",
                                  ec=ARM_B_COLOUR, alpha=0.9))

    ax.set_xlabel("Offered concurrency (requests outstanding)")
    ax.set_ylabel("NVML peak process VRAM (MB)")
    ax.set_title("Arm B: marginal VRAM per concurrent request")
    ax.grid(True, **GRID_KW)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.5, -0.20,
            "Arm C is deliberately absent. vLLM pre-allocates its KV pool to "
            "gpu_memory_utilization at startup,\nso its NVML figure reports the "
            "configured reservation and is flat with load — there is no slope "
            "to fit,\nand plotting it beside a demand curve would imply a "
            "comparison that does not exist.",
            transform=ax.transAxes, fontsize=7, color="#444444",
            ha="center", va="top")
    _require_plotted(ax, "fig4_vram_slope", "no vram_peak_mb in any level")
    return _finish(fig, ax, [arm_b], out_dir, "fig4_vram_slope")


# --------------------------------------------------------------------------
# fig 5 — ITL distribution
# --------------------------------------------------------------------------


def fig5_itl_distribution(runs: Sequence[RunSet], out_dir: str,
                          concurrency: int = 32) -> List[str]:
    """Inter-token latency histogram, log x. Two orders of magnitude apart."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    series: List[Tuple[RunSet, List[float], str]] = []
    for r, colour in zip(runs, (ARM_B_COLOUR, ARM_C_COLOUR)):
        try:
            records = _records_for_level(r, concurrency)
        except FileNotFoundError:
            continue
        vals: List[float] = []
        for rec in records:
            if _ok(rec):
                vals.extend(g * 1000.0 for g in _itls(rec) if g > 0)
        if vals:
            series.append((r, vals, colour))

    if series:
        lo = min(min(v) for _, v, _ in series)
        hi = max(max(v) for _, v, _ in series)
        bins = [10 ** e for e in _linspace(math.log10(max(lo, 1e-3)),
                                           math.log10(hi), 60)]
        for r, vals, colour in series:
            mean = sum(vals) / len(vals)
            ax.hist(vals, bins=bins, color=colour, alpha=0.55,
                    label=f"{r.display}  (mean {mean:.1f} ms, n={len(vals)})")
            ax.axvline(mean, color=colour, linestyle="--", linewidth=1.3)

    ax.set_xscale("log")
    ax.set_xlabel(f"Inter-token latency at concurrency {concurrency} (ms, log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"Inter-token latency distribution (C={concurrency})")
    ax.grid(True, which="both", **GRID_KW)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    _require_plotted(ax, "fig5_itl_distribution", "no inter-token gaps at this concurrency")
    return _finish(fig, ax, runs, out_dir, "fig5_itl_distribution",
                   f"C={concurrency}")


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


# --------------------------------------------------------------------------
# fig 6 — launch overhead
# --------------------------------------------------------------------------


def fig6_launch_overhead(trace_summary: Dict[str, Any], runs: Sequence[RunSet],
                         out_dir: str) -> List[str]:
    """cudaLaunchKernel CPU time vs GPU busy vs trace span, from Layer 3."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    span_us = float(trace_summary.get("trace_span_us") or 0.0)
    busy_us = float(trace_summary.get("gpu_busy_us") or 0.0)
    launch_us = 0.0
    for entry in trace_summary.get("cuda_runtime_breakdown") or []:
        if entry.get("name") == "cudaLaunchKernel":
            launch_us = float(entry.get("total_cpu_us") or 0.0)
            break

    labels = ["Trace span\n(wall)", "GPU busy\n(union of kernels)",
              "cudaLaunchKernel\n(CPU time)"]
    values = [span_us / 1e6, busy_us / 1e6, launch_us / 1e6]
    colours = ["#8C8C8C", ARM_C_COLOUR, ARM_B_COLOUR]

    bars = ax.bar(labels, values, color=colours, width=0.58)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1f} s", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)

    idle_frac = trace_summary.get("gpu_idle_fraction")
    if idle_frac is not None:
        ax.annotate(
            f"GPU idle {idle_frac * 100:.1f}% of the traced span",
            xy=(0.5, 0.90), xycoords="axes fraction", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF4E8", ec=ARM_B_COLOUR),
        )

    ax.set_ylabel("Seconds")
    ax.set_title("Arm B: where the traced window went")
    ax.grid(True, axis="y", **GRID_KW)
    ax.text(0.5, -0.16,
            "cudaLaunchKernel CPU time exceeding GPU busy time means the host "
            "spent longer dispatching\nwork than the device spent doing it. "
            "Launch cost is the bottleneck, not kernel duration.",
            transform=ax.transAxes, fontsize=7, color="#444444",
            ha="center", va="top")
    return _finish(fig, ax, runs, out_dir, "fig6_launch_overhead",
                   "Layer 3, bounded window")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def render_all(
    arm_b: Optional[RunSet],
    arm_c: Optional[RunSet],
    out_dir: str,
    cdf_concurrency: int = 32,
    trace_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runs = [r for r in (arm_b, arm_c) if r is not None]
    made: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    def attempt(name, fn):
        try:
            made[name] = fn()
        except Exception as exc:  # noqa: BLE001
            # A figure that cannot be built is recorded as skipped-with-reason
            # rather than silently missing from the output directory.
            skipped[name] = f"{type(exc).__name__}: {exc}"

    if runs:
        attempt("fig1_throughput_vs_concurrency",
                lambda: fig1_throughput_vs_concurrency(runs, out_dir))
        attempt("fig2_latency_cdf",
                lambda: fig2_latency_cdf(runs, out_dir, cdf_concurrency))
        attempt("fig3_gpu_utilisation_timeseries",
                lambda: fig3_gpu_utilisation_timeseries(runs, out_dir, cdf_concurrency))
        attempt("fig5_itl_distribution",
                lambda: fig5_itl_distribution(runs, out_dir, cdf_concurrency))
    if arm_b is not None:
        attempt("fig4_vram_slope", lambda: fig4_vram_slope(arm_b, out_dir))
    if trace_summary:
        attempt("fig6_launch_overhead",
                lambda: fig6_launch_overhead(trace_summary, runs, out_dir))
    else:
        skipped["fig6_launch_overhead"] = (
            "no --trace-analysis JSON supplied (Layer 3 output required)"
        )

    return {"figures": made, "skipped": skipped, "out_dir": out_dir}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render benchmark figures.")
    ap.add_argument("--arm-b", default=None, help="arm B *_summary.json")
    ap.add_argument("--arm-c", default=None, help="arm C *_summary.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="level used for the CDF / utilisation / ITL figures")
    ap.add_argument("--trace-analysis", default=None,
                    help="JSON from common.trace_analysis (for fig6)")
    args = ap.parse_args(argv)

    arm_b = load_run(args.arm_b, label="Arm B (HF threads)") if args.arm_b else None
    arm_c = load_run(args.arm_c, label="Arm C (vLLM)") if args.arm_c else None
    if arm_b is None and arm_c is None:
        print("supply at least one of --arm-b / --arm-c", file=sys.stderr)
        return 2

    trace_summary = None
    if args.trace_analysis:
        with open(args.trace_analysis) as fh:
            trace_summary = json.load(fh)

    result = render_all(arm_b, arm_c, args.out_dir,
                        cdf_concurrency=args.concurrency,
                        trace_summary=trace_summary)

    for name, paths in result["figures"].items():
        print(f"[plots] {name}: " + ", ".join(paths))
    for name, why in result["skipped"].items():
        print(f"[plots] SKIPPED {name}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
