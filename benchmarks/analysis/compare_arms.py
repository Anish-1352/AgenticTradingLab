"""Matched arm-to-arm comparison table (markdown + CSV).

    python -m analysis.compare_arms A_summary.json B_summary.json --out-dir out/

Compares only at concurrency levels present in **every** run, and only after
:func:`loader.require_comparable` has confirmed the runs share hardware,
fixture, config, generation length and software stack.

VRAM IS NOT IN THE COMPARISON TABLE
-----------------------------------
It cannot be, and the separation is structural rather than a footnote.

Arm B's NVML peak is real demand: it grows with concurrency because each
in-flight request holds its own KV cache. Arm C's is flat, because vLLM
pre-allocates its KV pool to ``gpu_memory_utilization`` of the device at
startup — the number reports what the pool was *configured* to reserve and
barely moves with load.

Putting those two in adjacent columns produces a reading like "arm C uses 3x
the memory", which is false: it uses whatever it was told to reserve. So VRAM
is emitted in a **separate per-run section** with the config value alongside
it, and the comparison table carries no memory column at all.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.loader import (  # noqa: E402
    ProvenanceMismatch,
    RunSet,
    describe_provenance,
    load_runs,
    require_comparable,
    shared_concurrencies,
)

__all__ = ["METRICS", "build_table", "render_markdown", "render_csv", "main"]

# (key, label, unit, scale, decimals). scale converts summary seconds -> unit.
METRICS = (
    ("ttft_p50", "TTFT p50", "ms", 1000.0, 1),
    ("ttft_p95", "TTFT p95", "ms", 1000.0, 1),
    ("ttft_p99", "TTFT p99", "ms", 1000.0, 1),
    ("e2e_p50", "e2e p50", "ms", 1000.0, 1),
    ("e2e_p95", "e2e p95", "ms", 1000.0, 1),
    ("e2e_p99", "e2e p99", "ms", 1000.0, 1),
    ("itl_mean", "ITL mean", "ms", 1000.0, 2),
    ("itl_p95", "ITL p95", "ms", 1000.0, 2),
    ("input_tok_per_s", "input throughput", "tok/s", 1.0, 1),
    ("output_tok_per_s", "output throughput", "tok/s", 1.0, 1),
    ("total_tok_per_s", "total throughput", "tok/s", 1.0, 1),
    ("completed_requests_per_s", "completed", "req/s", 1.0, 3),
    ("wall_time", "wall time", "s", 1.0, 2),
    ("completed", "completed", "requests", 1.0, 0),
    ("errored", "errors", "requests", 1.0, 0),
)

# Pulled from level["resources"] rather than the level itself.
RESOURCE_METRICS = (
    ("util_gpu_mean_pct", "GPU utilisation mean", "%", 1.0, 1),
    ("cpu_mean_pct_across_cores", "CPU utilisation mean", "%", 1.0, 1),
)

VRAM_METRICS = (
    ("vram_peak_mb", "NVML peak", "MB", 0),
    ("vram_steady_state_mb", "NVML steady-state", "MB", 0),
)


def _get(level: Dict[str, Any], key: str) -> Optional[float]:
    if key in level:
        return level.get(key)
    res = level.get("resources") or {}
    return res.get(key)


def _fmt(value: Optional[float], scale: float, decimals: int) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * scale:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def build_table(runs: Sequence[RunSet], concurrency: int) -> List[Dict[str, Any]]:
    """Rows of {metric, unit, <run label>: formatted value, ...} for one level."""
    rows: List[Dict[str, Any]] = []
    for key, label, unit, scale, decimals in METRICS + RESOURCE_METRICS:
        row: Dict[str, Any] = {"metric": label, "unit": unit}
        for r in runs:
            lv = r.level(concurrency) or {}
            row[r.display] = _fmt(_get(lv, key), scale, decimals)
        rows.append(row)
    return rows


def _ratio_note(runs: Sequence[RunSet], concurrency: int, key: str) -> Optional[str]:
    """Fold-change on output throughput between exactly two runs."""
    if len(runs) != 2:
        return None
    a = _get(runs[0].level(concurrency) or {}, key)
    b = _get(runs[1].level(concurrency) or {}, key)
    if not a or not b:
        return None
    return f"{b / a:.2f}x" if a else None


def render_markdown(
    runs: Sequence[RunSet],
    levels: Sequence[int],
    waived: Sequence[str] = (),
) -> str:
    prov = describe_provenance(runs)
    common = prov["common"]
    out = io.StringIO()
    w = out.write

    w("# Matched arm comparison\n\n")

    if waived:
        w("> **PROVENANCE GUARD WAIVED** for: `" + "`, `".join(waived) + "`.\n")
        w("> These runs differ on the listed field(s). The comparison below is\n")
        w("> NOT a clean single-variable measurement — read every delta with\n")
        w("> that in mind.\n\n")

    w("## Provenance\n\n")
    w("| Field | Value |\n|---|---|\n")
    for key, label in (
        ("gpu_uuid", "GPU UUID"), ("gpu_name", "GPU"),
        ("driver_version", "Driver"), ("cuda_version", "CUDA (torch)"),
        ("model", "Model"), ("fixture_name", "Fixture"),
        ("fixture_sha256", "Fixture sha256"), ("config_sha256", "Config sha256"),
        ("pip_freeze_sha256", "pip freeze sha256"),
        ("max_new_tokens", "max_new_tokens"), ("context_tokens", "Context tokens"),
        ("upstream_sha", "Upstream SHA"), ("branch_sha", "Branch SHA"),
    ):
        value = common.get(key)
        w(f"| {label} | `{value}` |\n" if value is not None
          else f"| {label} | *(differs or absent)* |\n")
    w("\n")

    w("| Run | Arm | Levels | Layer |\n|---|---|---|---|\n")
    for r in prov["runs"]:
        w(f"| `{r['run_id']}` | {r['arm']} | {r['concurrencies']} | "
          f"{r['profiling_layer']} |\n")
    w("\n")

    if not levels:
        w("**No shared concurrency levels.** The runs have no level in common, "
          "so there is nothing to compare at matched load.\n\n")
        return out.getvalue()

    for c in levels:
        w(f"## Concurrency {c}\n\n")
        rows = build_table(runs, c)
        headers = ["Metric", "Unit"] + [r.display for r in runs]
        w("| " + " | ".join(headers) + " |\n")
        w("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            cells = [row["metric"], row["unit"]] + [row[r.display] for r in runs]
            w("| " + " | ".join(str(x) for x in cells) + " |\n")
        w("\n")
        ratio = _ratio_note(runs, c, "output_tok_per_s")
        if ratio:
            w(f"Output-throughput ratio ({runs[1].display} / {runs[0].display}): "
              f"**{ratio}**\n\n")

    w(_vram_section(runs, levels))
    return out.getvalue()


def _vram_section(runs: Sequence[RunSet], levels: Sequence[int]) -> str:
    """Per-run VRAM, deliberately NOT a joint table. See module docstring."""
    out = io.StringIO()
    w = out.write
    w("## Memory (reported separately — NOT comparable across arms)\n\n")
    w("These figures are **not** placed in one table on purpose.\n\n")
    w("- **Arm B** allocates on demand: NVML peak grows with concurrency "
      "because each in-flight request holds its own KV cache. The number is a "
      "measurement.\n")
    w("- **Arm C** pre-allocates: vLLM reserves `gpu_memory_utilization` of the "
      "device for its KV pool at startup, so NVML reports the *configured* "
      "reservation and stays flat under load. The number is a setting.\n\n")
    w("Reading them side by side suggests arm C \"uses more memory\". It does "
      "not — it reserves what it was told to. Arm C's demand signal is "
      "`notes.kv_cache.usage_perc_peak`, which requires the vLLM stats path to "
      "have resolved.\n\n")

    for r in runs:
        gmu = r.manifest.get("extra", {}).get("gpu_memory_utilization")
        if gmu is None:
            gmu = (r.summary.get("vllm") or {}).get("gpu_memory_utilization")
        w(f"### {r.display}\n\n")
        if gmu is not None:
            w(f"`gpu_memory_utilization = {gmu}` — the NVML figures below are "
              f"bounded by this setting, not by demand.\n\n")
        w("| Concurrency | " + " | ".join(m[1] for m in VRAM_METRICS) + " |\n")
        w("|" + "|".join(["---"] * (1 + len(VRAM_METRICS))) + "|\n")
        for c in levels:
            lv = r.level(c) or {}
            cells = [_fmt(_get(lv, key), 1.0, dec) for key, _, _, dec in VRAM_METRICS]
            w(f"| {c} | " + " | ".join(cells) + " |\n")
        w("\n")

        # A steady-state that never populated is a real, explainable gap.
        missing_steady = [
            c for c in levels
            if (r.level(c) or {}).get("resources", {}).get("vram_steady_state_mb") is None
        ]
        if missing_steady:
            w(f"> Steady-state is `—` at concurrency {missing_steady}: the level "
              f"finished before the monitor's warm-up window elapsed, so no "
              f"post-warm-up samples exist. Peak is still valid.\n\n")

        kv = None
        for c in levels:
            kv = ((r.level(c) or {}).get("notes") or {}).get("kv_cache")
            if kv:
                break
        if kv is not None:
            src = kv.get("stats_source")
            if not src:
                w("> KV-cache usage and prefix-cache hit rate are **absent**: the "
                  "vLLM stats path did not resolve on this build "
                  "(`stats_source: null`). Absent is not zero — no demand-side "
                  "memory number is available for this arm.\n\n")
    return out.getvalue()


def render_csv(runs: Sequence[RunSet], levels: Sequence[int]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["concurrency", "metric", "unit"] + [r.display for r in runs])
    for c in levels:
        for row in build_table(runs, c):
            writer.writerow([c, row["metric"], row["unit"]]
                            + [row[r.display] for r in runs])
    # VRAM rows are tagged so a consumer cannot accidentally pivot them into
    # the same view as the comparison metrics.
    for c in levels:
        for key, label, unit, dec in VRAM_METRICS:
            writer.writerow(
                [c, f"[NOT COMPARABLE ACROSS ARMS] {label}", unit]
                + [_fmt(_get(r.level(c) or {}, key), 1.0, dec) for r in runs]
            )
    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Matched arm-to-arm comparison.")
    ap.add_argument("summaries", nargs="+", help="paths to *_summary.json")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--basename", default="compare_arms")
    ap.add_argument(
        "--allow-mismatch", nargs="*", default=[],
        help="waive the provenance guard for these fields. Recorded in the "
             "output; use only for a deliberate cross-condition comparison.",
    )
    args = ap.parse_args(argv)

    if len(args.summaries) < 2:
        print("need at least two summaries to compare", file=sys.stderr)
        return 2

    runs = load_runs(args.summaries, args.labels)
    try:
        require_comparable(runs, allow=args.allow_mismatch)
    except ProvenanceMismatch as exc:
        print(str(exc), file=sys.stderr)
        return 3

    levels = shared_concurrencies(runs)
    md = render_markdown(runs, levels, waived=args.allow_mismatch)
    csv_text = render_csv(runs, levels)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        md_path = os.path.join(args.out_dir, f"{args.basename}.md")
        csv_path = os.path.join(args.out_dir, f"{args.basename}.csv")
        with open(md_path, "w") as fh:
            fh.write(md)
        with open(csv_path, "w") as fh:
            fh.write(csv_text)
        print(f"[compare] {md_path}")
        print(f"[compare] {csv_path}")
    else:
        print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
