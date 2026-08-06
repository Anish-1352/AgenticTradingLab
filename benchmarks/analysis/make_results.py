"""Generate ``analysis/RESULTS.md`` from the run JSONs. Single command.

    python -m analysis.make_results \
        --arm-b results/armB_shared_summary.json \
        --arm-c results/armC_shared_summary.json \
        --trace-analysis results/armB_L3_trace.json \
        --exclude "results/smoke_c1_summary.json=superseded: different stack AND different GPU" \
        --out analysis/RESULTS.md --figures-dir analysis/figures

RESULTS.md is **generated, never hand-edited**. Every number in it is read from
a ``*_summary.json`` and attributed to a ``run_id``; nothing is transcribed.
That is the whole point — a hand-maintained results document drifts from the
data the moment one run is re-done, and the v1 phase of this study is a record
of what that costs.

Caveats are **derived from the data** wherever possible rather than written by
hand: a null ``stats_source``, a level whose wall time was shorter than the
monitor's warm-up window, an absent steady-state VRAM figure. Anything the data
cannot know — such as why a superseded run was excluded — is passed in
explicitly via ``--exclude`` so that the exclusion appears in the document with
its reason instead of a run silently vanishing.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis import compare_arms  # noqa: E402
from analysis import raw_stats  # noqa: E402
from analysis.loader import (  # noqa: E402
    ProvenanceMismatch,
    RunSet,
    describe_provenance,
    load_run,
    require_comparable,
    shared_concurrencies,
)

__all__ = ["build_results", "main"]

# The metric list this study set out to produce, with the layer that yields
# each. "measured" is decided by inspecting the artifacts, not asserted here.
METRIC_PLAN: Tuple[Tuple[str, str, str], ...] = (
    ("TTFT p50/p95/p99", "1", "summary.levels[].ttft_p*"),
    ("Inter-token latency (mean/p95, full distribution)", "1", "summary + raw"),
    ("End-to-end latency p50/p95/p99", "1", "summary.levels[].e2e_p*"),
    ("Input / output / total token throughput", "1", "summary.levels[]"),
    ("Completed requests per second", "1", "summary.levels[]"),
    ("VRAM peak and steady-state", "1", "summary.levels[].resources"),
    ("CPU utilisation", "1", "summary.levels[].resources"),
    ("GPU utilisation (kernel residency)", "1", "summary.levels[].resources"),
    ("Marginal VRAM per request (slope)", "1", "vram_slope across levels"),
    ("Kernel timeline / stream overlap", "2/3", "trace_analysis"),
    ("cudaLaunchKernel / sync / memcpy CPU breakdown", "3", "trace_analysis"),
    ("Prefill vs decode kernel time split", "3", "trace_analysis + raw boundary"),
    ("SM activity over time (duration-weighted)", "2", "nsys gpu-metrics"),
    ("Achieved occupancy", "4", "ncu"),
    ("Tensor Core / HMMA pipe utilisation", "4", "ncu"),
    ("Memory bandwidth achieved vs peak", "4", "ncu"),
    ("vLLM KV cache usage", "1", "summary.levels[].notes.kv_cache"),
    ("Prefix cache hit rate", "1", "summary.levels[].notes.kv_cache"),
)


# --------------------------------------------------------------------------
# derived caveats
# --------------------------------------------------------------------------


def detect_caveats(runs: Sequence[RunSet]) -> List[Dict[str, str]]:
    """Read caveats out of the artifacts rather than writing them by hand."""
    out: List[Dict[str, str]] = []

    # 1. VRAM is never cross-comparable when any arm pre-allocates.
    preallocating = [
        r for r in runs
        if (r.manifest.get("extra") or {}).get("gpu_memory_utilization") is not None
        or (r.summary.get("vllm") or {}).get("gpu_memory_utilization") is not None
    ]
    if preallocating and len(runs) > 1:
        names = ", ".join(r.display for r in preallocating)
        out.append({
            "id": "vram-not-comparable",
            "title": "VRAM is not comparable across arms",
            "body": (
                f"{names} pre-allocates its KV pool to `gpu_memory_utilization` "
                f"at startup, so its NVML figure reports the *configured* "
                f"reservation and stays flat under load. The other arm's NVML "
                f"figure grows with concurrency and is real demand. They are "
                f"reported in separate sections and must never be placed in "
                f"adjacent columns."
            ),
        })

    # 2. vLLM stats path unresolved -> no KV / hit-rate numbers at all.
    for r in runs:
        for lv in r.levels:
            kv = (lv.get("notes") or {}).get("kv_cache")
            if kv is not None and not kv.get("stats_source"):
                out.append({
                    "id": "kv-stats-unresolved",
                    "title": "No KV-cache or prefix-cache-hit-rate numbers",
                    "body": (
                        f"`{r.run_id}` reports `stats_source: null` — the vLLM "
                        f"0.26 metrics path did not resolve through any of the "
                        f"access paths `VllmIntrospector` probes. **Absent is "
                        f"not zero.** No KV-cache utilisation and no prefix-"
                        f"cache hit rate exist for this arm, which also means "
                        f"the prefix-cache ablation has no demand-side number "
                        f"to report. See `stats_probe_attempts` in the manifest."
                    ),
                })
                break
        else:
            continue
        break

    # 3. Level shorter than the monitor's warm-up window -> no steady-state.
    for r in runs:
        for lv in r.levels:
            res = lv.get("resources") or {}
            wall = lv.get("wall_time")
            warm = res.get("warmup_s")
            if (res.get("vram_steady_state_mb") is None and wall is not None
                    and warm is not None and wall < warm):
                out.append({
                    "id": f"steady-state-missing-{r.run_id}-c{lv.get('concurrency')}",
                    "title": (
                        f"No steady-state VRAM for {r.display} at C="
                        f"{lv.get('concurrency')}"
                    ),
                    "body": (
                        f"That level ran for {wall:.2f}s, shorter than the "
                        f"monitor's {warm:.0f}s warm-up window, so zero "
                        f"post-warm-up samples were collected and steady-state "
                        f"is reported as `—`. Peak VRAM for the level is still "
                        f"valid. This is a property of the run's duration, not "
                        f"a measurement failure."
                    ),
                })

    # 4. Bounded profiler window.
    for r in runs:
        frac = r.manifest.get("profiled_window_fraction")
        if frac is not None and frac < 0.999:
            out.append({
                "id": f"bounded-window-{r.run_id}",
                "title": f"Layer 3 trace for {r.display} is a bounded window",
                "body": (
                    f"The profiler's active window covered "
                    f"{frac * 100:.1f}% of the run's wall time. Kernel counts "
                    f"and CPU-time totals from that trace describe the window, "
                    f"not the whole run, and must not be scaled up to it."
                ),
            })

    # 5. Incomplete manifests.
    for r in runs:
        if r.manifest and not r.manifest.get("manifest_complete", True):
            out.append({
                "id": f"manifest-incomplete-{r.run_id}",
                "title": f"Manifest incomplete for {r.display}",
                "body": (
                    "Some provenance fields could not be collected: "
                    + "; ".join(r.manifest.get("collection_errors") or [])
                ),
            })
    return out


def measured_status(runs: Sequence[RunSet],
                    trace_summary: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Decide measured / not measured by inspecting what the artifacts contain."""
    def any_level(key: str) -> bool:
        for r in runs:
            for lv in r.levels:
                if lv.get(key) is not None:
                    return True
                if (lv.get("resources") or {}).get(key) is not None:
                    return True
        return False

    def kv_present() -> bool:
        for r in runs:
            for lv in r.levels:
                kv = (lv.get("notes") or {}).get("kv_cache")
                if kv and kv.get("stats_source"):
                    return True
        return False

    has_trace = bool(trace_summary)
    has_split = bool(
        has_trace
        and (trace_summary.get("prefill_decode") or {}).get("method_confidence")
        == "measured"
    )
    slope = any(
        (r.summary.get("vram_slope") or {}).get("available") for r in runs
    )

    checks = {
        "TTFT p50/p95/p99": any_level("ttft_p50"),
        "Inter-token latency (mean/p95, full distribution)": any_level("itl_mean"),
        "End-to-end latency p50/p95/p99": any_level("e2e_p50"),
        "Input / output / total token throughput": any_level("output_tok_per_s"),
        "Completed requests per second": any_level("completed_requests_per_s"),
        "VRAM peak and steady-state": any_level("vram_peak_mb"),
        "CPU utilisation": any_level("cpu_mean_pct_across_cores"),
        "GPU utilisation (kernel residency)": any_level("util_gpu_mean_pct"),
        "Marginal VRAM per request (slope)": slope,
        "Kernel timeline / stream overlap": has_trace,
        "cudaLaunchKernel / sync / memcpy CPU breakdown": has_trace,
        "Prefill vs decode kernel time split": has_split,
        "SM activity over time (duration-weighted)": False,
        "Achieved occupancy": False,
        "Tensor Core / HMMA pipe utilisation": False,
        "Memory bandwidth achieved vs peak": False,
        "vLLM KV cache usage": kv_present(),
        "Prefix cache hit rate": kv_present(),
    }

    rows = []
    for metric, layer, source in METRIC_PLAN:
        ok = checks.get(metric, False)
        rows.append({
            "metric": metric,
            "layer": layer,
            "status": "measured" if ok else "not yet measured",
            "source": source,
        })
    return rows


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------

FIGURE_CAPTIONS = {
    "fig1_throughput_vs_concurrency": (
        "Output token throughput against offered concurrency, log y. The two "
        "arms move in opposite directions with load, which is the study's "
        "headline: adding concurrency to a stack with no batching does not add "
        "throughput."
    ),
    "fig2_latency_cdf": (
        "Empirical CDF of end-to-end latency. Percentiles compress away the "
        "shape of a distribution; the CDF shows whether latency is a tight "
        "spike or a broad smear, which p50/p95/p99 alone cannot distinguish."
    ),
    "fig3_gpu_utilisation_timeseries": (
        "NVML GPU utilisation over the level, both arms on shared axes with "
        "time normalised to level start. NVML utilisation is the fraction of "
        "time at least one kernel was resident — it is NOT achieved occupancy, "
        "which requires ncu (Layer 4)."
    ),
    "fig4_vram_slope": (
        "Arm B only: NVML peak VRAM against concurrency, with a least-squares "
        "fit. The slope is the marginal cost of one concurrent request; the "
        "intercept is the fixed cost of weights plus runtime. Arm C cannot "
        "appear — it pre-allocates its KV pool, so its VRAM is a configuration "
        "value with no slope to fit."
    ),
    "fig5_itl_distribution": (
        "Inter-token latency histogram at matched concurrency, log x. ITL "
        "measures the gap between consecutive tokens, so N tokens yield N-1 "
        "samples; the first interval is TTFT and is excluded."
    ),
    "fig6_launch_overhead": (
        "Arm B, Layer 3: wall time of the traced window against GPU busy time "
        "(the union of kernel intervals, so stream overlap is counted once) "
        "and cudaLaunchKernel CPU time. Launch cost exceeding GPU busy time "
        "means the host spent longer dispatching work than the device spent "
        "doing it."
    ),
}


def build_results(
    runs: Sequence[RunSet],
    trace_summary: Optional[Dict[str, Any]] = None,
    figures_dir: Optional[str] = None,
    figures: Optional[Dict[str, Any]] = None,
    excluded: Sequence[Tuple[str, str]] = (),
    waived: Sequence[str] = (),
    command: str = "",
) -> str:
    out = io.StringIO()
    w = out.write
    prov = describe_provenance(runs)
    common = prov["common"]
    levels = shared_concurrencies(runs)

    w("# GPU serving benchmark — results\n\n")
    w("> **Generated file. Do not edit.** Regenerate with:\n>\n")
    w(f"> ```bash\n> {command or 'python -m analysis.make_results ...'}\n> ```\n\n")
    w("Every number below is read from a `*_summary.json` and attributed to a "
      "`run_id`. Nothing is transcribed by hand.\n\n")

    # ---- provenance ----
    w("## 1. Provenance\n\n")
    w("| Field | Value |\n|---|---|\n")
    for key, label in (
        ("gpu_uuid", "GPU UUID"), ("gpu_name", "GPU"),
        ("driver_version", "Driver"), ("cuda_version", "CUDA (torch-linked)"),
        ("model", "Model"), ("fixture_name", "Fixture"),
        ("fixture_sha256", "Fixture sha256"), ("context_tokens", "Context tokens"),
        ("max_new_tokens", "max_new_tokens"), ("config_sha256", "Config sha256"),
        ("pip_freeze_sha256", "pip freeze sha256"),
        ("torchvision_shim", "torchvision shim active"),
        ("upstream_sha", "Upstream SHA"), ("branch_sha", "Branch SHA"),
    ):
        value = common.get(key)
        w(f"| {label} | `{value}` |\n" if value is not None
          else f"| {label} | *(differs across runs, or absent)* |\n")
    w("\n")

    w("### Runs included\n\n")
    w("| run_id | Arm | Levels | Layer | Timestamp |\n|---|---|---|---|---|\n")
    for r in prov["runs"]:
        w(f"| `{r['run_id']}` | {r['arm']} | {r['concurrencies']} | "
          f"{r['profiling_layer']} | {r['utc_timestamp']} |\n")
    w("\n")

    if excluded:
        w("### Runs excluded\n\n")
        w("Listed rather than silently dropped — an unexplained gap in the "
          "results is indistinguishable from a deleted bad result.\n\n")
        w("| Run | Reason for exclusion |\n|---|---|\n")
        for path, reason in excluded:
            w(f"| `{os.path.basename(path)}` | {reason} |\n")
        w("\n")

    if waived:
        w("> **Provenance guard waived** for: `" + "`, `".join(waived) + "`. "
          "The comparison below is not a clean single-variable measurement.\n\n")

    # ---- comparison ----
    w("## 2. Matched comparison\n\n")
    if not levels:
        w("No concurrency level is present in every run, so there is no matched "
          "comparison to make.\n\n")
    else:
        w(f"Compared at levels present in every run: **{levels}**.\n\n")
        for c in levels:
            w(f"### Concurrency {c}\n\n")
            rows = compare_arms.build_table(runs, c)
            headers = ["Metric", "Unit"] + [r.display for r in runs]
            w("| " + " | ".join(headers) + " |\n")
            w("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in rows:
                cells = [row["metric"], row["unit"]] + [row[r.display] for r in runs]
                w("| " + " | ".join(str(x) for x in cells) + " |\n")
            w("\n")

    w(compare_arms._vram_section(runs, levels))

    # ---- per-request ----
    w("## 3. Per-request behaviour\n\n")
    w("From `*_raw.json`; not derivable from the summary.\n\n")
    for r in runs:
        for c in (levels[-1:] if levels else []):
            try:
                res = raw_stats.analyse(r, concurrency=c)
            except FileNotFoundError as exc:
                w(f"- `{r.run_id}` C={c}: raw records unavailable ({exc})\n")
                continue
            t = res["itl_trajectory"]
            o = res["ttft_vs_order"]
            v = res["output_lengths"]
            w(f"### {r.display} — C={c}\n\n")
            if t.get("available"):
                w(f"- **ITL trajectory:** {t['verdict']} "
                  f"(first decile {t['first_bin_ms']:.1f} ms -> last "
                  f"{t['last_bin_ms']:.1f} ms, drift {t['drift_pct']:+.1f}%)\n")
            if o.get("available"):
                rho = o["spearman_submit_vs_ttft"]
                w(f"- **Submission-order effect:** {o['verdict']}"
                  + (f" (Spearman rho {rho:.3f})\n" if rho is not None else "\n"))
            if v.get("available"):
                w(f"- **ignore_eos check:** "
                  + ("every request emitted exactly "
                     f"{v['expected']} tokens\n" if v["all_equal_expected"]
                     else f"**{v['mismatch_count']} request(s)** differ from "
                          f"max_new_tokens={v['expected']} — output throughput "
                          f"is not purely a stack property here\n"))
            w("\n")

    # ---- figures ----
    w("## 4. Figures\n\n")
    if figures_dir:
        made = (figures or {}).get("figures", {})
        skipped = (figures or {}).get("skipped", {})
        for name, caption in FIGURE_CAPTIONS.items():
            w(f"### {name}\n\n")
            if name in made:
                w(f"![{name}]({os.path.join(figures_dir, name + '.png')})\n\n")
                w(f"{caption}\n\n")
            else:
                why = skipped.get(name, "not generated")
                w(f"*Not generated: {why}*\n\n")
    else:
        w("*No figures directory supplied.*\n\n")

    # ---- measured vs not ----
    w("## 5. Measured vs not yet measured\n\n")
    w("Assessed by inspecting the artifacts, not asserted.\n\n")
    w("| Metric | Layer | Status | Source |\n|---|---|---|---|\n")
    for row in measured_status(runs, trace_summary):
        mark = "measured" if row["status"] == "measured" else "**not yet measured**"
        w(f"| {row['metric']} | {row['layer']} | {mark} | `{row['source']}` |\n")
    w("\n")

    # ---- caveats ----
    w("## 6. Caveats\n\n")
    caveats = detect_caveats(runs)
    if not caveats:
        w("No automatically-detectable caveats.\n\n")
    for i, cav in enumerate(caveats, 1):
        w(f"### 6.{i} {cav['title']}\n\n{cav['body']}\n\n")

    if trace_summary:
        pd = trace_summary.get("prefill_decode") or {}
        w(f"### 6.{len(caveats) + 1} Prefill/decode split provenance\n\n")
        w(f"Method: `{pd.get('method')}` "
          f"(confidence: {pd.get('method_confidence')}).\n\n")
        if pd.get("caveat"):
            w(f"{pd['caveat']}\n\n")
        if pd.get("method_confidence", "").startswith("inferred"):
            w("This split came from the kernel-name heuristic, whose own caveat "
              "says not to quote it. Supply `--raw` to `common.trace_analysis` "
              "so the boundary is measured from `t_first_token` instead.\n\n")

    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate analysis/RESULTS.md.")
    ap.add_argument("--arm-b", default=None)
    ap.add_argument("--arm-c", default=None)
    ap.add_argument("--summary", nargs="*", default=[],
                    help="additional summaries beyond --arm-b/--arm-c")
    ap.add_argument("--trace-analysis", default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "RESULTS.md"))
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--render-figures", action="store_true",
                    help="also render the figures (needs matplotlib)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="level for the CDF / utilisation / ITL figures; "
                         "defaults to the highest shared level")
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="PATH=REASON",
                    help="record an excluded run and why, repeatable")
    ap.add_argument("--allow-mismatch", nargs="*", default=[])
    args = ap.parse_args(argv)

    runs: List[RunSet] = []
    if args.arm_b:
        runs.append(load_run(args.arm_b, label="Arm B (HF threads)"))
    if args.arm_c:
        runs.append(load_run(args.arm_c, label="Arm C (vLLM)"))
    for p in args.summary:
        runs.append(load_run(p))
    if not runs:
        print("supply --arm-b and/or --arm-c", file=sys.stderr)
        return 2

    try:
        require_comparable(runs, allow=args.allow_mismatch)
    except ProvenanceMismatch as exc:
        print(str(exc), file=sys.stderr)
        return 3

    trace_summary = None
    if args.trace_analysis:
        with open(args.trace_analysis) as fh:
            trace_summary = json.load(fh)

    excluded: List[Tuple[str, str]] = []
    for item in args.exclude:
        path, _, reason = item.partition("=")
        excluded.append((path, reason or "no reason given"))

    figures = None
    if args.render_figures and args.figures_dir:
        from analysis import plots  # noqa: PLC0415

        levels = shared_concurrencies(runs)
        c = args.concurrency or (levels[-1] if levels else 1)
        arm_b = next((r for r in runs if r.arm == "B"), None)
        arm_c = next((r for r in runs if r.arm == "C"), None)
        figures = plots.render_all(arm_b, arm_c, args.figures_dir,
                                   cdf_concurrency=c, trace_summary=trace_summary)

    command = "python -m analysis.make_results " + " ".join(sys.argv[1:])
    doc = build_results(
        runs, trace_summary=trace_summary, figures_dir=args.figures_dir,
        figures=figures, excluded=excluded, waived=args.allow_mismatch,
        command=command,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(doc)
    print(f"[results] {args.out}")
    if figures:
        for name in figures["figures"]:
            print(f"[results] figure {name}")
        for name, why in figures["skipped"].items():
            print(f"[results] SKIPPED {name}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
