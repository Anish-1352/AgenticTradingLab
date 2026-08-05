#!/usr/bin/env python3
"""Ablation: vLLM prefix caching, across both fixture families.

Runs arm C **four** times as fresh subprocesses:

    shared_prefix + caching ON      shared_prefix + caching OFF
    low_overlap   + caching ON      low_overlap   + caching OFF

Everything else — concurrency, request count, max_new_tokens, model, seed — is
held identical.

THE HEADLINE IS THE GAP BETWEEN FIXTURES, NOT A SINGLE NUMBER
-------------------------------------------------------------
Quoting one prefix-caching speedup would be the single most misleading thing
this study could produce, because the number is almost entirely a property of
how much prefix the workload happens to share.

    shared_prefix   99.4% common prefix  ->  a deliberate UPPER BOUND
    low_overlap      0.3% common prefix  ->  the FLOOR

Real agent traffic sits **between** them. An ATL fleet where every agent carries
the same system prompt and market-context block but a different ticker leans
toward the upper bound; one where agents read distinct filings leans toward the
floor. Neither endpoint is a forecast of production.

So the report presents the pair as a **bracket** and says so explicitly. The
deliverable of this ablation is the interval, plus the observation that the
distance between its ends is itself the sensitivity of the result to workload
composition.

WHY NVML VRAM IS NOT THE MEMORY ANSWER HERE
-------------------------------------------
vLLM pre-allocates its KV pool to ``gpu_memory_utilization`` at startup, so NVML
process VRAM is roughly constant across all four conditions regardless of what
caching does. It is reported for continuity with arm B, but the *demand* signal
is ``kv_cache.usage_perc_peak``, and the direct measurement of the mechanism is
``prefix_cache_hit_rate``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ablation_util import (  # noqa: E402
    ARM_C,
    BENCH_ROOT,
    fmt,
    level_for,
    load_summary,
    pct_delta,
    render_table,
    run_condition,
    write_comparison,
)

FIXTURES = ("shared_prefix", "low_overlap")


def _metrics(level: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten one level into the quantities this ablation compares."""
    if not level:
        return {k: None for k in (
            "ttft_p50", "ttft_p95", "ttft_p99", "e2e_p50",
            "output_tok_per_s", "input_tok_per_s", "total_tok_per_s",
            "completed_requests_per_s", "itl_mean",
            "nvml_peak_mb", "nvml_steady_mb",
            "kv_usage_peak", "kv_usage_mean", "prefix_hit_rate",
            "completed", "errored",
        )}
    res = level.get("resources") or {}
    kv = (level.get("notes") or {}).get("kv_cache") or {}
    return {
        "ttft_p50": level.get("ttft_p50"),
        "ttft_p95": level.get("ttft_p95"),
        "ttft_p99": level.get("ttft_p99"),
        "e2e_p50": level.get("e2e_p50"),
        "itl_mean": level.get("itl_mean"),
        "output_tok_per_s": level.get("output_tok_per_s"),
        "input_tok_per_s": level.get("input_tok_per_s"),
        "total_tok_per_s": level.get("total_tok_per_s"),
        "completed_requests_per_s": level.get("completed_requests_per_s"),
        "completed": level.get("completed"),
        "errored": level.get("errored"),
        "nvml_peak_mb": res.get("vram_peak_mb"),
        "nvml_steady_mb": res.get("vram_steady_state_mb"),
        "kv_usage_peak": kv.get("usage_perc_peak"),
        "kv_usage_mean": kv.get("usage_perc_mean"),
        "prefix_hit_rate": kv.get("prefix_cache_hit_rate_final"),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concurrency", type=int, default=15,
                    help="single offered-load level, identical in all four runs")
    ap.add_argument("--n-requests", type=int, default=105)
    ap.add_argument("--out-dir", default=os.path.join(BENCH_ROOT, "results"))
    ap.add_argument("--fixtures-dir", default=os.path.join(BENCH_ROOT, "fixtures"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tag", default=None, help="suffix for run ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub-tokenizer", action="store_true",
                    help="dry-run only; passes through to the runner")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    c = args.concurrency

    print("=" * 78)
    print("ABLATION: vLLM prefix caching")
    print("=" * 78)
    print(f"concurrency        {c} (identical in all four runs)")
    print(f"requests per run   {args.n_requests}")
    print(f"conditions         {{shared_prefix, low_overlap}} x {{caching ON, OFF}}")
    print("each condition is a FRESH subprocess: new CUDA context, new KV pool,")
    print("new prefix cache — so no run can contaminate another.")

    runs: Dict[str, Dict[str, Any]] = {}
    for fixture in FIXTURES:
        for caching in (True, False):
            key = f"{fixture}__{'on' if caching else 'off'}"
            run_id = f"{stamp}-C-{fixture[:6]}-cache{'ON' if caching else 'OFF'}-c{c}"
            extra = [
                "--fixture", fixture,
                "--concurrency", str(c),
                "--n-requests", str(args.n_requests),
                "--fixtures-dir", args.fixtures_dir,
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--enable-prefix-caching" if caching else "--no-prefix-caching",
                "--no-resume",
            ]
            if args.config:
                extra += ["--config", args.config]
            if args.max_new_tokens is not None:
                extra += ["--max-new-tokens", str(args.max_new_tokens)]
            if args.stub_tokenizer:
                extra += ["--stub-tokenizer"]

            print(f"\n--- {fixture} / prefix caching {'ON' if caching else 'OFF'} ---")
            rec = run_condition(ARM_C, run_id, extra, args.out_dir, dry_run=args.dry_run)
            rec["fixture"] = fixture
            rec["caching"] = caching
            if not args.dry_run:
                rec["metrics"] = _metrics(level_for(load_summary(rec["summary_path"]), c))
            runs[key] = rec

    if args.dry_run:
        print("\n--dry-run: four conditions planned, no engine constructed.")
        for key, rec in runs.items():
            print(f"  {key:28s} rc={rec['returncode']}  id={rec['run_id']}")
        return 0 if all(r["ok"] for r in runs.values()) else 1

    # ---- per-fixture caching effect ----
    print("\n" + "=" * 78)
    print("CACHING EFFECT, WITHIN EACH FIXTURE")
    print("=" * 78)

    rows: List[List[str]] = []
    effects: Dict[str, Dict[str, Any]] = {}
    for fixture in FIXTURES:
        on = runs[f"{fixture}__on"].get("metrics") or {}
        off = runs[f"{fixture}__off"].get("metrics") or {}
        eff = {
            "ttft_p50_delta_pct": pct_delta(on.get("ttft_p50"), off.get("ttft_p50")),
            "ttft_p95_delta_pct": pct_delta(on.get("ttft_p95"), off.get("ttft_p95")),
            "output_tok_per_s_delta_pct": pct_delta(
                on.get("output_tok_per_s"), off.get("output_tok_per_s")),
            "req_per_s_delta_pct": pct_delta(
                on.get("completed_requests_per_s"), off.get("completed_requests_per_s")),
            "kv_usage_peak_delta_pct": pct_delta(
                on.get("kv_usage_peak"), off.get("kv_usage_peak")),
            "hit_rate_on": on.get("prefix_hit_rate"),
            "hit_rate_off": off.get("prefix_hit_rate"),
        }
        effects[fixture] = eff
        for label, m in (("caching ON", on), ("caching OFF", off)):
            rows.append([
                fixture, label,
                fmt((m.get("ttft_p50") or 0) * 1000 if m.get("ttft_p50") else None, " ms"),
                fmt((m.get("ttft_p95") or 0) * 1000 if m.get("ttft_p95") else None, " ms"),
                fmt((m.get("ttft_p99") or 0) * 1000 if m.get("ttft_p99") else None, " ms"),
                fmt(m.get("output_tok_per_s"), " tok/s"),
                fmt(m.get("completed_requests_per_s"), "", 3),
                fmt(m.get("nvml_peak_mb"), " MB", 0),
                fmt(m.get("kv_usage_peak"), "", 3),
                fmt(m.get("prefix_hit_rate"), "", 3),
            ])

    print(render_table(
        ["fixture", "condition", "TTFT p50", "TTFT p95", "TTFT p99",
         "out tok/s", "req/s", "NVML peak", "KV peak", "hit rate"],
        rows,
    ))

    print("\ndeltas (caching ON relative to OFF, same fixture):")
    print(render_table(
        ["fixture", "TTFT p50", "TTFT p95", "out tok/s", "req/s", "KV peak"],
        [[f,
          fmt(effects[f]["ttft_p50_delta_pct"], "%"),
          fmt(effects[f]["ttft_p95_delta_pct"], "%"),
          fmt(effects[f]["output_tok_per_s_delta_pct"], "%"),
          fmt(effects[f]["req_per_s_delta_pct"], "%"),
          fmt(effects[f]["kv_usage_peak_delta_pct"], "%")]
         for f in FIXTURES],
    ))

    # ---- the bracket ----
    upper = effects["shared_prefix"]
    lower = effects["low_overlap"]

    print("\n" + "=" * 78)
    print("HEADLINE: THE BRACKET")
    print("=" * 78)
    print("Prefix caching's benefit is a property of how much prefix the workload")
    print("shares. These two fixtures are the ENDS of the range, not predictions:")
    print()
    print(f"  UPPER BOUND  shared_prefix (99.4% common prefix)")
    print(f"               TTFT p50 {fmt(upper['ttft_p50_delta_pct'], '%')}"
          f"   throughput {fmt(upper['output_tok_per_s_delta_pct'], '%')}"
          f"   hit rate {fmt(upper['hit_rate_on'], '', 3)}")
    print(f"  LOWER BOUND  low_overlap (0.3% common prefix)")
    print(f"               TTFT p50 {fmt(lower['ttft_p50_delta_pct'], '%')}"
          f"   throughput {fmt(lower['output_tok_per_s_delta_pct'], '%')}"
          f"   hit rate {fmt(lower['hit_rate_on'], '', 3)}")
    print()
    print("Real ATL agent traffic sits BETWEEN these. Agents sharing a system")
    print("prompt and market-context block but differing in ticker approach the")
    print("upper bound; agents reading distinct filings approach the floor.")
    print("Quote the interval. A single prefix-caching number, taken from either")
    print("end, would not generalize to production and should not be presented as")
    print("if it did.")

    payload = {
        "ablation": "prefix_cache",
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "concurrency": c,
        "n_requests": args.n_requests,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "runs": runs,
        "effects_by_fixture": effects,
        "bracket": {
            "upper_bound_fixture": "shared_prefix",
            "lower_bound_fixture": "low_overlap",
            "upper": upper,
            "lower": lower,
            "interpretation": (
                "shared_prefix (99.4% common prefix) is a deliberate upper bound "
                "and low_overlap (0.3%) is the floor. Real agent traffic lies "
                "between them. Report the interval, not a point estimate — the "
                "distance between the ends is the result's sensitivity to "
                "workload composition."
            ),
        },
        "memory_caveat": (
            "NVML VRAM is roughly constant across conditions because vLLM "
            "pre-allocates the KV pool to gpu_memory_utilization at startup. "
            "kv_usage_peak is the demand signal; prefix_hit_rate is the direct "
            "measurement of the mechanism."
        ),
    }
    path = write_comparison(args.out_dir, f"ablation_prefix_cache_{stamp}", payload)
    print(f"\n[write] {path}")
    return 0 if all(r["ok"] for r in runs.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
