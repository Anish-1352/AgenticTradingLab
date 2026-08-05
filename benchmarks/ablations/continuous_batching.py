#!/usr/bin/env python3
"""Ablation: continuous batching, isolated by constraining the scheduler.

Continuous batching is the lever named first in the study brief, and it is the
one arm B lacks entirely. It is also the hardest to ablate, because **in vLLM
it cannot be turned off** — the scheduler always admits and retires sequences
per step.

So it is isolated from the other side, by capping how many sequences the
scheduler may run at once:

    --max-num-seqs 1        one sequence resident at a time. The scheduler still
                            runs, but has nothing to batch: effectively
                            sequential decoding.
    --max-num-seqs <high>   engine default; full continuous batching.

Offered load is held identical between the two (the client still keeps N
requests outstanding via the semaphore) — what changes is only whether the
server may service them concurrently. That is as close to a clean
batching on/off as vLLM permits, and the residual difference from a true
"no scheduler" baseline is noted in the output rather than papered over.

PREFIX CACHING IS HELD OFF IN BOTH ARMS
---------------------------------------
Deliberately, and not negotiable through a flag. With caching on, the batched
condition would also enjoy cross-request prefix reuse that the sequential
condition largely would not, and the measured gap would be
batching + caching — attributed entirely to batching. Holding caching off makes
the comparison about scheduling alone. The caching effect is measured
separately, and bracketed, in ``prefix_cache.py``.

THE RESULT IS A CURVE, NOT A NUMBER
-----------------------------------
Batching has nothing to work with at concurrency 1 and progressively more as
offered load rises, so the benefit should *grow* with concurrency. The shape of
that growth — and where it saturates — is the finding. A single-concurrency
speedup would hide exactly the information that makes the result actionable for
capacity planning.
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

DEFAULT_LEVELS = (1, 4, 8, 15, 32, 64)


def _metrics(level: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not level:
        return {k: None for k in (
            "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "itl_mean",
            "output_tok_per_s", "total_tok_per_s", "completed_requests_per_s",
            "wall_time", "kv_usage_peak", "nvml_peak_mb", "completed", "errored",
        )}
    res = level.get("resources") or {}
    kv = (level.get("notes") or {}).get("kv_cache") or {}
    return {
        "ttft_p50": level.get("ttft_p50"),
        "ttft_p95": level.get("ttft_p95"),
        "e2e_p50": level.get("e2e_p50"),
        "e2e_p95": level.get("e2e_p95"),
        "itl_mean": level.get("itl_mean"),
        "output_tok_per_s": level.get("output_tok_per_s"),
        "total_tok_per_s": level.get("total_tok_per_s"),
        "completed_requests_per_s": level.get("completed_requests_per_s"),
        "wall_time": level.get("wall_time"),
        "completed": level.get("completed"),
        "errored": level.get("errored"),
        "kv_usage_peak": kv.get("usage_perc_peak"),
        "nvml_peak_mb": res.get("vram_peak_mb"),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concurrency", type=int, nargs="*", default=list(DEFAULT_LEVELS),
                    help="offered-load levels; the benefit curve is the result")
    ap.add_argument("--n-requests", type=int, default=105)
    ap.add_argument("--fixture", default="low_overlap",
                    help="low_overlap by default: with caching held off, a "
                         "high-overlap fixture would still let paged-attention "
                         "block reuse leak into the batching comparison")
    ap.add_argument("--out-dir", default=os.path.join(BENCH_ROOT, "results"))
    ap.add_argument("--fixtures-dir", default=os.path.join(BENCH_ROOT, "fixtures"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--batched-max-num-seqs", type=int, default=None,
                    help="cap for the batched condition; omit for engine default")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub-tokenizer", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    levels = list(args.concurrency)

    print("=" * 78)
    print("ABLATION: continuous batching (via scheduler cap)")
    print("=" * 78)
    print(f"levels             {levels}")
    print(f"requests per level {args.n_requests}")
    print(f"fixture            {args.fixture}")
    print("prefix caching     OFF in BOTH conditions (held constant, not a flag)")
    print("conditions         --max-num-seqs 1  vs  engine default")
    print("offered load is identical in both; only the server's freedom to")
    print("service requests concurrently differs.")

    runs: Dict[str, Dict[str, Any]] = {}
    for label, seqs in (("sequential", 1), ("batched", args.batched_max_num_seqs)):
        run_id = f"{stamp}-C-batch-{label}"
        extra = [
            "--fixture", args.fixture,
            "--concurrency", *[str(c) for c in levels],
            "--n-requests", str(args.n_requests),
            "--fixtures-dir", args.fixtures_dir,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            # Held constant so caching cannot confound the batching result.
            "--no-prefix-caching",
            "--no-resume",
        ]
        if seqs is not None:
            extra += ["--max-num-seqs", str(seqs)]
        if args.config:
            extra += ["--config", args.config]
        if args.max_new_tokens is not None:
            extra += ["--max-new-tokens", str(args.max_new_tokens)]
        if args.stub_tokenizer:
            extra += ["--stub-tokenizer"]

        print(f"\n--- condition: {label} "
              f"(max_num_seqs={seqs if seqs is not None else 'engine default'}) ---")
        rec = run_condition(ARM_C, run_id, extra, args.out_dir, dry_run=args.dry_run)
        rec["condition"] = label
        rec["max_num_seqs"] = seqs
        if not args.dry_run:
            summary = load_summary(rec["summary_path"])
            rec["by_level"] = {
                str(c): _metrics(level_for(summary, c)) for c in levels
            }
        runs[label] = rec

    if args.dry_run:
        print("\n--dry-run: two conditions planned, no engine constructed.")
        for label, rec in runs.items():
            print(f"  {label:12s} rc={rec['returncode']}  "
                  f"max_num_seqs={rec['max_num_seqs']}  id={rec['run_id']}")
        return 0 if all(r["ok"] for r in runs.values()) else 1

    seq_by = runs["sequential"].get("by_level", {})
    bat_by = runs["batched"].get("by_level", {})

    print("\n" + "=" * 78)
    print("BATCHING BENEFIT vs OFFERED LOAD")
    print("=" * 78)

    rows: List[List[str]] = []
    curve: Dict[str, Dict[str, Any]] = {}
    for c in levels:
        s = seq_by.get(str(c), {}) or {}
        b = bat_by.get(str(c), {}) or {}
        entry = {
            "throughput_gain_pct": pct_delta(
                b.get("output_tok_per_s"), s.get("output_tok_per_s")),
            "req_per_s_gain_pct": pct_delta(
                b.get("completed_requests_per_s"), s.get("completed_requests_per_s")),
            "wall_time_reduction_pct": pct_delta(
                b.get("wall_time"), s.get("wall_time")),
            "ttft_p50_delta_pct": pct_delta(b.get("ttft_p50"), s.get("ttft_p50")),
            "e2e_p50_delta_pct": pct_delta(b.get("e2e_p50"), s.get("e2e_p50")),
            "sequential": s,
            "batched": b,
        }
        curve[str(c)] = entry
        rows.append([
            str(c),
            fmt(s.get("output_tok_per_s"), " tok/s"),
            fmt(b.get("output_tok_per_s"), " tok/s"),
            fmt(entry["throughput_gain_pct"], "%"),
            fmt(entry["req_per_s_gain_pct"], "%"),
            fmt((s.get("ttft_p50") or 0) * 1000 if s.get("ttft_p50") else None, " ms"),
            fmt((b.get("ttft_p50") or 0) * 1000 if b.get("ttft_p50") else None, " ms"),
            fmt(entry["e2e_p50_delta_pct"], "%"),
        ])

    print(render_table(
        ["conc", "seq out/s", "batch out/s", "throughput gain", "req/s gain",
         "seq TTFT p50", "batch TTFT p50", "e2e p50 delta"],
        rows,
    ))

    gains = [(int(c), e["throughput_gain_pct"]) for c, e in curve.items()
             if e["throughput_gain_pct"] is not None]
    print("\n" + "=" * 78)
    print("READING THE CURVE")
    print("=" * 78)
    if len(gains) >= 2:
        gains.sort()
        lo_c, lo_g = gains[0]
        hi_c, hi_g = max(gains, key=lambda kv: kv[1])
        print(f"  lowest load   c={lo_c}: {fmt(lo_g, '%')} throughput gain")
        print(f"  peak benefit  c={hi_c}: {fmt(hi_g, '%')} throughput gain")
        print()
        if hi_g > lo_g:
            print("  The benefit GROWS with offered load, which is the expected")
            print("  shape: at low concurrency the scheduler has nothing to batch.")
            print(f"  Quote the curve, and c={hi_c} as where it peaked in this sweep —")
            print("  not a single speedup number.")
        else:
            print("  The benefit did NOT grow with load. That is unexpected and")
            print("  worth investigating before reporting: check the errored counts")
            print("  per level (a partially-OOMed level inflates apparent throughput)")
            print("  and confirm max_num_seqs=1 actually constrained the scheduler.")
    else:
        print("  Too few comparable levels to describe a curve. Check the run")
        print("  records for failures before drawing any conclusion.")

    payload = {
        "ablation": "continuous_batching",
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": levels,
        "n_requests": args.n_requests,
        "fixture": args.fixture,
        "prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "runs": runs,
        "curve": curve,
        "method_note": (
            "vLLM cannot disable continuous batching. The sequential condition "
            "caps max_num_seqs=1, so the scheduler still runs but has nothing to "
            "batch. This is not identical to a scheduler-free baseline; residual "
            "per-step scheduling overhead is present in BOTH conditions and so "
            "cancels from the delta, but the sequential arm is not the same thing "
            "as arm B's thread-per-request loop."
        ),
        "confound_control": (
            "Prefix caching is off in both conditions. With it on, the batched "
            "arm would additionally benefit from cross-request prefix reuse and "
            "the combined effect would be misattributed to batching alone."
        ),
    }
    path = write_comparison(args.out_dir, f"ablation_continuous_batching_{stamp}", payload)
    print(f"\n[write] {path}")
    return 0 if all(r["ok"] for r in runs.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
