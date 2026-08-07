"""Per-request analysis from ``*_raw.json`` that the summary cannot express.

    python -m analysis.raw_stats run_summary.json --concurrency 32

The summary reports percentiles. Percentiles cannot answer three questions that
matter for the mechanism claim:

1. **Does inter-token latency degrade as a request generates, or is it
   flat-but-slow?** A rising trajectory means progressive queue build-up —
   pressure accumulating as the run proceeds. A flat trajectory means steady
   serialisation — every token pays the same contention cost from the first one.
   These imply different bottlenecks and different fixes, and a mean ITL is
   identical under both.

2. **Did later-submitted requests starve?** If TTFT climbs with submission
   order, the queue is FIFO-ish and late arrivals simply wait. If it does not,
   the cost is spread evenly and no request is being sacrificed.

3. **Did `ignore_eos` actually hold?** Every request must emit exactly
   ``max_new_tokens``. If some stopped early, output-token throughput is partly
   a function of the prompt rather than the serving stack, and cross-arm
   comparison of it is invalid.

Pure stdlib. No GPU, no torch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.loader import RunSet, load_run  # noqa: E402
from common.metrics import percentile  # noqa: E402

__all__ = [
    "distribution",
    "itl_trajectory",
    "ttft_vs_order",
    "verify_output_lengths",
    "analyse",
    "format_report",
]

DIST_PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _itls(record: Dict[str, Any]) -> List[float]:
    """N timestamps -> N-1 gaps. Matches common.metrics.RequestRecord.itl."""
    ts = record.get("per_token_timestamps") or []
    if len(ts) < 2:
        return []
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def _ok(record: Dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    return record.get("t_done") is not None


def _linfit(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    """Least-squares slope/intercept/r^2. Returns availability rather than raising."""
    n = len(xs)
    if n < 2:
        return {"available": False, "reason": "need at least 2 points"}
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"available": False, "reason": "no variance in x"}
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return {
        "available": True,
        "slope": slope,
        "intercept": intercept,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n": n,
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, implemented here to avoid a scipy dependency."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank for ties
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return (num / den) if den else None


# --------------------------------------------------------------------------
# 1. full distribution
# --------------------------------------------------------------------------


def distribution(values: Sequence[float], unit_scale: float = 1000.0) -> Dict[str, Any]:
    """Every percentile plus shape, not just p50/p95/p99."""
    vals = [v * unit_scale for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "cv_pct": (statistics.stdev(vals) / statistics.fmean(vals) * 100.0)
        if len(vals) > 1 and statistics.fmean(vals) else None,
        "percentiles": {f"p{p}": percentile(vals, p) for p in DIST_PERCENTILES},
        # max/min ratio makes a broad distribution obvious at a glance in a way
        # p50/p95/p99 alone does not.
        "spread_ratio": (max(vals) / min(vals)) if min(vals) > 0 else None,
    }


# --------------------------------------------------------------------------
# 2. ITL trajectory within a request
# --------------------------------------------------------------------------


def itl_trajectory(records: Sequence[Dict[str, Any]], bins: int = 10) -> Dict[str, Any]:
    """Mean ITL by normalised position within the request.

    Answers: does generation get slower as it proceeds (progressive queue
    build-up) or is it uniformly slow (steady serialisation)?
    """
    ok = [r for r in records if _ok(r)]
    per_bin: List[List[float]] = [[] for _ in range(bins)]
    all_pairs: List[Tuple[float, float]] = []

    for rec in ok:
        gaps = _itls(rec)
        if len(gaps) < bins:
            continue
        for idx, gap in enumerate(gaps):
            frac = idx / (len(gaps) - 1) if len(gaps) > 1 else 0.0
            b = min(int(frac * bins), bins - 1)
            per_bin[b].append(gap)
            all_pairs.append((frac, gap))

    if not all_pairs:
        return {"available": False,
                "reason": f"no request produced at least {bins} inter-token gaps"}

    bin_means = [statistics.fmean(v) * 1000.0 if v else None for v in per_bin]
    fit = _linfit([p[0] for p in all_pairs], [p[1] * 1000.0 for p in all_pairs])

    present = [m for m in bin_means if m is not None]
    first = present[0] if present else None
    last = present[-1] if present else None
    drift_pct = ((last - first) / first * 100.0) if (first and last) else None

    # The first bin is frequently an outlier — a batch is still filling, or the
    # first tokens escape before contention builds — so an endpoint-only drift
    # can label a flat trajectory as "degrading" on the strength of one bin.
    # Recomputing without it distinguishes a genuine trend from a start-up
    # transient.
    second = present[1] if len(present) > 1 else None
    drift_excl_first = (
        ((last - second) / second * 100.0) if (second and last) else None
    )

    # Deliberately blunt thresholds: they prompt the judgement, they do not
    # replace it. Bin means are printed so the reader can disagree.
    verdict = "indeterminate"
    if drift_pct is not None:
        if abs(drift_pct) < 10.0:
            verdict = "flat-but-slow (steady serialisation)"
        elif drift_excl_first is not None and abs(drift_excl_first) < 10.0:
            verdict = (
                "flat after an initial transient — the first bin differs, the "
                "rest is steady (NOT progressive build-up)"
            )
        elif drift_pct > 0:
            verdict = "degrading (progressive build-up)"
        else:
            verdict = "improving over the request"

    return {
        "available": True,
        "bins": bins,
        "bin_mean_itl_ms": bin_means,
        "bin_sample_counts": [len(v) for v in per_bin],
        "first_bin_ms": first,
        "last_bin_ms": last,
        "drift_pct": drift_pct,
        "drift_pct_excl_first_bin": drift_excl_first,
        "fit_vs_position": fit,
        "verdict": verdict,
        "requests_used": len([r for r in ok if len(_itls(r)) >= bins]),
        "interpretation": (
            "drift_pct is the change in mean inter-token latency from the first "
            "tenth of a request to the last. Near zero means every token pays "
            "the same contention cost (serialisation). Strongly positive means "
            "pressure accumulates as the run proceeds (queue build-up). The two "
            "imply different bottlenecks and are indistinguishable from a mean. "
            "drift_pct_excl_first_bin repeats the calculation without the first "
            "decile, which is often a start-up transient rather than a trend."
        ),
    }


# --------------------------------------------------------------------------
# 3. TTFT vs submission / completion order
# --------------------------------------------------------------------------


def ttft_vs_order(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did later-submitted requests wait longer, or was cost spread evenly?"""
    ok = [r for r in records if _ok(r) and r.get("t_first_token") is not None]
    if len(ok) < 3:
        return {"available": False, "reason": "need at least 3 completed requests"}

    by_submit = sorted(ok, key=lambda r: r["t_submit"])
    submit_idx = list(range(len(by_submit)))
    ttfts = [(r["t_first_token"] - r["t_submit"]) * 1000.0 for r in by_submit]

    # Completion rank of each request, in submission order: a value far from
    # its submission index means the queue reordered it.
    done_rank = {id(r): i for i, r in enumerate(sorted(ok, key=lambda r: r["t_done"]))}
    completion_idx = [done_rank[id(r)] for r in by_submit]

    rho = _spearman(submit_idx, ttfts)
    fit = _linfit([float(i) for i in submit_idx], ttfts)

    starving = rho is not None and rho > 0.5
    reversed_order = rho is not None and rho < -0.5
    return {
        "available": True,
        "n": len(ok),
        "spearman_submit_vs_ttft": rho,
        "fit_ttft_vs_submit_index": fit,
        "first_ttft_ms": ttfts[0],
        "last_ttft_ms": ttfts[-1],
        "order_preserved_fraction": (
            sum(1 for i, c in zip(submit_idx, completion_idx) if i == c) / len(ok)
        ),
        "verdict": (
            "later requests waited progressively longer (FIFO queueing)"
            if starving else
            "later requests were served FASTER — TTFT falls monotonically with "
            "submission order, consistent with joining an already-warm batch "
            "rather than queueing behind one"
            if reversed_order else
            "no strong submission-order penalty; cost spread across requests"
        ),
        "interpretation": (
            "Spearman rho near +1 means TTFT rises monotonically with "
            "submission order — late arrivals queue behind early ones. Near 0 "
            "means the contention cost is shared rather than borne by the tail."
        ),
    }


# --------------------------------------------------------------------------
# 4. ignore_eos verification
# --------------------------------------------------------------------------


def verify_output_lengths(
    records: Sequence[Dict[str, Any]], expected: Optional[int]
) -> Dict[str, Any]:
    """Every request must emit exactly ``max_new_tokens``.

    If not, output-token throughput is partly a function of the prompt rather
    than the serving stack, and comparing it across arms is invalid.
    """
    ok = [r for r in records if _ok(r)]
    lengths = [int(r.get("output_tokens") or 0) for r in ok]
    if not lengths:
        return {"available": False, "reason": "no completed requests"}

    counts: Dict[int, int] = {}
    for n in lengths:
        counts[n] = counts.get(n, 0) + 1

    mismatches = []
    if expected is not None:
        mismatches = [
            {"request_id": r.get("request_id"), "output_tokens": r.get("output_tokens")}
            for r in ok if int(r.get("output_tokens") or 0) != int(expected)
        ]

    return {
        "available": True,
        "expected": expected,
        "n": len(lengths),
        "distinct_lengths": sorted(counts),
        "length_counts": {str(k): v for k, v in sorted(counts.items())},
        "all_equal_expected": expected is not None and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "note": (
            "ignore_eos held; output length is a controlled variable."
            if expected is not None and not mismatches else
            "Output lengths differ from max_new_tokens — output_tok_per_s is "
            "NOT purely a property of the serving stack for these runs."
        ),
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _records_for_level(run: RunSet, concurrency: Optional[int]) -> List[Dict[str, Any]]:
    """Records belonging to one level.

    ``_raw.json`` accumulates every level of a sweep, so a run covering more
    than one level must be split by time window or the distributions blend two
    different load conditions into one.
    """
    records = run.records()
    if concurrency is None or len(run.concurrencies()) <= 1:
        return records

    # Levels run sequentially; bound this level by the wall window implied by
    # its own summary, using submit/done times.
    ordered = sorted((r for r in records if r.get("t_submit") is not None),
                     key=lambda r: r["t_submit"])
    level = run.level(concurrency)
    if level is None:
        return []
    want = int(level.get("requested") or level.get("completed") or 0)
    if not want:
        return ordered

    idx = run.concurrencies().index(concurrency)
    start = 0
    for c in run.concurrencies()[:idx]:
        prev = run.level(c) or {}
        start += int(prev.get("requested") or prev.get("completed") or 0)
    return ordered[start:start + want]


def analyse(
    run: RunSet, concurrency: Optional[int] = None, bins: int = 10
) -> Dict[str, Any]:
    records = _records_for_level(run, concurrency)
    ok = [r for r in records if _ok(r)]

    e2e = [r["t_done"] - r["t_submit"] for r in ok]
    ttft = [r["t_first_token"] - r["t_submit"] for r in ok
            if r.get("t_first_token") is not None]
    itl_all: List[float] = []
    for r in ok:
        itl_all.extend(_itls(r))

    expected = run.manifest.get("max_new_tokens")

    return {
        "run_id": run.run_id,
        "arm": run.arm,
        "concurrency": concurrency,
        "records_considered": len(records),
        "completed": len(ok),
        "errored": len(records) - len(ok),
        "e2e_ms": distribution(e2e),
        "ttft_ms": distribution(ttft),
        "itl_ms": distribution(itl_all),
        "itl_trajectory": itl_trajectory(records, bins=bins),
        "ttft_vs_order": ttft_vs_order(records),
        "output_lengths": verify_output_lengths(records, expected),
    }


def format_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A(f"PER-REQUEST ANALYSIS — {result['run_id']} (arm {result['arm']}, "
      f"concurrency {result['concurrency']})")
    A("=" * 78)
    A(f"  records {result['records_considered']}  completed {result['completed']}"
      f"  errored {result['errored']}")

    for key, label in (("e2e_ms", "e2e latency"), ("ttft_ms", "TTFT"),
                       ("itl_ms", "inter-token latency")):
        d = result[key]
        if not d.get("n"):
            continue
        A("")
        A(f"  {label} (ms), n={d['n']}")
        p = d["percentiles"]
        A(f"    min {p['p0']:.1f}  p25 {p['p25']:.1f}  p50 {p['p50']:.1f}  "
          f"p75 {p['p75']:.1f}  p95 {p['p95']:.1f}  p99 {p['p99']:.1f}  "
          f"max {p['p100']:.1f}")
        cv = d.get("cv_pct")
        A(f"    mean {d['mean']:.1f}  sd {d['stdev']:.1f}"
          + (f"  cv {cv:.1f}%" if cv is not None else "")
          + (f"  max/min {d['spread_ratio']:.1f}x"
             if d.get("spread_ratio") else ""))

    t = result["itl_trajectory"]
    A("")
    A("  ITL trajectory within a request")
    if not t.get("available"):
        A(f"    unavailable: {t.get('reason')}")
    else:
        A(f"    bin means (ms): " + ", ".join(
            "—" if m is None else f"{m:.1f}" for m in t["bin_mean_itl_ms"]))
        A(f"    first {t['first_bin_ms']:.1f} ms -> last {t['last_bin_ms']:.1f} ms"
          f"  (drift {t['drift_pct']:+.1f}%)")
        A(f"    VERDICT: {t['verdict']}")

    o = result["ttft_vs_order"]
    A("")
    A("  TTFT vs submission order")
    if not o.get("available"):
        A(f"    unavailable: {o.get('reason')}")
    else:
        rho = o["spearman_submit_vs_ttft"]
        A(f"    spearman rho {rho:.3f}" if rho is not None else "    rho —")
        A(f"    first {o['first_ttft_ms']:.1f} ms -> last {o['last_ttft_ms']:.1f} ms")
        A(f"    VERDICT: {o['verdict']}")

    v = result["output_lengths"]
    A("")
    A("  ignore_eos check")
    if not v.get("available"):
        A(f"    unavailable: {v.get('reason')}")
    else:
        A(f"    expected {v['expected']}  distinct lengths {v['distinct_lengths']}")
        if v["all_equal_expected"]:
            A("    OK — every request emitted exactly max_new_tokens")
        else:
            A(f"    !! {v['mismatch_count']} request(s) differ from max_new_tokens")
            for m in v["mismatches"][:5]:
                A(f"       {m['request_id']}: {m['output_tokens']}")
            A(f"    {v['note']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-request analysis from _raw.json.")
    ap.add_argument("summary", help="path to *_summary.json")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    run = load_run(args.summary)
    result = analyse(run, concurrency=args.concurrency, bins=args.bins)
    print(format_report(result))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n[raw_stats] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
