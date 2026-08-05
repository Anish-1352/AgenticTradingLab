"""Calibration log: the measured noise floor for the study.

A fixed, short arm-B configuration is run at the start of every session and its
headline numbers appended here, keyed by GPU UUID and timestamp.

WHY THIS EXISTS
---------------
Arm-to-arm differences only mean something against a known noise floor. Without
one, "arm C is 12% faster" is indistinguishable from "the two sweeps ran on
different days" — and on Colab they very often did, sometimes on different
physical cards. This measures session-to-session and card-to-card variance
directly, so a reported delta can be stated against it rather than against an
assumption that the environment was stable.

The analysis deliberately separates **within-UUID** spread from **between-UUID**
spread. If a metric's variation is mostly explained by which card you landed on,
then any cross-session comparison of that metric is a hardware comparison
wearing a software label, and the analysis says so.

Pure stdlib — no torch, no GPU, unit-testable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "CALIBRATION_METRICS",
    "append_record",
    "read_log",
    "record_from_summary",
    "analyze",
    "format_analysis",
]

DEFAULT_LOG = "/content/drive/MyDrive/atl_bench/calibration_log.jsonl"

# (key in the record, human label, unit, higher_is_better)
CALIBRATION_METRICS = (
    ("ttft_p50_ms", "TTFT p50", "ms", False),
    ("ttft_p95_ms", "TTFT p95", "ms", False),
    ("e2e_p50_ms", "e2e p50", "ms", False),
    ("output_tok_per_s", "output throughput", "tok/s", True),
    ("completed_requests_per_s", "requests/s", "req/s", True),
    ("vram_peak_mb", "VRAM peak", "MB", False),
)

# Above this coefficient of variation, a difference smaller than the noise is
# not a result. 5% is a deliberately blunt threshold — it exists to prompt a
# judgement, not to substitute for one.
CV_WARN_PCT = 5.0


def read_log(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Preemption can truncate the last line; keep the history.
                continue
    return out


def append_record(path: str, record: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def record_from_summary(
    summary_path: str,
    gpu_uuid: Optional[str],
    gpu_name: Optional[str] = None,
    branch_sha: Optional[str] = None,
    pip_freeze_sha256: Optional[str] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract the calibration metrics from a runner's ``*_summary.json``.

    Reads the level matching ``concurrency`` (or the sole level). Raises rather
    than guessing if the file has no usable level — a calibration entry built
    from the wrong level would silently widen the measured noise floor.
    """
    with open(summary_path) as fh:
        data = json.load(fh)

    levels = data.get("levels") or []
    if not levels:
        raise ValueError(f"{summary_path} contains no levels")

    level = None
    if concurrency is not None:
        for lv in levels:
            if int(lv.get("concurrency", -1)) == int(concurrency):
                level = lv
                break
        if level is None:
            raise ValueError(
                f"{summary_path} has no level at concurrency={concurrency} "
                f"(has {[lv.get('concurrency') for lv in levels]})"
            )
    else:
        if len(levels) > 1:
            raise ValueError(
                f"{summary_path} has {len(levels)} levels; pass --concurrency "
                f"to say which one is the calibration point"
            )
        level = levels[0]

    res = level.get("resources") or {}

    def ms(v: Optional[float]) -> Optional[float]:
        return None if v is None else float(v) * 1000.0

    return {
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "branch_sha": branch_sha,
        "pip_freeze_sha256": pip_freeze_sha256,
        "run_id": data.get("run_id"),
        "arm": data.get("arm"),
        "concurrency": level.get("concurrency"),
        "requested": level.get("requested"),
        "completed": level.get("completed"),
        "errored": level.get("errored"),
        "ttft_p50_ms": ms(level.get("ttft_p50")),
        "ttft_p95_ms": ms(level.get("ttft_p95")),
        "e2e_p50_ms": ms(level.get("e2e_p50")),
        "output_tok_per_s": level.get("output_tok_per_s"),
        "completed_requests_per_s": level.get("completed_requests_per_s"),
        "vram_peak_mb": res.get("vram_peak_mb"),
        "vram_steady_state_mb": res.get("vram_steady_state_mb"),
        "summary_path": summary_path,
    }


def _stats(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "stdev": sd,
        "min": min(vals),
        "max": max(vals),
        "spread_pct": ((max(vals) - min(vals)) / mean * 100.0) if mean else None,
        "cv_pct": (sd / mean * 100.0) if mean else None,
    }


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Spread across all calibration runs, overall and grouped by GPU UUID."""
    by_uuid: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_uuid.setdefault(r.get("gpu_uuid") or "UNKNOWN", []).append(r)

    metrics: Dict[str, Any] = {}
    for key, label, unit, higher_better in CALIBRATION_METRICS:
        overall = _stats([r.get(key) for r in records])
        groups = {
            uuid: _stats([r.get(key) for r in rs]) for uuid, rs in by_uuid.items()
        }

        # Does card identity explain the spread? Compare the mean within-card
        # CV against the spread of per-card means. If the latter dominates, the
        # variance is hardware, and cross-session comparison of this metric is
        # a cross-hardware comparison.
        card_means = [g["mean"] for g in groups.values() if g.get("n")]
        between_pct = None
        if len(card_means) > 1:
            gmean = statistics.fmean(card_means)
            between_pct = ((max(card_means) - min(card_means)) / gmean * 100.0) if gmean else None
        within_cvs = [g["cv_pct"] for g in groups.values()
                      if g.get("n", 0) > 1 and g.get("cv_pct") is not None]
        within_pct = statistics.fmean(within_cvs) if within_cvs else None

        hardware_dominated = bool(
            between_pct is not None and within_pct is not None
            and between_pct > max(within_pct, 1.0)
        )

        metrics[key] = {
            "label": label,
            "unit": unit,
            "higher_is_better": higher_better,
            "overall": overall,
            "by_uuid": groups,
            "between_uuid_spread_pct": between_pct,
            "mean_within_uuid_cv_pct": within_pct,
            "hardware_dominated": hardware_dominated,
            "noisy": bool(overall.get("cv_pct") is not None
                          and overall["cv_pct"] > CV_WARN_PCT),
        }

    return {
        "n_records": len(records),
        "n_distinct_gpus": len(by_uuid),
        "gpu_uuids": sorted(by_uuid),
        "metrics": metrics,
        "cv_warn_pct": CV_WARN_PCT,
    }


def format_analysis(analysis: Dict[str, Any]) -> str:
    lines: List[str] = []
    A = lines.append

    A("=" * 78)
    A("CALIBRATION ANALYSIS — measured noise floor")
    A("=" * 78)
    A(f"  runs recorded : {analysis['n_records']}")
    A(f"  distinct GPUs : {analysis['n_distinct_gpus']}")
    for u in analysis["gpu_uuids"]:
        A(f"                  {u}")

    if analysis["n_records"] < 2:
        A("")
        A("  Fewer than 2 runs — no spread to report yet. Run calibration at the")
        A("  start of each session; the noise floor is what makes a reported")
        A("  arm-to-arm delta meaningful.")
        return "\n".join(lines)

    A("")
    A(f"  {'metric':<22} {'n':>3} {'mean':>12} {'min':>12} {'max':>12} {'CV%':>7}")
    A("  " + "-" * 74)
    for key, m in analysis["metrics"].items():
        o = m["overall"]
        if not o.get("n"):
            continue
        cv = o.get("cv_pct")
        flag = "  <-- NOISY" if m["noisy"] else ""
        A(f"  {m['label'] + ' (' + m['unit'] + ')':<22} {o['n']:>3} "
          f"{o['mean']:>12.2f} {o['min']:>12.2f} {o['max']:>12.2f} "
          f"{(f'{cv:.2f}' if cv is not None else '—'):>7}{flag}")

    noisy = [m for m in analysis["metrics"].values() if m["noisy"]]
    if noisy:
        A("")
        A(f"  NOISY (CV > {analysis['cv_warn_pct']}%): a measured arm-to-arm")
        A("  difference smaller than this is not a result.")
        for m in noisy:
            A(f"    - {m['label']}: CV {m['overall']['cv_pct']:.2f}%, "
              f"range {m['overall']['min']:.2f}..{m['overall']['max']:.2f} {m['unit']}")

    hw = [m for m in analysis["metrics"].values() if m["hardware_dominated"]]
    if hw:
        A("")
        A("  " + "!" * 70)
        A("  HARDWARE-DOMINATED METRICS — variance tracks WHICH CARD, not the run")
        A("  " + "!" * 70)
        for m in hw:
            A(f"    - {m['label']}: between-card spread "
              f"{m['between_uuid_spread_pct']:.2f}% vs within-card CV "
              f"{m['mean_within_uuid_cv_pct']:.2f}%")
        A("")
        A("  For these metrics, a comparison spanning a GPU change is a")
        A("  CROSS-HARDWARE comparison. Either re-run both arms on one card, or")
        A("  report the delta with this spread stated alongside it.")

    if analysis["n_distinct_gpus"] > 1:
        A("")
        A("  Per-GPU breakdown:")
        for key, m in analysis["metrics"].items():
            rows = [(u, g) for u, g in m["by_uuid"].items() if g.get("n")]
            if len(rows) < 2:
                continue
            A(f"    {m['label']} ({m['unit']}):")
            for uuid, g in sorted(rows):
                cv = g.get("cv_pct")
                cv_txt = "—" if cv is None else f"{cv:.2f}%"
                A(f"      {uuid[:28]:<30} n={g['n']:<3} "
                  f"mean={g['mean']:.2f}  cv={cv_txt}")

    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibration log append / analyze.")
    ap.add_argument("--log", default=os.environ.get("ATL_CALIBRATION_LOG", DEFAULT_LOG))
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--append", metavar="SUMMARY_JSON",
                     help="extract metrics from a runner summary and append")
    sub.add_argument("--analyze", action="store_true",
                     help="report spread across all recorded runs")
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--gpu-name", default=None)
    ap.add_argument("--branch-sha", default=None)
    ap.add_argument("--pip-freeze-sha256", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.append:
        rec = record_from_summary(
            args.append, gpu_uuid=args.gpu_uuid, gpu_name=args.gpu_name,
            branch_sha=args.branch_sha, pip_freeze_sha256=args.pip_freeze_sha256,
            concurrency=args.concurrency,
        )
        append_record(args.log, rec)
        print(f"[calibration] appended to {args.log}")
        for key, label, unit, _ in CALIBRATION_METRICS:
            val = rec.get(key)
            print(f"  {label:<22} {'—' if val is None else f'{val:.2f}'} {unit}")
        return 0

    records = read_log(args.log)
    analysis = analyze(records)
    print(format_analysis(analysis))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(analysis, fh, indent=2)
        print(f"\n[calibration] analysis written to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
