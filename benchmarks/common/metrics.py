"""Per-request timing records and derived summary statistics.

Layer 1 instrumentation. Pure stdlib: importable and unit-testable with no
torch, no GPU, no numpy.

WHY THE INPUT/OUTPUT SPLIT IS LOAD-BEARING
------------------------------------------
v1 reported a single "True Token Throughput" computed as
``(input_tokens + output_tokens) / total_time``, and that number was then read
as if it described decode. With ``max_new_tokens=5`` over a ~2600-token
context, ~99.8% of that figure was prefill. The two are different hardware
regimes — prefill is a compute-bound GEMM over the whole context, decode is a
memory-bandwidth-bound GEMV per token — and averaging them produces a number
that describes neither.

``Summary`` therefore reports ``input_tok_per_s``, ``output_tok_per_s`` and
``total_tok_per_s`` as three separate fields. ``total_tok_per_s`` is retained
only for comparison against v1's figure; it is not a meaningful serving metric
and is documented as such wherever it is emitted.

TTFT / ITL DEFINITIONS
----------------------
Fixed precisely here because v1's per-token metric was wrong in two ways (it
divided the post-first-token window by the *total* token count rather than the
number of inter-token gaps, and it dropped whitespace-only streamer chunks from
the denominator while their elapsed time stayed in the numerator):

    TTFT = t_first_token - t_submit
    ITL  = [t[i] - t[i-1] for i in 1..len(per_token_timestamps)-1]
    e2e  = t_done - t_submit

``ITL`` has exactly ``len(per_token_timestamps) - 1`` entries: it measures the
*gaps between* tokens, so N tokens yield N-1 gaps. The first gap is not an ITL
sample — that interval is TTFT. Every emitted token contributes a timestamp,
including whitespace-only ones; filtering by content is what corrupted v1.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

__all__ = [
    "RequestRecord",
    "Summary",
    "percentile",
    "summarize",
    "write_results",
]


def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile. ``q`` in [0, 100].

    Matches numpy's default ('linear') method so figures are reproducible
    without taking a numpy dependency in the test path:

        rank = q/100 * (n - 1)
        result = x[floor(rank)] + frac * (x[ceil(rank)] - x[floor(rank)])

    Returns None for an empty input rather than raising — a concurrency level
    where every request errored has no percentile, and that must serialize as
    null rather than crash the summary.
    """
    if not values:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    rank = (q / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[int(rank)])
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


@dataclass
class RequestRecord:
    """One request's raw timing. All times are ``time.perf_counter()`` seconds.

    A single monotonic clock source is used throughout; perf_counter has no
    defined epoch, so these are only meaningful as differences. The wall-clock
    anchor for the run lives in the manifest.
    """

    request_id: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    t_submit: float = 0.0
    t_first_token: Optional[float] = None
    per_token_timestamps: List[float] = field(default_factory=list)
    t_done: Optional[float] = None
    error: Optional[str] = None

    # ---- derived ----

    @property
    def ok(self) -> bool:
        return self.error is None and self.t_done is not None

    @property
    def ttft(self) -> Optional[float]:
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.t_submit

    @property
    def e2e(self) -> Optional[float]:
        if self.t_done is None:
            return None
        return self.t_done - self.t_submit

    @property
    def itl(self) -> List[float]:
        """Inter-token latencies: N timestamps -> N-1 gaps."""
        ts = self.per_token_timestamps
        if len(ts) < 2:
            return []
        return [ts[i] - ts[i - 1] for i in range(1, len(ts))]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ttft"] = self.ttft
        d["e2e"] = self.e2e
        d["itl"] = self.itl
        d["ok"] = self.ok
        return d


@dataclass
class Summary:
    """Aggregate over one concurrency level.

    ``wall_time`` is the span from the earliest submit to the latest completion
    across the level. All *_per_s rates use it as the denominator, so they are
    offered-load throughputs for the level as a whole, not per-request rates.
    """

    run_id: str
    concurrency: int
    requested: int
    completed: int
    errored: int
    wall_time: float

    input_tokens_total: int
    output_tokens_total: int
    total_tokens: int

    input_tok_per_s: Optional[float]
    output_tok_per_s: Optional[float]
    total_tok_per_s: Optional[float]
    completed_requests_per_s: Optional[float]

    ttft_p50: Optional[float]
    ttft_p95: Optional[float]
    ttft_p99: Optional[float]

    e2e_p50: Optional[float]
    e2e_p95: Optional[float]
    e2e_p99: Optional[float]

    itl_mean: Optional[float]
    itl_p95: Optional[float]

    # Resource rollup from ResourceMonitor.summary(), attached per level.
    # Lives on the Summary rather than only in a runner's own bookkeeping so
    # that <run_id>_summary.json carries latency, throughput AND resource usage
    # in one object for every arm — a reader should not have to join two
    # structures to answer "what did this level cost".
    resources: Dict[str, Any] = field(default_factory=dict)

    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize(
    run_id: str,
    concurrency: int,
    records: List[RequestRecord],
    requested: Optional[int] = None,
) -> Summary:
    """Aggregate per-request records into a Summary.

    Only successful requests contribute to token counts and latency
    percentiles; ``errored`` carries the rest. A level where some requests OOM
    still produces a valid summary for those that completed, with the failure
    count visible — an OOM at high concurrency is a result, not a lost run.
    """
    ok = [r for r in records if r.ok]
    errored = len(records) - len(ok)

    if ok:
        t_start = min(r.t_submit for r in ok)
        t_end = max(r.t_done for r in ok)  # type: ignore[type-var]
        wall = max(t_end - t_start, 0.0)
    else:
        wall = 0.0

    in_tok = sum(r.prompt_tokens for r in ok)
    out_tok = sum(r.output_tokens for r in ok)

    def rate(n: int) -> Optional[float]:
        return (n / wall) if wall > 0 else None

    itls: List[float] = []
    for r in ok:
        itls.extend(r.itl)

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok if r.e2e is not None]

    return Summary(
        run_id=run_id,
        concurrency=concurrency,
        requested=requested if requested is not None else len(records),
        completed=len(ok),
        errored=errored,
        wall_time=wall,
        input_tokens_total=in_tok,
        output_tokens_total=out_tok,
        total_tokens=in_tok + out_tok,
        input_tok_per_s=rate(in_tok),
        output_tok_per_s=rate(out_tok),
        total_tok_per_s=rate(in_tok + out_tok),
        completed_requests_per_s=rate(len(ok)),
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_p99=percentile(ttfts, 99),
        e2e_p50=percentile(e2es, 50),
        e2e_p95=percentile(e2es, 95),
        e2e_p99=percentile(e2es, 99),
        itl_mean=_mean(itls),
        itl_p95=percentile(itls, 95),
        notes={
            "itl_sample_count": len(itls),
            "total_tok_per_s_warning": (
                "Sum of prefill and decode tokens over wall time. Reported only "
                "for comparison with the v1 figure; it conflates two different "
                "hardware regimes. Use input_tok_per_s / output_tok_per_s."
            ),
        },
    )


def attach_resources(summary: Summary, monitor_summary: Dict[str, Any]) -> Summary:
    """Fold a ``ResourceMonitor.summary()`` into a level's Summary.

    Called by every arm so the resource fields land in the same place with the
    same names, which is what makes an arm-vs-arm diff of the summary JSON
    meaningful.
    """
    summary.resources = dict(monitor_summary or {})
    return summary


def console_lines(summary: Summary) -> List[str]:
    """Human-readable rollup printed by every runner.

    Shared rather than per-arm so the console shows the same quantities for
    arm B and arm C. It deliberately surfaces the metrics that were requested
    but were previously only reachable by opening the JSON: e2e percentiles,
    ITL, total throughput, completed req/s, steady-state VRAM (not just peak),
    and CPU/GPU utilization.
    """
    r = summary.resources or {}

    def ms(v: Optional[float]) -> str:
        return "—" if v is None else f"{v * 1000:.1f} ms"

    def num(v: Optional[float], unit: str = "", nd: int = 1) -> str:
        return "—" if v is None else f"{v:.{nd}f}{unit}"

    lines = [
        f"  requests      {summary.completed}/{summary.requested} completed"
        f"  ({summary.errored} errored)  wall {summary.wall_time:.2f}s",
        f"  TTFT          p50 {ms(summary.ttft_p50)}   p95 {ms(summary.ttft_p95)}"
        f"   p99 {ms(summary.ttft_p99)}",
        f"  e2e           p50 {ms(summary.e2e_p50)}   p95 {ms(summary.e2e_p95)}"
        f"   p99 {ms(summary.e2e_p99)}",
        f"  ITL           mean {ms(summary.itl_mean)}  p95 {ms(summary.itl_p95)}"
        f"   (n={summary.notes.get('itl_sample_count', 0)})",
        f"  throughput    in {num(summary.input_tok_per_s, ' tok/s')}"
        f"   out {num(summary.output_tok_per_s, ' tok/s')}"
        f"   total {num(summary.total_tok_per_s, ' tok/s')}",
        f"  requests/s    {num(summary.completed_requests_per_s, '', 3)}",
        f"  VRAM          peak {num(r.get('vram_peak_mb'), ' MB', 0)}"
        f"   steady {num(r.get('vram_steady_state_mb'), ' MB', 0)}",
        f"  GPU util      mean {num(r.get('util_gpu_mean_pct'), '%')}"
        f"   max {num(r.get('util_gpu_max_pct'), '%')}"
        f"   (kernel residency, NOT occupancy)",
        f"  CPU util      mean {num(r.get('cpu_mean_pct_across_cores'), '%')}"
        f"   peak {num(r.get('cpu_peak_pct_across_cores'), '%')}"
        f"   ({r.get('cpu_core_count', '?')} cores)",
    ]
    return lines


def write_results(
    out_dir: str,
    run_id: str,
    records: List[RequestRecord],
    summaries: List[Summary],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write ``<run_id>_raw.json`` and ``<run_id>_summary.json``.

    Called after every concurrency level rather than once at the end, so a
    Colab preemption mid-sweep leaves the completed levels on disk. Each call
    rewrites both files with everything accumulated so far.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{run_id}_raw.json")
    sum_path = os.path.join(out_dir, f"{run_id}_summary.json")

    with open(raw_path, "w") as fh:
        json.dump(
            {"run_id": run_id, "records": [r.to_dict() for r in records]},
            fh,
            indent=2,
        )

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "levels": [s.to_dict() for s in summaries],
    }
    if extra:
        payload.update(extra)
    with open(sum_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return {"raw": raw_path, "summary": sum_path}
