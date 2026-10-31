"""Aggregate ``DecisionRecord`` telemetry into the Stage 1 serving metrics.

Only measured values are summed: a request whose server reported no token
usage contributes nothing to the token totals (it is counted in
``requests_without_usage`` instead of as zero), and throughput is computed only
over requests that have both a token count and a measured inference time.
Percentiles are nearest-rank, so every reported latency is one that occurred.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Union

from ._vllm_agent import DecisionRecord, DecisionStatus

_ACTIONS = ("BUY", "SELL", "HOLD")


def summarize_records(records: Iterable[Union[DecisionRecord, Dict[str, Any]]]) -> Dict[str, Any]:
    rows = [r.to_dict() if isinstance(r, DecisionRecord) else dict(r) for r in records]
    n = len(rows)

    def rate(pred) -> Optional[float]:
        return sum(1 for r in rows if pred(r)) / n if n else None

    with_usage = [r for r in rows if r["prompt_tokens"] is not None and r["completion_tokens"] is not None]
    timed = [r for r in with_usage if r["inference_seconds"]]
    latencies = sorted(r["latency_seconds"] for r in rows)
    reasons: Dict[str, int] = {}
    for r in rows:
        if r["reason"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    output_tokens = sum(r["completion_tokens"] for r in timed)
    inference_seconds = sum(r["inference_seconds"] for r in timed)
    return {
        "total_requests": n,
        "valid_decision_rate": rate(lambda r: r["status"] == DecisionStatus.VALID),
        "invalid_response_rate": rate(lambda r: r["status"] == DecisionStatus.REJECTED),
        "failed_inference_rate": rate(lambda r: r["status"] == DecisionStatus.FAILED),
        "timeout_rate": rate(lambda r: r["timed_out"]),
        "action_rates": {a: rate(lambda r, a=a: r["action"] == a) for a in _ACTIONS},
        "reasons": dict(sorted(reasons.items())),
        "attempts": sum(r["attempts"] for r in rows),
        "latency_p50_seconds": _nearest_rank(latencies, 0.50),
        "latency_p95_seconds": _nearest_rank(latencies, 0.95),
        "input_tokens": sum(r["prompt_tokens"] for r in with_usage),
        "output_tokens": sum(r["completion_tokens"] for r in with_usage),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in with_usage),
        "requests_without_usage": sum(1 for r in rows if r["attempts"] and r not in with_usage),
        "output_tokens_per_second": output_tokens / inference_seconds if inference_seconds else None,
    }


def _nearest_rank(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]
