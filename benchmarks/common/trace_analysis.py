"""Post-process a torch.profiler Chrome trace into a JSON summary.

Layer 3 analysis. Pure stdlib — operates on an already-parsed trace dict, so
it is unit-testable against a hand-built fake with no torch and no GPU.

Chrome trace shape consumed (torch.profiler ``export_chrome_trace``)::

    {"traceEvents": [
      {"ph": "X", "cat": "kernel",       "name": "...s16816gemm...",
       "ts": 1000.0, "dur": 250.0, "args": {"stream": 7, "grid": [64,1,1],
                                            "block": [128,1,1], "device": 0}},
      {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
       "ts": 990.0,  "dur": 12.0,  "args": {...}},
      ...
    ]}

All ``ts``/``dur`` are microseconds.

GPU BUSY IS A UNION, NOT A SUM
------------------------------
Summing kernel durations double-counts whenever kernels overlap on different
streams, and can trivially exceed wall time — which reads as >100% utilization
and is the classic way to conclude a GPU is saturated when it is not. Busy time
here is the measure of the *union* of kernel intervals; ``total_kernel_time_us``
(the plain sum) is reported alongside it, and the ratio between the two is
itself the concurrency signal: sum/union ~= 1.0 means kernels are serialized.

WHAT THIS CANNOT TELL YOU
-------------------------
Kernel *duration* is not occupancy. A kernel resident for 100% of the window at
8% achieved occupancy looks identical here to one at 90%. Achieved occupancy
and Tensor Core pipe utilization are Layer 4 (ncu) metrics. The
``tensor_core_gemm_share`` below is a share of *time in kernels whose names
indicate an HMMA GEMM* — it says those kernels ran, not how well they used the
tensor pipes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "merge_intervals",
    "union_duration",
    "analyze_trace",
    "load_trace",
    "analyze_trace_file",
]

# Kernel-name substrings (lowercased) that indicate an FP16 Tensor Core GEMM.
# 's16816gemm' is the cuBLAS/CUTLASS HMMA naming on Ampere; the others cover
# CUTLASS-generated and flash-attention kernels that also target the tensor
# pipes.
TENSOR_CORE_PATTERNS = ("s16816gemm", "h16816gemm", "hmma", "cutlass", "flash")
# Decode's matrix-vector products. Batch-1 decode degenerates to GEMV, which is
# memory-bandwidth-bound and does not use the tensor pipes at all.
GEMV_PATTERNS = ("gemv", "dot_kernel", "splitkreduce")

_KERNEL_CATS = {"kernel", "gpu_kernel", "cuda_kernel"}
_RUNTIME_CATS = {"cuda_runtime", "runtime", "cuda_driver"}
_MEMCPY_CATS = {"gpu_memcpy", "gpu_memset", "memcpy", "memset"}


def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/touching ``(start, end)`` intervals. Sorted output."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: List[Tuple[float, float]] = [tuple(ordered[0])]  # type: ignore[list-item]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def union_duration(intervals: Sequence[Tuple[float, float]]) -> float:
    """Total time covered by the union of intervals (overlap counted once)."""
    return float(sum(end - start for start, end in merge_intervals(intervals)))


def _cat(event: Dict[str, Any]) -> str:
    return str(event.get("cat", event.get("categories", ""))).lower()


def _is_complete(event: Dict[str, Any]) -> bool:
    # 'X' is a complete event (has both ts and dur). Instant/flow/metadata
    # events carry no duration and must not enter any time computation.
    return event.get("ph", "X") == "X" and "dur" in event and "ts" in event


def _grid_key(event: Dict[str, Any]) -> str:
    grid = (event.get("args") or {}).get("grid")
    if isinstance(grid, (list, tuple)):
        return "x".join(str(int(g)) for g in grid)
    return "unknown"


def _classify_name(name: str) -> Optional[str]:
    low = name.lower()
    # GEMV is checked first: a name can contain both a cutlass tag and gemv,
    # and in that case it is the vector path that characterises it.
    if any(p in low for p in GEMV_PATTERNS):
        return "gemv"
    if any(p in low for p in TENSOR_CORE_PATTERNS):
        return "tensor_core_gemm"
    return None


def _split_prefill_decode(
    kernels: List[Dict[str, Any]],
    decode_start_us: Optional[float],
) -> Dict[str, Any]:
    """Attribute kernel time to prefill vs decode.

    Preferred path is an explicit boundary: the runner knows ``t_first_token``
    for the profiled request and converts it into the trace clock, so the split
    is a fact rather than an inference. ``analyze_trace`` records which method
    was used, because a heuristic split and a measured one do not deserve equal
    confidence in a write-up.
    """
    if decode_start_us is not None:
        prefill = sum(k["dur"] for k in kernels if k["ts"] < decode_start_us)
        decode = sum(k["dur"] for k in kernels if k["ts"] >= decode_start_us)
        return {
            "method": "explicit_boundary",
            "decode_start_us": decode_start_us,
            "prefill_kernel_time_us": prefill,
            "decode_kernel_time_us": decode,
            "caveat": None,
        }

    # Fallback: classify by kernel shape. Decode is GEMV-dominated; prefill is
    # GEMM over the full context. Kernels matching neither are unattributed
    # rather than silently pushed into one bucket.
    prefill = 0.0
    decode = 0.0
    unattributed = 0.0
    for k in kernels:
        kind = _classify_name(k["name"])
        if kind == "tensor_core_gemm":
            prefill += k["dur"]
        elif kind == "gemv":
            decode += k["dur"]
        else:
            unattributed += k["dur"]
    return {
        "method": "heuristic_name_shape",
        "decode_start_us": None,
        "prefill_kernel_time_us": prefill,
        "decode_kernel_time_us": decode,
        "unattributed_kernel_time_us": unattributed,
        "caveat": (
            "No explicit boundary supplied; split inferred from kernel names "
            "(GEMM->prefill, GEMV->decode). Elementwise/norm/attention kernels "
            "are unattributed. Do not quote this split without the boundary."
        ),
    }


def analyze_trace(
    trace: Dict[str, Any],
    decode_start_us: Optional[float] = None,
    top_n: int = 15,
) -> Dict[str, Any]:
    """Analyze a parsed Chrome trace dict. Returns a JSON-serializable summary."""
    events = trace.get("traceEvents", [])

    kernels: List[Dict[str, Any]] = []
    runtime_events: List[Dict[str, Any]] = []
    memcpy_events: List[Dict[str, Any]] = []

    for ev in events:
        if not _is_complete(ev):
            continue
        cat = _cat(ev)
        rec = {
            "name": str(ev.get("name", "")),
            "ts": float(ev["ts"]),
            "dur": float(ev["dur"]),
            "args": ev.get("args") or {},
        }
        if cat in _KERNEL_CATS:
            kernels.append(rec)
        elif cat in _RUNTIME_CATS:
            runtime_events.append(rec)
        elif cat in _MEMCPY_CATS:
            memcpy_events.append(rec)

    total_kernel_time = float(sum(k["dur"] for k in kernels))
    intervals = [(k["ts"], k["ts"] + k["dur"]) for k in kernels]
    busy = union_duration(intervals)

    # Span is measured over GPU-side work only (kernels + memcpy). Including
    # host-side runtime events would stretch the window with CPU time during
    # which the GPU was legitimately idle, deflating the idle fraction's
    # meaning.
    gpu_events = kernels + memcpy_events
    if gpu_events:
        span_start = min(e["ts"] for e in gpu_events)
        span_end = max(e["ts"] + e["dur"] for e in gpu_events)
        span = max(span_end - span_start, 0.0)
    else:
        span_start = span_end = 0.0
        span = 0.0

    by_name: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    by_grid: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    tc_time = 0.0
    gemv_time = 0.0
    streams: set = set()

    for k in kernels:
        by_name[k["name"]]["count"] += 1
        by_name[k["name"]]["total_us"] += k["dur"]
        gk = _grid_key(k)
        by_grid[gk]["count"] += 1
        by_grid[gk]["total_us"] += k["dur"]

        kind = _classify_name(k["name"])
        if kind == "tensor_core_gemm":
            tc_time += k["dur"]
        elif kind == "gemv":
            gemv_time += k["dur"]

        stream = k["args"].get("stream")
        if stream is not None:
            streams.add(stream)

    runtime_breakdown: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_cpu_us": 0.0}
    )
    for ev in runtime_events:
        runtime_breakdown[ev["name"]]["count"] += 1
        runtime_breakdown[ev["name"]]["total_cpu_us"] += ev["dur"]

    def _top(d: Dict[str, Dict[str, float]], n: int) -> List[Dict[str, Any]]:
        items = sorted(d.items(), key=lambda kv: kv[1]["total_us"], reverse=True)[:n]
        return [
            {
                "name": name,
                "count": int(stats["count"]),
                "total_us": stats["total_us"],
                "share_of_kernel_time": (
                    stats["total_us"] / total_kernel_time if total_kernel_time else None
                ),
            }
            for name, stats in items
        ]

    def _share(x: float) -> Optional[float]:
        return (x / total_kernel_time) if total_kernel_time else None

    return {
        "kernel_count": len(kernels),
        "total_kernel_time_us": total_kernel_time,
        "gpu_busy_us": busy,
        "trace_span_us": span,
        "trace_span_start_us": span_start,
        "trace_span_end_us": span_end,
        "gpu_busy_fraction": (busy / span) if span > 0 else None,
        "gpu_idle_fraction": (1.0 - busy / span) if span > 0 else None,
        # >1.0 means kernels overlapped on separate streams; ~1.0 means they
        # were serialized end to end.
        "kernel_overlap_factor": (total_kernel_time / busy) if busy > 0 else None,
        "distinct_streams": sorted(streams, key=lambda s: str(s)),
        "distinct_stream_count": len(streams),
        "tensor_core_gemm_time_us": tc_time,
        "tensor_core_gemm_share": _share(tc_time),
        "gemv_time_us": gemv_time,
        "gemv_share": _share(gemv_time),
        "top_kernels_by_name": _top(by_name, top_n),
        "kernel_time_by_grid": _top(by_grid, top_n),
        "cuda_runtime_breakdown": sorted(
            (
                {
                    "name": name,
                    "count": int(stats["count"]),
                    "total_cpu_us": stats["total_cpu_us"],
                }
                for name, stats in runtime_breakdown.items()
            ),
            key=lambda d: d["total_cpu_us"],
            reverse=True,
        ),
        "cuda_runtime_total_cpu_us": float(
            sum(e["dur"] for e in runtime_events)
        ),
        "memcpy_count": len(memcpy_events),
        "memcpy_total_us": float(sum(e["dur"] for e in memcpy_events)),
        "prefill_decode": _split_prefill_decode(kernels, decode_start_us),
        "caveats": [
            "gpu_busy_us is the union of kernel intervals; total_kernel_time_us "
            "is the plain sum and double-counts stream overlap.",
            "Kernel residency is not occupancy. Achieved occupancy and Tensor "
            "Core pipe utilization require ncu (Layer 4).",
            "tensor_core_gemm_share is the time share of kernels whose names "
            "indicate an HMMA GEMM, not a measure of tensor pipe efficiency.",
        ],
    }


def load_trace(path: str) -> Dict[str, Any]:
    """Load a Chrome trace from ``.json`` or ``.json.gz``."""
    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def analyze_trace_file(
    path: str,
    out_path: Optional[str] = None,
    decode_start_us: Optional[float] = None,
    top_n: int = 15,
) -> Dict[str, Any]:
    result = analyze_trace(load_trace(path), decode_start_us=decode_start_us, top_n=top_n)
    result["source_trace"] = path
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Analyze a torch.profiler chrome trace.")
    ap.add_argument("trace", help="path to trace .json or .json.gz")
    ap.add_argument("--out", help="write JSON summary here")
    ap.add_argument(
        "--decode-start-us",
        type=float,
        default=None,
        help="trace-clock timestamp of first generated token; enables an exact "
        "prefill/decode split instead of the name heuristic",
    )
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args(argv)

    result = analyze_trace_file(
        args.trace, out_path=args.out, decode_start_us=args.decode_start_us, top_n=args.top_n
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
