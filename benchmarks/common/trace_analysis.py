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
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "merge_intervals",
    "union_duration",
    "analyze_trace",
    "load_trace",
    "analyze_trace_file",
    "prefill_boundary_from_records",
    "derive_decode_start_us",
    "iter_trace_events",
    "profiler_step_anchors",
    "anchor_decode_start_us",
    "analyze_trace_stream",
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


def prefill_boundary_from_records(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Locate the prefill/decode boundary from per-request timings.

    Under a burst arrival pattern every admitted request prefills at roughly the
    same time and then decodes. The **last** request to emit its first token
    therefore marks the end of the prefill phase: after that instant, no request
    is still prefilling, so GPU work is decode.

    All times are ``perf_counter`` seconds relative to the earliest submit in
    the set, because perf_counter has no defined epoch and only differences
    are meaningful.
    """
    ok = [
        r for r in records
        if not r.get("error")
        and r.get("t_done") is not None
        and r.get("t_first_token") is not None
    ]
    if not ok:
        return {"available": False,
                "reason": "no completed request carries a t_first_token"}

    t0 = min(r["t_submit"] for r in ok)
    last_first_token = max(r["t_first_token"] for r in ok)
    t_end = max(r["t_done"] for r in ok)

    wall = t_end - t0
    boundary_rel = last_first_token - t0
    return {
        "available": True,
        "n_requests": len(ok),
        "level_wall_s": wall,
        "prefill_end_rel_s": boundary_rel,
        "prefill_fraction_of_level": (boundary_rel / wall) if wall > 0 else None,
        "first_token_earliest_rel_s": min(r["t_first_token"] for r in ok) - t0,
        "definition": (
            "boundary = last request's first token. Before it at least one "
            "request is still prefilling; after it every request is decoding."
        ),
    }


def derive_decode_start_us(
    trace: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    window_start_rel_s: Optional[float] = None,
    section_wall_s: Optional[float] = None,
    window_length_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Map the measured boundary onto the trace clock.

    The two clocks do not share an epoch: ``_raw.json`` carries
    ``perf_counter`` seconds, a chrome trace carries its own microsecond
    timebase. They are aligned proportionally — the boundary's position within
    the profiled wall-clock window is mapped onto the same position within the
    trace's span.

    That proportional step is an assumption, and it is reported rather than
    hidden. Two cases matter:

    * **Full-run trace** (``section_wall_s`` absent or ~= the level wall): the
      mapping is direct and sound.
    * **Bounded window** (Layer 3 default): the window is stepped by request
      completions and therefore opens *after* prefill has finished. When
      ``window_start_rel_s`` shows the window began past the boundary, the
      honest answer is not a proportional map at all — the whole trace is
      decode, and that is returned instead.
    """
    boundary = prefill_boundary_from_records(records)
    if not boundary.get("available"):
        return {"available": False, "reason": boundary.get("reason"),
                "boundary": boundary}

    events = trace.get("traceEvents", [])
    spans = [
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if _is_complete(e) and _cat(e) in (_KERNEL_CATS | _MEMCPY_CATS)
    ]
    if not spans:
        return {"available": False, "reason": "trace contains no GPU events",
                "boundary": boundary}

    span_start = min(s for s, _ in spans)
    span_end = max(e for _, e in spans)
    span_us = span_end - span_start

    boundary_rel = boundary["prefill_end_rel_s"]

    # The traced window opened after prefill finished -> nothing in this trace
    # is prefill. Proportionally mapping would fabricate a prefill slice that
    # the capture cannot contain.
    if window_start_rel_s is not None and window_start_rel_s >= boundary_rel:
        return {
            "available": True,
            "decode_start_us": span_start,
            "method": "measured_boundary_window_after_prefill",
            "boundary": boundary,
            "window_start_rel_s": window_start_rel_s,
            "note": (
                "The profiled window opened at "
                f"{window_start_rel_s:.3f}s, after the prefill boundary at "
                f"{boundary_rel:.3f}s. This capture contains DECODE ONLY; no "
                "prefill kernels are present to attribute."
            ),
        }

    # A bounded window: the trace span covers [window_start, window_start +
    # window_length], NOT the whole level. Dividing the boundary by the level
    # wall would place it at ~41% of a capture that begins after 41% of the
    # level has already elapsed — attributing a large prefill slice to a trace
    # that contains almost none. Map WITHIN the window instead.
    if window_start_rel_s is not None and window_length_s:
        window_end_rel = window_start_rel_s + window_length_s
        frac = (boundary_rel - window_start_rel_s) / window_length_s
        frac = max(0.0, min(1.0, frac))
        return {
            "available": True,
            "decode_start_us": span_start + frac * span_us,
            "method": "measured_boundary_mapped_within_window",
            "boundary": boundary,
            "window_start_rel_s": window_start_rel_s,
            "window_end_rel_s": window_end_rel,
            "window_length_s": window_length_s,
            "prefill_fraction_of_window": frac,
            "trace_span_us": span_us,
            "assumption": (
                "The trace span is mapped onto the profiled window "
                f"[{window_start_rel_s:.2f}s, {window_end_rel:.2f}s], not onto "
                "the whole level. The boundary falls "
                f"{frac * 100:.2f}% into that window, so that is the share of "
                "the capture attributed to prefill."
            ),
        }

    reference_wall = section_wall_s or boundary["level_wall_s"]
    if not reference_wall:
        return {"available": False, "reason": "zero reference wall time",
                "boundary": boundary}

    frac = boundary_rel / reference_wall
    decode_start_us = span_start + max(0.0, min(1.0, frac)) * span_us

    return {
        "available": True,
        "decode_start_us": decode_start_us,
        "method": "measured_boundary_proportional_map",
        "boundary": boundary,
        "boundary_fraction_of_wall": frac,
        "trace_span_us": span_us,
        "assumption": (
            "The trace's span is assumed to cover the same wall-clock interval "
            "the per-request timings describe, so the boundary's fractional "
            "position transfers. Sound ONLY for a full-run capture. For a "
            "bounded window supply window_start_rel_s AND window_length_s — "
            "otherwise the boundary is divided by the level wall while the "
            "trace covers only part of it, and the prefill share is overstated."
        ),
    }


def _split_prefill_decode(
    kernels: List[Dict[str, Any]],
    decode_start_us: Optional[float],
    boundary_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attribute kernel time to prefill vs decode.

    Preferred path is a boundary **measured** from ``_raw.json``'s
    ``t_first_token`` (see :func:`derive_decode_start_us`), which makes the
    split a fact about the run rather than an inference from kernel names. The
    name heuristic remains only as a labelled fallback, and the output always
    states which produced the number — a heuristic split and a measured one do
    not deserve equal confidence in a write-up.
    """
    if decode_start_us is not None:
        prefill = sum(k["dur"] for k in kernels if k["ts"] < decode_start_us)
        decode = sum(k["dur"] for k in kernels if k["ts"] >= decode_start_us)
        meta = boundary_meta or {}
        return {
            "method": meta.get("method", "explicit_boundary"),
            "method_confidence": "measured",
            "decode_start_us": decode_start_us,
            "prefill_kernel_time_us": prefill,
            "decode_kernel_time_us": decode,
            "boundary_source": meta or {
                "note": "decode_start_us supplied directly by the caller"
            },
            "caveat": meta.get("note") or meta.get("assumption"),
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
        "method_confidence": "inferred — DO NOT QUOTE without the measured boundary",
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
    boundary_meta: Optional[Dict[str, Any]] = None,
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
        "prefill_decode": _split_prefill_decode(kernels, decode_start_us, boundary_meta),
        "caveats": [
            "gpu_busy_us is the union of kernel intervals; total_kernel_time_us "
            "is the plain sum and double-counts stream overlap.",
            "Kernel residency is not occupancy. Achieved occupancy and Tensor "
            "Core pipe utilization require ncu (Layer 4).",
            "tensor_core_gemm_share is the time share of kernels whose names "
            "indicate an HMMA GEMM, not a measure of tensor pipe efficiency.",
        ],
    }


def iter_trace_events(path: str, chunk_size: int = 1 << 22) -> Iterable[Dict[str, Any]]:
    """Stream ``traceEvents`` one object at a time, without loading the file.

    A real Layer 3 capture is far too large to parse whole: the arm B trace is
    228 MB gzipped and **3.76 GB** expanded, which as Python objects would not
    fit in a 17 GB host. ``json.load`` is therefore not an option, and neither
    is line-splitting — torch pretty-prints each event across ~7 lines.

    So the file is read in chunks and objects are pulled off the front of a
    rolling buffer with ``JSONDecoder.raw_decode``. Peak memory is one chunk
    plus one event.
    """
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    decoder = json.JSONDecoder()

    with opener(path, "rt") as fh:  # type: ignore[operator]
        buf = ""
        # Seek to the start of the traceEvents array.
        while '"traceEvents"' not in buf:
            chunk = fh.read(chunk_size)
            if not chunk:
                return
            buf += chunk
        idx = buf.index('"traceEvents"')
        idx = buf.index("[", idx) + 1
        buf = buf[idx:]

        while True:
            buf = buf.lstrip().lstrip(",").lstrip()
            if buf.startswith("]") or not buf:
                if buf.startswith("]"):
                    return
                chunk = fh.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
                continue
            try:
                obj, end = decoder.raw_decode(buf)
            except ValueError:
                chunk = fh.read(chunk_size)
                if not chunk:
                    return  # truncated tail; stop rather than raise
                buf += chunk
                continue
            buf = buf[end:]
            if isinstance(obj, dict):
                yield obj


def profiler_step_anchors(path: str) -> List[Dict[str, Any]]:
    """``ProfilerStep#N`` annotations: (step, trace ts, dur) in trace clock.

    These are the exact positions of the profiler's steps on the trace
    timebase, which makes clock alignment an *anchored* operation rather than a
    proportional guess. Each step's duration is the wall time between two
    consecutive ``prof.step()`` calls, and the runner steps once per request
    completion — so a step's duration can be matched against the gap between
    two completions in ``_raw.json`` to tie the two clocks together exactly.
    """
    out: List[Dict[str, Any]] = []
    for ev in iter_trace_events(path):
        name = str(ev.get("name", ""))
        if name.startswith("ProfilerStep#") and _is_complete(ev):
            try:
                step = int(name.split("#", 1)[1])
            except (IndexError, ValueError):
                continue
            out.append({"step": step, "ts": float(ev["ts"]),
                        "dur": float(ev["dur"])})
    return sorted(out, key=lambda d: d["step"])


def anchor_decode_start_us(
    steps: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Align the two clocks on a ProfilerStep, then place the boundary exactly.

    Strictly better than any proportional mapping: a step annotation is a real
    event present in both worlds. The runner calls ``prof.step()`` once per
    completion, so ``ProfilerStep#N`` begins at the N-th completion; matching a
    step's duration against the corresponding inter-completion gap confirms the
    correspondence before it is used.
    """
    boundary = prefill_boundary_from_records(records)
    if not boundary.get("available"):
        return {"available": False, "reason": boundary.get("reason")}
    if not steps:
        return {"available": False, "reason": "no ProfilerStep annotations in trace"}

    ok = [r for r in records if not r.get("error") and r.get("t_done") is not None]
    t0 = min(r["t_submit"] for r in ok)
    done = sorted(r["t_done"] - t0 for r in ok)

    # Find the step whose duration best matches an inter-completion gap.
    best = None
    for st in steps:
        n = st["step"]
        if not (1 <= n < len(done)):
            continue
        gap = (done[n] - done[n - 1]) * 1e6  # us
        if gap <= 0:
            continue
        err = abs(st["dur"] - gap) / gap
        if best is None or err < best["rel_error"]:
            best = {"step": n, "trace_ts_us": st["ts"], "trace_dur_us": st["dur"],
                    "wall_rel_s": done[n - 1], "gap_us": gap, "rel_error": err}

    if best is None:
        return {"available": False,
                "reason": "no ProfilerStep could be matched to a completion gap"}

    # ProfilerStep#n starts at completion n (1-indexed) -> done[n-1].
    offset_us = best["trace_ts_us"] - best["wall_rel_s"] * 1e6
    decode_start_us = offset_us + boundary["prefill_end_rel_s"] * 1e6

    return {
        "available": True,
        "decode_start_us": decode_start_us,
        "method": "anchored_on_profiler_step",
        "boundary": boundary,
        "anchor": best,
        "clock_offset_us": offset_us,
        "confidence_note": (
            f"Anchored on ProfilerStep#{best['step']}, whose duration matches "
            f"the corresponding inter-completion gap to within "
            f"{best['rel_error'] * 100:.2f}%. This is an exact alignment on a "
            f"shared event, not a proportional mapping."
        ),
    }


def load_trace(path: str) -> Dict[str, Any]:
    """Load a Chrome trace from ``.json`` or ``.json.gz``."""
    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def analyze_trace_stream(
    path: str,
    decode_start_us: Optional[float] = None,
    top_n: int = 15,
    boundary_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Same summary as :func:`analyze_trace`, computed without loading the file.

    Kernel intervals are accumulated into ``array('d')`` rather than a list of
    dicts — for the ~9M events in a real capture that is the difference between
    ~75 MB and several GB. Everything else aggregates into small dicts keyed by
    kernel name or grid shape, which are bounded by the number of *distinct*
    kernels rather than by the number of events.
    """
    from array import array

    starts, ends = array("d"), array("d")
    total_kernel_time = 0.0
    kernel_count = 0
    by_name: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])   # count, us
    by_grid: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    runtime: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    tc_time = gemv_time = 0.0
    runtime_total = 0.0
    memcpy_count = 0
    memcpy_total = 0.0
    streams: set = set()
    span_start = span_end = None
    prefill = decode = 0.0

    for ev in iter_trace_events(path):
        if not _is_complete(ev):
            continue
        cat = _cat(ev)
        ts = float(ev["ts"])
        dur = float(ev["dur"])

        if cat in _KERNEL_CATS:
            name = str(ev.get("name", ""))
            kernel_count += 1
            total_kernel_time += dur
            starts.append(ts)
            ends.append(ts + dur)
            by_name[name][0] += 1
            by_name[name][1] += dur
            gk = _grid_key(ev)
            by_grid[gk][0] += 1
            by_grid[gk][1] += dur
            kind = _classify_name(name)
            if kind == "tensor_core_gemm":
                tc_time += dur
            elif kind == "gemv":
                gemv_time += dur
            stream = (ev.get("args") or {}).get("stream")
            if stream is not None:
                streams.add(stream)
            if decode_start_us is not None:
                if ts < decode_start_us:
                    prefill += dur
                else:
                    decode += dur
            span_start = ts if span_start is None else min(span_start, ts)
            span_end = ts + dur if span_end is None else max(span_end, ts + dur)

        elif cat in _RUNTIME_CATS:
            name = str(ev.get("name", ""))
            runtime[name][0] += 1
            runtime[name][1] += dur
            runtime_total += dur

        elif cat in _MEMCPY_CATS:
            memcpy_count += 1
            memcpy_total += dur
            span_start = ts if span_start is None else min(span_start, ts)
            span_end = ts + dur if span_end is None else max(span_end, ts + dur)

    intervals = sorted(zip(starts, ends))
    busy = union_duration(intervals)
    span = (span_end - span_start) if (span_start is not None) else 0.0

    def _share(x: float) -> Optional[float]:
        return (x / total_kernel_time) if total_kernel_time else None

    def _top(d: Dict[str, List[float]], n: int) -> List[Dict[str, Any]]:
        items = sorted(d.items(), key=lambda kv: kv[1][1], reverse=True)[:n]
        return [{"name": k, "count": int(v[0]), "total_us": v[1],
                 "share_of_kernel_time": _share(v[1])} for k, v in items]

    if decode_start_us is not None:
        meta = boundary_meta or {}
        split = {
            "method": meta.get("method", "explicit_boundary"),
            "method_confidence": "measured",
            "decode_start_us": decode_start_us,
            "prefill_kernel_time_us": prefill,
            "decode_kernel_time_us": decode,
            "boundary_source": meta or {"note": "supplied by the caller"},
            "caveat": meta.get("confidence_note") or meta.get("note")
            or meta.get("assumption"),
        }
    else:
        split = {"method": "not_computed", "method_confidence": "none",
                 "caveat": "no boundary supplied; pass --raw to measure one"}

    return {
        "kernel_count": kernel_count,
        "total_kernel_time_us": total_kernel_time,
        "gpu_busy_us": busy,
        "trace_span_us": span,
        "trace_span_start_us": span_start or 0.0,
        "trace_span_end_us": span_end or 0.0,
        "gpu_busy_fraction": (busy / span) if span > 0 else None,
        "gpu_idle_fraction": (1.0 - busy / span) if span > 0 else None,
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
            ({"name": k, "count": int(v[0]), "total_cpu_us": v[1]}
             for k, v in runtime.items()),
            key=lambda d: d["total_cpu_us"], reverse=True,
        ),
        "cuda_runtime_total_cpu_us": runtime_total,
        "memcpy_count": memcpy_count,
        "memcpy_total_us": memcpy_total,
        "prefill_decode": split,
        "streamed": True,
        "caveats": [
            "gpu_busy_us is the union of kernel intervals; total_kernel_time_us "
            "is the plain sum and double-counts stream overlap.",
            "Kernel residency is not occupancy. Achieved occupancy and Tensor "
            "Core pipe utilization require ncu (Layer 4).",
            "tensor_core_gemm_share is the time share of kernels whose names "
            "indicate an HMMA GEMM, not a measure of tensor pipe efficiency.",
        ],
    }


def analyze_trace_file(
    path: str,
    out_path: Optional[str] = None,
    decode_start_us: Optional[float] = None,
    top_n: int = 15,
    boundary_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = analyze_trace(load_trace(path), decode_start_us=decode_start_us,
                           top_n=top_n, boundary_meta=boundary_meta)
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
        help="trace-clock timestamp of the prefill/decode boundary, supplied "
        "directly. Prefer --raw, which measures it.",
    )
    ap.add_argument(
        "--raw",
        default=None,
        help="path to the run's *_raw.json. MEASURES the prefill/decode "
        "boundary from t_first_token instead of guessing from kernel names.",
    )
    ap.add_argument(
        "--window-start-rel-s", type=float, default=None,
        help="when the profiled window opened, relative to the level's first "
        "submit. Lets a decode-only capture be identified as such.",
    )
    ap.add_argument("--section-wall-s", type=float, default=None)
    ap.add_argument(
        "--window-length-s", type=float, default=None,
        help="length of the profiled window (levels_detail[].profiling."
        "active_window_s). Required with --window-start-rel-s for a bounded "
        "capture: without it the boundary is divided by the LEVEL wall while "
        "the trace covers only the window, overstating the prefill share.",
    )
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument(
        "--stream", action="store_true",
        help="parse the trace incrementally instead of loading it whole. "
        "Required for real captures — a Layer 3 trace is multiple GB expanded.",
    )
    args = ap.parse_args(argv)

    if args.stream:
        return _main_stream(args)

    trace = load_trace(args.trace)
    decode_start_us = args.decode_start_us
    boundary_meta = None

    if args.raw:
        with open(args.raw) as fh:
            records = (json.load(fh) or {}).get("records") or []
        derived = derive_decode_start_us(
            trace, records,
            window_start_rel_s=args.window_start_rel_s,
            section_wall_s=args.section_wall_s,
            window_length_s=args.window_length_s,
        )
        if derived.get("available"):
            decode_start_us = derived["decode_start_us"]
            boundary_meta = derived
            print(f"[boundary] method={derived['method']} "
                  f"decode_start_us={decode_start_us:.1f}", file=sys.stderr)
        else:
            print(f"[boundary] could not measure: {derived.get('reason')} "
                  f"-- falling back to the NAME HEURISTIC, which its own "
                  f"caveat says not to quote.", file=sys.stderr)

    result = analyze_trace(trace, decode_start_us=decode_start_us,
                           top_n=args.top_n, boundary_meta=boundary_meta)
    result["source_trace"] = args.trace
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


def _main_stream(args) -> int:
    """Streaming path: anchor the clock on ProfilerStep, then analyze."""
    decode_start_us = args.decode_start_us
    boundary_meta = None

    if args.raw:
        with open(args.raw) as fh:
            records = (json.load(fh) or {}).get("records") or []

        print("[boundary] reading ProfilerStep anchors ...", file=sys.stderr)
        steps = profiler_step_anchors(args.trace)
        print(f"[boundary] {len(steps)} ProfilerStep annotation(s)", file=sys.stderr)

        anchored = anchor_decode_start_us(steps, records)
        if anchored.get("available"):
            decode_start_us = anchored["decode_start_us"]
            boundary_meta = anchored
            a = anchored["anchor"]
            print(f"[boundary] anchored on ProfilerStep#{a['step']} "
                  f"(match error {a['rel_error'] * 100:.2f}%)", file=sys.stderr)
        else:
            print(f"[boundary] anchoring failed: {anchored.get('reason')}",
                  file=sys.stderr)

    print("[trace] streaming ...", file=sys.stderr)
    result = analyze_trace_stream(
        args.trace, decode_start_us=decode_start_us, top_n=args.top_n,
        boundary_meta=boundary_meta,
    )
    result["source_trace"] = args.trace
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[trace] {args.out}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
