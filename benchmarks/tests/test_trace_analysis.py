"""Unit tests for trace_analysis against a hand-built fake chrome trace.

No GPU, no torch — the analyzer consumes a plain dict, which is precisely why
it was written to take one.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.trace_analysis import (  # noqa: E402
    analyze_trace,
    merge_intervals,
    union_duration,
)


def _kernel(name, ts, dur, stream=7, grid=(64, 1, 1)):
    return {
        "ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,
        "args": {"stream": stream, "grid": list(grid), "block": [128, 1, 1]},
    }


def _runtime(name, ts, dur):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur, "args": {}}


@pytest.fixture
def trace():
    """Two overlapping streams, a GEMM, GEMVs, and runtime API calls.

    Kernel intervals: (0,400) (300,600) (500,600) (700,800) (900,1000).
    Durations sum to 1000us; their union is 800us — (300,600) on stream 9
    overlaps the GEMM, and (500,600) sits inside it. sum/union = 1.25,
    deliberately >1 to exercise the overlap path. Wall span is 0..1000us, so
    busy = 0.8 and idle = 0.2.
    """
    return {
        "traceEvents": [
            # stream 7: prefill GEMM 0-400, then decode GEMVs
            _kernel("volta_s16816gemm_f16_128x128", 0, 400, stream=7, grid=(128, 8, 1)),
            _kernel("gemv2T_kernel_val", 500, 100, stream=7, grid=(1, 1, 1)),
            _kernel("gemv2T_kernel_val", 700, 100, stream=7, grid=(1, 1, 1)),
            # stream 9: overlaps the GEMM 300-600
            _kernel("elementwise_add", 300, 300, stream=9, grid=(32, 1, 1)),
            # stream 7: tail
            _kernel("layer_norm", 900, 100, stream=7, grid=(16, 1, 1)),
            # runtime
            _runtime("cudaLaunchKernel", 0, 12),
            _runtime("cudaLaunchKernel", 480, 8),
            _runtime("cudaStreamSynchronize", 990, 55),
            _runtime("cudaMemcpyAsync", 100, 5),
            # must be ignored: instant event, no duration
            {"ph": "i", "cat": "kernel", "name": "marker", "ts": 50},
            # must be ignored: metadata
            {"ph": "M", "name": "process_name", "args": {"name": "python"}},
        ]
    }


# --------------------------------------------------------------------------
# interval union
# --------------------------------------------------------------------------


def test_merge_intervals_merges_overlap_and_touch():
    assert merge_intervals([(0, 10), (5, 20)]) == [(0, 20)]
    assert merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]
    assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]


def test_merge_intervals_handles_containment_and_unsorted_input():
    assert merge_intervals([(0, 100), (10, 20)]) == [(0, 100)]
    assert merge_intervals([(50, 60), (0, 10)]) == [(0, 10), (50, 60)]


def test_union_duration_counts_overlap_once():
    assert union_duration([(0, 10), (5, 15)]) == pytest.approx(15.0)
    assert union_duration([]) == 0.0


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_sum_and_union_differ_under_overlap(trace):
    """Summing kernel durations double-counts concurrent streams.

    Reporting the sum as GPU busy time is how a device gets described as
    saturated when it is not.
    """
    r = analyze_trace(trace)
    assert r["total_kernel_time_us"] == pytest.approx(1000.0)
    assert r["gpu_busy_us"] == pytest.approx(800.0)
    assert r["kernel_overlap_factor"] == pytest.approx(1.25)


def test_span_and_idle_fraction(trace):
    r = analyze_trace(trace)
    assert r["trace_span_us"] == pytest.approx(1000.0)
    assert r["gpu_busy_fraction"] == pytest.approx(0.8)
    assert r["gpu_idle_fraction"] == pytest.approx(0.2)


def test_non_complete_events_are_excluded(trace):
    """Instant ('i') and metadata ('M') events carry no duration."""
    r = analyze_trace(trace)
    assert r["kernel_count"] == 5  # the 'marker' instant event is not one


def test_distinct_streams(trace):
    r = analyze_trace(trace)
    assert r["distinct_stream_count"] == 2
    assert set(r["distinct_streams"]) == {7, 9}


def test_tensor_core_and_gemv_shares(trace):
    r = analyze_trace(trace)
    assert r["tensor_core_gemm_time_us"] == pytest.approx(400.0)
    assert r["tensor_core_gemm_share"] == pytest.approx(0.4)
    assert r["gemv_time_us"] == pytest.approx(200.0)
    assert r["gemv_share"] == pytest.approx(0.2)


def test_kernel_grouping_by_name_and_grid(trace):
    r = analyze_trace(trace)
    top = {k["name"]: k for k in r["top_kernels_by_name"]}
    assert top["gemv2T_kernel_val"]["count"] == 2
    assert top["gemv2T_kernel_val"]["total_us"] == pytest.approx(200.0)

    grids = {g["name"]: g for g in r["kernel_time_by_grid"]}
    assert "128x8x1" in grids
    assert grids["1x1x1"]["count"] == 2


def test_top_n_is_respected(trace):
    r = analyze_trace(trace, top_n=2)
    assert len(r["top_kernels_by_name"]) == 2


def test_cuda_runtime_breakdown_is_the_launch_wait_metric(trace):
    """cudaLaunchKernel / Synchronize / Memcpy CPU time — the 'time spent
    launching, scheduling and waiting on kernels' figure."""
    r = analyze_trace(trace)
    bd = {e["name"]: e for e in r["cuda_runtime_breakdown"]}
    assert bd["cudaLaunchKernel"]["count"] == 2
    assert bd["cudaLaunchKernel"]["total_cpu_us"] == pytest.approx(20.0)
    assert bd["cudaStreamSynchronize"]["total_cpu_us"] == pytest.approx(55.0)
    assert r["cuda_runtime_total_cpu_us"] == pytest.approx(80.0)
    # sorted by cost so the dominant call is first
    assert r["cuda_runtime_breakdown"][0]["name"] == "cudaStreamSynchronize"


def test_prefill_decode_split_with_explicit_boundary(trace):
    """The trustworthy path: the runner supplies t_first_token."""
    r = analyze_trace(trace, decode_start_us=450.0)
    pd = r["prefill_decode"]
    assert pd["method"] == "explicit_boundary"
    assert pd["prefill_kernel_time_us"] == pytest.approx(700.0)  # 400 GEMM + 300 elementwise
    assert pd["decode_kernel_time_us"] == pytest.approx(300.0)
    assert pd["caveat"] is None


def test_prefill_decode_split_falls_back_and_says_so(trace):
    r = analyze_trace(trace)
    pd = r["prefill_decode"]
    assert pd["method"] == "heuristic_name_shape"
    assert pd["prefill_kernel_time_us"] == pytest.approx(400.0)
    assert pd["decode_kernel_time_us"] == pytest.approx(200.0)
    # elementwise_add + layer_norm are attributed to neither, not silently binned
    assert pd["unattributed_kernel_time_us"] == pytest.approx(400.0)
    assert "Do not quote this split" in pd["caveat"]


def test_empty_trace_does_not_crash():
    r = analyze_trace({"traceEvents": []})
    assert r["kernel_count"] == 0
    assert r["total_kernel_time_us"] == 0.0
    assert r["gpu_busy_fraction"] is None
    assert r["tensor_core_gemm_share"] is None


def test_caveats_are_always_emitted(trace):
    r = analyze_trace(trace)
    joined = " ".join(r["caveats"])
    assert "not occupancy" in joined
    assert "ncu" in joined
