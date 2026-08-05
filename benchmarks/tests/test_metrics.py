"""Unit tests for metrics: percentiles and the derived latency definitions.

No GPU, no torch. Run with:  pytest benchmarks/tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.metrics import RequestRecord, percentile, summarize  # noqa: E402


# --------------------------------------------------------------------------
# percentile
# --------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0]
    # rank = q/100 * (n-1) = 0.5*3 = 1.5 -> 2.0 + 0.5*(3.0-2.0) = 2.5
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 0) == pytest.approx(1.0)
    assert percentile(xs, 100) == pytest.approx(4.0)


def test_percentile_known_p95_p99():
    xs = [float(i) for i in range(1, 101)]  # 1..100
    # rank = 0.95*99 = 94.05 -> xs[94] + 0.05*(xs[95]-xs[94]) = 95 + 0.05 = 95.05
    assert percentile(xs, 95) == pytest.approx(95.05)
    assert percentile(xs, 99) == pytest.approx(99.01)


def test_percentile_single_value_and_empty():
    assert percentile([7.5], 50) == pytest.approx(7.5)
    assert percentile([7.5], 99) == pytest.approx(7.5)
    assert percentile([], 50) is None


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


def test_percentile_does_not_mutate_input():
    xs = [3.0, 1.0, 2.0]
    percentile(xs, 50)
    assert xs == [3.0, 1.0, 2.0]


# --------------------------------------------------------------------------
# per-request derived values
# --------------------------------------------------------------------------


def _rec(**kw):
    base = dict(
        request_id="r0",
        prompt_tokens=100,
        output_tokens=4,
        t_submit=0.0,
        t_first_token=1.0,
        per_token_timestamps=[1.0, 1.5, 2.0, 2.5],
        t_done=2.5,
    )
    base.update(kw)
    return RequestRecord(**base)


def test_ttft_is_first_token_minus_submit():
    assert _rec().ttft == pytest.approx(1.0)


def test_itl_has_n_minus_one_entries():
    """The v1 bug: dividing by token count instead of the number of gaps.

    Four timestamps describe three inter-token intervals, not four.
    """
    r = _rec()
    assert len(r.per_token_timestamps) == 4
    assert len(r.itl) == 3
    assert r.itl == pytest.approx([0.5, 0.5, 0.5])


def test_itl_empty_when_fewer_than_two_tokens():
    assert _rec(per_token_timestamps=[1.0]).itl == []
    assert _rec(per_token_timestamps=[]).itl == []


def test_itl_counts_zero_duration_chunks():
    """Whitespace-only chunks must contribute a timestamp.

    v1 skipped chunks failing `.strip()`, which removed them from the
    denominator while their elapsed time stayed in the numerator.
    """
    r = _rec(per_token_timestamps=[1.0, 1.0, 1.5], output_tokens=3)
    assert r.itl == pytest.approx([0.0, 0.5])


def test_e2e_and_ok_flag():
    r = _rec()
    assert r.e2e == pytest.approx(2.5)
    assert r.ok is True
    assert _rec(error="CUDA_OOM").ok is False


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def test_summary_reports_input_and_output_separately():
    """The headline v1 error: prefill and decode tokens summed into one rate.

    100 prompt tokens and 4 output tokens over a 2.5s window are three
    different rates, and the total is not a decode number.
    """
    s = summarize("run", 1, [_rec()])
    assert s.input_tokens_total == 100
    assert s.output_tokens_total == 4
    assert s.total_tokens == 104
    assert s.input_tok_per_s == pytest.approx(40.0)
    assert s.output_tok_per_s == pytest.approx(1.6)
    assert s.total_tok_per_s == pytest.approx(41.6)
    # The combined figure carries its own health warning wherever it is emitted.
    assert "total_tok_per_s_warning" in s.notes


def test_summary_wall_time_spans_first_submit_to_last_done():
    a = _rec(request_id="a", t_submit=0.0, t_done=2.0,
             per_token_timestamps=[1.0, 2.0], output_tokens=2)
    b = _rec(request_id="b", t_submit=1.0, t_done=5.0,
             per_token_timestamps=[3.0, 5.0], output_tokens=2)
    s = summarize("run", 2, [a, b])
    assert s.wall_time == pytest.approx(5.0)
    assert s.completed_requests_per_s == pytest.approx(0.4)


def test_summary_excludes_errored_requests_from_tokens_and_latency():
    ok = _rec(request_id="ok")
    bad = _rec(request_id="bad", error="CUDA_OOM")
    s = summarize("run", 2, [ok, bad])
    assert s.completed == 1
    assert s.errored == 1
    assert s.input_tokens_total == 100  # only the successful one
    assert s.ttft_p50 == pytest.approx(1.0)


def test_summary_all_errored_is_valid_not_a_crash():
    """A level that fully OOMs is a result, not a lost run."""
    s = summarize("run", 105, [_rec(error="CUDA_OOM"), _rec(error="CUDA_OOM")])
    assert s.completed == 0
    assert s.errored == 2
    assert s.wall_time == 0.0
    assert s.output_tok_per_s is None
    assert s.ttft_p50 is None


def test_summary_requested_defaults_to_record_count():
    s = summarize("run", 4, [_rec(), _rec()], requested=8)
    assert s.requested == 8
    assert summarize("run", 4, [_rec()]).requested == 1


def test_itl_samples_aggregate_across_requests():
    s = summarize("run", 2, [_rec(request_id="a"), _rec(request_id="b")])
    assert s.notes["itl_sample_count"] == 6  # 3 gaps x 2 requests
    assert s.itl_mean == pytest.approx(0.5)
