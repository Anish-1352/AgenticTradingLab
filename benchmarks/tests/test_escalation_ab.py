"""Pooling and the escalation report.

The single-pair result said output tokens rose and wall time got worse.
Pooling three pairs reversed both. These tests guard the pooling and the
Fisher test that made that difference legible, and pin the report to the data
rather than to prose written before it arrived.
"""

import io
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, ".."))
_ANALYSIS = os.path.join(_BENCH, "analysis")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from analysis import escalation_ab as ab  # noqa: E402
from analysis.tier_check import check_report  # noqa: E402

REPORT = os.path.join(_ANALYSIS, "ESCALATION_ON_FIRST_RETRY.md")
AB = os.path.join(_BENCH, "results", "escalation_ab.json")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(AB):
        pytest.skip("no pooled result committed")
    with io.open(AB, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------ fisher ------

def test_no_difference_is_not_significant():
    assert ab.fisher_exact_greater(7, 7, 7, 7) > 0.5


def test_a_wiped_out_tail_is_significant():
    assert ab.fisher_exact_greater(14, 28, 0, 42) < 0.001


def test_identical_empty_tails_are_certain():
    assert ab.fisher_exact_greater(0, 14, 0, 14) == pytest.approx(1.0)


def test_p_is_monotonic_in_the_effect():
    """A bigger gap must not produce a larger p."""
    a = ab.fisher_exact_greater(4, 10, 0, 14)
    b = ab.fisher_exact_greater(8, 6, 0, 14)
    assert b < a


def test_fisher_never_exceeds_one():
    for args in [(0, 1, 0, 1), (1, 0, 1, 0), (3, 3, 3, 3), (14, 28, 0, 42)]:
        assert 0.0 <= ab.fisher_exact_greater(*args) <= 1.0


# ------------------------------------------------------------ pooling -----

def test_pooling_sums_the_arms(r):
    for side in ("off", "on"):
        p = r[side]
        assert p["decisions"] == 42
        assert p["attempts"] == sum(
            int(k) * v for k, v in p["histogram"].items())
        assert sum(p["histogram"].values()) == p["decisions"]


def test_attempts_per_decision_is_the_pooled_ratio(r):
    for side in ("off", "on"):
        p = r[side]
        assert p["attempts_per_decision"] == pytest.approx(
            p["attempts"] / p["decisions"])


def test_the_tail_count_matches_the_histogram(r):
    for side in ("off", "on"):
        p = r[side]
        tail = sum(v for k, v in p["histogram"].items()
                   if int(k) >= ab.TAIL_THRESHOLD)
        assert p["tail_decisions"] == tail


# ------------------------------------------------------------ the claim ---

def test_the_flag_removed_the_tail_entirely(r):
    assert r["off"]["tail_decisions"] > 0
    assert r["on"]["tail_decisions"] == 0
    assert r["tail_p_one_sided"] < 0.01


def test_no_decision_exceeded_three_attempts_with_the_flag_on(r):
    assert max(int(k) for k in r["on"]["histogram"]) <= 3


def test_the_flag_reduces_calls_per_decision(r):
    assert r["deltas"]["attempts_per_decision"] < 0


def test_off_arm_variance_justifies_pairing():
    """If single runs were stable, pairing would be needless ceremony."""
    rates = []
    for name in ab.OFF_ARMS:
        d = ab._load(name)
        if d:
            rates.append(d["summary"]["attempts_per_decision"])
    assert len(rates) >= 2
    assert max(rates) - min(rates) > 0.5, (
        "off-arm spread is small; the paired design may be over-engineered")


# ------------------------------------------------------------ report ------

def test_report_is_not_stale():
    from analysis import make_escalation_report as mk
    with io.open(REPORT, encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "ESCALATION_ON_FIRST_RETRY.md is stale; regenerate with "
            "`python benchmarks/analysis/make_escalation_report.py`")


def test_report_is_fully_tagged():
    with io.open(REPORT, encoding="utf-8") as fh:
        assert check_report(fh.read())["ok"]


def test_report_verdicts_agree_with_the_measured_deltas(r):
    """The prose is generated from thresholds precisely because an earlier
    draft asserted 'cost is flat' against a -23.6% measurement."""
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    cheaper = r["deltas"]["cost_usd"] < -0.10
    assert ("| Cheaper | yes" in t) == cheaper


def test_report_does_not_claim_a_returns_result(r):
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "settles nothing" in t
    assert "[NOT MEASURED]" in t


def test_generator_runs_as_a_script():
    out = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_escalation_report.py"),
         "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
