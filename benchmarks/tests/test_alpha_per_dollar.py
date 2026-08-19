"""Cost per unit of alpha.

The tests that matter are the ones that stop this analysis overclaiming:
a single window must not produce a ranking, a negative Sharpe must not become
a flattering cost ratio, and the sample-size question must refuse to answer
rather than answering the easy version of itself.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import alpha_per_dollar as ap        # noqa: E402
from analysis import make_alpha_report as rep      # noqa: E402
from analysis import tier_check                    # noqa: E402

_MD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "analysis", "ALPHA_PER_DOLLAR.md"))


@pytest.fixture(scope="module")
def table():
    return ap.cost_per_alpha_table()


@pytest.fixture(scope="module")
def report_text():
    return rep.build_report()


# ------------------------------------------------- refusing to overclaim ----

def test_the_data_is_one_window(table):
    assert table["board"]["n_windows"] == 1
    assert table["board"]["single_window"] is True


def test_can_rank_is_false_on_a_single_window(table):
    """The gate. One window cannot order seven models, whatever the CIs say."""
    assert table["can_rank"] is False


def test_sample_size_question_refuses_rather_than_answering_the_easy_version(table):
    """Within-window precision is not across-window generalisation.

    Answering with the within-window SE would give a small confident number to
    a question the data cannot address.
    """
    need = table["windows_needed_for_that_gap"]
    assert need["available"] is False
    assert "unestimable" in need["reason"] or "cannot be estimated" in need["reason"]
    assert need["what_would_resolve_it"]


def test_report_does_not_declare_a_winner(report_text):
    lowered = report_text.lower()
    assert "cannot rank" in lowered or "too small to rank" in lowered
    for banned in ("best model is", "we recommend switching to",
                   "the winner is"):
        assert banned not in lowered


def test_report_states_both_framings_of_the_best_llm(report_text):
    """deepseek beat SPY on return and lost on Sharpe. Both must appear."""
    assert "raw return" in report_text
    assert "risk-adjusted" in report_text


# ------------------------------------------------------------ statistics ----

def test_sharpe_se_shrinks_with_more_observations():
    assert ap.sharpe_standard_error(1.0, 1000) < ap.sharpe_standard_error(1.0, 100)


def test_sharpe_se_grows_with_larger_sharpe():
    """Lo's SE is not constant in S — a big Sharpe is estimated less precisely."""
    assert ap.sharpe_standard_error(5.0, 160) > ap.sharpe_standard_error(0.5, 160)


def test_sharpe_se_undefined_for_too_few_observations():
    assert ap.sharpe_standard_error(1.0, 1) is None


def test_confidence_interval_brackets_the_estimate():
    lo, hi = ap.sharpe_confidence_interval(2.0, 160)
    assert lo < 2.0 < hi


def test_rank_correlation_detects_perfect_agreement():
    assert ap.rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_rank_correlation_detects_perfect_inversion():
    assert ap.rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_rank_correlation_handles_ties():
    assert ap.rank_correlation([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_rank_correlation_needs_three_points():
    assert ap.rank_correlation([1, 2], [1, 2]) is None


def test_overlap_detects_separable_and_overlapping_pairs():
    entries = [
        {"label": "a", "sharpe_ci": (0.0, 1.0)},
        {"label": "b", "sharpe_ci": (0.5, 1.5)},   # overlaps a
        {"label": "c", "sharpe_ci": (5.0, 6.0)},   # separable from both
    ]
    o = ap.pairwise_overlap(entries)
    assert o["n_pairs"] == 3 and o["n_overlapping"] == 1
    assert len(o["separable_pairs"]) == 2


# ------------------------------------------------ the cost/quality join -----

def test_cost_per_sharpe_is_undefined_for_nonpositive_sharpe(table):
    """A negative denominator would sort as if it were excellent."""
    bad = [e for e in table["llm"] if (e["sharpe_recomputed"] or 0) <= 0]
    assert bad, "expected at least one non-positive Sharpe in the seed data"
    for e in bad:
        assert e["cost_per_sharpe"] is None
        assert "undefined" in e["cost_per_sharpe_note"]


def test_cost_span_is_the_measured_price_spread(table):
    assert table["cost_span"] > 100


def test_baselines_are_zero_llm_cost(table):
    assert table["baselines"]
    assert all(e["run_cost_usd"] == 0.0 for e in table["baselines"])


def test_no_llm_beats_the_best_baseline_on_sharpe(table):
    """The headline. Recorded as a test so a data change surfaces it."""
    assert table["llm_beating_best_baseline"] == []


def test_cost_does_not_predict_sharpe(table):
    """|rho| small — the price spread buys no rank correlation."""
    assert abs(table["cost_sharpe_rank_correlation"]) < 0.5


def test_every_llm_entry_carries_its_uncertainty(table):
    for e in table["llm"]:
        assert e["sharpe_ci"] is not None
        assert e["n_observations"] > 100


# ----------------------------------------------------------- provenance -----

def test_report_has_no_untagged_numbers(report_text):
    result = tier_check.check_report(report_text)
    assert result["ok"], tier_check.format_violations(result["violations"])


def test_committed_report_matches_the_generator(report_text):
    assert os.path.exists(_MD), "run: python -m analysis.make_alpha_report"
    with open(_MD) as fh:
        assert fh.read() == report_text, (
            "ALPHA_PER_DOLLAR.md is stale; regenerate with "
            "`python -m analysis.make_alpha_report`")


def test_report_explains_why_the_ablations_were_not_run(report_text):
    assert "Why the three ablations were not run" in report_text
    assert "no power" in report_text or "uninterpretable" in report_text


def test_report_records_the_zero_cache_result(report_text):
    """One check instead of one refactor — the answer must be in the report."""
    assert "0 cached tokens" in report_text
    assert "Do not do the refactor" in report_text
