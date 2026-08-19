"""SERVING_COST.md — scope enforcement and the claims it must carry.

Scope creep is the failure mode for this document. The project has produced a
GPU benchmark, a latency audit, a cadence re-scope and a multi-model design,
and every one of them is easy to reach for and would obscure the cost answer.
So the exclusion is a test, not a resolution.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import make_serving_cost_report as rep   # noqa: E402
from analysis import tier_check                        # noqa: E402

_MD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "analysis", "SERVING_COST.md"))


@pytest.fixture(scope="module")
def text():
    return rep.build_report()


# ------------------------------------------------------------ the scope -----

def test_no_excluded_term_appears(text):
    """The whole point of this file."""
    lowered = text.lower()
    drift = [t for t in rep.EXCLUDED_TERMS if t.lower() in lowered]
    assert not drift, f"out-of-scope terms leaked into the report: {drift}"


def test_the_scope_guard_is_not_vacuous(text):
    """A guard that cannot fire proves nothing."""
    poisoned = text + "\nArm C reached 9.28 req/s on an A100 with vLLM.\n"
    lowered = poisoned.lower()
    hits = [t for t in rep.EXCLUDED_TERMS if t.lower() in lowered]
    assert {"arm C", "req/s", "A100", "vLLM"} <= set(hits)


def test_generator_refuses_to_write_out_of_scope(monkeypatch, tmp_path):
    """main() exits non-zero rather than emitting a drifted report."""
    monkeypatch.setattr(
        rep, "build_report",
        lambda: "# T\n\nArm C hit 9.28 req/s [MEASURED].\n")
    code = rep.main(["--out", str(tmp_path / "x.md")])
    assert code == 4
    assert not (tmp_path / "x.md").exists()


def test_generator_refuses_to_write_untagged(monkeypatch, tmp_path):
    monkeypatch.setattr(rep, "build_report",
                        lambda: "# T\n\nThe platform made 4,182 calls.\n")
    code = rep.main(["--out", str(tmp_path / "x.md")])
    assert code == 3
    assert not (tmp_path / "x.md").exists()


@pytest.mark.parametrize("category,term", [
    ("gpu benchmark", "cudaLaunchKernel"),
    ("gpu benchmark", "173x"),
    ("cadence re-scope", "Nof1"),
    ("latency audit", "fill price"),
    ("multi-model sweep", "models-per-card"),
    ("self-hosting economics", "crossover"),
])
def test_each_excluded_category_is_actually_guarded(category, term):
    assert term in rep.EXCLUDED_TERMS, f"{category} not guarded: {term}"


# ---------------------------------------------------- what it must carry ----

def test_states_no_saving_has_been_measured(text):
    assert "No cost reduction has been measured" in text


def test_carries_the_three_measured_spreads(text):
    for s in ("1.4x", "5.8x", "193x"):
        assert s in text, f"missing spread {s}"


def test_carries_pricing_verification(text):
    assert "1e-5" in text


def test_carries_calls_per_decision_exactly(text):
    assert "| 3 | 3.000 | yes |" in text
    assert "| 5 | 5.000 | yes |" in text
    assert "no retry inflation" in text


def test_carries_the_uncorrelated_cost_quality_result(text):
    assert "+0.071" in text
    assert "Spearman" in text


def test_carries_both_framings_of_the_best_model(text):
    """Raw return and risk-adjusted disagree; picking one would be dishonest."""
    assert "raw return" in text
    assert "risk-adjusted" in text
    assert "7.49%" in text and "5.95%" in text


def test_carries_the_n_equals_1_limit_prominently(text):
    assert "can_rank" in text and "False" in text
    assert "windows_needed" in text and "refuses" in text
    assert "Lo standard error" in text and "iid" in text
    assert "too small to rank" in text


def test_n1_limit_appears_in_the_same_section_as_the_result(text):
    """A limitation two sections away from its claim is not prominent."""
    result_at = text.index("+0.071")
    limit_at = text.index("cannot rank the models")
    assert 0 < (limit_at - result_at) < 3000


def test_carries_the_zero_cache_result_and_its_consequence(text):
    assert "0 cached tokens" in text
    assert "do not do the prompt-reordering refactor" in text.lower()
    assert "this model and this endpoint" in text


def test_carries_the_observability_gap(text):
    assert "`+=`" in text
    assert "quiet week" in text


def test_carries_the_sensitivity_ranking(text):
    for span in ("134.4x", "6.0x", "2.3x", "1.3x"):
        assert span in text, f"missing lever span {span}"


def test_carries_what_is_built_and_unmerged(text):
    assert "unmerged" in text
    for change in ("Per-call usage logging", "Backtest result cache",
                   "Leaderboard governance"):
        assert change in text


def test_says_each_built_saving_is_unmeasured(text):
    assert "unmeasured on purpose" in text


def test_has_the_required_five_sections(text):
    for heading in ("## 1. What a call costs",
                    "## 2. What that spend buys",
                    "## 3. What has been built",
                    "## 4. What is blocked",
                    "## 5. What cannot be measured yet"):
        assert heading in text, f"missing section: {heading}"


# ------------------------------------------------------------ provenance ----

def test_no_untagged_numbers(text):
    result = tier_check.check_report(text)
    assert result["ok"], tier_check.format_violations(result["violations"])


def test_committed_report_matches_the_generator(text):
    assert os.path.exists(_MD), "run: python -m analysis.make_serving_cost_report"
    with open(_MD) as fh:
        assert fh.read() == text, (
            "SERVING_COST.md is stale; regenerate with "
            "`python -m analysis.make_serving_cost_report`")
