"""Nof1-cadence re-scope.

The property that matters most: **the hourly figures still exist and still say
what they said.** The re-scope is a comparison, and a comparison with one side
deleted is just a replacement.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import cadence_model as cm            # noqa: E402
from analysis import make_cadence_report as rep     # noqa: E402
from analysis import tier_check                     # noqa: E402
from analysis.cost_model_lib import MEASURED        # noqa: E402

_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
_RESCOPE_MD = os.path.join(_ANALYSIS, "CADENCE_RESCOPE.md")
_ADVISOR_MD = os.path.join(_ANALYSIS, "ADVISOR_REPORT.md")


@pytest.fixture(scope="module")
def report_text():
    return rep.build_report()


# ------------------------------------------------------- the hourly figures --

def test_hourly_report_still_exists():
    """The re-scope must not have replaced the hourly analysis."""
    assert os.path.exists(_ADVISOR_MD)


def test_hourly_report_still_states_its_hourly_conclusion():
    with open(_ADVISOR_MD) as fh:
        text = fh.read()
    # The live-trading-cannot-reach-$50K conclusion is hourly-specific and must
    # survive intact; the re-scope contradicts it only at a different cadence.
    assert "live trading alone cannot" in text


def test_hourly_cadence_is_measured_not_assumed():
    assert cm.CADENCES["hourly"].tier == MEASURED
    assert cm.CADENCES["hourly"].decisions_per_day == 7


@pytest.mark.parametrize("key", ["nof1_equity", "nof1_crypto"])
def test_nof1_cadences_are_tagged_assumed(key):
    assert cm.CADENCES[key].tier == cm.ASSUMED
    assert "not confirmed" in cm.CADENCES[key].tier


# ------------------------------------------------------------ the inversion --

def test_nothing_crosses_budget_at_hourly_cadence():
    """The earlier conclusion, re-derived rather than asserted."""
    t = cm.cadence_table()
    hourly = [r for r in t["rows"] if r["cadence"] == "hourly"]
    assert hourly and not any(r["over_budget"] for r in hourly)


def test_most_of_the_table_crosses_at_crypto_cadence():
    t = cm.cadence_table()
    rows = [r for r in t["rows"]
            if r["cadence"] == "nof1_crypto" and r["calls_per_decision"] == 3]
    assert sum(1 for r in rows if r["over_budget"]) >= 5


def test_the_production_default_survives_every_cadence_at_one_call():
    """Only the cheapest model stays under budget across the board."""
    t = cm.cadence_table()
    nemo = [r for r in t["rows"]
            if r["db_model"] == "nemotron_3_nano_30b" and r["calls_per_decision"] == 1]
    assert nemo and not any(r["over_budget"] for r in nemo)


def test_calls_scale_linearly_with_cadence_and_depth():
    t = cm.cadence_table(agents=300)
    def calls(cad, depth):
        return next(r["calls_per_day"] for r in t["rows"]
                    if r["cadence"] == cad and r["calls_per_decision"] == depth)
    assert calls("hourly", 1) == 300 * 7
    assert calls("nof1_equity", 3) == 3 * calls("nof1_equity", 1)
    assert calls("nof1_crypto", 1) == 300 * 500


# ------------------------------------------------------------- crossover -----

def test_crossover_uses_measured_arm_c_throughput():
    x = cm.crossover_table()
    assert x["arm_c_rps"] == pytest.approx(9.2801, rel=1e-3)


def test_crossover_is_inversely_proportional_to_cost_per_call():
    x = cm.crossover_table()
    rows = x["rows"]
    # Cheapest model needs the most volume before a GPU pays for itself.
    assert rows[0]["crossover_calls_per_day"] > rows[-1]["crossover_calls_per_day"]


def test_self_hosting_does_not_pay_at_hourly_for_cheap_models():
    """The earlier conclusion holds where it was made."""
    x = cm.crossover_table()
    nemo = next(r for r in x["rows"] if r["db_model"] == "nemotron_3_nano_30b")
    assert nemo["by_cadence"]["hourly"]["past_crossover"] is False


def test_self_hosting_pays_for_most_models_at_nof1_equity():
    x = cm.crossover_table()
    past = [r for r in x["rows"] if r["by_cadence"]["nof1_equity"]["past_crossover"]]
    assert len(past) >= 6


def test_one_gpu_suffices_at_every_cadence():
    x = cm.crossover_table()
    assert all(c["gpus_for_300_agents"] == 1 for c in x["ceilings"].values())


def test_gpu_ceiling_falls_as_cadence_rises():
    x = cm.crossover_table()
    assert (x["ceilings"]["hourly"]["agents_per_gpu"]
            > x["ceilings"]["nof1_equity"]["agents_per_gpu"])


# ---------------------------------------------------------------- latency ----

def test_arm_b_at_c32_overruns_every_sub_hourly_bar():
    t = cm.latency_vs_bar_table()
    row = next(r for r in t["rows"] if r["arm"] == "B" and r["concurrency"] == 32)
    for bar in (60, 120, 150, 180, 300):
        assert row["fits"][bar]["p50"] is False
    assert row["fits"][3600]["p95"] is True   # and fitted at hourly


def test_arm_c_fits_every_candidate_bar():
    t = cm.latency_vs_bar_table()
    for row in [r for r in t["rows"] if r["arm"] == "C"]:
        for bar in t["bar_intervals_s"]:
            assert row["fits"][bar]["p95"] is True


def test_arm_b_c8_is_marginal_at_150s():
    """148.3s against a 150s bar is inside it, but not by anything usable."""
    t = cm.latency_vs_bar_table()
    row = next(r for r in t["rows"] if r["arm"] == "B" and r["concurrency"] == 8)
    assert row["fits"][150]["p50"] is True
    assert row["fits"][120]["p50"] is False
    assert 0.95 < row["fits"][150]["overrun_factor_p50"] < 1.0


def test_arm_a_carries_its_incomparability_exclusion():
    t = cm.latency_vs_bar_table()
    assert all(r["comparable"] is False for r in t["rows"] if r["arm"] == "A")
    assert "min_tokens" in t["arm_a_exclusion"]


# ---------------------------------------------------------------- blockers ---

def test_every_blocker_citation_is_live():
    """Audited against origin/main — what ships — not this branch."""
    verified = rep.verify_blockers()
    assert verified
    bad = [f"{b['file']}:{b['line']}" for b in verified if not b["verified"]]
    assert not bad, f"citations no longer match origin/main: {bad}"


def test_blockers_are_audited_against_main():
    assert all(b["ref"] == "origin/main" for b in rep.BLOCKERS)


def test_scheduler_and_bar_interval_are_both_named():
    ids = {b["id"] for b in rep.BLOCKERS}
    assert {"no_scheduler", "hourly_hardcoded", "fill_equals_decision_price"} <= ids


# ------------------------------------------------------------ fixture gap ----

def test_fixture_gap_separates_what_changes_from_what_holds():
    g = cm.FIXTURE_GAP
    assert g["claims_that_would_change"] and g["claims_that_hold"]
    joined = " ".join(g["claims_that_would_change"])
    assert "173x" in joined                      # throughput ratio does not transfer
    held = " ".join(g["claims_that_hold"])
    assert "launch-bound" in held                # mechanism findings do


def test_fixture_gap_is_flagged_not_closed():
    assert "never been executed" in cm.FIXTURE_GAP["never_run"]
    assert "TOP priority" in cm.FIXTURE_GAP["priority"]


# ------------------------------------------------------------------ report ---

def test_report_has_no_untagged_numbers(report_text):
    result = tier_check.check_report(report_text)
    assert result["ok"], tier_check.format_violations(result["violations"])


def test_committed_report_matches_the_generator(report_text):
    assert os.path.exists(_RESCOPE_MD), "run: python -m analysis.make_cadence_report"
    with open(_RESCOPE_MD) as fh:
        assert fh.read() == report_text, (
            "CADENCE_RESCOPE.md is stale; regenerate with "
            "`python -m analysis.make_cadence_report`")


def test_report_marks_cadence_as_assumed(report_text):
    assert "ASSUMED — not confirmed with advisor" in report_text


def test_report_says_the_hourly_figures_are_not_superseded(report_text):
    assert "not superseded" in report_text


def test_tier_checker_accepts_assumed():
    assert tier_check.check_report(
        "# T\n\n156 decisions/day [ASSUMED — not confirmed with advisor].\n")["ok"]


def test_tier_checker_still_rejects_untagged():
    assert not tier_check.check_report("# T\n\n156 decisions per day.\n")["ok"]
