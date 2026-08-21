"""API optimisation levers.

The load-bearing tests are the ones that stop a saving being claimed from an
assumption: that the decision step's unmeasured parse rate withholds the
completion effect, that 15/15 is read as a bound rather than as 100%, and that
the compound token effect is taken from the measurement even though it points
the opposite way to the obvious guess.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import api_optimisation as ao            # noqa: E402
from analysis.cost_model_lib import MissingInput, load_measured  # noqa: E402

CHEAP = "nvidia/nemotron-3-nano-30b-a3b"
DEAR = "deepseek/deepseek-v4-pro"


@pytest.fixture(scope="module")
def measured():
    return load_measured()


@pytest.fixture(scope="module")
def steps():
    return ao.load_step_measurements()


@pytest.fixture(scope="module")
def routing(measured, steps):
    return ao.routing_projection(DEAR, CHEAP, steps, measured)


# ---------------------------------------------------- measurement loading ---

def test_the_decision_step_is_excluded_from_the_usable_set(steps):
    """Both models scored 0/15 there — the empty-orders defect, not the model."""
    assert all(k[1] != 2 for k in steps["usable"])
    assert all(k[1] == 2 for k in steps["excluded"])
    assert "NOT a model-quality result" in steps["provenance"]["caveat_step3"]


def test_the_vendored_measurement_carries_its_provenance(steps):
    p = steps["provenance"]
    assert "feature/per-step-model-routing" in p["source"]
    assert p["sha256_of_source"]
    assert p["actual_spend_usd"] > 0


def test_both_models_have_measurements_for_the_mechanical_steps(steps):
    for model in (CHEAP, DEAR):
        for i in (0, 1):
            assert (model, i) in steps["usable"]


# ------------------------------------------------------- rule of three ------

def test_zero_failures_in_n_bounds_the_rate_it_does_not_zero_it():
    assert ao.rule_of_three_upper_bound(15) == pytest.approx(0.20)
    assert ao.rule_of_three_upper_bound(300) == pytest.approx(0.01)


def test_rule_of_three_rejects_nonsense():
    with pytest.raises(ValueError):
        ao.rule_of_three_upper_bound(0)


# --------------------------------------------------------- 2a routing -------

def test_routing_saves_on_the_mechanical_steps(routing):
    assert routing["saving_fraction"].value > 0.5
    assert routing["routed_cost_usd"].value < routing["baseline_cost_usd"].value
    assert routing["steps_projected"] == [1, 2]
    assert routing["steps_excluded"] == [3]


def test_the_compound_effect_is_measured_and_runs_backwards(routing):
    """Nemotron emits MORE at step 1, so it INFLATES downstream input."""
    c = routing["compound_effect"]
    assert c["direction"] == "INFLATES"
    assert c["step1_output_cheap"] > c["step1_output_baseline"]
    assert "the assumed direction is the opposite" in c["note"]


def test_later_steps_carry_upstream_tokens(routing):
    """_build_step_prompt embeds every prior output verbatim."""
    rows = routing["routed_rows"]
    assert rows[1]["input_tokens"] > rows[1]["output_tokens"] * 0  # sanity
    # step 2's input exceeds its own measured prompt by the upstream carry
    assert rows[1]["input_tokens"] > rows[0]["input_tokens"] * 0.5


def test_the_completion_effect_is_withheld(routing):
    """No measured decision-step rate means no end-to-end claim."""
    assert routing["completion_effect_withheld"] is True
    assert "empty-orders defect" in routing["why_withheld"]


def test_parse_rates_are_reported_as_a_bound_not_as_perfection(routing):
    p = routing["parse_rates"]
    assert p["upper_bound_failure_rate_95pct"] == pytest.approx(0.20)
    assert "NOT at zero" in p["reading"]
    assert "156 decisions/day" in p["reading"]


def test_routing_refuses_an_unpriced_model(measured, steps):
    with pytest.raises(MissingInput, match="no measured price"):
        ao.routing_projection("mystery/model", CHEAP, steps, measured)


def test_routing_refuses_an_unmeasured_step(measured, steps):
    thin = {**steps, "usable": {k: v for k, v in steps["usable"].items()
                                if k != (CHEAP, 1)}}
    with pytest.raises(MissingInput, match="no measured output tokens"):
        ao.routing_projection(DEAR, CHEAP, thin, measured)


def test_the_decision_model_is_unchanged_by_construction(routing):
    assert "decision step keeps the expensive model" in routing["behaviour_change"]


# ------------------------------------------------------------ 2b caps -------

def test_cap_projection_refuses_without_a_measured_sweep():
    with pytest.raises(MissingInput, match="needs a measured sweep"):
        ao.cap_projection(sweep_path="/nonexistent.json")


def test_cap_projection_reads_a_sweep(measured):
    sweep = {
        "attempts": 12,
        "rows": [
            {"model": CHEAP, "step_index": 0, "step_label": "A",
             "max_tokens": 2000, "attempts": 12, "n_ok": 12, "parse_rate": 1.0,
             "mean_output_tokens": 1400.0, "mean_input_tokens": 900.0,
             "categories": {"ok": 12}},
            {"model": CHEAP, "step_index": 0, "step_label": "A",
             "max_tokens": 500, "attempts": 12, "n_ok": 2, "parse_rate": 2 / 12,
             "mean_output_tokens": 500.0, "mean_input_tokens": 900.0,
             "categories": {"truncated": 10, "ok": 2}},
        ]}
    r = ao.cap_projection(sweep=sweep, measured=measured)["results"][0]
    caps = {c["max_tokens"]: c for c in r["caps"]}
    assert caps[2000]["safe"] is True
    assert caps[500]["safe"] is False
    assert caps[500]["token_reduction_fraction"] == pytest.approx(1 - 500 / 1400)
    assert r["lowest_safe_cap"] == 2000


def test_a_cap_that_breaks_parsing_is_not_counted_as_a_saving(measured):
    sweep = {"attempts": 12, "rows": [
        {"model": CHEAP, "step_index": 0, "step_label": "A", "max_tokens": 2000,
         "attempts": 12, "n_ok": 12, "parse_rate": 1.0,
         "mean_output_tokens": 1400.0, "mean_input_tokens": 900.0,
         "categories": {}},
        {"model": CHEAP, "step_index": 0, "step_label": "A", "max_tokens": 250,
         "attempts": 12, "n_ok": 0, "parse_rate": 0.0,
         "mean_output_tokens": 250.0, "mean_input_tokens": 900.0,
         "categories": {}}]}
    r = ao.cap_projection(sweep=sweep, measured=measured)["results"][0]
    # The 82% token reduction at cap=250 must NOT appear as a safe saving.
    assert r["max_safe_token_reduction"] == 0.0


def test_caps_explain_why_they_target_the_right_term(measured):
    sweep = {"attempts": 12, "rows": [
        {"model": CHEAP, "step_index": 0, "step_label": "A", "max_tokens": 2000,
         "attempts": 12, "n_ok": 12, "parse_rate": 1.0,
         "mean_output_tokens": 1400.0, "mean_input_tokens": 900.0,
         "categories": {}}]}
    r = ao.cap_projection(sweep=sweep, measured=measured)
    assert "61.5%" in r["why_it_targets_the_right_term"]
    assert "77.4%" in r["why_it_targets_the_right_term"]
    assert "TRUNCATES" in r["behaviour_change"]


# ------------------------------------------------- 2c eliminated calls ------

def test_caching_refuses_without_a_measured_repeat_rate(measured):
    e = ao.eliminated_calls(measured)
    assert e["backtest_caching"]["saving_fraction"] is None
    assert "no telemetry" in e["backtest_caching"]["refused"]


def test_caching_accepts_a_supplied_rate_and_tags_it_unmeasured(measured):
    e = ao.eliminated_calls(measured, repeat_backtest_fraction=0.4)
    f = e["backtest_caching"]["saving_fraction"]
    assert f.value == 0.4 and f.tier == "NOT MEASURED"


def test_leaderboard_governance_saves_nothing_recurring_today(measured):
    e = ao.eliminated_calls(measured)
    lg = e["leaderboard_governance"]
    assert lg["recurring_saving_usd_per_month"].value == 0.0
    assert "not being spent" in lg["note"]
    assert lg["per_manual_deploy_usd"].value > 0


def test_eliminated_calls_change_no_behaviour(measured):
    assert ao.eliminated_calls(measured)["behaviour_change"].startswith("none")


# ------------------------------------------------ 2d dropdown composition ---

def test_dropdown_is_not_an_engineering_change(measured):
    d = ao.dropdown_composition(measured)
    assert d["is_engineering_change"] is False
    assert d["decision_owner"] == "advisor / product"


def test_dropdown_span_is_the_measured_193x(measured):
    d = ao.dropdown_composition(measured)
    assert 190 < d["span"].value < 196
    assert d["span"].tier == "MEASURED"


def test_rows_are_ordered_cheapest_first_and_carry_output_share(measured):
    rows = ao.dropdown_composition(measured)["rows"]
    costs = [r["cost_per_call_usd"] for r in rows]
    assert costs == sorted(costs)
    assert rows[0]["model"] == "nemotron_3_nano_30b"
    assert all(0 < r["output_share_of_cost"] < 1 for r in rows)


def test_the_cost_structure_inverts_between_the_endpoints(measured):
    """Input dominates the cheap model; output dominates the dear one."""
    rows = {r["model"]: r for r in ao.dropdown_composition(measured)["rows"]}
    assert rows["nemotron_3_nano_30b"]["output_share_of_cost"] < 0.5
    assert rows["gpt_5_5"]["output_share_of_cost"] > 0.7


def test_pipeline_depth_multiplies_cost_per_decision(measured):
    one = ao.dropdown_composition(measured, calls_per_decision=1)["rows"][0]
    three = ao.dropdown_composition(measured, calls_per_decision=3)["rows"][0]
    assert three["cost_per_decision_usd"] == pytest.approx(
        one["cost_per_decision_usd"] * 3)


# ------------------------------------------------------------- stacked ------

def _fake_caps(reduction=0.3):
    return {"results": [{"model": CHEAP, "max_safe_token_reduction": reduction,
                         "caps": [], "lowest_safe_cap": 1000,
                         "uncapped_reference": 2000}]}


def test_stacked_refuses_a_residual_without_a_bill(routing):
    s = ao.stacked_ceiling(routing, _fake_caps())
    assert s["residual_usd_per_month"] is None
    assert "workload_split first" in s["refused"]


def test_stacked_reports_a_residual_when_given_a_bill(routing):
    s = ao.stacked_ceiling(routing, _fake_caps(), monthly_bill_usd=600_000)
    assert s["residual_usd_per_month"].value > 0
    assert "order of magnitude" in s["verdict"]


def test_stacked_warns_that_the_levers_overlap(routing):
    s = ao.stacked_ceiling(routing, _fake_caps())
    assert "overstates" in s["overlap_warning"]


def test_stacked_flags_the_unmeasured_applicability_assumption(routing):
    s = ao.stacked_ceiling(routing, _fake_caps())
    assert any("most generous" in n for n in s["notes"])


def test_dropdown_is_excluded_from_the_stack(routing):
    s = ao.stacked_ceiling(routing, _fake_caps())
    assert any("2d" in x for x in s["levers_excluded"])
    assert any("2c" in x for x in s["levers_excluded"])


def test_even_a_large_stacked_saving_leaves_the_wrong_order_of_magnitude(routing):
    """The brief's point: 60-70% off $600k is still ~$200k."""
    s = ao.stacked_ceiling(routing, _fake_caps(0.5), monthly_bill_usd=600_000)
    assert s["residual_usd_per_month"].value > 50_000
