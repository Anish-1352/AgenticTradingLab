"""Two-workload cost model.

Most of what is tested here is refusal. The brief is explicit that "where a
parameter is unknown, the model must refuse to compute rather than default",
because silent defaults are how an assumption becomes a finding — which has
already happened once in this project.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import workload_split as ws              # noqa: E402
from analysis.cost_model_lib import MissingInput, load_measured  # noqa: E402


@pytest.fixture(scope="module")
def measured():
    return load_measured()


@pytest.fixture(scope="module")
def all_models(measured):
    return list(measured["models"])


def _exp(models, **over):
    base = dict(n_agents=1000, n_distinct_models=len(models),
                cadence_seconds=150, calls_per_decision=1,
                experiment_duration_days=30, models=models)
    base.update(over)
    return base


def _plat(**over):
    base = dict(n_users=1000, agents_per_user=2, backtests_per_user_per_day=1.0,
                model_mix={"nemotron_3_nano_30b": 1.0},
                pipeline_depth_distribution={1: 1.0}, live_trading_on=False,
                bars_per_backtest=161)
    base.update(over)
    return base


# ------------------------------------------------------------- refusal ------

def test_experiment_refuses_every_blank_parameter(measured):
    with pytest.raises(MissingInput) as e:
        ws.experiment_cost({"n_agents": 100}, measured)
    msg = str(e.value)
    for k in ("n_distinct_models", "cadence_seconds", "calls_per_decision",
              "experiment_duration_days"):
        assert k in msg
    assert "would become a finding" in msg


def test_refusal_names_who_controls_each_blank(measured):
    with pytest.raises(MissingInput) as e:
        ws.platform_cost({"n_users": 10}, measured)
    assert ws.USER_DETERMINED in str(e.value)


def test_experiment_refuses_without_a_named_model_set(measured):
    """An even split over an unnamed set spans 193x — an invented figure."""
    with pytest.raises(MissingInput, match="193x"):
        ws.experiment_cost({
            "n_agents": 10, "n_distinct_models": 3, "cadence_seconds": 150,
            "calls_per_decision": 1, "experiment_duration_days": 30}, measured)


def test_platform_refuses_live_trading_without_cadence(measured):
    with pytest.raises(MissingInput, match="22x"):
        ws.platform_cost(_plat(live_trading_on=True), measured)


def test_platform_refuses_without_bars_per_backtest(measured):
    with pytest.raises(MissingInput, match="20x"):
        ws.platform_cost(_plat(bars_per_backtest=None), measured)


def test_an_unpriced_model_cannot_be_priced_by_analogy(measured):
    with pytest.raises(MissingInput, match="no measured cost"):
        ws.platform_cost(_plat(model_mix={"some/new-model": 1.0}), measured)


def test_a_mix_that_does_not_sum_to_one_is_rejected(measured):
    with pytest.raises(ValueError, match="sum to 1.0"):
        ws.platform_cost(_plat(model_mix={"nemotron_3_nano_30b": 0.5}), measured)


# ---------------------------------------------------- the 22x cadence term --

def test_nof1_cadence_is_about_22x_hourly(measured, all_models):
    hourly = ws.experiment_cost(_exp(all_models, cadence_seconds=3600), measured)
    nof1 = ws.experiment_cost(_exp(all_models, cadence_seconds=150), measured)
    ratio = nof1["total_cost_usd"].value / hourly["total_cost_usd"].value
    assert 23 < ratio < 25              # 3600/150 = 24 exactly
    assert hourly["decisions_per_agent_day"].value == pytest.approx(6.5)
    assert nof1["decisions_per_agent_day"].value == pytest.approx(156.0)


def test_both_cadences_remain_computable_side_by_side(measured, all_models):
    """The brief keeps hourly alongside Nof1; the ratio is the finding."""
    for cad in (3600, 150):
        r = ws.experiment_cost(_exp(all_models, cadence_seconds=cad), measured)
        assert r["total_cost_usd"].value > 0


# --------------------------------------------- experiment: levers vs required

def test_model_diversity_is_flagged_as_the_independent_variable():
    p = ws.EXPERIMENT_PARAMS["n_distinct_models"]
    assert p.required_by_experiment is True
    assert "INDEPENDENT VARIABLE" in p.description


def test_the_other_experiment_parameters_are_free():
    for name in ("n_agents", "cadence_seconds", "calls_per_decision",
                 "experiment_duration_days"):
        assert ws.EXPERIMENT_PARAMS[name].required_by_experiment is False
        assert ws.EXPERIMENT_PARAMS[name].control == ws.LAB_CONTROLLED


def test_sensitivity_marks_the_off_limits_parameter(measured, all_models):
    rows = ws.experiment_sensitivity(_exp(all_models), measured)
    required = [r for r in rows if r["required_by_experiment"]]
    assert len(required) == 1
    assert "INDEPENDENT VARIABLE" in required[0]["note"]


def test_model_choice_dominates_the_experiment_sensitivity(measured, all_models):
    rows = ws.experiment_sensitivity(_exp(all_models), measured)
    assert "model_mix" in rows[0]["parameter"]
    assert rows[0]["cost_ratio"] > 100


def test_longer_cadence_reduces_cost(measured, all_models):
    rows = ws.experiment_sensitivity(_exp(all_models), measured, factor=2.0)
    cad = [r for r in rows if r["parameter"] == "cadence_seconds"][0]
    assert cad["cost_ratio"] == pytest.approx(0.5)
    assert "inverse" in cad["note"]


# ------------------------------------------------------- the power question --

def test_the_power_question_is_priced_but_not_answered(measured):
    q = ws.power_question_options(measured)
    assert q["answerable_here"] is False
    assert "effect size" in q["why_not"]
    assert q["tier"] == "NOT MEASURED"


def test_fewer_models_is_priced_as_a_range_not_a_number(measured):
    """Which models are dropped matters more than how many."""
    q = ws.power_question_options(measured)
    three = [o for o in q["options"] if o["n_models"] == 3][0]
    assert three["cost_usd_high"] is not None
    assert three["cost_usd_high"] > three["cost_usd"] * 5
    assert "RANGE, not a figure" in three["note"]
    assert three["destroys_experiment"] is True


def test_dropping_the_cheap_models_can_cost_more_than_keeping_all_seven(measured):
    """Counterintuitive and worth surfacing: fewer models is not cheaper."""
    q = ws.power_question_options(measured)
    full = [o for o in q["options"] if o["option"].startswith("as described")][0]
    three = [o for o in q["options"] if o["n_models"] == 3][0]
    assert three["cost_usd_high"] > full["cost_usd"]


def test_agent_count_and_cadence_do_not_destroy_the_experiment(measured):
    q = ws.power_question_options(measured)
    for o in q["options"]:
        if o["n_models"] == 7:
            assert o["destroys_experiment"] is False


# ------------------------------------------------------------- platform -----

def test_every_platform_parameter_declares_who_controls_it():
    for p in ws.PLATFORM_PARAMS.values():
        assert p.control in (ws.LAB_CONTROLLED, ws.USER_DETERMINED)


def test_the_core_platform_parameters_are_user_determined():
    for name in ("n_users", "agents_per_user", "backtests_per_user_per_day",
                 "model_mix", "pipeline_depth_distribution", "live_trading_on"):
        assert ws.PLATFORM_PARAMS[name].control == ws.USER_DETERMINED


def test_the_lab_levers_are_named_explicitly():
    levers = " ".join(ws.PLATFORM_LEVERS)
    for expected in ("dropdown", "default model", "routing", "caching",
                     "leaderboard cadence", "quota"):
        assert expected in levers


def test_live_trading_dominates_when_enabled(measured):
    off = ws.platform_cost(_plat(), measured)
    on = ws.platform_cost(_plat(live_trading_on=True, cadence_seconds=150),
                          measured)
    assert on["live_calls_per_day"].value > off["backtest_calls_per_day"].value
    assert on["cost_per_month_usd"].value > off["cost_per_month_usd"].value


def test_platform_reports_cost_per_user(measured):
    r = ws.platform_cost(_plat(), measured)
    assert r["cost_per_user_per_month_usd"].value > 0


def test_dropdown_is_the_only_platform_lever_above_2x(measured):
    rows = ws.platform_sensitivity(
        _plat(live_trading_on=True, cadence_seconds=150), measured)
    assert "model_mix" in rows[0]["parameter"]
    assert rows[0]["cost_ratio"] > 100
    assert all(r["cost_ratio"] <= 2.01 for r in rows[1:])


# ---------------------------------------------------------- leaderboard -----

def test_leaderboard_recurring_cost_is_zero_because_nothing_schedules_it(measured):
    lb = ws.leaderboard_recurring(measured)
    assert lb["scheduled_today"] is False
    assert lb["recurring_cost_usd_per_month"].value == 0.0
    assert "no cron or CI schedule" in lb["scheduled_evidence"]


def test_the_contest_deploy_is_a_measured_one_off(measured):
    lb = ws.leaderboard_recurring(measured)
    c = lb["contest_window"]["cost_usd"]
    assert c.tier == "MEASURED"
    assert 30 < c.value < 40


def test_the_daily_window_refuses_without_a_bar_count(measured):
    """It is a rolling ONE-day window; 7 bars or 156 differ by 22x."""
    lb = ws.leaderboard_recurring(measured)
    assert lb["daily_window"]["cost_usd"] is None
    assert "22x" in lb["daily_window"]["refused"]


def test_a_hypothetical_schedule_is_labelled_hypothetical(measured):
    lb = ws.leaderboard_recurring(measured, bars_per_daily_window=156)
    f = lb["daily_window"]["cost_usd_per_month_if_scheduled"]
    assert "HYPOTHETICAL" in f.source


def test_report_renders(measured, all_models):
    text = ws.format_workload_report(
        ws.experiment_cost(_exp(all_models), measured),
        ws.platform_cost(_plat(), measured),
        ws.leaderboard_recurring(measured))
    assert "TWO WORKLOADS" in text
    assert "EXPERIMENT" in text and "PLATFORM" in text and "LEADERBOARD" in text
