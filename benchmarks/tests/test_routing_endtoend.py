"""End-to-end routing projection, the decision probe, and the two documents.

The load-bearing tests are the ones that keep the honest reading honest:
that the saving is reported end to end rather than on mechanical steps alone,
that a 100% observation is carried as a rule-of-three bound, that an empty
decision is never counted as a parse failure, and that the identical baseline
and routed completion rates are labelled an artifact rather than a finding.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import routing_endtoend as re_          # noqa: E402
from analysis.cost_model_lib import MissingInput      # noqa: E402
from analysis.tier_check import check_report          # noqa: E402

_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
PROBE = os.path.join(_RESULTS, "decision_step_probe.json")
REPORT = os.path.join(_ANALYSIS, "ROUTING_ENDTOEND.md")
MEMO = os.path.join(_ANALYSIS, "DECISION_routing_on_the_leaderboard.md")

CHEAP = "nvidia/nemotron-3-nano-30b-a3b"
MID = "deepseek/deepseek-v4-pro"
DEAR = "anthropic/claude-haiku-4-5"

_spec = importlib.util.spec_from_file_location(
    "_dsp", os.path.join(_ANALYSIS, "decision_step_probe.py"))
dsp = importlib.util.module_from_spec(_spec)
sys.modules["_dsp"] = dsp
_spec.loader.exec_module(dsp)


@pytest.fixture(scope="module")
def probe():
    with open(PROBE, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def proj():
    return re_.endtoend_projection(MID, CHEAP, decision_model=MID)


# ------------------------------------------------- the probe's instrument ---

def test_empty_orders_is_not_a_parse_failure():
    """A well-formed no-trade is a pipeline defect, not a model failure."""
    ok, cat, detail = dsp.classify_decision(
        '{"orders": []}', {"orders": []})
    assert ok is False
    assert cat == dsp.EMPTY_ORDERS
    assert detail == "orders"


def test_real_orders_are_usable():
    payload = {"orders": [{"symbol": "AAPL", "side": "buy", "qty": 3}]}
    ok, cat, _ = dsp.classify_decision(json.dumps(payload), payload)
    assert ok is True and cat == dsp.OK


def test_prose_only_is_no_json():
    ok, cat, _ = dsp.classify_decision("I would hold everything.", None)
    assert ok is False and cat == dsp.FAIL_NO_JSON


def test_prose_wrapped_json_still_counts_as_usable():
    """The production parser slices to the braces, so this survives."""
    payload = {"orders": [{"symbol": "AAPL", "side": "buy", "qty": 1}]}
    text = "Here you go:\n" + json.dumps(payload) + "\nHope that helps."
    ok, cat, _ = dsp.classify_decision(text, payload)
    assert ok is True and cat == dsp.OK_WRAPPED


def test_empty_response_is_distinct_from_truncation():
    assert dsp.classify_decision("", None)[1] == dsp.FAIL_EMPTY
    assert dsp.classify_decision('{"orders": [', None)[1] == dsp.FAIL_TRUNCATED


def test_four_distinct_regimes_are_defined():
    names = [r["_regime"] for r in dsp.REGIMES]
    assert len(names) == len(set(names)) >= 4
    assert "flat_range" in names


def test_regimes_carry_the_production_snapshot_shape():
    for r in dsp.REGIMES:
        assert set(r) >= {"timestamp", "portfolio", "current_holdings",
                          "recent_trades", "top_signals"}
        assert set(r["portfolio"]) == {
            "cash", "positions_value", "total_equity", "num_positions"}


def test_upstream_failure_returns_none_rather_than_raising():
    class Bad:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("429")
    steps = dsp.load_steps()
    assert dsp.build_real_context(Bad(), steps, "m", dsp.REGIMES[0],
                                  verbose=False) is None


# ----------------------------------------------------- the measured probe ---

def test_the_probe_used_real_upstream_not_placeholders(probe):
    for c in probe["contexts"]:
        sig = c["prior_outputs"][-1]["output"]
        assert sig, "empty upstream output"
        assert "..." not in json.dumps(sig), "placeholder leaked into context"


def test_every_context_came_from_the_cheap_model(probe):
    """That IS the routed configuration; it is also the projection's limit."""
    assert {c["upstream_model"] for c in probe["contexts"]} == {CHEAP}


def test_the_empty_decision_rate_is_zero_with_real_context(probe):
    """Phase 16's 15/15 empty result was a fixture artifact."""
    for s in probe["summary"]:
        assert s["n_empty_orders"] == 0


def test_cheap_model_failures_concentrate_in_the_flat_regime(probe):
    """Concentrate, NOT exclusively — an earlier draft claimed 'every failure'
    and one deepseek failure in sharp_selloff disproves it. The report counts
    rather than asserts for exactly this reason."""
    fails = [(r["regime"], r["attempts"] - r["n_usable"])
             for r in probe["rows"]
             if r["model"] in (CHEAP, MID) and r["n_usable"] < r["attempts"]]
    total = sum(f for _r, f in fails)
    flat = sum(f for reg, f in fails if reg == "flat_range")
    assert total > 0
    assert flat / total >= 0.75, f"only {flat}/{total} failures in flat_range"
    assert flat < total, "if every failure were flat_range, say so exactly"


def test_the_mid_tier_model_handles_the_flat_regime(probe):
    """So it is not 'the flat tape is hard' — it is the cheap models."""
    by = {r["model"]: {x["regime"]: x for x in probe["rows"]
                       if x["model"] == r["model"]}
          for r in probe["rows"]}
    hk = by[DEAR]["flat_range"]
    assert hk["n_usable"] == hk["attempts"]


def test_a_perfect_score_carries_a_rule_of_three_bound(probe):
    for s in probe["summary"]:
        if s["n_usable"] == s["n_total"]:
            assert s["rule_of_three_bound"] == pytest.approx(3 / s["n_total"])
        else:
            assert s["rule_of_three_bound"] is None


def test_qwen_is_recorded_as_unmeasured_not_as_zero(probe):
    """16/16 BadRequestError is a gateway rejection, not a model result."""
    unm = probe["_unmeasured"]
    assert "qwen/qwen3.7-plus" in unm
    assert "thinking_budget" in unm["qwen/qwen3.7-plus"]["reason"]
    assert not any(s["model"].startswith("qwen/") for s in probe["summary"])


def test_the_later_model_reused_the_stored_contexts(probe):
    assert probe["_merged"]["haiku_reused_contexts"] is True


# ------------------------------------------------------ the projection ------

def test_the_saving_is_end_to_end_not_mechanical_only(proj):
    assert proj["saving_fraction"].value < 0.5
    assert "END TO END" in proj["saving_fraction"].source


def test_the_mechanical_share_bounds_what_routing_can_touch(proj):
    share = proj["mechanical_share_of_baseline"].value
    assert 0 < share < 1
    assert proj["saving_fraction"].value < share


def test_protecting_a_dearer_decision_model_shrinks_the_saving():
    cheap_dec = re_.endtoend_projection(MID, CHEAP, decision_model=MID)
    dear_dec = re_.endtoend_projection(MID, CHEAP, decision_model=DEAR)
    assert dear_dec["saving_fraction"].value < cheap_dec["saving_fraction"].value
    assert (dear_dec["mechanical_share_of_baseline"].value
            < cheap_dec["mechanical_share_of_baseline"].value)


def test_the_compound_effect_inflates_the_decision_step(proj):
    assert proj["compound_direction"] == "INFLATES"
    base_in = proj["baseline_rows"][-1]["input_tokens"]
    routed_in = proj["routed_rows"][-1]["input_tokens"]
    assert routed_in > base_in
    assert proj["routed_rows"][-1]["cost_usd"] > proj["baseline_rows"][-1]["cost_usd"]
    assert any("INFLATES" in w for w in proj["warnings"])


def test_completion_is_the_product_of_step_rates(proj):
    rates = [r["parse_rate"] for r in proj["routed_rows"]]
    expected = 1.0
    for r in rates:
        expected *= r
    assert proj["routed_completion"].value == pytest.approx(expected)


def test_the_completion_floor_is_below_the_point_estimate(proj):
    assert proj["routed_completion_floor"].value < proj["routed_completion"].value


def test_baseline_and_routed_completion_match_by_construction(proj):
    """Not a finding. There is only one decision-step measurement and it was
    taken under cheap upstream, so both arms carry the same rate."""
    assert (proj["baseline_completion"].value
            == proj["routed_completion"].value)


def test_projection_refuses_an_unmeasured_decision_model():
    with pytest.raises(MissingInput, match="decision-step reliability"):
        re_.endtoend_projection(MID, CHEAP, decision_model="openai/gpt-5.5")


def test_projection_refuses_an_unpriced_model():
    with pytest.raises(MissingInput, match="no measured price"):
        re_.endtoend_projection("mystery/model", CHEAP)


def test_missing_probe_results_refuse_rather_than_fall_back():
    """The isolated-mode figures are not a substitute."""
    with pytest.raises(MissingInput, match="NOT a substitute"):
        re_.load_decision_measurements("/nonexistent.json")


# ------------------------------------------------- saving against cadence ---

def test_cadence_saving_is_withheld_without_the_pipeline_share(proj):
    out = re_.saving_against_cadence(proj, {"nof1": 142181.0})
    assert out["cadences"]["nof1"]["saving_usd"] is None
    assert "1.000 calls per decision" in out["refused"]


def test_cadence_saving_computes_when_the_share_is_supplied(proj):
    out = re_.saving_against_cadence(proj, {"nof1": 142181.0},
                                     pipeline_fraction=0.5)
    c = out["cadences"]["nof1"]
    assert c["saving_usd"] == pytest.approx(142181.0 * 0.5
                                            * proj["saving_fraction"].value)
    assert c["residual_usd"] < c["monthly_usd"]


def test_an_out_of_range_share_is_rejected(proj):
    with pytest.raises(ValueError):
        re_.saving_against_cadence(proj, {"a": 1.0}, pipeline_fraction=1.5)


# ------------------------------------------------------------ documents -----

def test_report_is_not_stale():
    from analysis import make_routing_report as mk
    with open(REPORT, "r", encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "ROUTING_ENDTOEND.md is stale; regenerate with "
            "`python benchmarks/analysis/make_routing_report.py`")


def test_report_and_memo_are_fully_tagged():
    for path in (REPORT, MEMO):
        with open(path, "r", encoding="utf-8") as fh:
            r = check_report(fh.read())
        assert r["ok"], (path, r["violations"][:3])


def test_report_leads_with_the_end_to_end_number():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "35.6%" in t and "76.2%" in t
    assert "not `76.2%`" in t


def test_report_states_the_upstream_limitation():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "artifact, not a finding" in t
    assert "same rate by construction" in t


def test_report_withholds_the_cadence_saving():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "WITHHELD" in t
    assert "1.000" in t


def test_memo_presents_both_readings_without_picking():
    with open(MEMO, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "Reading A" in t and "Reading B" in t
    assert "does not pick" in t
    for verdict in ("we recommend", "we should adopt", "the answer is"):
        assert verdict not in t.lower()


def test_memo_carries_the_attribution_gap():
    with open(MEMO, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "llm_decisions" in t
    assert "already weaker than it appears" in t


def test_memo_reframes_using_the_single_prompt_measurement():
    with open(MEMO, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "single-prompt agents" in t
    assert "exactly `$0`" in t


def test_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_routing_report.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "all tagged" in r.stdout
