"""Phase 27B evaluation: metrics, buckets, cost sensitivity, and the ATL run path.

The run path is exercised end to end: the real HTTP API, the unmodified SDK
loop, a frozen multi-day replay, ATL's fills and costs.
"""
import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.tests.test_protocol_api import client
from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.backend.domain.research.phase27b import (
    build_dataset, calendar_sessions, round_trip, session_frame, spell_cash)
from dashboard.backend.domain.research.phase27b_metrics import (
    score_window, probability_buckets, cost_sensitivity)
from dashboard.scripts.phase27b_experiment import (
    run_symbol_window, fit_pooled_gate, momentum_rule, validate_prediction)


@pytest.fixture(autouse=True)
def local_sdk_source(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "packaging/agentictrading/src"))


# ------------------------------------------------------------------ metrics

def test_score_window_reports_outcome_conditioned_metrics():
    y = [1, 0, 1, 0]
    p = [0.9, 0.8, 0.1, 0.2]
    net = [0.03, -0.01, 0.02, -0.02]
    mae, mfe = [-0.01, -0.02, -0.005, -0.03], [0.04, 0.01, 0.03, 0.0]
    m = score_window(y, p, net, mae, mfe)
    assert m["trade_coverage"] == 0.5 and m["precision_trade"] == 0.5 and m["recall_trade"] == 0.5
    assert m["mean_net_return_selected"] == pytest.approx(0.01)
    assert m["median_net_return_selected"] == pytest.approx(0.01)
    assert m["fraction_selected_positive"] == 0.5
    assert m["mean_mae_selected"] == pytest.approx(-0.015)
    assert m["mean_mfe_selected"] == pytest.approx(0.025)
    # long-only: direction accuracy given TRADE is TRADE precision, reported as such
    assert m["direction_accuracy_given_trade"] == m["precision_trade"]
    assert m["brier"] >= 0 and m["log_loss"] >= 0 and m["balanced_accuracy"] == pytest.approx(0.5)


def test_score_window_is_defined_when_nothing_is_selected():
    m = score_window([1, 0], [0.0, 0.0], [0.01, -0.01], [-0.01, -0.01], [0.01, 0.0])
    assert m["trade_coverage"] == 0.0
    for key in ("precision_trade", "mean_net_return_selected", "median_net_return_selected",
                "fraction_selected_positive", "mean_mae_selected", "mean_mfe_selected"):
        assert m[key] is None


def test_probability_buckets_are_exact_and_include_one_in_the_top_bucket():
    p = [0.0, 0.19, 0.2, 0.5, 0.79, 0.8, 1.0]
    net = [-0.01, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05]
    b = probability_buckets(p, net)
    assert [x["count"] for x in b] == [2, 1, 1, 1, 2]
    assert b[0]["mean_net_return"] == pytest.approx(-0.015)
    assert b[4]["median_net_return"] == pytest.approx(0.04)
    assert [x["low"] for x in b] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_empty_buckets_report_none_not_zero():
    b = probability_buckets([0.9], [0.01])
    assert b[0]["count"] == 0 and b[0]["mean_net_return"] is None


def test_cost_sensitivity_recomputes_net_with_atl_quotes_at_fixed_tau():
    s = small_spec()
    outcomes = [{"entry_price": 100.0, "exit_price": 101.0}, {"entry_price": 50.0, "exit_price": 49.0}]
    preds = [0.9, 0.9]
    sens = cost_sensitivity(outcomes, preds, s, multipliers=[0.5, 1.0, 2.0])
    assert [r["multiplier"] for r in sens] == [0.5, 1.0, 2.0]
    one = sens[1]
    assert one["mean_net_return_selected"] == pytest.approx(
        np.mean([round_trip(100.0, 101.0, s), round_trip(50.0, 49.0, s)]))
    assert sens[0]["mean_net_return_selected"] > one["mean_net_return_selected"] > sens[2]["mean_net_return_selected"]
    assert all(r["trade_threshold"] == s["trade_threshold"] for r in sens)


# -------------------------------------------------------------- model boundary

@pytest.mark.parametrize("value", [
    {"trade_probability": float("nan"), "direction": "NONE"},
    {"trade_probability": 1.2, "direction": "LONG"},
    {"trade_probability": 0.8, "direction": "SHORT"},
    {"trade_probability": 0.8, "direction": "NONE"},
    {"trade_probability": 0.2, "direction": "NONE", "forward_return": 0.1},
])
def test_invalid_model_outputs_fail_closed(value):
    with pytest.raises(ValueError):
        validate_prediction(value)


def test_the_run_path_cannot_receive_outcomes():
    params = set(inspect.signature(run_symbol_window).parameters)
    assert not params & {"outcomes", "labels", "hidden"}


def test_momentum_rule_is_the_frozen_rule():
    s = small_spec()
    up = {"timestamp": "t", "symbol": "A", "features": {f: 0.0 for f in s["feature_fields"]}}
    up["features"]["return_20"] = 0.01
    down = copy.deepcopy(up); down["features"]["return_20"] = -0.01
    flat = copy.deepcopy(up); flat["features"]["return_20"] = 0.0
    assert momentum_rule(up) == {"trade_probability": 1.0, "direction": "LONG"}
    assert momentum_rule(down)["direction"] == "NONE" and momentum_rule(flat)["direction"] == "NONE"


def test_pooled_gate_fits_on_training_ids_only():
    s = small_spec()
    raws = {"AAPL": tape(sessions=80, seed=20), "MSFT": tape(sessions=80, seed=21)}
    inputs, outcomes, _ = build_dataset(raws, raws["AAPL"], s)
    train = [r["record_id"] for r in inputs[: len(inputs) // 2]]
    before = fit_pooled_gate(inputs, outcomes, train, s).state()
    flipped = copy.deepcopy(outcomes)
    for o in flipped:
        if o["record_id"] not in set(train):
            o["trade_worthy"] = not o["trade_worthy"]
    assert fit_pooled_gate(inputs, flipped, train, s).state() == before


# ----------------------------------------------------------- the ATL run path

def _window(seed=30):
    s = small_spec()
    raw = tape(sessions=60, seed=seed)
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    sessions = session_frame(raw, s)
    calendar = calendar_sessions(sessions.index[0], sessions.index[-1], s)
    start = pd.Timestamp(inputs[0]["timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    end = pd.Timestamp(outcomes[-1]["exit_timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    return s, raw, inputs, outcomes, calendar, start, end


def test_multi_day_round_trips_match_hidden_cash_and_repeat(client, tmp_path):
    """Alternating LONG/NONE: every chosen row is its own ATL round trip."""
    s, raw, inputs, outcomes, calendar, start, end = _window()
    seen = []
    chosen = {r["timestamp"] for r in inputs[::2]}

    def every_other(payload):
        assert set(payload) == {"timestamp", "symbol", "features"}
        seen.append(payload["timestamp"])
        return ({"trade_probability": 1.0, "direction": "LONG"} if payload["timestamp"] in chosen
                else {"trade_probability": 0.0, "direction": "NONE"})

    runs = [run_symbol_window(client, tmp_path / str(i), raw, inputs, calendar, s, "AAPL", start, end,
                              every_other, {"experiment_id": "t", "repeat": i}) for i in range(2)]
    assert runs[0]["fingerprint"] == runs[1]["fingerprint"]
    assert len(seen) == 2 * len(inputs)
    trades = runs[0]["trades"]
    assert len(trades) == 2 * len(chosen)                 # every chosen entry bought and sold
    assert sum(t["total_fees"] for t in trades) > 0
    # Positions were held overnight: exits are sessions later than entries.
    buys = [pd.Timestamp(t["timestamp"]) for t in trades if t["side"].lower() == "buy"]
    sells = [pd.Timestamp(t["timestamp"]) for t in trades if t["side"].lower() == "sell"]
    assert all((x - b) > pd.Timedelta(days=4) for b, x in zip(buys, sells))
    expected = s["initial_cash"] + sum(o["cost_adjusted_forward_return"] * o["entry_price"]
                                       for o in outcomes if o["timestamp"] in chosen)
    assert runs[0]["metrics"]["final_equity"] == pytest.approx(expected)
    assert runs[0]["metrics"]["final_equity"] == pytest.approx(
        s["initial_cash"] + spell_cash([o for o in outcomes if o["timestamp"] in chosen], s))


def _pricey(raw, scale=3.8):
    """Same path at ~$380: two shares exceed the 25% cap on $3,000, one does not."""
    out = raw.copy()
    out[["open", "high", "low", "close"]] *= scale
    out.attrs = dict(raw.attrs)
    return out


def test_consecutive_longs_hold_through_instead_of_selling_and_rebuying(client, tmp_path):
    """ATL's position cap counts same-decision buys but not same-decision sells,
    so a sell-then-rebuy of a $380 name is judged as two shares and rejected.
    Holding through is also what the position is: one spell, one round trip."""
    s = small_spec()
    raw = _pricey(tape(sessions=60, seed=33))
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    sessions = session_frame(raw, s)
    calendar = calendar_sessions(sessions.index[0], sessions.index[-1], s)
    start = pd.Timestamp(inputs[0]["timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    end = pd.Timestamp(outcomes[-1]["exit_timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    assert 2 * outcomes[0]["entry_price"] > s["initial_cash"] * 0.25 > outcomes[0]["entry_price"]
    run = run_symbol_window(client, tmp_path, raw, inputs, calendar, s, "AAPL", start, end,
                            lambda p: {"trade_probability": 1.0, "direction": "LONG"},
                            {"experiment_id": "t"})
    assert [t["side"].lower() for t in run["trades"]] == ["buy", "sell"]
    assert len(run["predictions"]) == len(inputs)
    first, last = outcomes[0], outcomes[-1]
    assert run["metrics"]["final_equity"] == pytest.approx(
        s["initial_cash"] + round_trip(first["entry_price"], last["exit_price"], s) * first["entry_price"])
    assert run["metrics"]["final_equity"] == pytest.approx(s["initial_cash"] + spell_cash(outcomes, s))


def test_spell_cash_joins_only_back_to_back_rows():
    s = small_spec()
    t = [f"2024-01-{d:02d}T20:30:00+00:00" for d in (2, 9, 16, 23, 30)]
    row = lambda a, b, e, x: {"entry_timestamp": t[a], "exit_timestamp": t[b], "timestamp": t[a],
                              "entry_price": e, "exit_price": x,
                              "cost_adjusted_forward_return": round_trip(e, x, s)}
    joined = [row(0, 1, 100.0, 102.0), row(1, 2, 102.0, 101.0)]
    split = [row(0, 1, 100.0, 102.0), row(2, 3, 103.0, 101.0)]
    assert spell_cash(joined, s) == pytest.approx(round_trip(100.0, 101.0, s) * 100.0)
    assert spell_cash(split, s) == pytest.approx(round_trip(100.0, 102.0, s) * 100.0
                                                 + round_trip(103.0, 101.0, s) * 103.0)
    # Holding through never costs more than paying a round trip every week.
    assert spell_cash(joined, s) > sum(r["cost_adjusted_forward_return"] * r["entry_price"] for r in joined)
    assert spell_cash([], s) == 0.0


def test_hold_places_no_orders_and_pays_no_costs(client, tmp_path):
    s, raw, inputs, outcomes, calendar, start, end = _window(seed=31)
    run = run_symbol_window(client, tmp_path, raw, inputs, calendar, s, "AAPL", start, end,
                            lambda p: {"trade_probability": 0.0, "direction": "NONE"},
                            {"experiment_id": "t"})
    assert run["trades"] == []
    assert run["metrics"]["final_equity"] == pytest.approx(s["initial_cash"])
    assert len(run["predictions"]) == len(inputs)


def test_a_failed_model_sends_no_order_and_records_the_failure(client, tmp_path):
    s, raw, inputs, outcomes, calendar, start, end = _window(seed=32)
    with pytest.raises(ValueError):
        run_symbol_window(client, tmp_path, raw, inputs, calendar, s, "AAPL", start, end,
                          lambda p: {"trade_probability": 0.7, "direction": "NONE"}, {"experiment_id": "t"})
    assert (tmp_path / "failure.json").exists()
