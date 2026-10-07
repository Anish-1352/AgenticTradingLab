"""Phase 27D: the continuous target and its linear model. Synthetic data only.

The regressor must differ from the Phase 27C LogisticGate in its target alone:
same features, same train-only standardisation, same single L2 value.
"""
import copy
import math

import numpy as np
import pytest

from dashboard.backend.domain.research.phase27 import LogisticGate
from dashboard.backend.domain.research.phase27d import (
    LinearOpportunityRegressor, fit_linear_regressor, fit_train_mean, top_fraction_selection,
    validate_regression_prediction, TARGET)
from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.backend.domain.research.phase27b import build_dataset, model_payload


def _xy(n=400, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) * [1.0, 5.0, 0.1] + [0.0, 2.0, -1.0]
    y = 0.02 * (x[:, 0]) - 0.004 * (x[:, 1] - 2.0) + rng.normal(scale=noise, size=n)
    return x, y


def test_the_target_is_the_stored_cost_adjusted_net_return():
    assert TARGET == "cost_adjusted_forward_return"


def test_regressor_recovers_a_linear_relation_and_is_deterministic():
    x, y = _xy()
    a, b = LinearOpportunityRegressor(0.01).fit(x, y), LinearOpportunityRegressor(0.01).fit(x, y)
    assert a.state() == b.state()
    pa = a.predict(x)
    assert np.array_equal(pa, b.predict(x))
    assert np.corrcoef(pa, y)[0, 1] > 0.85


def test_standardisation_is_the_logistic_gates_and_uses_training_rows_only():
    x, y = _xy(seed=1)
    reg = LinearOpportunityRegressor(0.01).fit(x, y)
    gate = LogisticGate().fit(x, (y > 0).astype(int))
    assert np.allclose(reg.mean, gate.mean) and np.allclose(reg.scale, gate.scale)
    assert np.allclose(reg.mean, x.mean(axis=0)) and np.allclose(reg.scale, x.std(axis=0))
    flat = np.c_[x, np.full(len(x), 3.0)]                     # a constant column is scaled by 1, not 0
    assert LinearOpportunityRegressor(0.01).fit(flat, y).scale[-1] == 1.0


def test_ridge_solution_with_an_unpenalised_intercept():
    x, y = _xy(seed=2)
    l2 = 0.5
    reg = LinearOpportunityRegressor(l2).fit(x, y)
    z = np.c_[np.ones(len(x)), (x - x.mean(0)) / x.std(0)]
    penalty = l2 * np.eye(z.shape[1]); penalty[0, 0] = 0
    w = np.linalg.solve(z.T @ z / len(x) + penalty, z.T @ y / len(x))
    assert np.allclose(reg.weights, w)
    assert reg.weights[0] == pytest.approx(y.mean())          # intercept is the training mean


def _dataset():
    s = small_spec(regressor={"l2": 0.01})
    raw = tape(sessions=90, seed=40)
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, tape(sessions=90, seed=41), s)
    return s, inputs, outcomes


def test_fit_reads_the_continuous_target_of_training_rows_only():
    s, inputs, outcomes = _dataset()
    train = [r["record_id"] for r in inputs[: len(inputs) // 2]]
    before = fit_linear_regressor(inputs, outcomes, train, s).state()
    others = copy.deepcopy(outcomes)
    for o in others:
        o["trade_worthy"] = not o["trade_worthy"]              # the binary label is never read
        if o["record_id"] not in set(train):
            o[TARGET] = 99.0                                   # nor any non-training outcome
    assert fit_linear_regressor(inputs, others, train, s).state() == before
    moved = copy.deepcopy(outcomes)
    moved[0][TARGET] += 0.5
    assert fit_linear_regressor(inputs, moved, train, s).state() != before


def test_the_model_sees_only_the_allowlisted_payload():
    s, inputs, _ = _dataset()
    payload = model_payload(inputs[0], s)
    assert set(payload) == {"timestamp", "symbol", "features"}
    assert TARGET not in payload["features"] and set(payload["features"]) == set(s["feature_fields"])


def test_train_mean_uses_training_rows_only():
    s, inputs, outcomes = _dataset()
    train = [r["record_id"] for r in inputs[:10]]
    by_id = {o["record_id"]: o[TARGET] for o in outcomes}
    assert fit_train_mean(outcomes, train) == pytest.approx(np.mean([by_id[r] for r in train]))


def test_top_fraction_is_cross_sectional_per_timestamp_with_fixed_ties():
    rows = [{"record_id": f"{t}-{s}", "timestamp": t, "symbol": s}
            for t in ("t1", "t2") for s in "ABCDEFGHIJ"]
    pred = {r["record_id"]: (0.0 if r["timestamp"] == "t2" else ord(r["symbol"]) / 100) for r in rows}
    chosen = top_fraction_selection(rows, pred, 0.2)
    assert {r for r in chosen if r.startswith("t1")} == {"t1-J", "t1-I"}     # ceil(0.2 * 10) = 2
    assert {r for r in chosen if r.startswith("t2")} == {"t2-A", "t2-B"}     # all tied: symbol order
    odd = [{"record_id": f"x-{s}", "timestamp": "x", "symbol": s} for s in "ABCDEFG"]
    assert len(top_fraction_selection(odd, {r["record_id"]: 0.0 for r in odd}, 0.2)) == math.ceil(1.4)
    assert top_fraction_selection(rows, pred, 0.2) == chosen


@pytest.mark.parametrize("value", [
    {"predicted_net_return": float("nan"), "direction": "NONE"},
    {"predicted_net_return": float("inf"), "direction": "LONG"},
    {"predicted_net_return": True, "direction": "LONG"},
    {"predicted_net_return": "0.1", "direction": "LONG"},
    {"predicted_net_return": 0.01, "direction": "SHORT"},
    {"predicted_net_return": 0.01},
    {"predicted_net_return": 0.01, "direction": "LONG", "trade_probability": 0.6},
    None,
])
def test_invalid_regression_outputs_fail_closed(value):
    with pytest.raises(ValueError):
        validate_regression_prediction(value)


def test_a_valid_regression_output_passes_unchanged():
    assert validate_regression_prediction({"predicted_net_return": -0.01, "direction": "NONE"}) == \
        {"predicted_net_return": -0.01, "direction": "NONE"}


def test_regression_metrics_are_exact_on_a_known_case():
    from dashboard.backend.domain.research.phase27d import regression_metrics
    pred, net = [0.01, -0.02, 0.03, 0.0], [0.02, -0.01, -0.01, 0.0]
    m = regression_metrics(pred, net, reference=[0.005] * 4)
    err = np.subtract(pred, net)
    assert m["rows"] == 4
    assert m["mae"] == pytest.approx(np.abs(err).mean())
    assert m["rmse"] == pytest.approx(np.sqrt((err ** 2).mean()))
    assert m["r2"] == pytest.approx(1 - (err ** 2).sum() / ((np.array(net) - np.mean(net)) ** 2).sum())
    assert m["oos_r2_vs_reference"] == pytest.approx(
        1 - (err ** 2).sum() / ((np.array(net) - 0.005) ** 2).sum())
    # sign: predicted > 0 vs realised > 0 -> agree, agree, disagree, agree (0 is non-positive)
    assert m["sign_accuracy"] == pytest.approx(0.75)
    assert m["prediction_mean"] == pytest.approx(np.mean(pred))
    assert m["prediction_std"] == pytest.approx(np.std(pred))
    assert m["realized_mean"] == pytest.approx(np.mean(net)) and m["realized_std"] == pytest.approx(np.std(net))
    assert regression_metrics(pred, net)["oos_r2_vs_reference"] is None
