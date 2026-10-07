"""Phase 27D baselines through the unchanged Phase 27B/27C ATL path."""
import json
import math

import pandas as pd
import pytest

from dashboard.backend.tests.test_protocol_api import client
from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.scripts.phase27b_experiment import evaluate, validate_prediction

CONTINUOUS = ["train_mean", "linear_gt0", "linear_top20"]


@pytest.fixture(autouse=True)
def local_sdk_source(monkeypatch):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")


def _spec(baselines=CONTINUOUS):
    return small_spec(universe=["AAPL", "MSFT", "KO"], baselines=baselines,
                      regressor={"l2": 0.01}, ranking_gate={"fraction": 0.2}, folds=[
        {"id": "f1", "train": ["2024-01-01", "2024-03-18"], "validation": ["2024-03-18", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-06"]},
        {"id": "f2", "train": ["2024-01-01", "2024-04-15"], "validation": ["2024-04-15", "2024-05-06"],
         "test": ["2024-05-06", "2024-06-10"]}])


def _tapes(seed=80):
    return {s: tape(sessions=120, seed=seed + i) for i, s in enumerate(["AAPL", "MSFT", "KO"])}, \
        tape(sessions=120, seed=seed + 9)


def _sealed(root, fold, baseline):
    return [json.loads(l) for l in (root / "runs" / fold / baseline / "predictions.jsonl").read_text().splitlines()]


def test_continuous_baselines_run_through_atl_with_cash_matching_outcomes(client, tmp_path):
    raws, market = _tapes()
    report = evaluate(client, tmp_path, raws, market, {"sha256": "s" * 64}, _spec(), "code")
    assert {w["baseline"] for w in report["windows"]} == set(CONTINUOUS)
    for w in report["windows"]:
        assert w["reproducible"] and w["cash_matches_outcomes"]
        assert w["regression"]["rows"] == w["classification"]["n"]
    for fold in ("f1", "f2"):
        gt0, top = _sealed(tmp_path, fold, "linear_gt0"), _sealed(tmp_path, fold, "linear_top20")
        assert [p["predicted_net_return"] for p in gt0] == [p["predicted_net_return"] for p in top]
        assert all((p["direction"] == "LONG") == (p["predicted_net_return"] > 0) for p in gt0)
        by_time = pd.DataFrame(top).groupby("timestamp")
        for _, g in by_time:
            assert (g["direction"] == "LONG").sum() == math.ceil(0.2 * len(g))
        mean = _sealed(tmp_path, fold, "train_mean")
        assert len({p["predicted_net_return"] for p in mean}) == 1


def test_sealed_continuous_predictions_rebuild_identically(client, tmp_path):
    raws, market = _tapes(81)
    for k in (1, 2):
        evaluate(client, tmp_path / str(k), raws, market, {"sha256": "s" * 64}, _spec(["linear_gt0"]), "code")
    for fold in ("f1", "f2"):
        assert (tmp_path / "1" / "runs" / fold / "linear_gt0" / "predictions.jsonl").read_bytes() == \
            (tmp_path / "2" / "runs" / fold / "linear_gt0" / "predictions.jsonl").read_bytes()


def test_prices_after_the_evaluated_window_cannot_change_any_prediction(client, tmp_path):
    raws, market = _tapes(82)
    spec = _spec(["linear_gt0"])
    evaluate(client, tmp_path / "a", raws, market, {"sha256": "s" * 64}, spec, "code")
    cut = pd.Timestamp("2024-06-20", tz="America/New_York")
    late = {s: r.copy() for s, r in raws.items()}
    for frame in late.values():
        after = frame.index.tz_convert("America/New_York") >= cut
        frame.loc[after, ["open", "high", "low", "close"]] *= 1.5
        frame.attrs = dict(next(iter(raws.values())).attrs)
    evaluate(client, tmp_path / "b", late, market, {"sha256": "s" * 64}, spec, "code")
    for fold in ("f1", "f2"):
        assert _sealed(tmp_path / "a", fold, "linear_gt0") == _sealed(tmp_path / "b", fold, "linear_gt0")


def test_default_baselines_are_still_phase27bs(client, tmp_path):
    raws, market = _tapes(83)
    spec = small_spec(universe=["AAPL"], folds=_spec()["folds"][:1])
    report = evaluate(client, tmp_path, {"AAPL": raws["AAPL"]}, market, {"sha256": "s" * 64}, spec, "code")
    assert {w["baseline"] for w in report["windows"]} == {"always_hold", "momentum_20", "logistic"}


def test_an_unknown_baseline_is_refused(client, tmp_path):
    raws, market = _tapes(84)
    with pytest.raises(ValueError, match="baseline"):
        evaluate(client, tmp_path, raws, market, {"sha256": "s" * 64}, _spec(["xgboost"]), "code")


def test_the_run_contract_accepts_both_prediction_kinds_and_nothing_else():
    assert validate_prediction({"trade_probability": 0.7, "direction": "LONG"})["direction"] == "LONG"
    assert validate_prediction({"predicted_net_return": -0.002, "direction": "NONE"})["direction"] == "NONE"
    with pytest.raises(ValueError):
        validate_prediction({"predicted_net_return": 0.01, "trade_probability": 0.7, "direction": "LONG"})
