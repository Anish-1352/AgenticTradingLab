"""Phase 27B: frozen source reading, ranking signal, and the full evaluator."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.tests.test_protocol_api import client
from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.backend.domain.research.phase27 import digest
from dashboard.backend.domain.research.phase27b import read_sources
from dashboard.backend.domain.research.phase27b_metrics import rank_signal
from dashboard.scripts.phase27b_experiment import evaluate


@pytest.fixture(autouse=True)
def local_sdk_source(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "packaging/agentictrading/src"))


def _write_source(directory, frames):
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for symbol, frame in frames.items():
        path = directory / f"{symbol}-5m.csv"
        frame.rename_axis("timestamp").to_csv(path)
        files[symbol] = {"file": path.name, "sha256": digest(path.read_bytes())}
    manifest = {"files": files, "adjustment": "split", "feed": "sip",
                "frame_attrs": {"bar_open_stamped_minutes": 5}}
    (directory / "source-manifest.json").write_text(json.dumps(manifest))
    return directory / "source-manifest.json"


def test_sources_are_hash_checked_and_round_trip(tmp_path):
    frames = {"AAPL": tape(sessions=5, seed=40), "SPY": tape(sessions=5, seed=41)}
    path = _write_source(tmp_path, frames)
    raws, meta = read_sources(path, ["AAPL"], "SPY")
    assert set(raws) == {"AAPL", "SPY"}
    pd.testing.assert_frame_equal(raws["AAPL"][["open", "close"]], frames["AAPL"][["open", "close"]],
                                  check_freq=False, check_names=False)
    (tmp_path / "AAPL-5m.csv").write_text((tmp_path / "AAPL-5m.csv").read_text() + "\n")
    with pytest.raises(ValueError):
        read_sources(path, ["AAPL"], "SPY")


def test_sources_refuse_a_missing_symbol(tmp_path):
    path = _write_source(tmp_path, {"SPY": tape(sessions=5, seed=42)})
    with pytest.raises(ValueError):
        read_sources(path, ["AAPL"], "SPY")


def test_rank_signal_detects_perfect_inverse_and_no_order():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    good = rank_signal(p, [0, 0, 1, 1], [-0.02, -0.01, 0.01, 0.02])
    assert good["auc_trade"] == 1.0 and good["spearman_probability_net"] == pytest.approx(1.0)
    bad = rank_signal(p, [1, 1, 0, 0], [0.02, 0.01, -0.01, -0.02])
    assert bad["auc_trade"] == 0.0 and bad["spearman_probability_net"] == pytest.approx(-1.0)


def test_rank_signal_is_undefined_for_constant_predictions_or_one_class():
    r = rank_signal([0.0, 0.0, 0.0], [0, 1, 0], [0.01, 0.02, -0.01])
    assert r["auc_trade"] == 0.5 and r["spearman_probability_net"] is None
    r = rank_signal([0.1, 0.9], [1, 1], [0.01, 0.02])
    assert r["auc_trade"] is None


def _eval_spec():
    return small_spec(universe=["AAPL", "MSFT"], folds=[
        {"id": "f1", "train": ["2024-01-01", "2024-03-18"], "validation": ["2024-03-18", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-06"]},
        {"id": "f2", "train": ["2024-01-01", "2024-04-15"], "validation": ["2024-04-15", "2024-05-06"],
         "test": ["2024-05-06", "2024-06-10"]}])


def test_evaluator_runs_every_window_reproducibly_and_matches_hidden_cash(client, tmp_path):
    s = _eval_spec()
    raws = {"AAPL": tape(sessions=120, seed=50), "MSFT": tape(sessions=120, seed=51)}
    market = tape(sessions=120, seed=52)
    report = evaluate(client, tmp_path, raws, market, {"sha256": "s" * 64}, s, "code")
    windows = report["windows"]
    assert {w["fold"] for w in windows} == {"f1", "f2"}
    assert {w["baseline"] for w in windows} == {"always_hold", "momentum_20", "logistic"}
    for w in windows:
        assert w["reproducible"] and w["cash_matches_outcomes"]
        assert set(w["per_symbol"]) <= {"AAPL", "MSFT"}
        assert len(w["probability_buckets"]) == 5
        assert [c["multiplier"] for c in w["cost_sensitivity"]] == s["cost_sensitivity_multipliers"]
        if w["baseline"] == "always_hold":
            assert w["financial"]["fills"] == 0 and w["financial"]["fees"] == 0
            assert w["financial"]["cumulative_return"] == pytest.approx(0.0)
    # The test predictions of each fold were sealed before labels were joined.
    for fold in ("f1", "f2"):
        sealed = tmp_path / "runs" / fold / "logistic" / "predictions.jsonl"
        assert sealed.exists() and "trade_worthy" not in sealed.read_text()
    saved = json.loads((tmp_path / "experiment.json").read_text())
    assert saved["experiment_id"] == report["experiment_id"]
    assert saved["pooled"]["logistic"]["n"] == sum(w["classification"]["n"] for w in windows
                                                  if w["baseline"] == "logistic")
