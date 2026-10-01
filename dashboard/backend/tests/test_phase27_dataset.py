"""Test-first research contract; no network or model serving."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from dashboard.backend.domain.research.phase27 import (
    load_spec, build_dataset, build_splits, classify, make_outcome,
    model_payload, write_dataset, read_source, LogisticGate, score_predictions,
)


def tape():
    times = [t for d in pd.bdate_range("2023-01-03", periods=15)
             for t in pd.date_range(str(d.date()) + " 09:30", periods=78, freq="5min", tz="America/New_York")]
    p = 100 + np.sin(np.arange(len(times)) / 17)
    frame = pd.DataFrame({"open": p, "high": p + .3, "low": p - .3,
                          "close": p + .05, "volume": 1000 + np.arange(len(p))},
                         index=pd.DatetimeIndex(times).tz_convert("UTC"))
    frame.attrs["bar_open_stamped_minutes"] = 5
    return frame


def test_one_spec_is_explicit():
    s = load_spec()
    assert s["universe"] == ["AAPL"] and s["horizon_minutes"] == 60
    assert len(s["folds"]) == 3 and s["allow_short"] is False
    assert s["costs"]["buy_slippage_rate"] > 0


@pytest.mark.parametrize("net,expected", [(.001, (False, "NONE")), (.0010001, (True, "LONG")),
                                          (-.0010001, (False, "NONE")), (0, (False, "NONE"))])
def test_long_only_label_boundary(net, expected):
    assert classify(net, .001) == expected


def test_outcome_matches_round_trip_cash_costs():
    s = load_spec()
    bars = pd.DataFrame({"high": [103.], "low": [99.], "close": [101.]})
    outcome = make_outcome(100., 102., bars, s)
    assert outcome["forward_return"] == pytest.approx(.02)
    assert outcome["cost_adjusted_forward_return"] == pytest.approx(.0187)
    assert outcome["maximum_adverse_excursion"] == pytest.approx(-.01)
    assert outcome["maximum_favorable_excursion"] == pytest.approx(.03)
    assert outcome["future_volatility"] > 0
    assert outcome["trade_worthy"] and outcome["direction"] == "LONG"


def test_future_mutation_cannot_change_inputs_or_feature_availability():
    raw = tape()
    s = load_spec()
    inputs, outcomes, _ = build_dataset(raw, s)
    t = pd.Timestamp(inputs[12]["timestamp"])
    changed = raw.copy()
    changed.loc[changed.index >= t, ["high", "low", "close"]] *= 1.1
    changed.loc[changed.index >= t, "open"] *= 1.1
    changed.loc[changed.index >= t, "volume"] *= 10
    other, _, _ = build_dataset(changed, s)
    assert [r for r in inputs if pd.Timestamp(r["timestamp"]) <= t] == [r for r in other if pd.Timestamp(r["timestamp"]) <= t]
    assert all(pd.Timestamp(r["feature_timestamp"]) <= pd.Timestamp(r["timestamp"]) for r in inputs)
    assert all(pd.Timestamp(o["outcome_available_at"]) > pd.Timestamp(o["timestamp"]) for o in outcomes)
    assert not set(outcomes[0]).difference({"record_id", "timestamp", "symbol"}).intersection(inputs[0])


def test_generation_is_deterministic_and_threshold_cannot_change_features():
    s = load_spec()
    a = build_dataset(tape(), s)
    assert a == build_dataset(tape(), s)
    other = copy.deepcopy(s)
    other["trade_threshold"] = .5
    b = build_dataset(tape(), other)
    assert a[0] == b[0]
    assert a[1] != b[1]


@pytest.mark.parametrize("field", ["forward_return", "future_volatility", "maximum_adverse_excursion", "maximum_favorable_excursion", "trade_worthy"])
def test_model_payload_rejects_hidden_fields(field):
    row = build_dataset(tape(), load_spec())[0][0]
    row["features"][field] = 1
    with pytest.raises(ValueError, match="schema"):
        model_payload(row, load_spec())


def test_purge_and_embargo_use_outcome_availability():
    s = load_spec()
    s["folds"] = [{"id": "boundary", "train": ["2023-01-01", "2023-02-01"],
                   "validation": ["2023-02-01", "2023-03-01"], "test": ["2023-03-01", "2023-04-01"]}]
    times = ["2023-01-15T12:00Z", "2023-01-31T23:30Z", "2023-02-01T12:00Z", "2023-02-02T12:00Z", "2023-03-01T12:00Z", "2023-03-02T12:00Z"]
    inputs = [{"record_id": str(i), "timestamp": t} for i,t in enumerate(times)]
    outcomes = [{"record_id": str(i), "outcome_available_at": (pd.Timestamp(t) + pd.Timedelta(hours=1)).isoformat()} for i,t in enumerate(times)]
    fold = build_splits(inputs, outcomes, s)[0]
    assert fold["train_ids"] == ["0"]
    assert fold["validation_ids"] == ["3"]
    assert fold["test_ids"] == ["5"]
    assert fold["purged_ids"] == ["1"]
    assert set(fold["embargoed_ids"]) == {"2", "4"}
    assert build_splits(inputs, outcomes, s) == [fold]


def test_files_have_hashes_and_hidden_outcomes_are_separate(tmp_path):
    s = load_spec()
    inputs, outcomes, quality = build_dataset(tape(), s)
    manifest = write_dataset(tmp_path, inputs, outcomes, quality, s, {"sha256": "source-hash"}, "code-hash")
    assert manifest["row_count"] == len(inputs)
    assert set(manifest["files"]) == {"inputs.jsonl", "outcomes.jsonl", "splits.json", "spec.json"}
    assert "future_volatility" not in (tmp_path / "inputs.jsonl").read_text()
    assert "future_volatility" in (tmp_path / "outcomes.jsonl").read_text()
    assert manifest == write_dataset(tmp_path, inputs, outcomes, quality, s, {"sha256": "source-hash"}, "code-hash", created_at=manifest["created_at"])


def test_source_hash_is_checked(tmp_path):
    (tmp_path / "AAPL-5m.csv").write_text("changed")
    (tmp_path / "source-manifest.json").write_text(json.dumps({"sha256": "not-the-hash"}))
    with pytest.raises(ValueError, match="hash"):
        read_source(tmp_path / "source-manifest.json")


def test_classifier_is_repeatable_and_predict_does_not_refit():
    x = np.array([[-2., 0], [-1., 1], [1., 0], [2., 1]])
    y = np.array([0, 0, 1, 1])
    a = LogisticGate().fit(x, y)
    b = LogisticGate().fit(x, y)
    assert a.predict(x).tolist() == b.predict(x).tolist()
    before = a.state()
    assert a.predict(np.array([[1., 0]]))[0] == a.predict(np.array([[1., 0], [999., 999.]]))[0]
    assert a.state() == before


def test_metrics_are_defined_for_hold_and_single_class():
    result = score_predictions([0, 1], [0., 0.], [.01, -.02], ["LONG", "NONE"])
    assert result["trade_coverage"] == 0
    assert result["precision_trade"] is None
    assert result["recall_trade"] == 0
    assert result["brier"] == .5
    assert result["balanced_accuracy"] == .5
    assert result["conditional_net_return"] is None
    assert len(result["calibration"]) == 5
    assert score_predictions([0, 0], [0., 0.], [0., 0.], ["NONE", "NONE"])["balanced_accuracy"] is None


def test_logistic_fit_has_no_floating_point_errors_on_large_finite_tape():
    # Realistic feature magnitudes and enough rows to exercise the BLAS path.
    n = 1000
    t = np.arange(n, dtype=float)
    x = np.c_[150 + 20 * np.sin(t / 70), .01 * np.sin(t), .02 * np.cos(t / 3),
              .001 + .002 * (1 + np.sin(t / 5)), 1 + .5 * np.sin(t / 4), .0012 + .00001 * np.cos(t)]
    with np.errstate(all='raise'):
        model = LogisticGate().fit(x, (np.sin(t / 7) > .7).astype(float))
        p = model.predict(x)
    assert np.isfinite(model.weights).all() and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize('part,bounds', [
    ('train', ['2023-01-01', '2023-06-02']),
    ('validation', ['2023-06-01', '2023-07-02']),
    ('test', ['2023-08-01', '2023-07-01']),
])
def test_split_rejects_overlapping_or_reversed_partitions(part, bounds):
    spec = load_spec()
    spec['folds'][0][part] = bounds
    with pytest.raises(ValueError, match='chronological'):
        build_splits([], [], spec)
