"""Phase 27B research contract: weekly 5-session long/flat task, multi-symbol.

Test-first. No network, no model serving. Synthetic tapes only.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.research.phase27b import (
    load_spec, session_frame, build_features, decision_schedule, classify,
    make_outcome, build_dataset, build_splits, write_dataset, model_payload,
)
from dashboard.backend.domain.trading.execution import calculate_transaction_costs
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
from dashboard.backend.infrastructure.llm.validator import DJIA_30

NY = "America/New_York"


def tape(start="2024-01-02", sessions=60, seed=0, drift=0.0):
    """Regular-hours 5m bars. Holidays inside the range are deliberately
    present in the raw tape; the builder must drop them by calendar."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=sessions)
    times = [t for d in days for t in pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", tz=NY)]
    steps = rng.normal(drift, 0.001, len(times))
    close = 100 * np.exp(np.cumsum(steps))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0002, len(times)))
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                          "low": np.minimum(open_, close) * 0.999, "close": close,
                          "volume": rng.integers(1000, 5000, len(times)).astype(float)},
                         index=pd.DatetimeIndex(times).tz_convert("UTC"))
    frame.attrs["bar_open_stamped_minutes"] = 5
    return frame


def small_spec(**overrides):
    """The frozen spec with a short warm-up so small tapes yield rows."""
    s = load_spec()
    s = {**s, "warmup_sessions": 6, "feature_windows": {"short": 3, "long": 5}, **overrides}
    return s


# ------------------------------------------------------------------ the spec

def test_spec_is_frozen_and_explicit():
    s = load_spec()
    assert s["phase_id"] == "27B" and s["task_version"]
    assert set(s["universe"]) <= set(DJIA_30) and 10 <= len(s["universe"]) <= 20
    assert s["market_symbol"] == "SPY" and s["market_symbol"] not in s["universe"]
    assert s["horizon_sessions"] == 5 and s["decision_every_sessions"] == 5
    assert s["decision_bar_end_local"] == "15:30" and s["allow_short"] is False
    assert s["trade_threshold"] == 0.0025 and s["embargo_sessions"] == 5
    assert s["price_adjustment"] == "split"
    assert len(s["folds"]) >= 6 and s["seed"]
    assert s["costs"]["buy_slippage_rate"] > 0 and s["costs"]["commission_rate"] > 0


# -------------------------------------------------------- the decision clock

def test_session_price_is_the_1530_completed_bar_and_entry_is_the_next_open():
    raw, s = tape(sessions=3), small_spec()
    sessions = session_frame(raw, s)
    d = sessions.index[0]
    local = raw.index.tz_convert(NY)
    day = raw[local.date == pd.Timestamp(d).date()]
    day_local = day.index.tz_convert(NY)
    t = pd.Timestamp(f"{d} 15:30", tz=NY).tz_convert("UTC")
    row = sessions.loc[d]
    assert row["decision_timestamp"] == t
    # The completed 14:30-15:30 hourly bar closes with the 15:25 5m bar.
    assert row["price"] == pytest.approx(day.loc[t - pd.Timedelta(minutes=5), "close"])
    # Volume is everything completed by 15:30, and nothing after.
    before = day[(day_local.hour * 60 + day_local.minute) < 930]
    assert row["volume"] == pytest.approx(before["volume"].sum())
    # The executable price is the open of the source bar AT the boundary.
    assert row["entry_open"] == pytest.approx(day.loc[t, "open"])


def test_holidays_and_early_closes_are_not_sessions():
    raw, s = tape("2024-01-02", sessions=40), small_spec()
    days = {str(d) for d in session_frame(raw, s).index}
    assert "2024-01-15" not in days and "2024-02-19" not in days   # MLK, Presidents
    early = tape("2024-07-01", sessions=5)
    days = {str(d) for d in session_frame(early, s).index}
    assert "2024-07-03" not in days and "2024-07-04" not in days    # early close, holiday


# ------------------------------------------------------------------ features

def test_feature_definitions_on_a_known_path():
    raw, s = tape(sessions=30, seed=1), small_spec()
    sessions = session_frame(raw, s)
    feats = build_features(sessions, sessions, s)
    p, v = sessions["price"], sessions["volume"]
    d = sessions.index[-1]
    f = feats.loc[d]
    assert f["return_1"] == pytest.approx(p.iloc[-1] / p.iloc[-2] - 1)
    assert f["return_5"] == pytest.approx(p.iloc[-1] / p.iloc[-6] - 1)
    logs = np.log(p).diff()
    assert f["volatility_short"] == pytest.approx(logs.iloc[-3:].std(ddof=0))
    assert f["volatility_long"] == pytest.approx(logs.iloc[-5:].std(ddof=0))
    assert f["volume_ratio"] == pytest.approx(v.iloc[-1] / v.iloc[-5:].mean())
    assert f["volume_change_1"] == pytest.approx(np.log(v.iloc[-1] / v.iloc[-2]))
    assert f["sma_gap"] == pytest.approx(p.iloc[-1] / p.iloc[-5:].mean() - 1)
    assert f["market_return_1"] == pytest.approx(p.iloc[-1] / p.iloc[-2] - 1)  # market == self here
    assert list(feats.columns) == s["feature_fields"]


def test_market_features_use_the_same_1530_timestamp():
    raw, mkt, s = tape(seed=2), tape(seed=3), small_spec()
    a, m = session_frame(raw, s), session_frame(mkt, s)
    feats = build_features(a, m, s)
    d = feats.dropna().index[-1]
    i = list(m.index).index(d)
    assert feats.loc[d, "market_return_1"] == pytest.approx(m["price"].iloc[i] / m["price"].iloc[i - 1] - 1)
    assert (a["decision_timestamp"] == m["decision_timestamp"]).all()


# -------------------------------------------------------------------- labels

@pytest.mark.parametrize("net,expected", [(0.0025, (False, "NONE")), (0.0025001, (True, "LONG")),
                                          (-0.05, (False, "NONE")), (0.0, (False, "NONE"))])
def test_long_only_label_boundary(net, expected):
    assert classify(net, 0.0025) == expected


def _atl_round_trip(entry, exit_price, costs, multiplier=1.0):
    scaled = {**costs, "commission_rate": costs["commission_rate"] * multiplier,
              "buy_slippage_rate": costs["buy_slippage_rate"] * multiplier,
              "sell_slippage_rate": costs["sell_slippage_rate"] * multiplier}
    profile = TransactionCostProfile(**scaled)
    q = lambda side, px: calculate_transaction_costs(side=side, reference_price=px, shares=1,
                                                     transaction_cost_profile=profile)["net_cash_impact"]
    return (q("buy", entry) + q("sell", exit_price)) / entry


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0])
def test_outcome_net_return_is_atl_round_trip_cash(multiplier):
    s = small_spec()
    path = pd.DataFrame({"high": [103.0], "low": [99.0], "close": [101.0]})
    o = make_outcome(100.0, 102.0, path, s, cost_multiplier=multiplier)
    assert o["forward_return"] == pytest.approx(0.02)
    assert o["cost_adjusted_forward_return"] == pytest.approx(_atl_round_trip(100.0, 102.0, s["costs"], multiplier))
    assert o["cost_adjusted_forward_return"] < o["forward_return"]


def test_mae_mfe_are_exact_on_known_paths():
    s = small_spec()
    down_up = pd.DataFrame({"high": [101.0, 106.0], "low": [97.0, 100.0], "close": [98.0, 105.0]})
    o = make_outcome(100.0, 104.0, down_up, s)
    assert o["maximum_adverse_excursion"] == pytest.approx(-0.03)
    assert o["maximum_favorable_excursion"] == pytest.approx(0.06)
    only_up = pd.DataFrame({"high": [102.0], "low": [100.5], "close": [101.0]})
    o = make_outcome(100.0, 101.5, only_up, s)
    assert o["maximum_adverse_excursion"] == 0.0 and o["maximum_favorable_excursion"] == pytest.approx(0.02)
    # The exit price itself counts when it is the extreme.
    o = make_outcome(100.0, 95.0, pd.DataFrame({"high": [100.2], "low": [99.0], "close": [99.5]}), s)
    assert o["maximum_adverse_excursion"] == pytest.approx(-0.05)


def test_mae_starts_at_entry_not_before_it():
    raw, s = tape(sessions=30, seed=4), small_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    t = pd.Timestamp(inputs[0]["timestamp"])
    crashed = raw.copy()
    crashed.loc[crashed.index < t, "low"] *= 0.5     # a pre-entry crash
    _, other, _ = build_dataset({"AAPL": crashed}, crashed, s)
    assert other[0]["maximum_adverse_excursion"] == outcomes[0]["maximum_adverse_excursion"]


def test_forward_return_starts_from_the_executable_open_not_the_decision_close():
    raw, s = tape(sessions=30, seed=5), small_spec()
    inputs, _, _ = build_dataset({"AAPL": raw}, raw, s)
    t = pd.Timestamp(inputs[0]["timestamp"])
    gapped = raw.copy()
    # The executable price moves; the decision close does not. The high moves
    # with it so the bar stays valid OHLC -- the source validator rejects
    # an open above its own high, which is correct.
    gapped.loc[t, "open"] *= 1.02
    gapped.loc[t, "high"] = max(gapped.loc[t, "high"], gapped.loc[t, "open"])
    new_inputs, new_out, _ = build_dataset({"AAPL": gapped}, gapped, s)
    assert new_inputs[0] == inputs[0]
    entry = gapped.loc[t, "open"]
    exit_t = pd.Timestamp(new_out[0]["exit_timestamp"])
    assert new_out[0]["entry_price"] == pytest.approx(entry)
    assert new_out[0]["forward_return"] == pytest.approx(gapped.loc[exit_t, "open"] / entry - 1)


# ----------------------------------------------------------------- the schedule

def test_decisions_are_every_fifth_session_and_never_overlap():
    raw, s = tape(sessions=60, seed=6), small_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    sessions = list(session_frame(raw, s).index)
    idx = [sessions.index(pd.Timestamp(r["timestamp"]).tz_convert(NY).date()) for r in inputs]
    assert all(b - a == 5 for a, b in zip(idx, idx[1:]))
    for a, b in zip(outcomes, outcomes[1:]):
        assert pd.Timestamp(b["entry_timestamp"]) >= pd.Timestamp(a["exit_timestamp"])
        assert pd.Timestamp(b["entry_timestamp"]) == pd.Timestamp(a["exit_timestamp"])  # shared endpoint


def test_schedule_is_anchored_by_calendar_not_by_data_or_outcomes():
    s = small_spec()
    a = decision_schedule(list(range(40)), s)
    assert a == decision_schedule(list(range(40)), s)
    assert all(e - d == s["horizon_sessions"] for d, e in a)
    assert a[0][0] == s["warmup_sessions"]


# -------------------------------------------------------------- leakage tests

def test_future_perturbation_cannot_change_inputs_but_changes_outcomes():
    raw, mkt, s = tape(sessions=60, seed=7), tape(sessions=60, seed=8), small_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, mkt, s)
    k = len(inputs) // 2
    t = pd.Timestamp(inputs[k]["timestamp"])
    changed_raw, changed_mkt = raw.copy(), mkt.copy()
    for frame in (changed_raw, changed_mkt):
        future = frame.index >= t      # the 15:30 source bar opens at t: it is future
        # A rising, non-uniform factor. A single constant would scale entry and
        # exit alike and leave every return unchanged, so could not show that
        # outcomes respond to the future at all.
        ramp = 1.1 + 0.001 * np.arange(future.sum())
        frame.loc[future, ["open", "high", "low", "close"]] = (
            frame.loc[future, ["open", "high", "low", "close"]].mul(ramp, axis=0))
        frame.loc[future, "volume"] *= 7
    other_in, other_out, _ = build_dataset({"AAPL": changed_raw}, changed_mkt, s)
    assert other_in[: k + 1] == inputs[: k + 1]
    assert other_out[k]["forward_return"] != outcomes[k]["forward_return"]


def test_feature_timestamp_never_exceeds_decision_timestamp():
    raw, s = tape(sessions=40, seed=9), small_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw, "MSFT": tape(sessions=40, seed=10)}, raw, s)
    for r, o in zip(inputs, outcomes):
        assert pd.Timestamp(r["feature_timestamp"]) <= pd.Timestamp(r["timestamp"])
        assert pd.Timestamp(o["outcome_available_at"]) > pd.Timestamp(r["timestamp"])


@pytest.mark.parametrize("field", ["forward_return", "cost_adjusted_forward_return",
    "maximum_adverse_excursion", "maximum_favorable_excursion", "future_volatility",
    "trade_worthy", "direction", "entry_price", "exit_timestamp"])
def test_model_payload_rejects_hidden_fields(field):
    raw, s = tape(sessions=30, seed=11), small_spec()
    inputs, _, _ = build_dataset({"AAPL": raw}, raw, s)
    payload = model_payload(inputs[0], s)
    assert set(payload) == {"timestamp", "symbol", "features"}
    with pytest.raises(ValueError):
        model_payload({**inputs[0], field: 1}, s)


# -------------------------------------------------------------------- splits

def _split_spec():
    return small_spec(folds=[
        {"id": "a", "train": ["2024-01-01", "2024-02-15"], "validation": ["2024-02-15", "2024-03-01"],
         "test": ["2024-03-01", "2024-04-01"]},
        {"id": "b", "train": ["2024-01-01", "2024-03-15"], "validation": ["2024-03-15", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-01"]}])


def test_purge_uses_outcome_availability_and_embargo_uses_sessions():
    raw, s = tape(sessions=90, seed=12), _split_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    exits = {o["record_id"]: pd.Timestamp(o["outcome_available_at"]) for o in outcomes}
    starts = {r["record_id"]: pd.Timestamp(r["timestamp"]) for r in inputs}
    sessions = list(session_frame(raw, s).index)
    for fold in build_splits(inputs, outcomes, sessions, s):
        for part in ("train", "validation", "test"):
            begin, end = [pd.Timestamp(x, tz="UTC") for x in fold["bounds"][part]]
            for rid in fold[part + "_ids"]:
                assert begin <= starts[rid] < end and exits[rid] < end
        for rid in fold["purged_ids"]:
            assert any(exits[rid] >= pd.Timestamp(fold["bounds"][p][1], tz="UTC")
                       for p in ("train", "validation", "test")
                       if pd.Timestamp(fold["bounds"][p][0], tz="UTC") <= starts[rid] < pd.Timestamp(fold["bounds"][p][1], tz="UTC"))
        # embargo: no selected validation/test row in the first 5 sessions of its partition
        for part in ("validation", "test"):
            begin = pd.Timestamp(fold["bounds"][part][0]).date()
            first = [d for d in sessions if d >= begin][: s["embargo_sessions"]]
            for rid in fold[part + "_ids"]:
                assert starts[rid].tz_convert(NY).date() not in first
        ids = [set(fold[p + "_ids"]) for p in ("train", "validation", "test")]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_test_windows_are_disjoint_and_strictly_after_training():
    raw, s = tape(sessions=90, seed=13), _split_spec()
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    folds = build_splits(inputs, outcomes, list(session_frame(raw, s).index), s)
    tests = [set(f["test_ids"]) for f in folds]
    assert not (tests[0] & tests[1])
    t = {r["record_id"]: pd.Timestamp(r["timestamp"]) for r in inputs}
    for f in folds:
        if f["train_ids"] and f["test_ids"]:
            assert max(t[i] for i in f["train_ids"]) < min(t[i] for i in f["test_ids"])


def test_splits_reject_overlapping_partitions():
    raw, s = tape(sessions=40, seed=14), _split_spec()
    bad = {**s, "folds": [{"id": "x", "train": ["2024-01-01", "2024-03-01"],
                           "validation": ["2024-02-15", "2024-03-15"], "test": ["2024-03-15", "2024-04-01"]}]}
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, bad)
    with pytest.raises(ValueError):
        build_splits(inputs, outcomes, list(session_frame(raw, bad).index), bad)


# --------------------------------------------------------------- determinism

def test_dataset_is_deterministic_and_outcomes_live_in_a_separate_file(tmp_path):
    raw, s = tape(sessions=60, seed=15), _split_spec()
    source = {"sha256": "x" * 64}
    built = [build_dataset({"AAPL": raw, "MSFT": tape(sessions=60, seed=16)}, raw, s) for _ in range(2)]
    assert built[0][0] == built[1][0] and built[0][1] == built[1][1]
    sessions = list(session_frame(raw, s).index)
    a = write_dataset(tmp_path / "a", *built[0], sessions, s, source, "code", created_at="fixed")
    b = write_dataset(tmp_path / "b", *built[1], sessions, s, source, "code", created_at="fixed")
    assert a["dataset_id"] == b["dataset_id"] and a["files"] == b["files"]
    inputs_text = (tmp_path / "a" / "inputs.jsonl").read_text()
    for hidden in ("forward_return", "trade_worthy", "maximum_adverse_excursion", "entry_price"):
        assert hidden not in inputs_text
    assert (tmp_path / "a" / "outcomes.jsonl").exists() and (tmp_path / "a" / "splits.json").exists()


def test_threshold_changes_labels_but_never_features():
    raw, s = tape(sessions=60, seed=17), small_spec()
    a_in, a_out, _ = build_dataset({"AAPL": raw}, raw, s)
    b_in, b_out, _ = build_dataset({"AAPL": raw}, raw, {**s, "trade_threshold": -1.0})
    assert a_in == b_in
    assert all(o["trade_worthy"] for o in b_out)


def test_a_symbol_data_gap_never_stretches_a_feature_window():
    """Lookbacks count calendar sessions. When one symbol misses a session the
    market held, a window reaching across the gap is missing data, not a window
    one session longer -- the row is dropped, never silently re-measured."""
    s = small_spec()
    full, market = tape(sessions=60, seed=4), tape(sessions=60, seed=5)
    gap_day = pd.Timestamp("2024-02-06").date()
    gapped = full[full.index.tz_convert(NY).date != gap_day]
    rows_full, _, _ = build_dataset({"X": full}, market, s)
    rows_gap, _, quality = build_dataset({"X": gapped}, market, s)
    by_t = {r["timestamp"]: r["features"] for r in rows_full}
    assert rows_gap and len(rows_gap) < len(rows_full)
    for r in rows_gap:
        assert r["features"] == pytest.approx(by_t[r["timestamp"]])
    # Every decision whose longest lookback reached the gap is gone.
    calendar = [d for d in session_frame(market, s).index]
    g = calendar.index(gap_day)
    reach = s["feature_windows"]["long"]
    kept = {pd.Timestamp(r["timestamp"]).tz_convert(NY).date() for r in rows_gap}
    assert not any(g <= calendar.index(d) <= g + reach for d in kept)
    assert quality["symbols"]["X"].get("warmup_or_missing_features", 0) >= 1
