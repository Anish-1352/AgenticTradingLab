"""Phase 27C reuses the Phase 27B pipeline; these pin the few seams it needs.

Named specs, CLI spec selection, per-window-class pooling and a corporate-action
audit. No new builder, labeler or executor exists.
"""
import numpy as np
import pandas as pd
import pytest

from dashboard.backend.tests.test_protocol_api import client
from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.backend.domain.research.phase27b import load_spec
from dashboard.scripts import phase27b_experiment, phase27b_freeze
from dashboard.scripts.phase27b_experiment import evaluate
from dashboard.scripts.phase27b_freeze import split_audit

NY = "America/New_York"


@pytest.fixture(autouse=True)
def local_sdk_source(monkeypatch):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")


def test_load_spec_defaults_to_27b_and_reads_named_specs_only_from_the_research_dir():
    assert load_spec()["phase_id"] == "27B"
    assert load_spec("phase27b-v1.json") == load_spec()
    for bad in ("../research/phase27b-v1.json", "/etc/passwd", "phase27b-v1"):
        with pytest.raises(ValueError):
            load_spec(bad)


@pytest.mark.parametrize("module, extra", [
    (phase27b_freeze, ["--output", "/tmp/x"]),
    (phase27b_experiment, ["--output", "/tmp/x", "--source-manifest", "/tmp/m.json"]),
])
def test_both_clis_take_a_spec_and_default_to_27b(module, extra):
    assert module.parse_args(extra).spec == "phase27b-v1.json"
    assert module.parse_args([*extra, "--spec", "phase27c-v1.json"]).spec == "phase27c-v1.json"


def test_evaluator_pools_each_window_class_separately(client, tmp_path):
    s = small_spec(universe=["AAPL", "MSFT"], folds=[
        {"id": "f1", "window_class": "legacy", "train": ["2024-01-01", "2024-03-18"],
         "validation": ["2024-03-18", "2024-04-01"], "test": ["2024-04-01", "2024-05-06"]},
        {"id": "f2", "window_class": "new_historical", "train": ["2024-01-01", "2024-04-15"],
         "validation": ["2024-04-15", "2024-05-06"], "test": ["2024-05-06", "2024-06-10"]}])
    raws = {"AAPL": tape(sessions=120, seed=50), "MSFT": tape(sessions=120, seed=51)}
    report = evaluate(client, tmp_path, raws, tape(sessions=120, seed=52), {"sha256": "s" * 64}, s, "code")
    assert {w["fold"]: w["window_class"] for w in report["windows"]} == {"f1": "legacy", "f2": "new_historical"}
    by_class = report["pooled_by_class"]
    assert set(by_class) == {"legacy", "new_historical"}
    for cls, fold in (("legacy", "f1"), ("new_historical", "f2")):
        n = sum(w["classification"]["n"] for w in report["windows"]
                if w["fold"] == fold and w["baseline"] == "logistic")
        assert by_class[cls]["logistic"]["n"] == n


def test_a_fold_without_a_class_is_reported_as_unclassified(client, tmp_path):
    s = small_spec(universe=["AAPL"], folds=[
        {"id": "f1", "train": ["2024-01-01", "2024-03-18"], "validation": ["2024-03-18", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-06"]}])
    report = evaluate(client, tmp_path, {"AAPL": tape(sessions=120, seed=53)},
                      tape(sessions=120, seed=54), {"sha256": "s" * 64}, s, "code")
    assert set(report["pooled_by_class"]) == {"unclassified"}


def _split(raw, day, ratio):
    """Divide every price before ``day`` by nothing (raw) or by ``ratio`` (adjusted)."""
    out = raw.copy()
    before = out.index.tz_convert(NY).date < pd.Timestamp(day).date()
    out.loc[before, ["open", "high", "low", "close"]] *= ratio
    out.attrs = dict(raw.attrs)
    return out


def test_split_audit_flags_an_unadjusted_split_and_passes_an_adjusted_one():
    base, s = tape(sessions=20, seed=60), small_spec()
    raw = _split(base, "2024-01-16", 4.0)          # pre-split prices 4x: a raw 4:1 split
    adjusted = base                                  # one continuous adjusted series
    event = [{"date": "2024-01-16", "ratio": 4}]
    flagged = split_audit(raw, event, s)
    assert flagged[0]["mechanical_return"] == pytest.approx(-0.75, abs=0.02) and not flagged[0]["ok"]
    clean = split_audit(adjusted, event, s)
    assert abs(clean[0]["mechanical_return"]) < 0.05 and clean[0]["ok"]


def test_a_fold_subset_runs_only_those_folds_under_the_unchanged_spec(client, tmp_path):
    """Parallel workers split folds, never the spec: every worker's report
    carries the same spec hash and dataset id, so their windows can be joined."""
    s = small_spec(universe=["AAPL"], folds=[
        {"id": "f1", "train": ["2024-01-01", "2024-03-18"], "validation": ["2024-03-18", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-06"]},
        {"id": "f2", "train": ["2024-01-01", "2024-04-15"], "validation": ["2024-04-15", "2024-05-06"],
         "test": ["2024-05-06", "2024-06-10"]}])
    raws, market = {"AAPL": tape(sessions=120, seed=55)}, tape(sessions=120, seed=56)
    whole = evaluate(client, tmp_path / "whole", raws, market, {"sha256": "s" * 64}, s, "code")
    part = evaluate(client, tmp_path / "part", raws, market, {"sha256": "s" * 64}, s, "code", fold_ids=["f2"])
    assert {w["fold"] for w in part["windows"]} == {"f2"} and part["fold_ids"] == ["f2"]
    assert (part["spec_hash"], part["dataset_id"]) == (whole["spec_hash"], whole["dataset_id"])
    one = lambda r: [w for w in r["windows"] if w["fold"] == "f2" and w["baseline"] == "logistic"][0]
    assert one(part)["classification"] == one(whole)["classification"]
    with pytest.raises(ValueError):
        evaluate(client, tmp_path / "bad", raws, market, {"sha256": "s" * 64}, s, "code", fold_ids=["nope"])


def test_the_experiment_cli_takes_a_fold_subset():
    args = ["--output", "/tmp/x", "--source-manifest", "/tmp/m.json"]
    assert phase27b_experiment.parse_args(args).folds is None
    assert phase27b_experiment.parse_args([*args, "--folds", "2018Q1,2018Q2"]).folds == ["2018Q1", "2018Q2"]


def _window(seed):
    from dashboard.backend.domain.research.phase27b import build_dataset, calendar_sessions, session_frame
    s = small_spec()
    raw = tape(sessions=60, seed=seed)
    inputs, outcomes, _ = build_dataset({"AAPL": raw}, raw, s)
    sessions = session_frame(raw, s)
    calendar = calendar_sessions(sessions.index[0], sessions.index[-1], s)
    start = pd.Timestamp(inputs[0]["timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    end = pd.Timestamp(outcomes[-1]["exit_timestamp"]).tz_convert(s["timezone"]).date().isoformat()
    return s, raw, inputs, calendar, start, end


def test_a_prepared_tape_runs_exactly_like_the_raw_tape(client, tmp_path):
    """Whole-tape aggregation is a property of the symbol, not of the run.
    Preparing it once must not change a single byte ATL sees or records."""
    from dashboard.backend.domain.research.phase27b_replay import prepare_symbol_tape
    from dashboard.scripts.phase27b_experiment import run_symbol_window
    s, raw, inputs, calendar, start, end = _window(70)
    every_other = {r["timestamp"] for r in inputs[::2]}
    gate = lambda p: ({"trade_probability": 1.0, "direction": "LONG"} if p["timestamp"] in every_other
                      else {"trade_probability": 0.0, "direction": "NONE"})
    plain = run_symbol_window(client, tmp_path / "a", raw, inputs, calendar, s, "AAPL", start, end, gate, {})
    prepared = prepare_symbol_tape(raw, "AAPL", s)
    fast = run_symbol_window(client, tmp_path / "b", raw, inputs, calendar, s, "AAPL", start, end, gate, {},
                             prepared=prepared)
    assert fast["fingerprint"] == plain["fingerprint"]
    assert fast["trades"] and fast["trades"] == plain["trades"]


def test_evaluate_prepares_each_symbol_once_not_once_per_run(client, tmp_path, monkeypatch):
    from dashboard.backend.domain.research import phase27b_replay
    calls = []
    real = phase27b_replay.prepare_symbol_tape
    monkeypatch.setattr(phase27b_replay, "prepare_symbol_tape",
                        lambda raw, symbol, spec: calls.append(symbol) or real(raw, symbol, spec))
    s = small_spec(universe=["AAPL", "MSFT"], folds=[
        {"id": "f1", "train": ["2024-01-01", "2024-03-18"], "validation": ["2024-03-18", "2024-04-01"],
         "test": ["2024-04-01", "2024-05-06"]},
        {"id": "f2", "train": ["2024-01-01", "2024-04-15"], "validation": ["2024-04-15", "2024-05-06"],
         "test": ["2024-05-06", "2024-06-10"]}])
    raws = {"AAPL": tape(sessions=120, seed=71), "MSFT": tape(sessions=120, seed=72)}
    evaluate(client, tmp_path, raws, tape(sessions=120, seed=73), {"sha256": "s" * 64}, s, "code")
    assert sorted(calls) == ["AAPL", "MSFT"]     # 2 folds x 3 baselines x 2 repeats would be 24
