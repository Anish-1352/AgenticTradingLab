"""The Phase 27C preregistration scales Phase 27B's task; it may not change it."""
from datetime import date

import pytest

from dashboard.backend.domain.research.phase27b import load_spec
from dashboard.backend.infrastructure.llm.validator import DJIA_30

C, B = load_spec("phase27c-v1.json"), load_spec("phase27b-v1.json")


def test_every_frozen_task_field_is_phase27b_verbatim():
    frozen = C["frozen_from_phase27b"]
    for key in ("task_version", "feature_fields", "horizon_sessions", "decision_every_sessions",
                "trade_threshold", "decision_bar_end_local", "costs", "classifier", "momentum_rule",
                "quantity", "initial_cash", "embargo_sessions", "price_adjustment", "seed"):
        assert key in frozen
    for key in frozen:
        assert C[key] == B[key], key
    assert C["phase_id"] == "27C" and C["base_task_spec"] == "phase27b-v1.json" and C["base_commit"]


def test_universe_is_a_fixed_rule_over_the_djia30_constant():
    assert C["candidate_universe"] == DJIA_30
    assert len(set(C["universe"])) == len(C["universe"]) and set(C["universe"]) <= set(DJIA_30)
    assert set(C["universe"]) | set(C["excluded_symbols"]) == set(DJIA_30)
    assert not set(C["universe"]) & set(C["excluded_symbols"])
    assert all(reason for reason in C["excluded_symbols"].values())
    assert C["market_symbol"] == "SPY" and "SPY" not in C["universe"]


def test_calendar_extends_phase27b_and_agrees_where_they_overlap():
    first, last = B["excluded_sessions"][0], B["excluded_sessions"][-1]
    overlap = [d for d in C["excluded_sessions"] if first <= d <= last]
    assert overlap == B["excluded_sessions"]
    assert C["excluded_sessions"] == sorted(C["excluded_sessions"])
    start, end = C["data_window"]
    assert all(start <= d < end for d in C["excluded_sessions"])


def test_folds_are_chronological_disjoint_and_classed():
    folds = C["folds"]
    tests = [f["test"] for f in folds]
    assert tests == sorted(tests)
    assert all(a[1] <= b[0] for a, b in zip(tests, tests[1:]))
    for f in folds:
        assert f["window_class"] in {"new_historical", "legacy", "sealed"}
        assert f["train"][0] == C["data_window"][0]
        assert f["train"][1] <= f["validation"][0] < f["validation"][1] <= f["test"][0] < f["test"][1]
    legacy = {f["id"]: f for f in folds if f["window_class"] == "legacy"}
    assert sorted(legacy) == [f["id"] for f in B["folds"]]
    for f in B["folds"]:
        assert legacy[f["id"]]["test"] == f["test"] and legacy[f["id"]]["validation"] == f["validation"]


def test_sealed_windows_start_after_everything_phase27b_ever_fetched():
    b_end = date.fromisoformat(B["data_window"][1])
    sealed = [f for f in C["folds"] if f["window_class"] == "sealed"]
    assert sealed and all(date.fromisoformat(f["test"][0]) > b_end for f in sealed)
    new = [f for f in C["folds"] if f["window_class"] == "new_historical"]
    first_legacy = min(f["test"][0] for f in C["folds"] if f["window_class"] == "legacy")
    assert new and all(f["test"][1] <= first_legacy for f in new)


def test_the_decision_rules_are_written_down_before_results():
    rule = C["task_signal_rule"]
    assert set(rule["checks"]) == {"above_chance", "ci_excludes_chance", "not_one_symbol_or_quarter",
                                   "ranking_agrees"}
    assert C["evidence_sets"]["primary"] == ["new_historical", "sealed"]
    assert "legacy" in C["evidence_sets"]["never_pooled_with_primary"]
    assert C["uncertainty"]["draws"] == 2000 and C["uncertainty"]["seed"] == 2727
    assert C["probability_ranking"]["fixed_buckets"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert C["canonical_cost_multiplier"] == 1.0
