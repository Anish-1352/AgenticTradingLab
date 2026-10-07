"""The Phase 27D preregistration changes the target and nothing else."""
from dashboard.backend.domain.research.phase27b import load_spec

D, C = load_spec("phase27d-v1.json"), load_spec("phase27c-v1.json")


def test_every_phase27c_task_and_data_field_is_verbatim():
    frozen = D["frozen_from_phase27c"]
    for key in ("universe", "excluded_symbols", "folds", "feature_fields", "horizon_sessions",
                "decision_every_sessions", "trade_threshold", "costs", "classifier", "momentum_rule",
                "data_window", "excluded_sessions", "price_adjustment", "decision_bar_end_local",
                "embargo_sessions", "seed", "initial_cash", "quantity", "uncertainty", "market_symbol"):
        assert key in frozen, key
    for key in frozen:
        assert D[key] == C[key], key
    assert D["phase_id"] == "27D" and D["base_spec"] == "phase27c-v1.json"
    assert D["base_commit"] == "1dc7726f" and D["code_commit"]


def test_the_target_is_the_stored_net_return_untransformed():
    t = D["target"]
    assert t["field"] == "cost_adjusted_forward_return" and t["transform"].startswith("none")
    assert set(t["diagnostics_only"]) == {"maximum_adverse_excursion", "maximum_favorable_excursion",
                                         "future_volatility"}


def test_the_regressor_matches_the_logistic_gate_in_everything_but_loss():
    assert D["regressor"]["l2"] == C["classifier"]["l2"]
    assert D["baselines"] == ["train_mean", "linear_gt0", "linear_top20"]
    assert D["reused_from_phase27c"]["baselines"] == ["always_hold", "momentum_20", "logistic"]
    assert D["ranking_gate"] == {"fraction": 0.2, "scope": D["ranking_gate"]["scope"]}
    assert "cross-sectional" in D["ranking_gate"]["scope"]


def test_no_evidence_is_called_sealed_and_rules_are_written_down():
    assert D["primary_classes"] is None
    assert "REUSED_HISTORICAL" in D["evidence_classification"]
    assert "2026Q4" in D["evidence_classification"]["FUTURE_SEALED_CONFIRMATION"]
    assert set(D["continuous_signal_rule"]["checks"]) == {
        "rho_positive", "rho_ci_excludes_zero", "spread_positive", "spread_ci_excludes_zero",
        "not_concentrated", "beats_binary"}
    assert "SEALED_CONFIRMED = NO" in D["readiness_rule"]
    for banned in ("horizon search", "feature search", "target transformation search",
                   "outlier removal based on results", "hyperparameter sweep", "LLM training"):
        assert banned in D["prohibited"]
    assert D["phase27c_dataset_sha256"] and set(D["phase27c_dataset_sha256"]) == {
        "inputs.jsonl", "outcomes.jsonl", "splits.json", "sessions.json"}
