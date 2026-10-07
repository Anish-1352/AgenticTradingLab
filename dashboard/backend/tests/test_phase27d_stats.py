"""Phase 27D statistics: continuous ranking, Q5-Q1 spread, paired comparison, the rules."""
import numpy as np
import pytest

from dashboard.backend.domain.research.phase27c_stats import window_quintile_table
from dashboard.backend.domain.research.phase27d_stats import (
    window_quintiles, quintile_table, quintile_spread_with_ci, paired_difference_with_ci,
    leave_one_out_spearman, continuous_signal, beats_binary, readiness)


def _panel(dates=80, names=12, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(dates), names)
    w = d // 20                                                # four "windows"
    score = rng.normal(size=d.size)
    net = 0.01 * (signal * score + rng.normal(size=d.size) + np.repeat(rng.normal(size=dates), names))
    return d, w, score, net


def test_quintiles_are_cut_within_window_on_predictions_exactly_as_in_27c():
    d, w, score, net = _panel()
    q = window_quintiles(w, score)
    z = np.zeros(d.size)
    assert [int((q == k).sum()) for k in range(5)] == \
        [r["count"] for r in window_quintile_table(w, score, d, z, net, z, z)]
    rng = np.random.default_rng(1)
    assert np.array_equal(q, window_quintiles(w, score))                  # deterministic
    assert np.array_equal(q, window_quintiles(w, score)) and \
        np.array_equal(window_quintiles(w, score), window_quintiles(w, score.copy()))
    shuffled_net = rng.permutation(net)                                   # outcomes cannot move a row
    assert quintile_table(w, score, d, shuffled_net, z, z)[0]["count"] == quintile_table(w, score, d, net, z, z)[0]["count"]


def test_quintile_table_reports_prediction_and_realised_outcomes():
    d, w, score, net = _panel(seed=2)
    rows = quintile_table(w, score, d, net, net - 0.01, net + 0.01)
    assert [r["quintile"] for r in rows] == [1, 2, 3, 4, 5]
    for r in rows:
        assert set(r) >= {"count", "unique_dates", "mean_prediction", "mean_net_return", "median_net_return",
                          "positive_fraction", "mean_mae", "mean_mfe"}
    assert rows[-1]["mean_prediction"] > rows[0]["mean_prediction"]
    assert rows[-1]["mean_net_return"] > rows[0]["mean_net_return"]


def test_spread_ci_is_date_clustered_deterministic_and_signed():
    d, w, score, net = _panel(signal=1.0, seed=3)
    a = quintile_spread_with_ci(w, score, d, net, draws=300, seed=5)
    assert a == quintile_spread_with_ci(w, score, d, net, draws=300, seed=5)
    assert a["spread"] > 0 and a["ci_low"] > 0 and a["dates"] == 80
    flat = quintile_spread_with_ci(w, -score, d, net, draws=300, seed=5)
    assert flat["spread"] == pytest.approx(-a["spread"])


def test_spread_ci_is_not_shrunk_by_copying_rows_within_a_date():
    d, w, score, net = _panel(names=1, dates=200, signal=0.3, seed=4)
    w = d // 50
    one = quintile_spread_with_ci(w, score, d, net, draws=400, seed=1)
    ten = quintile_spread_with_ci(np.repeat(w, 10), np.repeat(score, 10), np.repeat(d, 10),
                                  np.repeat(net, 10), draws=400, seed=1)
    width = lambda r: r["ci_high"] - r["ci_low"]
    assert ten["spread"] == pytest.approx(one["spread"])
    assert width(ten) == pytest.approx(width(one), rel=0.1)


def test_paired_difference_uses_the_same_resampled_dates_for_both_models():
    d, w, score, net = _panel(seed=6)
    rng = np.random.default_rng(7)
    weak = score + rng.normal(scale=3.0, size=score.size)
    r = paired_difference_with_ci(w, d, score, weak, net, draws=300, seed=2)
    assert r["delta_spearman"] > 0 and r["delta_spearman_ci_low"] > 0
    assert r["delta_spread"] > 0
    same = paired_difference_with_ci(w, d, score, score, net, draws=300, seed=2)
    assert same["delta_spearman"] == 0 and same["delta_spearman_ci_low"] == 0 == same["delta_spearman_ci_high"]


def test_leave_one_out_spearman_drops_each_group_once():
    d, w, score, net = _panel(seed=8)
    groups = np.tile(np.array(list("abcdefghijkl")), 80)
    out = leave_one_out_spearman(groups, d, score, net, draws=50, seed=0)
    assert sorted(out) == list("abcdefghijkl") and all(v["rows"] == 880 for v in out.values())


GOOD = dict(rho=0.04, rho_ci_low=0.01, spread=0.004, spread_ci_low=0.001,
            loo_year_min=0.02, loo_quarter_min=0.03, loo_symbol_min=0.03, beats="YES")


@pytest.mark.parametrize("change, expected", [
    ({}, "PROMISING"),
    ({"rho_ci_low": -0.01}, "INCONCLUSIVE"),
    ({"spread_ci_low": -0.001}, "INCONCLUSIVE"),
    ({"beats": "INCONCLUSIVE"}, "INCONCLUSIVE"),
    ({"beats": "NO"}, "INCONCLUSIVE"),
    ({"loo_year_min": -0.001}, "WEAK"),                    # significant, but rests on one year
    ({"loo_symbol_min": 0.0}, "WEAK"),
    ({"rho_ci_low": -0.01, "loo_year_min": -0.001}, "INCONCLUSIVE"),
    ({"rho": 0.0, "rho_ci_low": -0.03}, "WEAK"),
    ({"rho": -0.02, "rho_ci_low": -0.05}, "WEAK"),
    ({"spread": -0.001, "spread_ci_low": -0.004}, "WEAK"),
    ({"rho": None, "rho_ci_low": None}, "WEAK"),
])
def test_continuous_signal_is_the_preregistered_truth_table(change, expected):
    label, checks = continuous_signal(**{**GOOD, **change})
    assert label == expected
    assert set(checks) == {"rho_positive", "rho_ci_excludes_zero", "spread_positive",
                           "spread_ci_excludes_zero", "not_concentrated", "beats_binary"}


@pytest.mark.parametrize("delta_rho, low, high, delta_spread, expected", [
    (0.03, 0.01, 0.05, 0.002, "YES"),
    (0.03, 0.01, 0.05, -0.001, "INCONCLUSIVE"),
    (0.03, -0.01, 0.07, 0.002, "INCONCLUSIVE"),
    (0.0, -0.02, 0.02, 0.002, "NO"),
    (-0.02, -0.05, 0.01, 0.002, "NO"),
])
def test_beats_binary(delta_rho, low, high, delta_spread, expected):
    assert beats_binary(delta_rho, low, high, delta_spread) == expected


@pytest.mark.parametrize("phase_pass, signal, sealed, expected", [
    (True, "PROMISING", "NO", {"MODEL_TARGET_READY": "YES", "SEALED_CONFIRMED": "NO", "PHASE_28_READY": "NO"}),
    (True, "PROMISING", "YES", {"MODEL_TARGET_READY": "YES", "SEALED_CONFIRMED": "YES", "PHASE_28_READY": "YES"}),
    (True, "INCONCLUSIVE", "YES", {"MODEL_TARGET_READY": "NO", "SEALED_CONFIRMED": "YES", "PHASE_28_READY": "NO"}),
    (False, "PROMISING", "NO", {"MODEL_TARGET_READY": "NO", "SEALED_CONFIRMED": "NO", "PHASE_28_READY": "NO"}),
])
def test_readiness_separates_the_target_from_sealed_confirmation(phase_pass, signal, sealed, expected):
    assert readiness(phase_pass, signal, sealed) == expected
