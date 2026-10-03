"""Phase 27C statistics: date-clustered uncertainty, dependence, ranking, the decision rule.

Synthetic inputs only. The rule tests pin the preregistered definitions so they
cannot drift after results are seen.
"""
import numpy as np
import pytest

from dashboard.backend.domain.research.phase27c_stats import (
    auc_with_ci, spearman_with_ci, dependence, bucket_table, window_quintile_table,
    leave_one_out, task_signal, sealed_confirmation, phase28_ready, FIXED_EDGES,
)


def _panel(dates=60, names=10, signal=0.0, seed=0, shared=0.0):
    """Rows = dates x names. ``shared`` is the weight of a per-date common shock."""
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(dates), names)
    common = np.repeat(rng.normal(size=dates), names)
    score = rng.normal(size=d.size)
    latent = signal * score + shared * common + rng.normal(size=d.size)
    y = (latent > 0).astype(int)
    p = 1 / (1 + np.exp(-score))
    return d, y, p, latent * 0.01


def test_auc_ci_is_deterministic_and_counts_dates_not_rows():
    d, y, p, _ = _panel(signal=1.0)
    a, b = auc_with_ci(d, y, p, draws=300, seed=7), auc_with_ci(d, y, p, draws=300, seed=7)
    assert a == b
    assert a["dates"] == 60 and a["rows"] == 600
    assert a["ci_low"] < a["auc"] < a["ci_high"]
    assert a["ci_low"] > 0.5                       # a strong signal is distinguishable


def test_cluster_ci_is_not_shrunk_by_copying_rows_within_a_date():
    """Ten identical copies of every row add no information. A row bootstrap
    would narrow the interval by ~sqrt(10); a date bootstrap must not."""
    d, y, p, _ = _panel(dates=80, names=1, signal=0.3, seed=3)
    one = auc_with_ci(d, y, p, draws=400, seed=1)
    ten = auc_with_ci(np.repeat(d, 10), np.repeat(y, 10), np.repeat(p, 10), draws=400, seed=1)
    assert ten["auc"] == pytest.approx(one["auc"])
    width = lambda r: r["ci_high"] - r["ci_low"]
    assert width(ten) == pytest.approx(width(one), rel=0.05)


def test_auc_is_undefined_not_fabricated_with_one_class():
    r = auc_with_ci([1, 1, 2], [1, 1, 1], [0.2, 0.4, 0.6], draws=50, seed=0)
    assert r["auc"] is None and r["ci_low"] is None


def test_spearman_ci_sign_follows_the_relationship():
    d, _, p, net = _panel(signal=1.0, seed=4)
    r = spearman_with_ci(d, p, net, draws=300, seed=2)
    assert r["rho"] > 0 and r["ci_low"] > 0
    flipped = spearman_with_ci(d, p, -net, draws=300, seed=2)
    assert flipped["rho"] == pytest.approx(-r["rho"])


def test_dependence_separates_date_shocks_from_independent_rows():
    d, y, _, net = _panel(dates=120, names=12, shared=0.0, seed=5)
    syms = np.tile(np.arange(12), 120)
    independent = dependence(d, syms, y, net)
    d2, y2, _, net2 = _panel(dates=120, names=12, shared=3.0, seed=5)
    clustered = dependence(d2, syms, y2, net2)
    assert independent["dates"] == 120 and independent["rows"] == 1440
    assert abs(independent["label_icc"]) < 0.05
    assert clustered["label_icc"] > 0.3
    assert clustered["mean_pairwise_net_corr"] > independent["mean_pairwise_net_corr"] + 0.3
    # Kish design effect: effective rows = n / (1 + (m - 1) * ICC).
    m = clustered["mean_rows_per_date"]
    assert clustered["effective_rows"] == pytest.approx(1440 / (1 + (m - 1) * clustered["label_icc"]))
    assert clustered["effective_rows"] < independent["effective_rows"]


def test_fixed_buckets_are_the_phase27b_edges_and_report_dates_labels_and_excursions():
    assert FIXED_EDGES == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p = [0.1, 0.1, 0.5, 0.9, 1.0]
    rows = bucket_table(p, dates=[1, 2, 2, 3, 3], labels=[0, 1, 1, 0, 1],
                        net=[-0.01, 0.02, 0.01, -0.02, 0.03],
                        mae=[-0.02, -0.01, -0.01, -0.03, 0.0], mfe=[0.0, 0.03, 0.02, 0.0, 0.04])
    first, top = rows[0], rows[-1]
    assert first["count"] == 2 and first["unique_dates"] == 2 and first["trade_rate"] == 0.5
    assert top["count"] == 2 and top["unique_dates"] == 1          # 1.0 belongs to the top bucket
    assert top["mean_mae"] == pytest.approx(-0.015) and top["mean_mfe"] == pytest.approx(0.02)
    assert rows[1]["count"] == 0 and rows[1]["mean_net_return"] is None


def test_window_quintiles_are_cut_on_predictions_only():
    rng = np.random.default_rng(8)
    n = 200
    windows, p = np.repeat(["a", "b"], n // 2), rng.uniform(size=n)
    dates, y, net = np.arange(n) // 4, rng.integers(0, 2, n), rng.normal(size=n)
    zero = np.zeros(n)
    base = window_quintile_table(windows, p, dates, y, net, zero, zero)
    shuffled = window_quintile_table(windows, p, dates, rng.permutation(y), rng.permutation(net), zero, zero)
    assert [r["count"] for r in base] == [r["count"] for r in shuffled] == [40] * 5
    assert [r["mean_probability"] for r in base] == [r["mean_probability"] for r in shuffled]


def test_leave_one_out_drops_each_group_once():
    d, y, p, _ = _panel(signal=1.0, seed=9)
    groups = np.tile(np.array(list("abcdefghij")), 60)
    out = leave_one_out(groups, d, y, p, draws=50, seed=0)
    assert sorted(out) == list("abcdefghij")
    assert all(r["rows"] == 540 for r in out.values())


def _primary(auc, low):
    return {"auc": auc, "ci_low": low, "ci_high": auc + (auc - low)}


GOOD = dict(primary=_primary(0.56, 0.52), loo_symbol_min=0.54, loo_quarter_min=0.53,
            spearman_rho=0.04, top_minus_bottom_net=0.002)


@pytest.mark.parametrize("change, expected", [
    ({}, "PROMISING"),
    ({"primary": _primary(0.55, 0.49)}, "INCONCLUSIVE"),          # CI includes chance
    ({"loo_quarter_min": 0.50}, "WEAK"),                           # edge rests on one quarter
    ({"loo_symbol_min": 0.49}, "WEAK"),                            # edge rests on one symbol
    ({"primary": _primary(0.55, 0.49), "loo_symbol_min": 0.49}, "INCONCLUSIVE"),
    ({"spearman_rho": -0.01}, "INCONCLUSIVE"),                     # ranking diagnostics disagree
    ({"top_minus_bottom_net": -0.001}, "INCONCLUSIVE"),
    ({"primary": _primary(0.50, 0.46)}, "WEAK"),                   # at chance
    ({"primary": _primary(0.47, 0.43)}, "WEAK"),
    ({"primary": {"auc": None, "ci_low": None, "ci_high": None}}, "WEAK"),
])
def test_task_signal_rule_is_the_preregistered_truth_table(change, expected):
    label, checks = task_signal(**{**GOOD, **change})
    assert label == expected
    assert set(checks) == {"above_chance", "ci_excludes_chance", "not_one_symbol_or_quarter",
                           "ranking_agrees"}


@pytest.mark.parametrize("sealed, expected", [
    (None, "BLOCKED"),
    (_primary(0.56, 0.51), "PASS"),
    (_primary(0.53, 0.48), "INSUFFICIENT"),
    (_primary(0.45, 0.40), "INSUFFICIENT"),
])
def test_sealed_confirmation(sealed, expected):
    assert sealed_confirmation(sealed) == expected


@pytest.mark.parametrize("phase_pass, signal, sealed_auc, expected", [
    (True, "PROMISING", 0.53, "YES"),
    (True, "PROMISING", 0.49, "NO"),
    (True, "INCONCLUSIVE", 0.60, "NO"),
    (False, "PROMISING", 0.60, "NO"),
    (True, "PROMISING", None, "NO"),
])
def test_phase28_readiness(phase_pass, signal, sealed_auc, expected):
    assert phase28_ready(phase_pass, signal, sealed_auc) == expected
