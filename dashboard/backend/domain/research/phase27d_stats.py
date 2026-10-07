"""Phase 27D statistics over sealed held-out predictions of a continuous target.

Same discipline as Phase 27C: every interval resamples whole decision dates with
a fixed seed, quintiles are cut inside each test window on predictions alone,
and the decision rules at the bottom are the preregistered ones in
``phase27d-v1.json``, pinned by truth-table tests.
"""
import numpy as np
import pandas as pd

from dashboard.backend.domain.research.phase27b_metrics import rank_signal
from dashboard.backend.domain.research.phase27c_stats import DRAWS, LEVEL, SEED, _date_draws, _interval

__all__ = ["window_quintiles", "quintile_table", "spearman", "quintile_spread_with_ci",
           "paired_difference_with_ci", "leave_one_out_spearman", "continuous_signal", "beats_binary",
           "readiness"]


def spearman(predictions, realized):
    p = np.asarray(predictions, float)
    return rank_signal(p, np.zeros(p.size, int), np.asarray(realized, float))["spearman_probability_net"]


def window_quintiles(windows, predictions):
    """0..4 per row: Phase 27C's within-window quintile cut (rank 'first', row order breaks ties)."""
    frame = pd.DataFrame({"w": np.asarray(windows), "p": np.asarray(predictions, float)})
    ranks = frame.groupby("w")["p"].rank(method="first")
    sizes = frame.groupby("w")["p"].transform("size")
    return np.minimum((5 * (ranks - 1) / sizes).astype(int), 4).to_numpy()


def quintile_table(windows, predictions, dates, net, mae, mfe):
    q = window_quintiles(windows, predictions)
    p, net = np.asarray(predictions, float), np.asarray(net, float)
    mae, mfe, d = np.asarray(mae, float), np.asarray(mfe, float), np.asarray(dates)
    out = []
    for k in range(5):
        m = q == k
        stat = lambda a, fn: float(fn(a[m])) if m.any() else None
        out.append({"quintile": k + 1, "count": int(m.sum()), "unique_dates": int(pd.Series(d[m]).nunique()),
                    "mean_prediction": stat(p, np.mean), "mean_net_return": stat(net, np.mean),
                    "median_net_return": stat(net, np.median), "positive_fraction": stat(net > 0, np.mean),
                    "mean_mae": stat(mae, np.mean), "mean_mfe": stat(mfe, np.mean)})
    return out


def _spread(q, net, idx):
    top, bottom = net[idx][q[idx] == 4], net[idx][q[idx] == 0]
    return float(top.mean() - bottom.mean()) if top.size and bottom.size else None


def quintile_spread_with_ci(windows, predictions, dates, net, draws=DRAWS, seed=SEED, level=LEVEL):
    """Q5 minus Q1 realised mean net return; membership fixed, dates resampled."""
    q, net = window_quintiles(windows, predictions), np.asarray(net, float)
    every = np.arange(net.size)
    point = _spread(q, net, every)
    low, high = _interval(point, [_spread(q, net, i) for i in _date_draws(dates, draws, seed)], level)
    return {"spread": point, "ci_low": low, "ci_high": high, "rows": int(net.size),
            "dates": int(pd.Series(np.asarray(dates)).nunique()), "draws": draws, "seed": seed}


def paired_difference_with_ci(windows, dates, predictions_a, predictions_b, net, draws=DRAWS, seed=SEED,
                              level=LEVEL):
    """Model A minus model B on the same rows, both measured on each resampled set of dates."""
    a, b, net = np.asarray(predictions_a, float), np.asarray(predictions_b, float), np.asarray(net, float)
    qa, qb = window_quintiles(windows, a), window_quintiles(windows, b)
    every = np.arange(net.size)

    def deltas(i):
        ra, rb = spearman(a[i], net[i]), spearman(b[i], net[i])
        sa, sb = _spread(qa, net, i), _spread(qb, net, i)
        return (None if ra is None or rb is None else ra - rb,
                None if sa is None or sb is None else sa - sb)

    point_rho, point_spread = deltas(every)
    samples = [deltas(i) for i in _date_draws(dates, draws, seed)]
    rho_low, rho_high = _interval(point_rho, [s[0] for s in samples], level)
    spread_low, spread_high = _interval(point_spread, [s[1] for s in samples], level)
    return {"delta_spearman": point_rho, "delta_spearman_ci_low": rho_low, "delta_spearman_ci_high": rho_high,
            "delta_spread": point_spread, "delta_spread_ci_low": spread_low, "delta_spread_ci_high": spread_high,
            "rows": int(net.size), "dates": int(pd.Series(np.asarray(dates)).nunique())}


def _spearman_with_ci(dates, predictions, net, draws, seed, level):
    p, net = np.asarray(predictions, float), np.asarray(net, float)
    point = spearman(p, net)
    low, high = _interval(point, [spearman(p[i], net[i]) for i in _date_draws(dates, draws, seed)], level)
    return {"rho": point, "ci_low": low, "ci_high": high, "rows": int(p.size),
            "dates": int(pd.Series(np.asarray(dates)).nunique())}


def leave_one_out_spearman(groups, dates, predictions, net, draws=DRAWS, seed=SEED, level=LEVEL):
    """Primary Spearman with each group (year, quarter or symbol) removed. Diagnosis only."""
    g, d = np.asarray(groups), np.asarray(dates)
    p, net = np.asarray(predictions, float), np.asarray(net, float)
    return {str(k): _spearman_with_ci(d[g != k], p[g != k], net[g != k], draws, seed, level)
            for k in sorted(pd.unique(g))}


def beats_binary(delta_rho, delta_rho_ci_low, delta_rho_ci_high, delta_spread):
    """CONTINUOUS_BEATS_BINARY: a ranking gain whose interval excludes zero, and a wider spread."""
    if delta_rho is None or delta_rho <= 0:
        return "NO"
    if delta_rho_ci_low is not None and delta_rho_ci_low > 0 and delta_spread is not None and delta_spread > 0:
        return "YES"
    return "INCONCLUSIVE"


def continuous_signal(rho, rho_ci_low, spread, spread_ci_low, loo_year_min, loo_quarter_min, loo_symbol_min,
                      beats):
    """The preregistered CONTINUOUS_SIGNAL rule.

    PROMISING needs all six checks. WEAK: no positive ranking or spread, or both
    intervals exclude zero yet one year, quarter or symbol carries the result.
    Anything else is INCONCLUSIVE.
    """
    positive = lambda v: v is not None and v > 0
    checks = {
        "rho_positive": positive(rho),
        "rho_ci_excludes_zero": positive(rho_ci_low),
        "spread_positive": positive(spread),
        "spread_ci_excludes_zero": positive(spread_ci_low),
        "not_concentrated": all(positive(v) for v in (loo_year_min, loo_quarter_min, loo_symbol_min)),
        "beats_binary": beats == "YES",
    }
    if all(checks.values()):
        return "PROMISING", checks
    significant = checks["rho_ci_excludes_zero"] and checks["spread_ci_excludes_zero"]
    if not checks["rho_positive"] or not checks["spread_positive"] or (significant and not checks["not_concentrated"]):
        return "WEAK", checks
    return "INCONCLUSIVE", checks


def readiness(phase_pass, signal, sealed_confirmed):
    """A promising target is not a confirmed one: Phase 28 needs both."""
    target = "YES" if phase_pass and signal == "PROMISING" else "NO"
    return {"MODEL_TARGET_READY": target, "SEALED_CONFIRMED": sealed_confirmed,
            "PHASE_28_READY": "YES" if target == "YES" and sealed_confirmed == "YES" else "NO"}
