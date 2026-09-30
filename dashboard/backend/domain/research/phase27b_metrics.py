"""Phase 27B held-out metrics: classification, outcome-conditioned, buckets, costs.

Every function here runs AFTER predictions are sealed. None of them feeds a
number back into a model, a threshold, or a split.
"""
import numpy as np

from dashboard.backend.domain.research.phase27 import score_predictions
from dashboard.backend.domain.research.phase27b import classify, round_trip

__all__ = ["score_window", "probability_buckets", "cost_sensitivity", "rank_signal"]

BUCKETS = 5


def _selected_stat(values, chosen, fn):
    values = np.asarray(values, float)[chosen]
    return float(fn(values)) if values.size else None


def score_window(labels, probabilities, net_returns, mae, mfe, threshold=0.5):
    """Phase 27's classification metrics plus what the selected trades actually did."""
    p = np.asarray(probabilities, float)
    chosen = p >= threshold
    directions = ["LONG" if y else "NONE" for y in labels]
    base = score_predictions(labels, probabilities, net_returns, directions)
    net = np.asarray(net_returns, float)
    base.update({
        "mean_net_return_selected": _selected_stat(net, chosen, np.mean),
        "median_net_return_selected": _selected_stat(net, chosen, np.median),
        "fraction_selected_positive": _selected_stat(net > 0, chosen, np.mean),
        "mean_mae_selected": _selected_stat(mae, chosen, np.mean),
        "mean_mfe_selected": _selected_stat(mfe, chosen, np.mean),
        # Long-only: a chosen trade is LONG by construction, so "direction
        # accuracy given TRADE" is TRADE precision. Reported under both names
        # so nobody reads it as an independent result.
        "direction_accuracy_given_trade": base["precision_trade"],
        "mean_net_return_all": float(net.mean()) if net.size else None,
    })
    base.pop("conditional_net_return", None)
    return base


def probability_buckets(probabilities, net_returns, bins=BUCKETS):
    """Held-out outcome by predicted probability: does confidence buy anything?"""
    p, net = np.asarray(probabilities, float), np.asarray(net_returns, float)
    out = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        mask = (p >= low) & ((p < high) if i < bins - 1 else (p <= 1.0))
        values = net[mask]
        out.append({"low": low, "high": high, "count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()) if mask.any() else None,
                    "mean_net_return": float(values.mean()) if values.size else None,
                    "median_net_return": float(np.median(values)) if values.size else None,
                    "fraction_positive": float((values > 0).mean()) if values.size else None})
    return out


def cost_sensitivity(outcomes, probabilities, spec, multipliers=None, threshold=0.5):
    """Re-price the SAME selected trades at scaled costs. Sensitivity only.

    Labels are recomputed at the fixed preregistered tau; the model, its
    predictions and the canonical 1x result are untouched.
    """
    multipliers = multipliers or spec["cost_sensitivity_multipliers"]
    p = np.asarray(probabilities, float)
    chosen = p >= threshold
    rows = []
    for m in multipliers:
        net = np.array([round_trip(o["entry_price"], o["exit_price"], spec, m) for o in outcomes], float)
        labels = [classify(x, spec["trade_threshold"])[0] for x in net]
        rows.append({"multiplier": m, "trade_threshold": spec["trade_threshold"],
                     "trade_label_rate": float(np.mean(labels)) if labels else None,
                     "selected": int(chosen.sum()),
                     "mean_net_return_selected": _selected_stat(net, chosen, np.mean),
                     "median_net_return_selected": _selected_stat(net, chosen, np.median),
                     "fraction_selected_positive": _selected_stat(net > 0, chosen, np.mean),
                     "sum_net_return_selected": float(net[chosen].sum()) if chosen.any() else 0.0})
    return rows


def rank_signal(probabilities, labels, net_returns):
    """Threshold-free answer to "does higher P(TRADE) mean a better outcome?".

    ``auc_trade`` is the chance a TRADE row outranks a HOLD row (ties count
    half), None with one class present. ``spearman_probability_net`` is the rank
    correlation of probability with realised net return, None when the
    probabilities are constant -- a constant ranks nothing.
    """
    import pandas as pd
    p, y = np.asarray(probabilities, float), np.asarray(labels, int)
    pos, neg = int(y.sum()), int((1 - y).sum())
    auc = None
    if pos and neg:
        ranks = pd.Series(p).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    rho = None
    net = np.asarray(net_returns, float)
    if len(p) > 1 and np.ptp(p) > 0 and np.ptp(net) > 0:
        # Spearman = Pearson on average ranks. Computed directly: pandas'
        # method="spearman" needs scipy, which is not a project dependency.
        rp = pd.Series(p).rank(method="average").to_numpy()
        rn = pd.Series(net).rank(method="average").to_numpy()
        rho = float(np.corrcoef(rp, rn)[0, 1])
    return {"auc_trade": auc, "spearman_probability_net": rho}
