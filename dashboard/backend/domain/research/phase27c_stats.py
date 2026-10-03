"""Phase 27C statistics over sealed held-out predictions.

Stocks on one decision date share that date's market, so a row is not an
independent observation. Every interval here resamples whole decision dates
(a cluster bootstrap) with a fixed seed. The decision rule at the bottom is the
preregistered one in ``phase27c-v1.json``; tests pin it as a truth table.

Nothing here feeds a number back into a model, a threshold or a split.
"""
import numpy as np
import pandas as pd

from dashboard.backend.domain.research.phase27b_metrics import rank_signal

__all__ = ["FIXED_EDGES", "auc_with_ci", "spearman_with_ci", "dependence", "bucket_table",
           "window_quintile_table", "leave_one_out", "task_signal", "sealed_confirmation",
           "phase28_ready"]

# Phase 27B's five equal-width probability buckets; the top one includes 1.0.
FIXED_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DRAWS, SEED, LEVEL = 2000, 2727, 0.95


def _date_draws(dates, draws, seed):
    """Row indices for each bootstrap draw: whole dates, sampled with replacement."""
    codes, _ = pd.factorize(pd.Series(np.asarray(dates)), sort=True)
    members = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        picks = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[i] for i in picks])


def _interval(point, samples, level):
    samples = np.asarray([s for s in samples if s is not None and np.isfinite(s)], float)
    if point is None or not samples.size:
        return None, None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def auc_with_ci(dates, labels, probabilities, draws=DRAWS, seed=SEED, level=LEVEL):
    """Pooled ROC AUC with a decision-date cluster-bootstrap percentile interval."""
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    point = rank_signal(p, y, np.zeros_like(p))["auc_trade"]
    samples = [] if point is None else [
        rank_signal(p[i], y[i], np.zeros(i.size))["auc_trade"] for i in _date_draws(dates, draws, seed)]
    low, high = _interval(point, samples, level)
    return {"auc": point, "ci_low": low, "ci_high": high, "rows": int(y.size),
            "dates": int(pd.Series(np.asarray(dates)).nunique()), "draws": draws, "seed": seed,
            "undefined_draws": int(sum(s is None for s in samples))}


def spearman_with_ci(dates, probabilities, net_returns, draws=DRAWS, seed=SEED, level=LEVEL):
    """Rank correlation of P(TRADE) with realised net return, date-clustered interval."""
    p, net = np.asarray(probabilities, float), np.asarray(net_returns, float)
    y = np.zeros(p.size, int)
    point = rank_signal(p, y, net)["spearman_probability_net"]
    samples = [] if point is None else [
        rank_signal(p[i], y[i], net[i])["spearman_probability_net"] for i in _date_draws(dates, draws, seed)]
    low, high = _interval(point, samples, level)
    return {"rho": point, "ci_low": low, "ci_high": high, "rows": int(p.size),
            "dates": int(pd.Series(np.asarray(dates)).nunique())}


def dependence(dates, symbols, labels, net_returns):
    """How far rows are from independent observations.

    ``label_icc`` is the one-way ANOVA intraclass correlation of the TRADE label
    by decision date. With m rows per date the Kish design effect is
    1 + (m - 1) * ICC and ``effective_rows`` is n divided by it.
    """
    frame = pd.DataFrame({"d": np.asarray(dates), "s": np.asarray(symbols),
                          "y": np.asarray(labels, float), "net": np.asarray(net_returns, float)})
    n, k = len(frame), frame["d"].nunique()
    sizes = frame.groupby("d").size()
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)            # unbalanced-group size
    grand = frame["y"].mean()
    means = frame.groupby("d")["y"].transform("mean")
    msb = (sizes * (frame.groupby("d")["y"].mean() - grand) ** 2).sum() / (k - 1)
    msw = ((frame["y"] - means) ** 2).sum() / (n - k)
    icc = float((msb - msw) / (msb + (m0 - 1) * msw)) if (msb + (m0 - 1) * msw) > 0 else 0.0
    m = float(sizes.mean())
    wide = frame.pivot_table(index="d", columns="s", values="net")
    corr = wide.corr(min_periods=10).to_numpy()
    pairs = corr[np.triu_indices_from(corr, 1)]
    pairs = pairs[np.isfinite(pairs)]
    return {"rows": int(n), "dates": int(k), "symbols": int(frame["s"].nunique()),
            "mean_rows_per_date": m, "rows_per_symbol": frame.groupby("s").size().to_dict(),
            "mean_pairwise_net_corr": float(pairs.mean()) if pairs.size else None,
            "label_icc": icc, "design_effect": 1 + (m - 1) * icc,
            "effective_rows": n / (1 + (m - 1) * icc)}


def _group_rows(mask, dates, labels, net, mae, mfe, probabilities):
    v = lambda a: np.asarray(a, float)[mask]
    stat = lambda a, fn: float(fn(a)) if a.size else None
    return {"count": int(mask.sum()), "unique_dates": int(pd.Series(np.asarray(dates)[mask]).nunique()),
            "mean_probability": stat(v(probabilities), np.mean),
            "trade_rate": stat(v(labels), np.mean),
            "mean_net_return": stat(v(net), np.mean), "median_net_return": stat(v(net), np.median),
            "mean_mae": stat(v(mae), np.mean), "mean_mfe": stat(v(mfe), np.mean)}


def bucket_table(probabilities, dates, labels, net, mae, mfe, edges=FIXED_EDGES):
    """Outcomes by fixed probability bucket (Phase 27B's edges)."""
    p = np.asarray(probabilities, float)
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= low) & ((p < high) if i < len(edges) - 2 else (p <= high))
        rows.append({"low": low, "high": high, **_group_rows(mask, dates, labels, net, mae, mfe, p)})
    return rows


def window_quintile_table(windows, probabilities, dates, labels, net, mae, mfe):
    """Outcomes by within-window prediction quintile.

    Quintiles are cut on the predictions alone, inside each test window, so a
    model that is confident everywhere in one quarter is compared with itself.
    Ties break by row order, which is (timestamp, symbol), so the cut is fixed.
    """
    frame = pd.DataFrame({"w": np.asarray(windows), "p": np.asarray(probabilities, float)})
    ranks = frame.groupby("w")["p"].rank(method="first")
    sizes = frame.groupby("w")["p"].transform("size")
    quintile = np.minimum((5 * (ranks - 1) / sizes).astype(int), 4).to_numpy()
    return [{"quintile": q + 1, **_group_rows(quintile == q, dates, labels, net, mae, mfe, frame["p"])}
            for q in range(5)]


def leave_one_out(groups, dates, labels, probabilities, draws=DRAWS, seed=SEED):
    """Pooled AUC with each group (a symbol or a quarter) removed in turn.

    Diagnosis of concentration only: the sealed predictions are reused, nothing
    is refit.
    """
    g, d = np.asarray(groups), np.asarray(dates)
    y, p = np.asarray(labels), np.asarray(probabilities)
    return {str(k): auc_with_ci(d[g != k], y[g != k], p[g != k], draws=draws, seed=seed)
            for k in sorted(pd.unique(g))}


def task_signal(primary, loo_symbol_min, loo_quarter_min, spearman_rho, top_minus_bottom_net):
    """The preregistered TASK_SIGNAL rule, applied to the new-evidence set.

    PROMISING needs all four checks. WEAK: at or below chance, or an interval
    that excludes chance but rests on one symbol or one quarter. Anything else
    is INCONCLUSIVE.
    """
    auc, low = primary.get("auc"), primary.get("ci_low")
    checks = {
        "above_chance": auc is not None and auc > 0.5,
        "ci_excludes_chance": low is not None and low > 0.5,
        "not_one_symbol_or_quarter": (loo_symbol_min is not None and loo_quarter_min is not None
                                      and loo_symbol_min > 0.5 and loo_quarter_min > 0.5),
        "ranking_agrees": (spearman_rho is not None and top_minus_bottom_net is not None
                           and spearman_rho > 0 and top_minus_bottom_net > 0),
    }
    if all(checks.values()):
        return "PROMISING", checks
    if not checks["above_chance"] or (checks["ci_excludes_chance"] and not checks["not_one_symbol_or_quarter"]):
        return "WEAK", checks
    return "INCONCLUSIVE", checks


def sealed_confirmation(sealed):
    """PASS only when the sealed windows alone exclude chance; BLOCKED without them."""
    if sealed is None:
        return "BLOCKED"
    low = sealed.get("ci_low")
    return "PASS" if (sealed.get("auc") or 0) > 0.5 and low is not None and low > 0.5 else "INSUFFICIENT"


def phase28_ready(phase_27c_pass, signal, sealed_auc):
    """YES only for a passing phase, a PROMISING signal, and sealed AUC above chance."""
    return "YES" if (phase_27c_pass and signal == "PROMISING"
                     and sealed_auc is not None and sealed_auc > 0.5) else "NO"
