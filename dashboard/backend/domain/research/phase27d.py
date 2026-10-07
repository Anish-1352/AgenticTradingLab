"""Phase 27D: the same task with a continuous target.

Phase 27C's LogisticGate learned TRADE/HOLD, a threshold on the five-session
net return. Here the model learns that net return itself -- the hidden
``cost_adjusted_forward_return`` already stored for every row, unchanged.
Everything else (features, standardisation, regularisation, folds, execution)
is Phase 27C's, so a difference in ranking is a difference in target.
"""
import math

import numpy as np

from dashboard.backend.domain.research.phase27b import model_payload

__all__ = ["TARGET", "LinearOpportunityRegressor", "fit_linear_regressor", "fit_train_mean",
           "top_fraction_selection", "validate_regression_prediction", "regression_metrics"]

TARGET = "cost_adjusted_forward_return"


class LinearOpportunityRegressor:
    """Ridge regression on training-standardised features.

    Minimises mean squared error plus ``l2`` times the squared weights,
    intercept unpenalised -- the LogisticGate's standardisation and penalty
    with squared loss in place of log loss. Closed form, so deterministic.
    """

    def __init__(self, l2=0.01):
        self.l2 = l2

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        z = np.c_[np.ones(len(x)), (x - self.mean) / self.scale]
        penalty = self.l2 * np.eye(z.shape[1])
        penalty[0, 0] = 0
        # Explicit reductions, as in LogisticGate: no platform BLAS paths.
        gram = np.einsum("ij,ik->jk", z, z, optimize=False) / len(x)
        rhs = np.einsum("ij,i->j", z, y, optimize=False) / len(x)
        self.weights = np.linalg.solve(gram + penalty, rhs)
        return self

    def predict(self, x):
        z = np.c_[np.ones(len(x)), (np.asarray(x, float) - self.mean) / self.scale]
        return np.einsum("ij,j->i", z, self.weights, optimize=False)

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "l2": self.l2, "target": TARGET}


def fit_linear_regressor(inputs, outcomes, train_ids, spec):
    """One pooled regressor across symbols, fitted on training rows only."""
    rows = {r["record_id"]: r for r in inputs}
    train = set(train_ids)
    target = {o["record_id"]: o[TARGET] for o in outcomes if o["record_id"] in train}
    x = [[model_payload(rows[rid], spec)["features"][f] for f in spec["feature_fields"]] for rid in train_ids]
    return LinearOpportunityRegressor(spec["regressor"]["l2"]).fit(x, [target[rid] for rid in train_ids])


def fit_train_mean(outcomes, train_ids):
    """The trivial baseline: the mean training net return, predicted for every row."""
    train = set(train_ids)
    return float(np.mean([o[TARGET] for o in outcomes if o["record_id"] in train]))


def top_fraction_selection(rows, predictions, fraction):
    """record_ids in the top ``fraction`` of predictions at each decision timestamp.

    Ranked within one timestamp only -- the names a decision at that moment can
    see -- never across a window, which would rank against later dates. Takes
    ceil(fraction * n) per timestamp; ties break by symbol.
    """
    by_time = {}
    for r in rows:
        by_time.setdefault(r["timestamp"], []).append(r)
    chosen = set()
    for group in by_time.values():
        k = math.ceil(fraction * len(group))
        ranked = sorted(group, key=lambda r: (-predictions[r["record_id"]], r["symbol"]))
        chosen.update(r["record_id"] for r in ranked[:k])
    return chosen


def validate_regression_prediction(value):
    """The continuous model's contract: a finite predicted net return and LONG/NONE."""
    if not isinstance(value, dict) or set(value) != {"predicted_net_return", "direction"}:
        raise ValueError("invalid regression prediction schema")
    p = value["predicted_net_return"]
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p):
        raise ValueError("invalid predicted net return")
    if value["direction"] not in ("LONG", "NONE"):
        raise ValueError("invalid long-only direction")
    return {"predicted_net_return": float(p), "direction": value["direction"]}


def regression_metrics(predictions, realized, reference=None):
    """Point errors of a return forecast. Secondary: ranking, not fit, is the question.

    ``r2`` is against the realised mean of these rows; ``oos_r2_vs_reference``
    against a reference forecast such as the training-mean predictor. Sign
    accuracy counts a prediction <= 0 as non-positive, with no tuned threshold.
    """
    p, y = np.asarray(predictions, float), np.asarray(realized, float)
    err = p - y
    sse = float((err ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    out = {"rows": int(y.size), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": 1 - sse / tss if tss > 0 else None, "oos_r2_vs_reference": None,
           "sign_accuracy": float(((p > 0) == (y > 0)).mean()),
           "prediction_mean": float(p.mean()), "prediction_std": float(p.std()),
           "realized_mean": float(y.mean()), "realized_std": float(y.std())}
    if reference is not None:
        ref = float(((y - np.asarray(reference, float)) ** 2).sum())
        out["oos_r2_vs_reference"] = 1 - sse / ref if ref > 0 else None
    return out
