"""Deterministic examples, hidden labels, chronological splits and gate metrics.

There is no portfolio simulator here. Outcome costs call ATL's existing
execution quote function; financial evaluation runs through the public API.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars, summarize_aggregation_quality
from dashboard.backend.domain.trading.execution import calculate_transaction_costs
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def load_spec():
    return json.loads(Path(__file__).with_name("phase27-v0.json").read_text())


def read_source(manifest_path):
    manifest_path = Path(manifest_path)
    meta = json.loads(manifest_path.read_text())
    path = manifest_path.with_name("AAPL-5m.csv")
    if digest(path.read_bytes()) != meta["sha256"]:
        raise ValueError("source hash mismatch")
    raw = pd.read_csv(path, index_col="timestamp")
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.attrs.update(meta["frame_attrs"])
    return raw, meta


def regular_source(raw, spec):
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("source timestamps must be unique and increasing")
    values = raw[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("invalid source values")
    if ((raw.high < raw[["open", "close"]].max(axis=1)) | (raw.low > raw[["open", "close"]].min(axis=1))).any():
        raise ValueError("invalid source OHLC")
    local = raw.index.tz_convert(spec["timezone"])
    minutes = local.hour * 60 + local.minute
    keep = (minutes >= 570) & (minutes < 960) & (local.dayofweek < 5)
    keep &= ~np.isin(local.strftime("%Y-%m-%d"), spec["excluded_sessions"])
    result = raw.loc[keep].copy()
    result.attrs.update(raw.attrs)
    return result


def completed_hours(raw, spec):
    bars = aggregate_bars(raw, source_timeframe="5m", decision_timeframe="60m", timezone=spec["timezone"])
    quality = summarize_aggregation_quality({"AAPL": bars})
    return bars.loc[bars.is_complete & (bars.expected_source_bars == 12)].copy(), quality


def classify(net, tau):
    trade = bool(net > tau)
    return trade, "LONG" if trade else "NONE"


def round_trip(entry, exit_price, spec):
    profile = TransactionCostProfile(**spec["costs"])
    quote = lambda side, price: calculate_transaction_costs(side=side, reference_price=price,
                        shares=spec["quantity"], transaction_cost_profile=profile)
    buy, sell = quote("buy", entry), quote("sell", exit_price)
    return (buy["net_cash_impact"] + sell["net_cash_impact"]) / (entry * spec["quantity"])


def make_outcome(entry, exit_price, bars, spec):
    net = round_trip(entry, exit_price, spec)
    trade, direction = classify(net, spec["trade_threshold"])
    path = np.array([entry, *bars.close.to_list(), exit_price], dtype=float)
    return {
        "forward_return": float(exit_price / entry - 1),
        "cost_adjusted_forward_return": float(net),
        "maximum_adverse_excursion": float(min(0, bars.low.min() / entry - 1, exit_price / entry - 1)),
        "maximum_favorable_excursion": float(max(0, bars.high.max() / entry - 1, exit_price / entry - 1)),
        "future_volatility": float(np.sqrt(np.square(np.diff(np.log(path))).sum())),
        "trade_worthy": trade, "direction": direction,
    }


def build_dataset(raw, spec):
    source = regular_source(raw, spec)
    hours, quality = completed_hours(source, spec)
    returns = hours.close.pct_change(fill_method=None)
    log_returns = np.log(hours.close).diff()
    features = pd.DataFrame({
        "close": hours.close,
        "return_1": returns,
        "return_3": hours.close.pct_change(3, fill_method=None),
        "volatility_6": log_returns.rolling(6).std(ddof=0),
        "volume_ratio_6": hours.volume / hours.volume.rolling(6).mean(),
        "estimated_cost": [-round_trip(c, c, spec) for c in hours.close],
    }, index=hours.index)
    inputs, outcomes = [], []
    missing_features = missing_horizon = 0
    for t, row in features.iterrows():
        if not np.isfinite(row.to_numpy(float)).all():
            missing_features += 1
            continue
        end = t + pd.Timedelta(minutes=spec["horizon_minutes"])
        expected = pd.date_range(t, end, freq="5min")
        if not expected.isin(source.index).all() or t.tz_convert(spec["timezone"]).date() != end.tz_convert(spec["timezone"]).date():
            missing_horizon += 1
            continue
        record_id = digest(["AAPL", t.isoformat()])[:24]
        inputs.append({"record_id": record_id, "timestamp": t.isoformat(), "feature_timestamp": t.isoformat(),
                       "symbol": "AAPL", "features": {name: float(row[name]) for name in spec["feature_fields"]}})
        outcome = make_outcome(float(source.loc[t, "open"]), float(source.loc[end, "open"]), source.loc[expected[:-1]], spec)
        outcomes.append({"record_id": record_id, "timestamp": t.isoformat(), "symbol": "AAPL",
                         "entry_timestamp": t.isoformat(), "outcome_available_at": end.isoformat(), **outcome})
    quality.update({"raw_rows": len(raw), "regular_source_rows": len(source), "warmup_or_missing_features": missing_features,
                    "no_complete_intraday_horizon": missing_horizon, "rows": len(inputs)})
    return inputs, outcomes, quality


def model_payload(row, spec):
    expected = {"record_id", "timestamp", "feature_timestamp", "symbol", "features"}
    if set(row) != expected or set(row["features"]) != set(spec["feature_fields"]):
        raise ValueError("model input schema violation")
    if pd.Timestamp(row["feature_timestamp"]) > pd.Timestamp(row["timestamp"]):
        raise ValueError("future feature timestamp")
    if not np.isfinite(list(row["features"].values())).all():
        raise ValueError("non-finite model input")
    return json.loads(canonical({"timestamp": row["timestamp"], "symbol": row["symbol"], "features": row["features"]}))


def build_splits(inputs, outcomes, spec):
    exits = {r["record_id"]: pd.Timestamp(r["outcome_available_at"]) for r in outcomes}
    folds = []
    for definition in spec["folds"]:
        bounds = [[pd.Timestamp(t, tz="UTC") for t in definition[part]] for part in ("train", "validation", "test")]
        if not (bounds[0][0] < bounds[0][1] <= bounds[1][0] < bounds[1][1] <= bounds[2][0] < bounds[2][1]):
            raise ValueError("partitions must be chronological and non-overlapping")
        fold = {"id": definition["id"], "bounds": definition, "purged_ids": [], "embargoed_ids": []}
        for part in ("train", "validation", "test"):
            begin, end = [pd.Timestamp(t, tz="UTC") for t in definition[part]]
            start = begin + (pd.Timedelta(hours=spec["embargo_hours"]) if part != "train" else pd.Timedelta(0))
            selected = []
            for row in inputs:
                t, rid = pd.Timestamp(row["timestamp"]), row["record_id"]
                if not begin <= t < end:
                    continue
                if t < start:
                    fold["embargoed_ids"].append(rid)
                elif exits[rid] >= end:
                    fold["purged_ids"].append(rid)
                else:
                    selected.append(rid)
            fold[part + "_ids"] = selected
        folds.append(fold)
    return folds


def write_dataset(directory, inputs, outcomes, quality, spec, source, code_hash, *, created_at=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    splits = build_splits(inputs, outcomes, spec)
    for filename, rows in (("inputs.jsonl", inputs), ("outcomes.jsonl", outcomes)):
        (directory / filename).write_text("".join(canonical(r) + "\n" for r in rows))
    for filename, value in (("spec.json", spec), ("splits.json", splits)):
        (directory / filename).write_text(canonical(value) + "\n")
    files = {f: digest((directory / f).read_bytes()) for f in ("inputs.jsonl", "outcomes.jsonl", "splits.json", "spec.json")}
    manifest = {"dataset_id": digest([source["sha256"], code_hash, files]),
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                "source": source, "code_hash": code_hash, "spec": spec, "files": files,
                "row_count": len(inputs), "class_balance": dict(Counter("TRADE" if o["trade_worthy"] else "HOLD" for o in outcomes)),
                "direction_balance": dict(Counter(o["direction"] for o in outcomes)), "missing_data": quality,
                "first_timestamp": inputs[0]["timestamp"] if inputs else None, "last_timestamp": inputs[-1]["timestamp"] if inputs else None}
    (directory / "manifest.json").write_text(canonical(manifest) + "\n")
    return manifest


class LogisticGate:
    def __init__(self, iterations=600, learning_rate=.1, l2=.01):
        self.iterations, self.learning_rate, self.l2 = iterations, learning_rate, l2

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        z = np.c_[np.ones(len(x)), (x - self.mean) / self.scale]
        self.weights = np.zeros(z.shape[1])
        # Explicit reductions avoid platform BLAS floating-point exceptions on
        # finite small-feature matrices; no optimizer or objective changes.
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(np.einsum("ij,j->i", z, self.weights, optimize=False), -40, 40)))
            penalty = self.l2 * self.weights
            penalty[0] = 0
            self.weights -= self.learning_rate * (np.einsum("ij,i->j", z, p - y, optimize=False) / len(x) + penalty)
        return self

    def predict(self, x):
        z = np.c_[np.ones(len(x)), (np.asarray(x, float) - self.mean) / self.scale]
        return 1 / (1 + np.exp(-np.clip(np.einsum("ij,j->i", z, self.weights, optimize=False), -40, 40)))

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "iterations": self.iterations, "learning_rate": self.learning_rate, "l2": self.l2}


def score_predictions(labels, probabilities, net_returns, directions):
    y, p, net = np.asarray(labels, int), np.asarray(probabilities, float), np.asarray(net_returns, float)
    if len(y) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid predictions")
    chosen = p >= .5
    tp, positives, selected = int(((y == 1) & chosen).sum()), int(y.sum()), int(chosen.sum())
    recall = tp / positives if positives else None
    specificity = float(((y == 0) & ~chosen).sum() / (y == 0).sum()) if (y == 0).any() else None
    calibration, ece = [], 0.
    for i in range(5):
        mask = (p >= i / 5) & ((p < (i + 1) / 5) if i < 4 else (p <= 1))
        confidence, fraction = (float(p[mask].mean()), float(y[mask].mean())) if mask.any() else (None, None)
        calibration.append({"low": i / 5, "high": (i + 1) / 5, "count": int(mask.sum()), "mean_probability": confidence, "trade_fraction": fraction})
        if mask.any():
            ece += float(mask.mean()) * abs(confidence - fraction)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {"n": len(y), "trade_labels": positives, "hold_labels": len(y) - positives,
            "precision_trade": tp / selected if selected else None, "recall_trade": recall,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "brier": float(np.square(p - y).mean()), "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
            "ece": ece, "calibration": calibration, "trade_coverage": float(chosen.mean()),
            "conditional_net_return": float(net[chosen].mean()) if selected else None,
            "direction_accuracy_given_trade": float((np.asarray(directions)[chosen] == "LONG").mean()) if selected else None}
