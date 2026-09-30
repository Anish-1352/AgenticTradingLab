"""Phase 27B: a weekly, five-session, long/flat TRADE/HOLD task over several symbols.

Builds on Phase 27's harness rather than beside it: the same source filter,
the same ATL cost quote, the same separated inputs/outcomes files, and the
same purge rule. What changes is the clock. ATL has no daily environment, so a
"session" is observed at the completed 14:30-15:30 ET hourly bar, the last full
decision bar of a regular session, and fills at the 15:30 source open.

Decisions are taken every fifth full session and held for five, so one lot per
symbol is open at a time. That keeps each step to at most two orders and one
share under ATL's $3,000 capital cap and 25% position limit, and it makes the
labels non-overlapping within a symbol. There is no portfolio simulator here.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars
from dashboard.backend.domain.research.phase27 import canonical, classify, digest, regular_source
from dashboard.backend.domain.trading.execution import calculate_transaction_costs
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile

__all__ = ["load_spec", "read_sources", "session_frame", "calendar_sessions", "build_features", "decision_schedule",
           "classify", "round_trip", "make_outcome", "build_dataset", "build_splits",
           "write_dataset", "model_payload"]

_INPUT_KEYS = {"record_id", "timestamp", "feature_timestamp", "symbol", "features"}


def load_spec():
    return json.loads(Path(__file__).with_name("phase27b-v1.json").read_text())


def read_sources(manifest_path, universe, market_symbol):
    """Frozen per-symbol 5m tapes, each checked against its manifest hash.

    Never downloads. A missing symbol or a changed byte fails closed: a
    provider re-download may differ, so only the frozen bytes are evidence.
    """
    manifest_path = Path(manifest_path)
    meta = json.loads(manifest_path.read_text())
    raws = {}
    for symbol in [*universe, market_symbol]:
        entry = meta["files"].get(symbol)
        if entry is None:
            raise ValueError(f"frozen source lacks {symbol}")
        path = manifest_path.with_name(entry["file"])
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"source hash mismatch for {symbol}")
        frame = pd.read_csv(path, index_col="timestamp")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.attrs.update(meta.get("frame_attrs", {}))
        raws[symbol] = frame
    return raws, meta


def _decision_minute(spec):
    hour, minute = (int(x) for x in spec["decision_bar_end_local"].split(":"))
    return hour * 60 + minute


def session_frame(raw, spec):
    """One row per full session: the 15:30 observation and its executable open.

    ``price`` and ``volume`` use only source bars that closed by 15:30.
    ``entry_open`` is the open of the source bar that begins at 15:30, which is
    the price ATL fills at; it is future information for the decision and is
    kept here only to build outcomes.
    """
    source = regular_source(raw, spec)
    tz = spec["timezone"]
    hours = aggregate_bars(source, source_timeframe="5m", decision_timeframe="60m", timezone=tz)
    hours = hours.loc[hours.is_complete & (hours.expected_source_bars == 12)]
    end = _decision_minute(spec)
    local_hours = hours.index.tz_convert(tz)
    decision_bars = hours.loc[(local_hours.hour * 60 + local_hours.minute) == end]
    local_source = source.index.tz_convert(tz)
    minute = local_source.hour * 60 + local_source.minute
    completed_volume = source.loc[minute < end, "volume"].groupby(local_source[minute < end].date).sum()
    rows = {}
    for t, bar in decision_bars.iterrows():
        day = t.tz_convert(tz).date()
        if t not in source.index:
            continue   # no executable source bar at the boundary
        rows[day] = {"decision_timestamp": t, "price": float(bar.close),
                     "volume": float(completed_volume.get(day, np.nan)),
                     "entry_open": float(source.loc[t, "open"])}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "session"
    return frame.sort_index()


def calendar_sessions(first, last, spec):
    """Full sessions between two dates, from the declared calendar -- not the data.

    Anchoring the weekly schedule here means a data gap drops a row instead of
    shifting every later decision for that symbol.
    """
    excluded = set(spec["excluded_sessions"])
    return [d.date() for d in pd.bdate_range(first, last) if d.date().isoformat() not in excluded]


def round_trip(entry, exit_price, spec, cost_multiplier=1.0):
    """Net cash return of buying at ``entry`` and selling at ``exit_price``, via ATL's quote."""
    costs = dict(spec["costs"])
    for key in ("commission_rate", "buy_slippage_rate", "sell_slippage_rate"):
        costs[key] = costs[key] * cost_multiplier
    profile = TransactionCostProfile(**costs)
    quote = lambda side, px: calculate_transaction_costs(
        side=side, reference_price=px, shares=spec["quantity"], transaction_cost_profile=profile)
    buy, sell = quote("buy", entry), quote("sell", exit_price)
    return (buy["net_cash_impact"] + sell["net_cash_impact"]) / (entry * spec["quantity"])


def build_features(sessions, market, spec):
    short, long_ = spec["feature_windows"]["short"], spec["feature_windows"]["long"]
    p, v = sessions["price"], sessions["volume"]
    logs = np.log(p).diff()
    m = market["price"]
    feats = pd.DataFrame({
        "return_1": p.pct_change(1, fill_method=None),
        "return_5": p.pct_change(5, fill_method=None),
        "return_10": p.pct_change(10, fill_method=None),
        "return_20": p.pct_change(20, fill_method=None),
        "volatility_short": logs.rolling(short).std(ddof=0),
        "volatility_long": logs.rolling(long_).std(ddof=0),
        "volume_ratio": v / v.rolling(long_).mean(),
        "volume_change_1": np.log(v / v.shift(1)),
        "sma_gap": p / p.rolling(long_).mean() - 1,
        # The market series is differenced on its own sessions, then aligned
        # to the asset's: both observe the bar that closes at the same 15:30.
        "market_return_1": m.pct_change(1, fill_method=None).reindex(sessions.index),
        "market_return_5": m.pct_change(5, fill_method=None).reindex(sessions.index),
        "estimated_cost": [-round_trip(x, x, spec) if np.isfinite(x) else np.nan for x in p],
    }, index=sessions.index)
    return feats[spec["feature_fields"]]


def decision_schedule(sessions, spec):
    """(decision, exit) index pairs: every ``decision_every_sessions`` from the warm-up."""
    step, horizon = spec["decision_every_sessions"], spec["horizon_sessions"]
    return [(i, i + horizon) for i in range(spec["warmup_sessions"], len(sessions) - horizon, step)]


def spell_cash(chosen, spec, cost_multiplier=1.0):
    """Cash a book of chosen rows earns when back-to-back LONGs are held through.

    Rows whose exit is the next row's entry form one spell: one buy at the
    first entry, one sell at the last exit. Each row's own label still charges
    a full round trip, so this is never below the sum of the rows' net cash.
    """
    total, spell = 0.0, None
    for row in sorted(chosen, key=lambda r: r["entry_timestamp"]):
        if spell and spell["exit_timestamp"] == row["entry_timestamp"]:
            spell = {**spell, "exit_timestamp": row["exit_timestamp"], "exit_price": row["exit_price"]}
            continue
        if spell:
            total += _spell_value(spell, spec, cost_multiplier)
        spell = dict(row)
    return total + (_spell_value(spell, spec, cost_multiplier) if spell else 0.0)


def _spell_value(spell, spec, cost_multiplier):
    entry = spell["entry_price"]
    return round_trip(entry, spell["exit_price"], spec, cost_multiplier) * entry * spec["quantity"]


def make_outcome(entry, exit_price, path, spec, cost_multiplier=1.0):
    """Hidden outcome of a long position entered at ``entry``, exited at ``exit_price``.

    ``path`` holds only the source bars strictly between entry and exit, so the
    excursions start at the actual fill and never look before it.
    """
    net = round_trip(entry, exit_price, spec, cost_multiplier)
    trade, direction = classify(net, spec["trade_threshold"])
    closes = np.array([entry, *path["close"].to_list(), exit_price], dtype=float)
    return {
        "forward_return": float(exit_price / entry - 1),
        "cost_adjusted_forward_return": float(net),
        "maximum_adverse_excursion": float(min(0.0, path["low"].min() / entry - 1, exit_price / entry - 1)),
        "maximum_favorable_excursion": float(max(0.0, path["high"].max() / entry - 1, exit_price / entry - 1)),
        "future_volatility": float(np.sqrt(np.square(np.diff(np.log(closes))).sum())),
        "trade_worthy": trade, "direction": direction,
    }


def _expected_path(entry_t, exit_t, calendar, i, spec):
    """Every regular source bar the position lives through, entry bar included."""
    tz, end = spec["timezone"], _decision_minute(spec)
    stamps = []
    for k in range(i, i + spec["horizon_sessions"] + 1):
        day = calendar[k]
        bars = pd.date_range(f"{day} 09:30", f"{day} 15:55", freq="5min", tz=tz).tz_convert("UTC")
        stamps.extend(b for b in bars if entry_t <= b < exit_t)
    return pd.DatetimeIndex(stamps)


def build_dataset(raw_by_symbol, market_raw, spec):
    """Point-in-time inputs and separately held outcomes for every scheduled decision."""
    market = session_frame(market_raw, spec)
    if market.empty:
        raise ValueError("market symbol has no full sessions")
    calendar = calendar_sessions(market.index[0], market.index[-1], spec)
    schedule = decision_schedule(calendar, spec)
    inputs, outcomes = [], []
    quality = {"calendar_sessions": len(calendar), "scheduled_decisions_per_symbol": len(schedule),
               "symbols": {}}
    for symbol in sorted(raw_by_symbol):
        raw = raw_by_symbol[symbol]
        source = regular_source(raw, spec)
        sessions = session_frame(raw, spec)
        # On the calendar, so a session this symbol lacks is a hole in every
        # window that reaches it -- never a window one session longer.
        feats = build_features(sessions.reindex(calendar), market, spec)
        counts = Counter()
        for i, j in schedule:
            day, exit_day = calendar[i], calendar[j]
            if day not in sessions.index or exit_day not in sessions.index or day not in market.index:
                counts["missing_session"] += 1
                continue
            row = feats.loc[day]
            if not np.isfinite(row.to_numpy(float)).all():
                counts["warmup_or_missing_features"] += 1
                continue
            t = sessions.at[day, "decision_timestamp"]
            x_t = sessions.at[exit_day, "decision_timestamp"]
            expected = _expected_path(t, x_t, calendar, i, spec)
            if not expected.isin(source.index).all() or x_t not in source.index:
                counts["incomplete_horizon"] += 1
                continue
            entry, exit_price = float(source.at[t, "open"]), float(source.at[x_t, "open"])
            record_id = digest([symbol, t.isoformat()])[:24]
            inputs.append({"record_id": record_id, "timestamp": t.isoformat(),
                           "feature_timestamp": t.isoformat(), "symbol": symbol,
                           "features": {f: float(row[f]) for f in spec["feature_fields"]}})
            outcomes.append({"record_id": record_id, "timestamp": t.isoformat(), "symbol": symbol,
                             "session": day.isoformat(), "entry_timestamp": t.isoformat(),
                             "exit_timestamp": x_t.isoformat(), "outcome_available_at": x_t.isoformat(),
                             "entry_price": entry, "exit_price": exit_price,
                             **make_outcome(entry, exit_price, source.loc[expected], spec)})
            counts["rows"] += 1
        quality["symbols"][symbol] = {"raw_rows": len(raw), "regular_source_rows": len(source),
                                      "full_sessions_observed": len(sessions), **counts}
    quality["rows"] = len(inputs)
    return inputs, outcomes, quality


def model_payload(row, spec):
    """The only view of a row a model may see: timestamp, symbol, named features."""
    if set(row) != _INPUT_KEYS or set(row["features"]) != set(spec["feature_fields"]):
        raise ValueError("model input schema violation")
    if pd.Timestamp(row["feature_timestamp"]) > pd.Timestamp(row["timestamp"]):
        raise ValueError("future feature timestamp")
    if not np.isfinite(list(row["features"].values())).all():
        raise ValueError("non-finite model input")
    return json.loads(canonical({"timestamp": row["timestamp"], "symbol": row["symbol"],
                                 "features": row["features"]}))


def build_splits(inputs, outcomes, sessions, spec):
    """Chronological expanding folds with outcome-availability purge and session embargo."""
    exits = {r["record_id"]: pd.Timestamp(r["outcome_available_at"]) for r in outcomes}
    tz = spec["timezone"]
    folds = []
    for definition in spec["folds"]:
        bounds = [[pd.Timestamp(t, tz="UTC") for t in definition[p]] for p in ("train", "validation", "test")]
        if not (bounds[0][0] < bounds[0][1] <= bounds[1][0] < bounds[1][1] <= bounds[2][0] < bounds[2][1]):
            raise ValueError("partitions must be chronological and non-overlapping")
        fold = {"id": definition["id"], "bounds": definition, "purged_ids": [], "embargoed_ids": []}
        for part, (begin, end) in zip(("train", "validation", "test"), bounds):
            embargoed = set()
            if part != "train":
                embargoed = set([d for d in sessions if d >= begin.date()][: spec["embargo_sessions"]])
            selected = []
            for row in inputs:
                t, rid = pd.Timestamp(row["timestamp"]), row["record_id"]
                if not begin <= t < end:
                    continue
                if t.tz_convert(tz).date() in embargoed:
                    fold["embargoed_ids"].append(rid)
                elif exits[rid] >= end:
                    fold["purged_ids"].append(rid)
                else:
                    selected.append(rid)
            fold[part + "_ids"] = selected
        folds.append(fold)
    return folds


def write_dataset(directory, inputs, outcomes, quality, sessions, spec, source, code_hash, *, created_at=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    splits = build_splits(inputs, outcomes, sessions, spec)
    for filename, rows in (("inputs.jsonl", inputs), ("outcomes.jsonl", outcomes)):
        (directory / filename).write_text("".join(canonical(r) + "\n" for r in rows))
    for filename, value in (("spec.json", spec), ("splits.json", splits),
                            ("sessions.json", [d.isoformat() for d in sessions])):
        (directory / filename).write_text(canonical(value) + "\n")
    names = ("inputs.jsonl", "outcomes.jsonl", "splits.json", "spec.json", "sessions.json")
    files = {f: digest((directory / f).read_bytes()) for f in names}
    manifest = {"dataset_id": digest([source["sha256"], code_hash, files]),
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                "source": source, "code_hash": code_hash, "spec": spec, "files": files,
                "row_count": len(inputs),
                "class_balance": dict(Counter("TRADE" if o["trade_worthy"] else "HOLD" for o in outcomes)),
                "direction_balance": dict(Counter(o["direction"] for o in outcomes)),
                "quality": quality,
                "first_timestamp": inputs[0]["timestamp"] if inputs else None,
                "last_timestamp": inputs[-1]["timestamp"] if inputs else None}
    (directory / "manifest.json").write_text(canonical(manifest) + "\n")
    return manifest
