"""Adapt a frozen Phase 27 tape to ATL's existing MarketDataset contract."""
import pandas as pd
from dashboard.backend.domain.backtesting.market_data_store import MarketDataset
from dashboard.backend.domain.backtesting.bar_aggregation import ExecutionFill
from dashboard.backend.domain.backtesting.replay import FrozenReplay
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
from .phase27 import regular_source, completed_hours, digest


def make_replay(raw, inputs, spec, start, end):
    if not inputs:
        raise ValueError("empty replay")
    source = regular_source(raw, spec)
    hours, quality = completed_hours(source, spec)
    entries = [pd.Timestamp(row["timestamp"]) for row in inputs]
    steps = sorted(set(entries) | {t + pd.Timedelta(minutes=spec["horizon_minutes"]) for t in entries})
    if not pd.DatetimeIndex(steps).isin(source.index).all() or not pd.DatetimeIndex(steps).isin(hours.index).all():
        raise ValueError("missing decision or execution bar")
    if any(not (pd.Timestamp(start, tz="UTC") <= t < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)) for t in steps):
        raise ValueError("replay outside requested window")
    source = source.loc[(source.index >= steps[0].normalize()) & (source.index <= steps[-1])].copy()
    decisions = hours.loc[steps].copy()
    # The final row is execution-only. Once the terminal position is flattened,
    # the experiment ends at that open, not the source bar's later close.
    source_times = list(source.index[source.index < steps[-1]])
    source_cache = {"AAPL": source.close.to_dict()}
    dataset = MarketDataset(("AAPL", start, (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()),
        {"AAPL": decisions}, steps, {"AAPL": decisions.close.to_dict()},
        source_data={"AAPL": source}, source_timestamps=source_times, source_price_cache=source_cache,
        execution_fills=[ExecutionFill(t, "open", t) for t in steps], source_timeframe="5m", decision_timeframe="60m", data_quality=quality)
    metadata = {"type": "phase27-frozen", "spec_hash": digest(spec), "entry_ids": [r["record_id"] for r in inputs],
                "fill_policy": "next_source_open", "terminal_policy": "flatten at fixed horizon; no overnight positions"}
    return FrozenReplay(dataset, TransactionCostProfile(**spec["costs"]), metadata, start, end, ("AAPL",))
