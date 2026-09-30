"""Adapt one symbol's frozen tape to ATL's MarketDataset for a multi-day schedule.

Generalizes Phase 27's ``make_replay`` from a one-hour, AAPL-only horizon to a
named symbol whose exits fall sessions later. Steps are exactly the entry and
exit boundaries; everything between them is valued, not decided, by ATL.
"""
import pandas as pd

from dashboard.backend.domain.backtesting.bar_aggregation import (
    ExecutionFill, aggregate_bars, summarize_aggregation_quality)
from dashboard.backend.domain.backtesting.market_data_store import MarketDataset
from dashboard.backend.domain.backtesting.replay import FrozenReplay
from dashboard.backend.domain.research.phase27 import digest, regular_source
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile

__all__ = ["make_symbol_replay"]


def make_symbol_replay(raw, symbol, entries, exits, spec, start, end):
    """Frozen replay whose steps are ``entries`` and ``exits`` (UTC timestamps)."""
    if not entries:
        raise ValueError("empty replay")
    source = regular_source(raw, spec)
    hours = aggregate_bars(source, source_timeframe="5m", decision_timeframe="60m", timezone=spec["timezone"])
    quality = summarize_aggregation_quality({symbol: hours})
    hours = hours.loc[hours.is_complete & (hours.expected_source_bars == 12)]
    steps = sorted(set(entries) | set(exits))
    index = pd.DatetimeIndex(steps)
    if not index.isin(source.index).all() or not index.isin(hours.index).all():
        raise ValueError("missing decision or execution bar")
    window_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    if any(not (pd.Timestamp(start, tz="UTC") <= t < window_end) for t in steps):
        raise ValueError("replay outside requested window")
    source = source.loc[(source.index >= steps[0].normalize()) & (source.index <= steps[-1])].copy()
    decisions = hours.loc[steps].copy()
    # The final step is execution-only: once the last lot is flattened at that
    # open, the run ends there rather than at the source bar's later close.
    source_times = list(source.index[source.index < steps[-1]])
    dataset = MarketDataset(
        (symbol, start, (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()),
        {symbol: decisions}, steps, {symbol: decisions.close.to_dict()},
        source_data={symbol: source}, source_timestamps=source_times,
        source_price_cache={symbol: source.close.to_dict()},
        execution_fills=[ExecutionFill(t, "open", t) for t in steps],
        source_timeframe="5m", decision_timeframe="60m", data_quality=quality)
    metadata = {"type": "phase27b-frozen", "spec_hash": digest(spec), "symbol": symbol,
                "entries": [t.isoformat() for t in entries], "fill_policy": "next_source_open",
                "terminal_policy": "each lot is sold at its fixed five-session exit"}
    return FrozenReplay(dataset, TransactionCostProfile(**spec["costs"]), metadata, start, end, (symbol,))
