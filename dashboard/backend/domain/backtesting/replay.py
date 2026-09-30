"""Trusted process-local frozen datasets; no file paths accepted over HTTP.

An offline experiment registers a tape before starting runs. Ordinary server
processes have an empty registry and reject replay requests rather than falling
back to live market data. Frames follow MarketDataset's read-only convention.
"""
from dataclasses import dataclass
from threading import RLock
from typing import Any

from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile


@dataclass(frozen=True)
class FrozenReplay:
    dataset: Any
    costs: TransactionCostProfile
    metadata: dict
    start_date: str
    end_date: str
    symbols: tuple


_replays = {}
_lock = RLock()


def register_replay(identifier, replay):
    if not isinstance(replay, FrozenReplay) or not replay.dataset.timestamps:
        raise ValueError("nonempty frozen replay required")
    with _lock:
        if identifier in _replays:
            raise ValueError("replay identifier already registered")
        _replays[identifier] = replay


def resolve_replay(identifier, start, end, symbols):
    with _lock:
        replay = _replays.get(identifier)
    if replay is None or replay.start_date != start or replay.end_date != end or set(replay.symbols) != set(symbols):
        raise ValueError("unregistered replay or mismatched window/universe")
    return replay


def clear_replays():
    with _lock:
        _replays.clear()
