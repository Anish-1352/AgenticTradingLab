"""Backtest result cache — keyed on everything that changes the result.

Users iterate on a prompt and re-run the same window repeatedly; each re-run
currently re-pays every LLM call. This maps a fully-specified configuration to
the ``run_id`` of a previous run with that exact configuration.

AN INDEX, NOT A COPY
--------------------
A hit returns a prior ``run_id``; the equity curve, trades and metadata are
read from where they already live. Nothing is duplicated, so a cached result
cannot drift from the run it names, and invalidation is deleting one row.

THE KEY IS DELIBERATELY TOO STRICT
-----------------------------------
A missed hit costs one backtest. A wrong hit shows a user results that do not
match the configuration in front of them, which costs trust and is very hard to
notice. So every input that could move a number is in the key, including ones
that rarely change (``llm_max_output_tokens`` truncates responses;
``runtime_config`` selects a different decision engine entirely), and anything
unrecognised is included verbatim rather than dropped.

``CACHE_KEY_VERSION`` is part of every key. Bump it when a code change alters
what a run produces from unchanged inputs — that invalidates every existing
entry at once, without a migration.

WHAT A HIT CHANGES SEMANTICALLY
--------------------------------
``temperature=0`` does **not** make a hosted API deterministic: batching and
hardware vary, so a genuine re-run would return a different sample. A cache hit
therefore returns *the first run's* result rather than a fresh draw. That is a
real change in meaning, not an implementation detail, so a hit is always
labelled with the run and date it came from (``cached_from``) and callers are
expected to surface it and to offer an explicit bypass.

DEFAULT OFF
-----------
``ATL_BACKTEST_CACHE`` defaults to disabled: with it unset, behaviour is
byte-identical to before this module existed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CACHE_KEY_VERSION",
    "CacheKeyInputs",
    "compute_cache_key",
    "cache_enabled",
    "cache_ttl_hours",
    "cache_scope",
    "CacheLookup",
]

# Bump to invalidate every existing entry when a code change alters what a run
# produces from unchanged inputs.
CACHE_KEY_VERSION = 1

_ENV_ENABLED = "ATL_BACKTEST_CACHE"
_ENV_TTL_HOURS = "ATL_BACKTEST_CACHE_TTL_HOURS"
_ENV_SCOPE = "ATL_BACKTEST_CACHE_SCOPE"

_TRUTHY = {"1", "true", "yes", "on", "enabled"}

# Per-user is the safe default. Global sharing would let two users with
# byte-identical configurations share one result — a larger saving, but a
# product decision about whether one user's compute may serve another, not an
# engineering one. See the PR notes.
_SCOPES = ("user", "global")
DEFAULT_SCOPE = "user"

# 0 disables expiry. A TTL exists so a cached result cannot outlive a
# market-data correction: the inputs would be unchanged, so the key would still
# match, but the underlying bars would not be the ones the run used.
DEFAULT_TTL_HOURS = 24 * 7


def cache_enabled() -> bool:
    """False unless explicitly enabled — unset means current behaviour."""
    raw = os.getenv(_ENV_ENABLED)
    if raw is None:
        return False
    return str(raw).strip().lower() in _TRUTHY


def cache_scope() -> str:
    raw = (os.getenv(_ENV_SCOPE) or DEFAULT_SCOPE).strip().lower()
    return raw if raw in _SCOPES else DEFAULT_SCOPE


def cache_ttl_hours() -> float:
    raw = os.getenv(_ENV_TTL_HOURS)
    if raw is None or not str(raw).strip():
        return float(DEFAULT_TTL_HOURS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_TTL_HOURS)
    return value if value >= 0 else float(DEFAULT_TTL_HOURS)


def _canonical(value: Any) -> Any:
    """Order-independent, type-stable form for hashing.

    Dict key order and float formatting must not change the hash, or the same
    configuration would produce different keys on different runs and the cache
    would simply never hit.
    """
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # repr of a float is stable within a Python version; normalising
        # integral floats keeps 1000 and 1000.0 from diverging.
        return int(value) if value.is_integer() else round(value, 10)
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


@dataclass(frozen=True)
class CacheKeyInputs:
    """Everything that changes a backtest's result.

    Anything omitted here is an assertion that it cannot move a number. Adding
    a field is safe (it only splits keys); removing one risks a wrong hit.
    """

    session_id: Optional[str]
    model: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    symbols: Sequence[str]
    bar_interval: Optional[str]
    initial_capital: Optional[float]
    data_source: Optional[str]
    mode: Optional[str]
    strategy_prompt: Optional[str]
    pipeline: Optional[Any]
    runtime_type: Optional[str]
    runtime_config: Optional[Dict[str, Any]]
    max_output_tokens: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

    def payload(self, scope: Optional[str] = None) -> Dict[str, Any]:
        scope = scope or cache_scope()
        body: Dict[str, Any] = {
            "v": CACHE_KEY_VERSION,
            "scope": scope,
            # Symbols are sorted: the same universe in a different order is the
            # same universe. Everything else is order-sensitive by nature.
            "symbols": sorted(str(s) for s in (self.symbols or [])),
            "model": self.model,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "bar_interval": self.bar_interval,
            "initial_capital": self.initial_capital,
            "data_source": self.data_source,
            "mode": self.mode,
            # Full prompt text, not a summary: a one-character edit must miss.
            "strategy_prompt": self.strategy_prompt,
            "pipeline": self.pipeline,
            "runtime_type": self.runtime_type,
            "runtime_config": self.runtime_config,
            "max_output_tokens": self.max_output_tokens,
            "extra": self.extra,
        }
        if scope == "user":
            body["session_id"] = self.session_id
        return _canonical(body)


def compute_cache_key(inputs: CacheKeyInputs, scope: Optional[str] = None) -> str:
    """SHA-256 over the canonical payload."""
    blob = json.dumps(inputs.payload(scope), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheLookup:
    """Result of a cache probe.

    ``cached_from`` is what the UI shows. A hit is never silent: the caller is
    expected to render "cached result from <date>" and offer a bypass, because
    a hit returns an earlier sample rather than a fresh one.
    """

    hit: bool
    run_id: Optional[str] = None
    cached_at: Optional[str] = None
    reason: Optional[str] = None

    @property
    def cached_from(self) -> Optional[str]:
        if not self.hit:
            return None
        return f"cached result from {self.cached_at}" if self.cached_at else \
            "cached result"


def is_expired(created_at: Optional[str], ttl_hours: Optional[float] = None,
               now: Optional[datetime] = None) -> bool:
    """True when an entry has outlived the TTL. ttl_hours == 0 means never."""
    ttl = cache_ttl_hours() if ttl_hours is None else ttl_hours
    if not ttl:
        return False
    if not created_at:
        # Unknown age is treated as expired: re-running costs money, but
        # serving a result of unknown vintage costs trust.
        return True
    try:
        stamp = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current - stamp > timedelta(hours=ttl)
