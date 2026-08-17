"""Leaderboard refresh governance — decide which models to redeploy, and when.

``refresh_daily_leaderboard.py --models`` redeploys every ``llm_agent`` entry
daily, whether or not anything changed and whether or not any user is looking.
At measured per-call costs the expensive models dominate that spend and no user
is attached to it.

This module answers one question — *should this entry be refreshed on this
run?* — and answers it the same way as today unless configuration says
otherwise.

DEFAULTS REPRODUCE CURRENT BEHAVIOUR, EXACTLY
----------------------------------------------
With no configuration present, ``should_refresh`` returns True for every entry
and ``skip_reason`` is None. The saving comes from an operator setting a cadence
or enabling change-detection, not from this code quietly deciding to skip work.
That matters because a leaderboard that silently stops updating looks identical
to one that is updating with unchanged numbers.

THREE INDEPENDENT CONTROLS
---------------------------
* **change detection** (``skip_unchanged``) — if the evaluation window and every
  agent config are byte-identical to the last completed refresh, the result
  would be the same run over the same data, so there is nothing to buy. This is
  the only control that is safe to enable blindly.
* **per-model cadence** — expensive models weekly, cheap ones daily. A pure
  cost/freshness trade, made per entry.
* **window length** — how many days the evaluation covers. Longer windows cost
  proportionally more, since the run replays every bar.

Cadence and window are deliberately *not* inferred from price. Which models are
worth refreshing often is a judgement about what the leaderboard is for, and
belongs in config where a human can see it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "GovernanceConfig",
    "RefreshDecision",
    "load_governance",
    "config_fingerprint",
    "should_refresh",
    "plan_refresh",
]

_ENV_PREFIX = "ATL_LEADERBOARD_"
_TRUTHY = {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class GovernanceConfig:
    """Refresh policy. Every field defaults to today's behaviour."""

    # False → refresh regardless of whether anything changed (current behaviour).
    skip_unchanged: bool = False
    # entry_id / model → minimum days between refreshes. Absent → every run.
    cadence_days: Optional[Dict[str, int]] = None
    # None → whatever the leaderboard config already computes.
    window_days: Optional[int] = None

    @property
    def active(self) -> bool:
        """True when any control is engaged — used to log why work was skipped."""
        return bool(self.skip_unchanged or self.cadence_days or self.window_days)


def _env(name: str) -> Optional[str]:
    raw = os.getenv(_ENV_PREFIX + name)
    return raw if raw is not None and str(raw).strip() != "" else None


def load_governance(config: Optional[Dict[str, Any]] = None) -> GovernanceConfig:
    """Read policy from the leaderboard config, with env overrides.

    Config file first (visible in review), env second (operable without a
    deploy). Absent everywhere means current behaviour.
    """
    block = ((config or {}).get("governance") or {}) if config else {}

    skip = block.get("skip_unchanged", False)
    raw_skip = _env("SKIP_UNCHANGED")
    if raw_skip is not None:
        skip = str(raw_skip).strip().lower() in _TRUTHY

    cadence = block.get("cadence_days") or None
    raw_cadence = _env("CADENCE_DAYS")
    if raw_cadence is not None:
        try:
            parsed = json.loads(raw_cadence)
            if isinstance(parsed, dict):
                cadence = {str(k): int(v) for k, v in parsed.items()}
        except (ValueError, TypeError):
            pass

    window = block.get("window_days")
    raw_window = _env("WINDOW_DAYS")
    if raw_window is not None:
        try:
            window = int(raw_window)
        except (TypeError, ValueError):
            pass
    if window is not None:
        try:
            window = int(window)
            if window <= 0:
                window = None
        except (TypeError, ValueError):
            window = None

    return GovernanceConfig(
        skip_unchanged=bool(skip),
        cadence_days={str(k): int(v) for k, v in (cadence or {}).items()} or None,
        window_days=window,
    )


def config_fingerprint(entry: Dict[str, Any], window: Dict[str, Any]) -> str:
    """Hash of everything that would change this entry's result.

    Covers the evaluation window and the entry's own configuration. If this is
    unchanged since the last completed refresh, re-running buys nothing: the
    same agent over the same dates against the same bars.

    Whole-entry rather than a field list — a new config key that affects
    behaviour must not slip past because nobody remembered to add it here.
    """
    payload = {
        "entry": entry,
        "window": {
            "start_date": window.get("start_date"),
            "end_date": window.get("end_date"),
            "initial_capital": window.get("initial_capital"),
            "session_id": window.get("session_id"),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RefreshDecision:
    entry_id: str
    refresh: bool
    reason: str
    fingerprint: Optional[str] = None


def _days_since(stamp: Optional[str], now: Optional[datetime] = None
                ) -> Optional[float]:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return (current - parsed).total_seconds() / 86400.0


def should_refresh(
    entry: Dict[str, Any],
    window: Dict[str, Any],
    governance: GovernanceConfig,
    *,
    last_fingerprint: Optional[str] = None,
    last_refreshed_at: Optional[str] = None,
    now: Optional[datetime] = None,
) -> RefreshDecision:
    """Decide for one entry. Defaults to True with no policy configured."""
    entry_id = str(entry.get("id") or entry.get("model") or "unknown")
    fingerprint = config_fingerprint(entry, window)

    # Cadence is checked before change-detection: an operator who asked for
    # weekly wants weekly even if something changed, otherwise a trivial config
    # edit silently restores daily spend.
    cadence = (governance.cadence_days or {})
    interval = cadence.get(entry_id)
    if interval is None:
        interval = cadence.get(str(entry.get("model") or ""))
    if interval:
        age = _days_since(last_refreshed_at, now)
        if age is not None and age < float(interval):
            return RefreshDecision(
                entry_id, False,
                f"cadence: refreshed {age:.1f}d ago, interval {interval}d",
                fingerprint)

    if governance.skip_unchanged and last_fingerprint:
        if last_fingerprint == fingerprint:
            return RefreshDecision(
                entry_id, False,
                "unchanged: window and agent config identical to last refresh",
                fingerprint)

    return RefreshDecision(entry_id, True, "refresh", fingerprint)


def plan_refresh(
    entries: Sequence[Dict[str, Any]],
    window: Dict[str, Any],
    governance: GovernanceConfig,
    *,
    history: Optional[Dict[str, Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> List[RefreshDecision]:
    """Decide for every entry. ``history`` maps entry_id → last refresh record."""
    history = history or {}
    out: List[RefreshDecision] = []
    for entry in entries:
        entry_id = str(entry.get("id") or entry.get("model") or "unknown")
        prev = history.get(entry_id) or {}
        out.append(should_refresh(
            entry, window, governance,
            last_fingerprint=prev.get("fingerprint"),
            last_refreshed_at=prev.get("refreshed_at"),
            now=now,
        ))
    return out


def resolve_window(config: Dict[str, Any], governance: GovernanceConfig
                   ) -> Dict[str, Any]:
    """Apply a configured window length, or leave the config untouched.

    Returns a copy: callers pass the result on to the existing resolver, so a
    None ``window_days`` produces byte-identical inputs to today.
    """
    if not governance.window_days:
        return dict(config)
    end = config.get("end_date")
    try:
        end_date = datetime.fromisoformat(str(end))
    except (TypeError, ValueError):
        return dict(config)
    start_date = end_date - timedelta(days=int(governance.window_days))
    updated = dict(config)
    updated["start_date"] = start_date.date().isoformat()
    return updated
