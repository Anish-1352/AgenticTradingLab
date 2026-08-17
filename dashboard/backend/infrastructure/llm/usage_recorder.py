"""Per-LLM-call usage capture.

``agent_runs`` accumulates ``llm_calls`` / ``input_tokens`` / ``output_tokens``
with ``+=`` and writes three totals once per run. The per-call distribution is
destroyed before it reaches the database, and no query brings it back — so a
claim like "this change cut cost 30%" is currently unfalsifiable. This module
records each call as it happens so that claim becomes checkable.

DESIGN: A CONTEXTVAR, NOT A THREADED PARAMETER
-----------------------------------------------
The three call sites live in the LLM layer and have no idea which run they are
serving; the ``run_id`` is known only to the engine, several frames up. Passing
it down would mean changing the signature of every function in between, on a
production path, for observability. Instead the engine installs a recorder for
the duration of a run and the call sites append to whatever is active.

A ``ContextVar`` rather than a module global: it is correct under asyncio and
threads, and — importantly — a recorder installed by one run cannot leak into a
concurrent one. When nothing is installed, ``record_call`` is a no-op, so the
call sites behave exactly as before outside a backtest.

ROWS ARE BUFFERED, NOT WRITTEN PER CALL
----------------------------------------
Persistence happens once at end of run, matching how ``insert_trades`` and
``insert_decisions`` already work. A database round trip per LLM call would add
write amplification to a hot path for no benefit — the call it accompanies
already took seconds.

FAILURE IS NEVER THE RUN'S PROBLEM
-----------------------------------
Recording is observability. If it breaks, the backtest must still finish: every
entry point here swallows its own exceptions and counts them. Losing a usage
row is an acceptable outcome; losing a user's backtest to a logging bug is not.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

__all__ = [
    "LLMCallUsage",
    "UsageRecorder",
    "recording",
    "active_recorder",
    "record_call",
    "usage_logging_enabled",
    "extract_cached_input_tokens",
]

# Kill switch. Recording is additive — it changes no result, no prompt, and no
# model — so it is on by default, because deliverables that depend on it cannot
# be produced from a system that is not recording. Set to a falsey value to
# disable entirely.
_ENV_FLAG = "ATL_LLM_CALL_USAGE"
_FALSEY = {"0", "false", "no", "off", "disabled", ""}


def usage_logging_enabled() -> bool:
    """True unless explicitly disabled via ``ATL_LLM_CALL_USAGE``."""
    raw = os.getenv(_ENV_FLAG)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


@dataclass
class LLMCallUsage:
    """One provider call. ``call_index`` is per-run and monotonic."""

    call_index: int
    step_label: Optional[str]
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: Optional[int]
    latency_ms: Optional[float]
    timestamp: str
    error: Optional[str]

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class UsageRecorder:
    """Buffers per-call usage for one run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.calls: List[LLMCallUsage] = []
        # Recording failures are counted rather than raised, so a run can
        # report "usage logging degraded" instead of dying.
        self.record_errors = 0

    def record(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        step_label: Optional[str] = None,
        model: Optional[str] = None,
        cached_input_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Optional[LLMCallUsage]:
        try:
            entry = LLMCallUsage(
                call_index=len(self.calls),
                step_label=step_label,
                model=model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cached_input_tokens=(
                    None if cached_input_tokens is None
                    else int(cached_input_tokens)
                ),
                latency_ms=(None if latency_ms is None else float(latency_ms)),
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=(str(error)[:500] if error else None),
            )
        except Exception:  # pragma: no cover - defensive
            self.record_errors += 1
            return None
        self.calls.append(entry)
        return entry

    def rows(self) -> List[Dict[str, Any]]:
        return [c.to_row() for c in self.calls]

    def totals(self) -> Dict[str, int]:
        """Aggregate over recorded calls.

        Compared against ``agent_runs``' independently accumulated totals by
        ``test_llm_call_usage``: if these two ever disagree, one of them is
        wrong and the per-call table cannot be trusted for attribution.
        """
        return {
            "llm_calls": len(self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cached_input_tokens": sum(
                c.cached_input_tokens or 0 for c in self.calls),
        }

    def __len__(self) -> int:
        return len(self.calls)


_active: ContextVar[Optional[UsageRecorder]] = ContextVar(
    "atl_llm_usage_recorder", default=None
)


def active_recorder() -> Optional[UsageRecorder]:
    return _active.get()


@contextmanager
def recording(recorder: Optional[UsageRecorder]) -> Iterator[Optional[UsageRecorder]]:
    """Install ``recorder`` for the duration of the block.

    Passing ``None`` explicitly suppresses recording for the block, which is
    what the kill switch does — callers do not need to branch.
    """
    token = _active.set(recorder)
    try:
        yield recorder
    finally:
        _active.reset(token)


def record_call(
    *,
    input_tokens: int,
    output_tokens: int,
    step_label: Optional[str] = None,
    model: Optional[str] = None,
    cached_input_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Record one call against the active recorder, or do nothing.

    A no-op when no recorder is installed, which is the case for every code
    path outside a backtest — chat, strategy synthesis, and the API all behave
    exactly as before.
    """
    recorder = _active.get()
    if recorder is None:
        return
    try:
        recorder.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            step_label=step_label,
            model=model,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency_ms,
            error=error,
        )
    except Exception:  # pragma: no cover - defensive
        try:
            recorder.record_errors += 1
        except Exception:
            pass


def extract_cached_input_tokens(response: Any) -> Optional[int]:
    """Provider-reported cached prompt tokens, or ``None`` if not reported.

    Deliberately returns ``None`` rather than ``0`` when the provider says
    nothing: "no cache field in the response" and "the cache was offered and
    returned nothing" are different facts, and collapsing them would make a
    provider that does not report caching look identical to one that reports a
    miss. Deliverable 3 needs to tell those apart.

    Both spellings are checked because ATL speaks to several gateways behind an
    Anthropic-shaped client: Anthropic uses ``cache_read_input_tokens``,
    OpenAI-compatible gateways nest ``cached_tokens`` under
    ``prompt_tokens_details``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    direct = getattr(usage, "cache_read_input_tokens", None)
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens")
        if cached is not None:
            try:
                return int(cached)
            except (TypeError, ValueError):
                return None
    if isinstance(usage, dict):
        for key in ("cache_read_input_tokens", "cached_input_tokens"):
            if key in usage:
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    return None
    return None


@contextmanager
def timed_call() -> Iterator[Dict[str, Optional[float]]]:
    """Measure wall time around a provider call.

    Yields a dict that gains ``latency_ms`` on exit, so the call site can pass
    it straight to ``record_call`` without arithmetic.
    """
    holder: Dict[str, Optional[float]] = {"latency_ms": None}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["latency_ms"] = (time.perf_counter() - start) * 1000.0
