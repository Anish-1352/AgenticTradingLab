#!/usr/bin/env python3
"""Measure retry amplification at the source, not by counting calls.

    python benchmarks/analysis/retry_amplification.py --dry-run
    DATABASE_PATH=$PWD/local_atl.db python benchmarks/analysis/retry_amplification.py \
        --model nvidia/nemotron-3-nano-30b-a3b --start 2026-04-01 --end 2026-04-04

WHY THIS EXISTS
----------------
Phase 20 measured 2.71 LLM requests per bar in one run and 1.00 in another,
and attributed the excess to retries by reasoning about call sites — the log
had been filtered before capture, so the mechanism was DERIVED. This measures
it: the retry TRIGGER is instrumented directly, so an attempt is known to be a
retry because the thing that causes retries fired, not because a count came out
high.

THE ONLY RETRY TRIGGER IN THIS LOOP IS AN EMPTY RESPONSE
----------------------------------------------------------
``portfolio_manager.py:366-410`` loops ``range(no_text_retries + 1)`` = 5
attempts. The loop advances on exactly one condition: ``_extract_response_text``
raising ``AttributeError`` whose message contains "No text content". Anything
else re-raises and leaves the loop. So unparseable output, timeouts and
provider errors do NOT cause retries here — they abort. A tracer that offered
those as categories would be offering categories the code cannot produce.

The fifth attempt is a "rescue" that disables reasoning for one call.

WHAT IS COUNTED WHERE
----------------------
``self.llm_calls += 1`` sit INSIDE the loop, so an ATL retry is billed and
counted as a call. That is deliberate — ``credits.py`` documents ``llm_calls``
as a billing counter precisely because a retry costs real money — and it is
also why retries are invisible: nothing downstream can separate them.

THERE ARE TWO RETRY LAYERS, AND ``llm_calls`` ONLY SEES ONE
------------------------------------------------------------
Underneath ATL's loop, the Anthropic SDK runs its own. ``SyncAPIClient.request``
loops ``for retries_taken in range(max_retries + 1)`` around
``self._client.send(...)``, retrying timeouts, connection errors and retryable
statuses. The providers build their client as ``anthropic_cls(api_key=...,
base_url=...)`` and pass neither ``timeout`` nor ``max_retries``, so the SDK
defaults apply: ``Timeout(connect=5.0, read=600, write=600, pool=600)`` and
``max_retries=2``.

Two consequences, and both are measured here rather than argued:

* **A single decision can stall for a very long time.** 600s read timeout x 3
  SDK attempts x 5 ATL attempts is the ceiling for one bar. That is what killed
  the first run of this probe. ``--timeout-seconds`` installs a real client-side
  timeout so the probe is bounded.
* **``llm_calls`` undercounts the requests actually issued.** An SDK-level retry
  never returns to ``portfolio_manager``, so it is never counted there. Counting
  at ``httpx.Client.send`` — the one call the SDK's retry loop wraps — sees
  every request that reached the provider.

So this module reports two rates: ATL attempts per decision (what the backend
can in principle see) and transport requests per decision (what actually went
over the wire).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED_DB = os.path.join(_REPO_ROOT, "dashboard", "storage", "data", "backtest.db")
SEED_DB_SHA256 = "414bf53cb056b2c60bbdf4d963dffd1ffd7998fcb6235cf9bda963bd52504c80"

# From portfolio_manager.py: `no_text_retries = 4`, loop is range(4 + 1).
MAX_ATTEMPTS_PER_DECISION = 5


def guard_seed_db() -> str:
    """Same guard as the latency tracer: the backend migrates at import time."""
    target = os.getenv("DATABASE_PATH")
    if not target:
        raise SystemExit(
            "DATABASE_PATH is not set. dashboard.backend.database migrates at "
            "import time against the default path, which is the committed seed "
            "DB.\n  export DATABASE_PATH=\"$PWD/local_atl.db\"")
    if os.path.abspath(target) == os.path.abspath(SEED_DB):
        raise SystemExit("DATABASE_PATH points at the committed seed DB.")
    return target


def seed_db_sha256() -> str:
    import hashlib
    h = hashlib.sha256()
    with open(SEED_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RetryRecorder:
    """Per-decision attempt log, keyed off the trigger rather than the count."""

    def __init__(self, on_attempt=None) -> None:
        self.on_attempt = on_attempt
        self.decision_index = -1
        self.attempts: List[Dict[str, Any]] = []
        self.triggers: Counter = Counter()
        self._pending: Optional[Dict[str, Any]] = None

    def new_decision(self) -> None:
        self.decision_index += 1

    def attempt_started(self) -> None:
        idx = sum(1 for a in self.attempts
                  if a["decision_index"] == self.decision_index)
        self._pending = {
            "decision_index": self.decision_index,
            "attempt_index": idx,
            "is_retry": idx > 0,
            "is_rescue": idx == MAX_ATTEMPTS_PER_DECISION - 1,
            "seconds": None, "input_tokens": 0, "output_tokens": 0,
            "outcome": None,
        }

    def attempt_finished(self, seconds: float) -> None:
        if self._pending is not None:
            self._pending["seconds"] = seconds
            self.attempts.append(self._pending)
            self._pending = None
            if self.on_attempt is not None:
                # Flush after every attempt: a hung provider call has no
                # timeout to rescue it, so a run can die with everything
                # measured so far still only in memory.
                self.on_attempt()

    def record_usage(self, in_tok: int, out_tok: int) -> None:
        if self.attempts:
            self.attempts[-1]["input_tokens"] = in_tok
            self.attempts[-1]["output_tokens"] = out_tok

    def record_outcome(self, outcome: str) -> None:
        if self.attempts:
            self.attempts[-1]["outcome"] = outcome
        if outcome != "text_returned":
            self.triggers[outcome] += 1

    def current_key(self):
        """(decision, attempt) a transport request belongs to.

        ``_pending`` is set for the duration of the request, so a send that
        happens inside an attempt is attributed to that attempt — including the
        SDK's own retries, which all fire while the same ``_pending`` is open.
        """
        if self._pending is not None:
            return self._pending["decision_index"], self._pending["attempt_index"]
        return self.decision_index, None


_PATCHES: List[tuple] = []


def _set(owner, name, value):
    _PATCHES.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


class BudgetExceeded(RuntimeError):
    """Raised to stop a run whose provider has stopped responding."""


class TransportRecorder:
    """Counts HTTP requests at ``httpx.Client.send``.

    This is the layer ``llm_calls`` cannot see. The SDK's retry loop wraps
    exactly this call, so one entry here is one request that reached the
    provider, whether ATL asked for it or the SDK retried on its own.
    """

    def __init__(self, key_fn=None) -> None:
        self.key_fn = key_fn or (lambda: (None, None))
        self.requests: List[Dict[str, Any]] = []
        self.by_status: Counter = Counter()
        self.errors: Counter = Counter()

    def record(self, request, response, seconds: float, error) -> None:
        decision, attempt = self.key_fn()
        # The SDK stamps its own retry index on the outgoing request, so an
        # SDK-level retry is identifiable without inferring it from timing.
        sdk_retry = request.headers.get("x-stainless-retry-count")
        try:
            sdk_retry = int(sdk_retry) if sdk_retry is not None else None
        except (TypeError, ValueError):
            sdk_retry = None
        status = getattr(response, "status_code", None)
        row = {
            "decision_index": decision,
            "attempt_index": attempt,
            "sdk_retry_count": sdk_retry,
            "is_sdk_retry": bool(sdk_retry),
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "seconds": seconds,
            "error": type(error).__name__ if error is not None else None,
        }
        self.requests.append(row)
        if error is not None:
            self.errors[type(error).__name__] += 1
        elif status is not None:
            self.by_status[str(status)] += 1

    def summary(self) -> Dict[str, Any]:
        total = len(self.requests)
        sdk_retries = sum(1 for r in self.requests if r["is_sdk_retry"])
        secs = sum(r["seconds"] for r in self.requests)
        return {
            "transport_requests": total,
            "sdk_level_retries": sdk_retries,
            "sdk_retry_share": sdk_retries / total if total else None,
            "status_counts": dict(self.by_status),
            "transport_errors": dict(self.errors),
            "transport_seconds": secs,
            "max_request_seconds": max((r["seconds"] for r in self.requests),
                                       default=None),
        }


def install_transport_counter(tr: "TransportRecorder") -> None:
    orig_send = httpx.Client.send

    def counting_send(self, request, *a, **kw):
        t0 = time.perf_counter()
        try:
            response = orig_send(self, request, *a, **kw)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            tr.record(request, None, time.perf_counter() - t0, exc)
            raise
        tr.record(request, response, time.perf_counter() - t0, None)
        return response

    _set(httpx.Client, "send", counting_send)


def install_client_timeout(seconds: float,
                           sdk_max_retries: Optional[int] = None) -> Dict[str, Any]:
    """Give the SDK client a real client-side timeout.

    The providers construct ``anthropic_cls(api_key=..., base_url=...)`` with no
    ``timeout``, so every request inherits the SDK default 600s read timeout.
    Patching the constructor rather than the provider keeps this probe read-only
    against ``dashboard/`` — it changes nothing on disk, only the client this
    process builds.
    """
    import anthropic

    orig_init = anthropic.Anthropic.__init__
    applied: Dict[str, Any] = {
        "read_timeout_seconds": seconds,
        "connect_timeout_seconds": min(10.0, seconds),
        "sdk_max_retries": sdk_max_retries,
        "clients_patched": 0,
        "sdk_default_timeout": str(_sdk_default_timeout()),
        "sdk_default_max_retries": _sdk_default_max_retries(),
    }

    def patched_init(self, *a, **kw):
        kw.setdefault("timeout", httpx.Timeout(seconds,
                                               connect=min(10.0, seconds)))
        if sdk_max_retries is not None:
            kw.setdefault("max_retries", sdk_max_retries)
        applied["clients_patched"] += 1
        return orig_init(self, *a, **kw)

    _set(anthropic.Anthropic, "__init__", patched_init)
    return applied


def _sdk_default_timeout():
    from anthropic import _base_client
    return _base_client.DEFAULT_TIMEOUT


def _sdk_default_max_retries():
    from anthropic import _base_client
    return _base_client.DEFAULT_MAX_RETRIES


def instrument(rec: RetryRecorder, deadline: Optional[float] = None
               ) -> Dict[str, bool]:
    from dashboard.backend.domain.backtesting import portfolio_manager as pm

    attached: Dict[str, bool] = {}

    orig_request = getattr(pm, "_request_trading_decision", None)
    if orig_request:
        def timed_request(*a, **kw):
            if deadline is not None and time.perf_counter() > deadline:
                raise BudgetExceeded(
                    "wall-clock budget exhausted before this attempt")
            rec.attempt_started()
            t0 = time.perf_counter()
            try:
                return orig_request(*a, **kw)
            finally:
                rec.attempt_finished(time.perf_counter() - t0)
        _set(pm, "_request_trading_decision", timed_request)
    attached["request"] = bool(orig_request)

    orig_usage = getattr(pm, "_extract_token_usage", None)
    if orig_usage:
        def timed_usage(response):
            i, o = orig_usage(response)
            rec.record_usage(i, o)
            return i, o
        _set(pm, "_extract_token_usage", timed_usage)
    attached["usage"] = bool(orig_usage)

    # THE trigger. An AttributeError carrying "No text content" is the only
    # thing that advances the retry loop, so recording it here makes a retry
    # MEASURED rather than inferred from a call count.
    orig_extract = getattr(pm, "_extract_response_text", None)
    if orig_extract:
        def watched_extract(response):
            try:
                text = orig_extract(response)
            except AttributeError as exc:
                rec.record_outcome("no_text_content"
                                   if "No text content" in str(exc)
                                   else f"attribute_error:{str(exc)[:60]}")
                raise
            except Exception as exc:  # noqa: BLE001
                rec.record_outcome(f"other:{type(exc).__name__}")
                raise
            rec.record_outcome("text_returned")
            return text
        _set(pm, "_extract_response_text", watched_extract)
    attached["extract"] = bool(orig_extract)

    orig_state = getattr(pm.PortfolioManager, "get_portfolio_state", None)
    if orig_state:
        def bar_boundary(self, *a, **kw):
            rec.new_decision()
            return orig_state(self, *a, **kw)
        _set(pm.PortfolioManager, "get_portfolio_state", bar_boundary)
    attached["bar_boundary"] = bool(orig_state)
    return attached


def restore() -> None:
    for owner, name, original in reversed(_PATCHES):
        setattr(owner, name, original)
    _PATCHES.clear()


def summarise(rec: RetryRecorder, model: str,
              tr: Optional["TransportRecorder"] = None) -> Dict[str, Any]:
    per_decision: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in rec.attempts:
        per_decision[a["decision_index"]].append(a)
    # Only decisions that actually issued a request count as denominators; a
    # bar the loop skipped never had an attempt and must not dilute the rate.
    decisions = {k: v for k, v in per_decision.items() if v}

    total = len(rec.attempts)
    retries = sum(1 for a in rec.attempts if a["is_retry"])
    retry_tok_in = sum(a["input_tokens"] for a in rec.attempts if a["is_retry"])
    retry_tok_out = sum(a["output_tokens"] for a in rec.attempts if a["is_retry"])
    retry_secs = sum(a["seconds"] or 0 for a in rec.attempts if a["is_retry"])
    all_secs = sum(a["seconds"] or 0 for a in rec.attempts)

    dist = Counter(len(v) for v in decisions.values())
    out = {
        "model": model,
        "decisions_with_requests": len(decisions),
        "total_attempts": total,
        "attempts_per_decision": total / len(decisions) if decisions else None,
        "retries": retries,
        "retry_share_of_attempts": retries / total if total else None,
        "retry_share_of_seconds": retry_secs / all_secs if all_secs else None,
        "rescue_calls": sum(1 for a in rec.attempts if a["is_rescue"]),
        "attempts_histogram": {str(k): v for k, v in sorted(dist.items())},
        "triggers": dict(rec.triggers),
        "retry_input_tokens": retry_tok_in,
        "retry_output_tokens": retry_tok_out,
        "total_input_tokens": sum(a["input_tokens"] for a in rec.attempts),
        "total_output_tokens": sum(a["output_tokens"] for a in rec.attempts),
        "total_seconds": all_secs,
        "retry_seconds": retry_secs,
    }
    if tr is not None:
        t = tr.summary()
        out.update(t)
        n = t["transport_requests"]
        out["transport_requests_per_decision"] = (
            n / len(decisions) if decisions else None)
        # >1 means the SDK issued requests that never came back to ATL, so
        # llm_calls understates what the provider was actually asked for.
        out["transport_requests_per_atl_attempt"] = n / total if total else None
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Measure retry amplification.")
    ap.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-04-04")
    ap.add_argument("--symbols", default="AAPL,MSFT")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-seconds", type=float, default=900.0,
                    help="stop issuing new attempts after this much wall time. "
                         "Checked between attempts, so it bounds the run, not "
                         "an individual request; --timeout-seconds bounds that.")
    ap.add_argument("--timeout-seconds", type=float, default=90.0,
                    help="client-side read timeout to install on the SDK "
                         "client. The backend sets none, so requests otherwise "
                         "inherit the SDK default of 600s and one hung call "
                         "stalls the whole backtest. This is applied in-process "
                         "only; nothing under dashboard/ is modified.")
    ap.add_argument("--sdk-max-retries", type=int, default=None,
                    help="override the SDK's own retry count. Left unset the "
                         "SDK default (2) applies and is measured at the "
                         "transport layer rather than assumed.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    return ap


def estimate(symbols: str, start: str, end: str, model: str) -> Dict[str, Any]:
    from datetime import date
    from analysis.cost_model_lib import PRICE_BY_DB_MODEL
    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    span = (date(y1, m1, d1) - date(y0, m0, d0)).days or 1
    bars = max(1, int(span * 5 / 7)) * 7
    p = {v["slug"]: v for v in PRICE_BY_DB_MODEL.values()}.get(model)
    # Pessimistic: assume the retry loop runs to its cap on every decision.
    worst = bars * MAX_ATTEMPTS_PER_DECISION
    cost = (worst * ((5000 / 1e6) * p["in"] + (2000 / 1e6) * p["out"])
            if p else None)
    return {"approx_bars": bars, "worst_case_attempts": worst,
            "max_cost_usd": cost, "priced": p is not None,
            "assumption": f"every decision retries to the {MAX_ATTEMPTS_PER_DECISION}-attempt cap"}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    est = estimate(args.symbols, args.start, args.end, args.model)
    print(f"RETRY PROBE  {args.model}  {args.start}..{args.end}  "
          f"symbols={args.symbols}")
    print(f"  ~{est['approx_bars']} bars; worst case "
          f"{est['worst_case_attempts']} attempts")
    print("  max cost " + (f"${est['max_cost_usd']:.4f}" if est["priced"]
                           else "unpriced") + f"  ({est['assumption']})")
    if args.dry_run:
        print("\n--dry-run: nothing run, nothing spent.")
        return 0
    if not args.yes:
        if input(f"\nSpend up to ${est['max_cost_usd']:.4f}? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            print("Aborted; nothing spent.")
            return 1

    guard_seed_db()
    before = seed_db_sha256()

    partial: Dict[str, Any] = {}

    def _flush():
        if not args.out:
            return
        partial["summary"] = summarise(rec, args.model, tr)
        partial["attempts"] = rec.attempts
        partial["transport_requests_log"] = tr.requests
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(partial, fh, indent=1, default=str)

    rec = RetryRecorder(on_attempt=_flush)
    tr = TransportRecorder(key_fn=rec.current_key)
    deadline = time.perf_counter() + args.budget_seconds
    attached = instrument(rec, deadline=deadline)
    install_transport_counter(tr)
    timeout_cfg = install_client_timeout(args.timeout_seconds,
                                         args.sdk_max_retries)
    print(f"  client timeout installed: read={args.timeout_seconds}s "
          f"(SDK default {timeout_cfg['sdk_default_timeout']}, "
          f"max_retries default {timeout_cfg['sdk_default_max_retries']})")
    partial.update({
        "label": args.label or args.model.split("/")[-1],
        "model": args.model, "start": args.start, "end": args.end,
        "symbols": args.symbols.split(","), "timers_attached": attached,
        "seed_db_sha256_before": before, "cost_ceiling": est,
        "client_timeout": timeout_cfg,
        "complete": False,
    })
    run_id, results = None, None
    try:
        from dashboard.backend.domain.backtesting.engine import HourlyBacktester
        engine = HourlyBacktester(
            args.start, args.end, "retry-probe", use_llm=True,
            model=args.model, symbols=args.symbols.split(","))
        engine.load_data()
        engine.calculate_indicators()
        run_id, results = engine.run_agent_backtest()
        partial["complete"] = True
    except BudgetExceeded as exc:
        print(f"\n  BUDGET STOP: {exc}. Reporting what was measured; the run "
              f"is marked incomplete.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        # A run that dies mid-way still measured everything up to that point,
        # and losing it is how the first attempt at this ended.
        partial["aborted_with"] = f"{type(exc).__name__}: {exc}"
        print(f"\n  RUN ABORTED: {type(exc).__name__}: {exc}. Reporting what "
              f"was measured; the run is marked incomplete.", file=sys.stderr)
    finally:
        restore()

    after = seed_db_sha256()
    if after != before or after != SEED_DB_SHA256:
        raise AssertionError("seed DB changed during the run")

    payload = dict(partial)
    payload.update({
        "run_id": run_id,
        "bars": len(results) if results else None,
        "seed_db_sha256_after": after,
        "summary": summarise(rec, args.model, tr),
        "attempts": rec.attempts,
        "transport_requests_log": tr.requests,
    })
    s = payload["summary"]
    print(f"\n  decisions {s['decisions_with_requests']}  attempts "
          f"{s['total_attempts']}  per-decision "
          f"{s['attempts_per_decision']:.2f}")
    print(f"  retries {s['retries']} "
          f"({(s['retry_share_of_attempts'] or 0):.1%} of attempts, "
          f"{(s['retry_share_of_seconds'] or 0):.1%} of LLM seconds)")
    print(f"  histogram (attempts->decisions): {s['attempts_histogram']}")
    print(f"  triggers: {s['triggers'] or 'none'}")
    print(f"  transport requests {s['transport_requests']} "
          f"({(s['transport_requests_per_decision'] or 0):.2f}/decision, "
          f"{(s['transport_requests_per_atl_attempt'] or 0):.2f}/ATL attempt)")
    print(f"  SDK-level retries {s['sdk_level_retries']}  "
          f"statuses {s['status_counts'] or 'none'}  "
          f"errors {s['transport_errors'] or 'none'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"  wrote {args.out} (complete={payload['complete']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
