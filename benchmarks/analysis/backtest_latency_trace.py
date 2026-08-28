#!/usr/bin/env python3
"""Where a backtest's wall clock actually goes.

    python benchmarks/analysis/backtest_latency_trace.py --dry-run
    python benchmarks/analysis/backtest_latency_trace.py \
        --start 2026-04-01 --end 2026-04-03 --model nvidia/nemotron-3-nano-30b-a3b

THE HYPOTHESIS THIS TESTS IS NOT ASSUMED
------------------------------------------
The standing explanation for a ~5 minute backtest is 161 sequential LLM calls
at 1-3s each. That is plausible arithmetic and it is not a measurement. If LLM
time is 60% of the wall clock, the other 40% is unexamined and this reports it.

HOW IT INSTRUMENTS WITHOUT TOUCHING dashboard/
------------------------------------------------
Every timer is a monkeypatched wrapper installed at runtime by ``instrument()``
and removed by ``restore()``. Nothing under ``dashboard/`` is edited, and the
audit stays read-only as the brief requires. The wrappers only record
``perf_counter`` deltas — they never change an argument or a return value.

WHAT THE PHASES MEAN
---------------------
``llm``          the provider call itself, wall time around messages.create
``parse``        production response parsing
``market_data``  Alpaca fetch — measured to settle whether it is per bar or
                 prefetched, which the code says is prefetched
``portfolio``    get_portfolio_state, per bar
``execute``      execute_actions + update_equity, per bar
``db``           insert_run / equity writes
``unattributed`` wall clock minus everything above. This is the number the
                 hypothesis cannot predict and the one worth looking at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED_DB = os.path.join(_REPO_ROOT, "dashboard", "storage", "data",
                       "backtest.db")
SEED_DB_SHA256 = "414bf53cb056b2c60bbdf4d963dffd1ffd7998fcb6235cf9bda963bd52504c80"


def guard_seed_db() -> str:
    """Refuse to import the backend until DATABASE_PATH points elsewhere.

    ``dashboard.backend.database`` runs schema migrations AT IMPORT TIME
    against whatever DATABASE_PATH resolves to. With it unset that is the
    committed seed database, and merely importing the module rewrites the file
    — which happened once while this script was being written, and is why the
    check is a hard failure rather than a note in the docstring.
    """
    target = os.getenv("DATABASE_PATH")
    if not target:
        raise SystemExit(
            "DATABASE_PATH is not set. Importing dashboard.backend.database "
            "runs migrations at import time against the default path, which "
            "is the committed seed DB, and rewrites it.\n"
            '  export DATABASE_PATH="$PWD/local_atl.db"')
    if os.path.abspath(target) == os.path.abspath(SEED_DB):
        raise SystemExit(
            f"DATABASE_PATH points at the committed seed DB ({SEED_DB}). "
            f"Refusing: this script's import alone would migrate it.")
    return target


def seed_db_sha256() -> str:
    import hashlib
    h = hashlib.sha256()
    with open(SEED_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_seed_db_untouched() -> None:
    actual = seed_db_sha256()
    if actual != SEED_DB_SHA256:
        raise AssertionError(
            f"seed DB changed during this run: expected {SEED_DB_SHA256}, "
            f"got {actual}")


class Timings:
    """Per-phase call counts and wall time. Not thread-safe by design: the
    engine's bar loop is single-threaded, and that is itself the finding."""

    def __init__(self) -> None:
        self.total: Dict[str, float] = defaultdict(float)
        self.count: Dict[str, int] = defaultdict(int)
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.wall: Optional[float] = None

    def record(self, phase: str, seconds: float) -> None:
        self.total[phase] += seconds
        self.count[phase] += 1
        if len(self.samples[phase]) < 5000:
            self.samples[phase].append(seconds)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"wall_seconds": self.wall, "phases": {}}
        accounted = 0.0
        for phase, secs in sorted(self.total.items(), key=lambda kv: -kv[1]):
            s = sorted(self.samples[phase])
            out["phases"][phase] = {
                "total_seconds": secs,
                "calls": self.count[phase],
                "mean_seconds": secs / self.count[phase] if self.count[phase] else None,
                "median_seconds": s[len(s) // 2] if s else None,
                "max_seconds": s[-1] if s else None,
                "share_of_wall": (secs / self.wall) if self.wall else None,
            }
            accounted += secs
        if self.wall:
            out["accounted_seconds"] = accounted
            out["unattributed_seconds"] = self.wall - accounted
            out["unattributed_share"] = (self.wall - accounted) / self.wall
        return out


_PATCHES: List[tuple] = []


def _wrap(owner: Any, name: str, phase: str, timings: Timings) -> bool:
    """Time one attribute in place. Returns False if it is not there."""
    original = getattr(owner, name, None)
    if original is None or not callable(original):
        return False

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timings.record(phase, time.perf_counter() - t0)

    timed.__name__ = getattr(original, "__name__", name)
    setattr(owner, name, timed)
    _PATCHES.append((owner, name, original))
    return True


def instrument(timings: Timings) -> Dict[str, bool]:
    """Install every timer. Returns which ones actually attached.

    A phase that fails to attach is reported rather than silently absent — a
    missing timer would show up as unattributed time and be read as overhead
    that does not exist.
    """
    from dashboard.backend.domain.backtesting import portfolio_manager as pm
    from dashboard.backend.infrastructure.llm import backtest_harness as bh
    from dashboard.backend.infrastructure.market_data import alpaca_bars as ab
    from dashboard.backend import database as db

    attached = {
        "llm": _wrap(pm, "_request_trading_decision", "llm", timings),
        "parse": _wrap(pm, "_parse_llm_response", "parse", timings),
        "market_data": _wrap(ab.AlpacaDataLoader, "fetch_bars", "market_data",
                             timings),
        "portfolio": _wrap(pm.PortfolioManager, "get_portfolio_state",
                           "portfolio", timings),
        "execute": _wrap(pm.PortfolioManager, "execute_actions", "execute",
                         timings),
        "update_equity": _wrap(pm.PortfolioManager, "update_equity",
                               "update_equity", timings),
        "extract_text": _wrap(bh, "extract_response_text", "extract_text",
                              timings),
    }
    for cls_name in ("BacktestDatabase", "Database"):
        cls = getattr(db, cls_name, None)
        if cls is not None:
            attached["db_insert_run"] = _wrap(cls, "insert_run", "db", timings)
            attached["db_equity_point"] = _wrap(cls, "insert_equity_point",
                                                "db", timings)
            attached["db_equity_points"] = _wrap(cls, "insert_equity_points",
                                                 "db", timings)
            attached["db_trades"] = _wrap(cls, "insert_trades", "db", timings)
            break
    return attached


def restore() -> None:
    for owner, name, original in reversed(_PATCHES):
        setattr(owner, name, original)
    _PATCHES.clear()


def format_trace(payload: Dict[str, Any]) -> str:
    t = payload["timings"]
    L = ["=" * 78, "BACKTEST LATENCY TRACE", "=" * 78,
         f"window {payload['start']} -> {payload['end']}  "
         f"symbols={payload['symbols']}  model={payload['model']}",
         f"bars={payload.get('bars')}  "
         f"construct {payload.get('construct_seconds', 0):.2f}s  "
         f"load_data {payload.get('load_data_seconds', 0):.2f}s  "
         f"indicators {payload.get('indicators_seconds', 0):.2f}s",
         f"bar-loop wall {t['wall_seconds']:.1f}s", "",
         f"  {'phase':<16}{'total s':>10}{'calls':>8}{'mean s':>10}"
         f"{'median s':>11}{'share':>9}"]
    L.append("  " + "-" * 62)
    for phase, d in t["phases"].items():
        L.append(f"  {phase:<16}{d['total_seconds']:>10.2f}{d['calls']:>8}"
                 f"{d['mean_seconds'] or 0:>10.3f}"
                 f"{d['median_seconds'] or 0:>11.3f}"
                 f"{(d['share_of_wall'] or 0):>9.1%}")
    L.append("  " + "-" * 62)
    L.append(f"  {'unattributed':<16}{t['unattributed_seconds']:>10.2f}"
             f"{'':>8}{'':>10}{'':>11}{t['unattributed_share']:>9.1%}")
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Time one real backtest end to end, by phase.")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-04-03")
    ap.add_argument("--symbols", default="AAPL,MSFT")
    ap.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    return ap


def estimate(symbols: str, start: str, end: str, model: str) -> Dict[str, Any]:
    """Bars are hourly over US cash sessions; ~7 per trading day."""
    from datetime import date
    from analysis.cost_model_lib import PRICE_BY_DB_MODEL

    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    span_days = (date(y1, m1, d1) - date(y0, m0, d0)).days or 1
    trading_days = max(1, int(span_days * 5 / 7))
    bars = trading_days * 7
    by_slug = {v["slug"]: v for v in PRICE_BY_DB_MODEL.values()}
    p = by_slug.get(model)
    cost = None
    if p:
        cost = bars * ((5000 / 1e6) * p["in"] + (2000 / 1e6) * p["out"])
    return {"span_days": span_days, "trading_days": trading_days,
            "approx_bars": bars, "approx_llm_calls": bars,
            "max_cost_usd": cost, "priced": p is not None,
            "assumption": "~7 hourly bars per trading day; output at the cap"}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    est = estimate(args.symbols, args.start, args.end, args.model)

    print(f"TRACE  {args.start} -> {args.end}  symbols={args.symbols}  "
          f"model={args.model}")
    print(f"  ~{est['trading_days']} trading days, ~{est['approx_bars']} bars "
          f"=> ~{est['approx_llm_calls']} LLM calls")
    print(f"  max cost " + (f"${est['max_cost_usd']:.4f}" if est["priced"]
                            else "unpriced")
          + f"   ({est['assumption']})")

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

    timings = Timings()
    attached = instrument(timings)
    missing = [k for k, v in attached.items() if not v]
    if missing:
        print(f"  WARNING: timers that did not attach: {missing} — their time "
              f"will appear as unattributed", file=sys.stderr)

    payload: Dict[str, Any] = {
        "start": args.start, "end": args.end,
        "symbols": args.symbols.split(","), "model": args.model,
        "timers_attached": attached, "estimate": est,
    }

    try:
        from dashboard.backend.domain.backtesting.engine import HourlyBacktester
        t0 = time.perf_counter()
        engine = HourlyBacktester(
            args.start, args.end, "latency-trace",
            use_llm=True, model=args.model,
            symbols=args.symbols.split(","),
        )
        payload["construct_seconds"] = time.perf_counter() - t0

        # load_data and calculate_indicators are OUTSIDE the bar loop and are
        # timed separately: they are startup, not per-bar work, and folding
        # them into the loop's wall clock would misattribute them.
        t1 = time.perf_counter()
        engine.load_data()
        payload["load_data_seconds"] = time.perf_counter() - t1

        t1b = time.perf_counter()
        engine.calculate_indicators()
        payload["indicators_seconds"] = time.perf_counter() - t1b

        t2 = time.perf_counter()
        run_id, results = engine.run_agent_backtest()
        timings.wall = time.perf_counter() - t2
        payload["run_id"] = run_id
        payload["bars"] = len(results) if results else None
        payload["llm_calls_reported"] = getattr(
            getattr(engine, "manager", None), "llm_calls", None)
    finally:
        restore()

    assert_seed_db_untouched()
    payload["seed_db_sha256_before"] = before
    payload["seed_db_sha256_after"] = seed_db_sha256()
    payload["timings"] = timings.to_dict()
    print("\n" + format_trace(payload))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
