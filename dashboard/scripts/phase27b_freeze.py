"""Freeze the Phase 27B source tape once, with explicit split adjustment.

    python -m dashboard.scripts.phase27b_freeze --output /abs/path/frozen-dir

ATL's loader never sets ``adjustment``, so Alpaca returns raw prices. Across
the AMZN (2022) and NVDA (2024) splits that would fabricate -95% and -90% bars,
so this requests split-adjusted bars directly from the same provider. It is the
only Phase 27B code that touches the network; the experiment CLI reads the
frozen bytes and checks their hashes.

Before writing the manifest it verifies the spec's holiday/early-close list
against the tape and that one share of every symbol fits ATL's position limit.
Either failure exits nonzero, so a wrong calendar or an untradable name is a
decision someone makes explicitly rather than a silent row loss.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd

TZ = "America/New_York"
FULL_SESSION_BARS = 78
# A full session's 15:55 bar carries the market-on-close auction. Measured on
# the frozen SPY tape: every early close sits at <= 0.08x the day's median bar,
# every full session at >= 2.56x. 1.0 is in the middle of that gap. Bar counts
# cannot do this job: a liquid ETF prints every five minutes after a 13:00
# close, so an early close has as many bars as a full session.
AUCTION_RATIO = 1.0


def _regular(frame):
    local = frame.index.tz_convert(TZ)
    minutes = local.hour * 60 + local.minute
    return frame.loc[(minutes >= 570) & (minutes < 960)]


def verify_calendar(raw, spec, start, end):
    """Classify every weekday in [start, end) against the declared calendar.

    ``end`` is exclusive, as it is in the provider request that fetched the
    tape: no bars on the end date is not a market closure.
    """
    import numpy as np
    regular = _regular(raw)
    local = regular.index.tz_convert(TZ)
    days = pd.Index(local.date)
    counts = pd.Series(1, index=days).groupby(level=0).sum()
    median = regular["volume"].groupby(days).median()
    closing = regular["volume"].where(local.strftime("%H:%M") == "15:55", 0.0).groupby(days).sum()
    auction = (closing / median.replace(0, np.nan)).fillna(0.0)
    excluded = set(spec["excluded_sessions"])
    report = {"excluded": {}, "unexpected_closures": [], "unexpected_early_closes": [],
              "incomplete_sessions": []}
    for day in pd.bdate_range(start, end, inclusive="left"):
        key, d = day.date().isoformat(), day.date()
        n = int(counts.get(d, 0))
        early = n > 0 and float(auction.get(d, 0.0)) < AUCTION_RATIO
        if key in excluded:
            report["excluded"][key] = ("closed" if n == 0 else
                                       "early_close" if early else "UNEXPECTED_FULL_SESSION")
        elif n == 0:
            report["unexpected_closures"].append(key)
        elif early:
            report["unexpected_early_closes"].append(key)
        elif n < FULL_SESSION_BARS:
            report["incomplete_sessions"].append({"session": key, "bars": n})
    report["ok"] = (not report["unexpected_closures"] and not report["unexpected_early_closes"]
                    and "UNEXPECTED_FULL_SESSION" not in report["excluded"].values())
    return report


def symbol_data_gaps(raw, spec, start, end):
    """Every session where a symbol's tape falls short of a verified calendar.

    Run per traded symbol after the market symbol has verified the calendar, so
    a short, truncated or absent session here is the provider's gap, not a
    holiday. The builder drops any row these touch; this makes the loss visible.
    """
    r = verify_calendar(raw, spec, start, end)
    gaps = [{"session": d, "kind": "missing_session"} for d in r["unexpected_closures"]]
    gaps += [{"session": d, "kind": "truncated_session"} for d in r["unexpected_early_closes"]]
    gaps += [{"session": g["session"], "kind": "incomplete_session", "bars": g["bars"]}
             for g in r["incomplete_sessions"]]
    return sorted(gaps, key=lambda g: g["session"])


def price_feasibility(frames, spec, max_position_weight=0.25):
    """One share must fit under ATL's per-position limit on the starting capital."""
    limit = max_position_weight * spec["initial_cash"]
    symbols = {}
    for symbol, frame in frames.items():
        top = float(_regular(frame)["high"].max())
        symbols[symbol] = {"max_regular_high": top, "feasible": top <= limit}
    return {"limit": limit, "symbols": symbols,
            "infeasible": sorted(s for s, v in symbols.items() if not v["feasible"])}


def fetch(symbol, start, end):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=pd.Timestamp(start, tz=TZ), end=pd.Timestamp(end, tz=TZ),
                               adjustment=Adjustment.SPLIT, feed=DataFeed.SIP)
    frame = client.get_stock_bars(request).df
    if frame.empty:
        raise ValueError(f"no bars returned for {symbol}")
    frame = frame.xs(symbol, level="symbol")[["open", "high", "low", "close", "volume"]]
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    frame.index.name = "timestamp"
    return frame.sort_index()


def main():
    from dashboard.backend.domain.research.phase27 import digest
    from dashboard.backend.domain.research.phase27b import load_spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = load_spec()
    start, end = spec["data_window"]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames, files = {}, {}
    for symbol in [*spec["universe"], spec["market_symbol"]]:
        frame = fetch(symbol, start, end)
        path = out / f"{symbol}-5m.csv"
        frame.to_csv(path)
        frames[symbol] = frame
        files[symbol] = {"file": path.name, "sha256": digest(path.read_bytes()), "rows": len(frame),
                         "first": frame.index[0].isoformat(), "last": frame.index[-1].isoformat()}
        print(f"  {symbol}: {len(frame)} bars", flush=True)
    calendar = verify_calendar(frames[spec["market_symbol"]], spec, start, end)
    per_symbol_gaps = {s: symbol_data_gaps(f, spec, start, end) for s, f in frames.items()}
    feasibility = price_feasibility({s: frames[s] for s in spec["universe"]}, spec)
    manifest = {"files": files, "adjustment": "split", "feed": "sip", "timeframe": "5m",
                "window": [start, end], "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "frame_attrs": {"bar_open_stamped_minutes": 5},
                "calendar_verification": {"authority": spec["market_symbol"], **calendar},
                "data_gaps_per_symbol": per_symbol_gaps, "price_feasibility": feasibility}
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"calendar ok={calendar['ok']}  infeasible={feasibility['infeasible']}  "
          f"gap sessions={ {s: len(g) for s, g in per_symbol_gaps.items()} }")
    if not calendar["ok"] or feasibility["infeasible"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
