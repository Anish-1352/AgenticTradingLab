#!/usr/bin/env python3
"""The BEFORE number: what the platform actually spends, from llm_call_usage.

    python dashboard/scripts/llm_spend_baseline.py --dry-run
    python dashboard/scripts/llm_spend_baseline.py --since 2026-08-01 --until 2026-08-31

WHY THIS EXISTS
---------------
``agent_runs`` accumulates ``llm_calls`` / ``input_tokens`` / ``output_tokens``
with ``+=`` and writes three totals once per run. That is enough to know what a
run cost in aggregate and not enough to attribute a saving to the change that
produced it: after enabling the backtest cache you can see the bill fall, but
you cannot show it fell *because of the cache* rather than because users ran
fewer backtests that week.

``llm_call_usage`` fixes that by keeping one row per call. This script is the
consumer: it turns those rows into the baseline that a later measurement is
compared against.

**It reports only what was recorded.** On an empty table it says so and
estimates nothing — an estimated baseline would make every later "we cut cost
X%" claim unfalsifiable in exactly the way the per-call table exists to
prevent.

THE THREE BREAKDOWNS, AND WHY EACH
-----------------------------------
* **By workload** (backtest / leaderboard / live-paper) — these scale on
  different variables and only one of them is user-driven. A bill that grows
  because users backtest more is a different problem from one that grows
  because a nightly job got more expensive.
* **By model** — cost per call spans two orders of magnitude, so the mix
  dominates the total. A shift in mix looks identical to a shift in volume in
  an aggregate figure.
* **By pipeline step** — the split ``agent_runs`` destroyed. It answers which
  step of a multi-step decision is expensive, which is the only way to know
  whether trimming a prompt is worth doing before doing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Direct execution needs the repo root importable. Use the shared bootstrap
# rather than touching sys.path here: it adds only the repo root and creates no
# module aliases, so every backend module keeps one canonical identity.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.paths import DEFAULT_DB_PATH  # noqa: E402

# USD per million tokens (input, output). A model absent here is counted in
# tokens and excluded from the dollar total rather than priced at a guess — a
# fabricated price in a baseline propagates into every later delta.
PRICES: Dict[str, Tuple[float, float]] = {
    "nvidia/nemotron-3-nano-30b-a3b": (0.05, 0.20),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "google/gemini-3.1-pro": (2.0, 12.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "openai/gpt-5.5": (5.0, 30.0),
}

# agent_runs.mode values, grouped into the three workloads that scale
# differently. Anything unrecognised is surfaced under its own name rather than
# folded into a bucket it may not belong in.
WORKLOAD_BY_MODE = {
    "backtest": "backtest",
    "leaderboard": "leaderboard",
    "paper_baseline": "live_paper",
    "paper": "live_paper",
    "live": "live_paper",
}


def price_for(model: Optional[str]) -> Optional[Tuple[float, float]]:
    if not model:
        return None
    name = str(model).strip()
    if name in PRICES:
        return PRICES[name]
    # ":free" slugs bill nothing. Substring-matching them to the paid rate
    # would invent a charge that never happened.
    if name.endswith(":free"):
        return (0.0, 0.0)
    return None


def load_calls(db_path: str, since: Optional[str], until: Optional[str]
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-call rows joined to their run's mode. Also reports table state."""
    state: Dict[str, Any] = {"db_exists": os.path.exists(db_path),
                             "table_exists": False}
    if not state["db_exists"]:
        return [], state
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        state["table_exists"] = "llm_call_usage" in tables
        if not state["table_exists"]:
            return [], state
        sql = ("SELECT u.*, r.mode, r.session_id, r.agent_name "
               "FROM llm_call_usage u "
               "LEFT JOIN agent_runs r ON r.run_id = u.run_id WHERE 1=1")
        params: List[Any] = []
        if since:
            sql += " AND u.timestamp >= ?"
            params.append(since)
        if until:
            sql += " AND u.timestamp <= ?"
            params.append(f"{until}T23:59:59Z" if len(until) == 10 else until)
        return [dict(r) for r in conn.execute(sql, params)], state
    finally:
        conn.close()


def _bucket() -> Dict[str, Any]:
    return {"calls": 0, "input": 0, "output": 0, "cached": 0,
            "cached_reported": 0, "errors": 0, "cost": 0.0, "unpriced": 0,
            "latency_ms_total": 0.0, "latency_n": 0}


def _add(b: Dict[str, Any], c: Dict[str, Any]) -> None:
    b["calls"] += 1
    b["input"] += int(c.get("input_tokens") or 0)
    b["output"] += int(c.get("output_tokens") or 0)
    if c.get("cached_input_tokens") is not None:
        b["cached"] += int(c["cached_input_tokens"])
        b["cached_reported"] += 1
    if c.get("error"):
        b["errors"] += 1
    if c.get("latency_ms") is not None:
        b["latency_ms_total"] += float(c["latency_ms"])
        b["latency_n"] += 1
    p = price_for(c.get("model"))
    if p is None:
        b["unpriced"] += 1
    else:
        b["cost"] += (int(c.get("input_tokens") or 0) / 1e6) * p[0] + (
            int(c.get("output_tokens") or 0) / 1e6) * p[1]


def summarise(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_workload: Dict[str, Dict[str, Any]] = defaultdict(_bucket)
    by_model: Dict[str, Dict[str, Any]] = defaultdict(_bucket)
    by_step: Dict[str, Dict[str, Any]] = defaultdict(_bucket)
    runs: set = set()
    unpriced_models: set = set()
    stamps: List[str] = []

    for c in calls:
        runs.add(c.get("run_id"))
        mode = str(c.get("mode") or "unknown")
        _add(by_workload[WORKLOAD_BY_MODE.get(mode, mode)], c)
        _add(by_model[str(c.get("model") or "(unrecorded)")], c)
        # "pipeline:2:Risk" -> "pipeline"; the kind is the useful axis, and the
        # full label stays available in the raw rows.
        _add(by_step[str(c.get("step_label") or "(unlabelled)").split(":")[0]], c)
        if price_for(c.get("model")) is None:
            unpriced_models.add(str(c.get("model")))
        if c.get("timestamp"):
            stamps.append(str(c["timestamp"]))

    total_cost = sum(b["cost"] for b in by_model.values())
    return {
        "calls": len(calls),
        "runs": len([r for r in runs if r]),
        "total_cost_usd": total_cost,
        "by_workload": dict(by_workload),
        "by_model": dict(by_model),
        "by_step": dict(by_step),
        "unpriced_models": sorted(m for m in unpriced_models if m),
        "observed_first": min(stamps) if stamps else None,
        "observed_last": max(stamps) if stamps else None,
        "cache_reporting_calls": sum(
            b["cached_reported"] for b in by_model.values()),
        "cached_tokens_total": sum(b["cached"] for b in by_model.values()),
    }


def _rows(title: str, buckets: Dict[str, Dict[str, Any]], total: float
          ) -> List[str]:
    out = [f"### {title}", "",
           "| Key | Calls | Input tok | Output tok | Errors | Cost USD | Share |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for key, b in sorted(buckets.items(), key=lambda kv: -kv[1]["cost"]):
        share = f"{(b['cost'] / total * 100):.1f}%" if total else "—"
        cost = f"{b['cost']:.4f}" + ("*" if b["unpriced"] else "")
        out.append(f"| `{key}` | {b['calls']:,} | {b['input']:,} | "
                   f"{b['output']:,} | {b['errors']:,} | {cost} | {share} |")
    out.append("")
    return out


def render(summary: Dict[str, Any], *, db_path: str, since: Optional[str],
           until: Optional[str], state: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    A("# LLM spend baseline")
    A("")
    A(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
      f"from `llm_call_usage`.")
    A("")
    A("| Window | |")
    A("|---|---|")
    A(f"| source | `{db_path}` |")
    A(f"| since | {since or '(all recorded history)'} |")
    A(f"| until | {until or '(now)'} |")
    A(f"| first call seen | {summary.get('observed_first') or '—'} |")
    A(f"| last call seen | {summary.get('observed_last') or '—'} |")
    A(f"| runs | {summary['runs']} |")
    A(f"| calls | {summary['calls']} |")
    A("")

    if not summary["calls"]:
        A("## No baseline yet")
        A("")
        if not state["db_exists"]:
            A(f"The database does not exist at `{db_path}`.")
        elif not state["table_exists"]:
            A("`llm_call_usage` does not exist in this database. The per-call "
              "logging patch has not landed here, or nothing has run since it "
              "did — the table is created on first write.")
        else:
            A("`llm_call_usage` exists but holds no rows in this window.")
        A("")
        A("**No baseline is reported and none is estimated.** A number here "
          "would be a guess, and every later \"cost fell X%\" would inherit "
          "it. Run a backtest or one leaderboard cycle with "
          "`ATL_LLM_CALL_USAGE` enabled (it is on by default), then re-run "
          "this.")
        A("")
        A("Deliberately NOT used as a substitute: the `agent_runs` aggregate. "
          "It carries run-level totals but no per-call, per-step or per-model "
          "split, so it cannot answer the questions this baseline exists to "
          "answer, and quietly falling back to it would produce a baseline "
          "that looks equivalent and is not.")
        return "\n".join(L) + "\n"

    A(f"**Total: ${summary['total_cost_usd']:.4f}** across "
      f"{summary['calls']:,} calls in {summary['runs']} run(s).")
    A("")
    L.extend(_rows("By workload", summary["by_workload"],
                   summary["total_cost_usd"]))
    L.extend(_rows("By model", summary["by_model"], summary["total_cost_usd"]))
    L.extend(_rows("By pipeline step", summary["by_step"],
                   summary["total_cost_usd"]))

    if summary["unpriced_models"]:
        A(f"\\* Rows marked with an asterisk contain calls from models with no "
          f"price entry: {', '.join('`' + m + '`' for m in summary['unpriced_models'])}. "
          f"Their tokens are counted; their dollars are excluded rather than "
          f"guessed.")
        A("")

    A("### Prompt caching")
    A("")
    if not summary["cache_reporting_calls"]:
        A("No call reported a `cached_input_tokens` value. That is the "
          "provider saying nothing about caching — **not** a measured zero "
          "hit rate. The two are different and only the second would be a "
          "result.")
    else:
        A(f"{summary['cache_reporting_calls']:,} of {summary['calls']:,} calls "
          f"reported cache information, totalling "
          f"{summary['cached_tokens_total']:,} cached input tokens.")
    A("")
    A("### How to use this")
    A("")
    A("Capture this before enabling a cost change, and again after a "
      "comparable period. The comparison is only valid if the two windows "
      "carry similar user activity — a fall in the bill during a quiet week "
      "is not a saving, and this script cannot tell the two apart. The "
      "workload table is the place to check that: if `backtest` call volume "
      "moved a lot, the periods are not comparable.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline LLM spend from llm_call_usage.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--since", default=None, help="ISO date, inclusive")
    ap.add_argument("--until", default=None, help="ISO date, inclusive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report table state and exit without writing files")
    args = ap.parse_args(argv)

    calls, state = load_calls(args.db, args.since, args.until)
    summary = summarise(calls)
    text = render(summary, db_path=args.db, since=args.since,
                  until=args.until, state=state)

    if args.dry_run:
        print(text)
        print(f"[dry-run] db_exists={state['db_exists']} "
              f"table_exists={state['table_exists']} calls={summary['calls']} "
              f"— nothing written")
        return 0

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[baseline] {args.out}")
    else:
        print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[json] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
