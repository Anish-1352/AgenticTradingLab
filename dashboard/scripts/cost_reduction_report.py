#!/usr/bin/env python3
"""Generate cost_reduction_report.md from llm_call_usage.

    python dashboard/scripts/cost_reduction_report.py --out cost_reduction_report.md
    python dashboard/scripts/cost_reduction_report.py --since 2026-08-01 --until 2026-08-17

Reads only what was recorded. Every figure here is a measurement over a stated
window, or it is absent — there are no modelled savings in this report, because
a modelled saving is indistinguishable from a hoped-for one once it is in a
table.

WHY THERE IS NO "BEFORE" UNTIL THERE IS ONE
--------------------------------------------
Per-call logging landed with this batch, so the first run of this script has
nothing to compare against: there is no recorded history from before the
instrumentation existed. It says so rather than reaching for the seed data,
which was produced under a different configuration and would make an
apples-to-oranges delta look like a result.

Once two windows exist, ``--since``/``--until`` bound them and the report
compares like with like.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Direct execution needs the repo root importable. Use the shared bootstrap
# rather than touching sys.path here: it adds only the repo root and creates no
# module aliases, so every backend module keeps one canonical identity.
if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.paths import DEFAULT_DB_PATH  # noqa: E402

# USD per million tokens. Used only where a model is recognised; unrecognised
# models are counted in tokens and reported as cost-unknown rather than being
# priced at a guess.
PRICES: Dict[str, tuple] = {
    "nvidia/nemotron-3-nano-30b-a3b": (0.05, 0.20),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "google/gemini-3.1-pro": (2.0, 12.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "openai/gpt-5.5": (5.0, 30.0),
}


def _price(model: Optional[str]) -> Optional[tuple]:
    if not model:
        return None
    name = str(model).strip()
    if name in PRICES:
        return PRICES[name]
    # ":free" slugs bill nothing; pricing them at the paid rate would fabricate
    # a charge that never happened.
    if name.endswith(":free"):
        return (0.0, 0.0)
    return None


def load_calls(db_path: str, since: Optional[str], until: Optional[str]
               ) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "llm_call_usage" not in tables:
            return []
        sql = ("SELECT u.*, r.mode, r.agent_name FROM llm_call_usage u "
               "LEFT JOIN agent_runs r ON r.run_id = u.run_id WHERE 1=1")
        params: List[Any] = []
        if since:
            sql += " AND u.timestamp >= ?"
            params.append(since)
        if until:
            sql += " AND u.timestamp <= ?"
            params.append(until + "T23:59:59Z" if len(until) == 10 else until)
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def summarise(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0, "cached": 0,
                 "cached_reported": 0, "errors": 0, "cost": 0.0,
                 "cost_known": True})
    by_step: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0})
    runs = set()

    for c in calls:
        runs.add(c.get("run_id"))
        model = c.get("model") or "(unrecorded)"
        m = by_model[model]
        m["calls"] += 1
        m["input"] += int(c.get("input_tokens") or 0)
        m["output"] += int(c.get("output_tokens") or 0)
        if c.get("cached_input_tokens") is not None:
            m["cached"] += int(c["cached_input_tokens"])
            m["cached_reported"] += 1
        if c.get("error"):
            m["errors"] += 1
        step = str(c.get("step_label") or "(unlabelled)").split(":")[0]
        s = by_step[step]
        s["calls"] += 1
        s["input"] += int(c.get("input_tokens") or 0)
        s["output"] += int(c.get("output_tokens") or 0)

    for model, m in by_model.items():
        price = _price(model)
        if price is None:
            m["cost_known"] = False
            m["cost"] = None
        else:
            m["cost"] = (m["input"] / 1e6) * price[0] + (m["output"] / 1e6) * price[1]

    total_cost = sum(m["cost"] or 0.0 for m in by_model.values())
    unknown = [k for k, m in by_model.items() if not m["cost_known"]]
    return {
        "calls": len(calls),
        "runs": len([r for r in runs if r]),
        "by_model": dict(by_model),
        "by_step": dict(by_step),
        "total_cost_usd": total_cost,
        "models_without_price": unknown,
        "cached_reported_calls": sum(
            m["cached_reported"] for m in by_model.values()),
        "cached_tokens_total": sum(m["cached"] for m in by_model.values()),
    }


def render(summary: Dict[str, Any], *, db_path: str, since: Optional[str],
           until: Optional[str]) -> str:
    L: List[str] = []
    A = L.append
    A("# Serving cost reduction — measured")
    A("")
    A(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
      f"from `llm_call_usage`.")
    A("")
    A("| Measurement window | |")
    A("|---|---|")
    A(f"| source | `{db_path}` |")
    A(f"| since | {since or '(all recorded history)'} |")
    A(f"| until | {until or '(now)'} |")
    A(f"| runs covered | {summary['runs']} |")
    A(f"| calls recorded | {summary['calls']} |")
    A("")

    if not summary["calls"]:
        A("## No data yet")
        A("")
        A("`llm_call_usage` is empty for this window. Per-call logging landed "
          "with this batch, so there is no recorded history from before it "
          "existed.")
        A("")
        A("**No before/after can be reported yet, and none is estimated here.** "
          "Run a backtest (or one leaderboard refresh cycle) with "
          "`ATL_LLM_CALL_USAGE` enabled — it is on by default — then re-run "
          "this script. Comparing against the pre-instrumentation seed data "
          "would compare different configurations and is deliberately not done.")
        return "\n".join(L) + "\n"

    A("## Cost by model")
    A("")
    A("| Model | Calls | Input tokens | Output tokens | Errors | Cost USD |")
    A("|---|---:|---:|---:|---:|---:|")
    for model, m in sorted(summary["by_model"].items(),
                           key=lambda kv: -(kv[1]["cost"] or 0)):
        cost = "unpriced" if not m["cost_known"] else f"{m['cost']:.6f}"
        A(f"| `{model}` | {m['calls']:,} | {m['input']:,} | {m['output']:,} | "
          f"{m['errors']:,} | {cost} |")
    A("")
    A(f"**Total: ${summary['total_cost_usd']:.4f}** over "
      f"{summary['calls']:,} calls across {summary['runs']} run(s).")
    if summary["models_without_price"]:
        A("")
        A(f"Unpriced models (counted in tokens, excluded from the total): "
          f"{', '.join('`' + m + '`' for m in summary['models_without_price'])}. "
          f"Adding a guessed price would put a fabricated number in the total.")
    A("")

    A("## Where the tokens go, by call site")
    A("")
    A("| Step kind | Calls | Input tokens | Output tokens |")
    A("|---|---:|---:|---:|")
    for step, s in sorted(summary["by_step"].items(), key=lambda kv: -kv[1]["calls"]):
        A(f"| `{step}` | {s['calls']:,} | {s['input']:,} | {s['output']:,} |")
    A("")
    A("This is the split that was impossible before per-call logging: "
      "`agent_runs` summed it away.")
    A("")

    A("## Prompt caching")
    A("")
    reported = summary["cached_reported_calls"]
    if not reported:
        A("No call in this window reported a `cached_input_tokens` field. That "
          "means the provider said nothing about caching — **not** that the "
          "cache missed. The two are different facts and are not collapsed "
          "here.")
    else:
        A(f"{reported:,} of {summary['calls']:,} calls reported cache "
          f"information, totalling {summary['cached_tokens_total']:,} cached "
          f"input tokens.")
    A("")

    A("## What this report does not claim")
    A("")
    A("- **No modelled savings.** Only what was recorded appears above.")
    A("- **No before/after across a config change** unless two windows are "
      "given, because a delta between different configurations is not a "
      "saving.")
    A("- **Cost uses list prices**, which change without notice; token counts "
      "are what was actually billed by the provider.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Report measured LLM cost.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--since", default=None, help="ISO date, inclusive")
    ap.add_argument("--until", default=None, help="ISO date, inclusive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    calls = load_calls(args.db, args.since, args.until)
    summary = summarise(calls)
    text = render(summary, db_path=args.db, since=args.since, until=args.until)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[report] {args.out}")
    else:
        print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[json] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
