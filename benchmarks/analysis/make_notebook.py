"""Build cost_model.ipynb.

The notebook is generated rather than hand-authored so its measured constants
cannot drift from the seed database, and so the tests can rebuild and re-check
it. Run:

    python -m analysis.make_notebook
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import (  # noqa: E402
    UNMEASURED_INPUTS, calls_to_reach_budget, load_measured,
)

__all__ = ["build_notebook", "main"]

UNMEASURED_PARAMS = [
    "n_users", "n_agents", "decisions_per_agent_per_day",
    "backtests_per_user_per_day", "calls_per_decision", "model_mix",
    "trading_days_per_month", "calendar_days_per_month",
    "infrastructure_usd_per_month",
]


def _md(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True)}


def _code(source: str) -> Dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def _wrap(param: str) -> str:
    """Comment block naming what would resolve one unmeasured parameter."""
    text = UNMEASURED_INPUTS[param]
    out, line = [], "#   "
    for word in text.split():
        if len(line) + len(word) + 1 > 78:
            out.append(line)
            line = "#   " + word
        else:
            line = f"{line}{word} " if line.endswith(" ") else f"{line} {word}"
    out.append(line.rstrip())
    return "\n".join(out)


def build_notebook() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    # Computed at build time so the prose cannot drift from the seed DB.
    _rows = calls_to_reach_budget(load_measured(), 1.0)
    model_span = _rows[-1]["cost_per_call"] / _rows[0]["cost_per_call"]

    cells.append(_md("""
# ATL cost model — parameterised

This notebook answers *"what would ATL cost per month?"* — but only once you
supply the inputs that no artifact in this repository contains.

**It will not guess.** Every unmeasured parameter starts as `None`, and the
model raises `MissingInput` naming exactly what is blank and what would resolve
it. A silent default is how an assumption becomes a finding, so there are none.

Run all cells as-is: it completes end to end and tells you what it needs.
Fill in the second cell and re-run to get numbers.

| Tag | Meaning |
|---|---|
| `MEASURED` | Read from a run artifact or the committed seed database |
| `DERIVED` | Arithmetic on measured values |
| `NOT MEASURED` | Blank below — you supply it |
"""))

    cells.append(_md("""
## 1. Measured inputs — pre-filled, do not edit

Loaded live from the committed seed database and the run artifacts. Token
counts come from seven leaderboard runs (one per model, ~161 calls each,
identical prompt). Pricing is verified at load time by recomputing every run's
cost from its own stored token totals and comparing with the `est_cost_usd`
the run stored — the loader raises if any run disagrees by more than 1e-5 USD.
"""))

    cells.append(_code("""
import os, sys

# Works whether the notebook is launched from benchmarks/ or benchmarks/analysis/
_here = os.getcwd()
_bench = _here if os.path.basename(_here) == "benchmarks" else os.path.dirname(_here)
if _bench not in sys.path:
    sys.path.insert(0, _bench)

from analysis.cost_model_lib import (
    MissingInput, load_measured, cost_per_call, monthly_breakdown,
    per_model_comparison, sensitivity, measured_calls_per_backtest,
    calls_to_reach_budget, UNMEASURED_INPUTS,
)

MEASURED_DATA = load_measured()          # [MEASURED] seed DB + results/*.json
CALLS_PER_BACKTEST = measured_calls_per_backtest(MEASURED_DATA)

print(f"seed DB runs: {MEASURED_DATA['total_runs']}  "
      f"with LLM usage: {MEASURED_DATA['runs_with_llm']}")
print(f"pricing verified on {MEASURED_DATA['price_verification']['checked']} runs, "
      f"max disagreement {MEASURED_DATA['price_verification']['max_delta_usd']:.2e} USD")
print()
print(f"{'model':<34}{'in tok':>9}{'out tok':>9}{'$/M in':>9}{'$/M out':>9}{'$/call':>11}")
print("-" * 81)
for name, m in sorted(MEASURED_DATA["models"].items(),
                      key=lambda kv: cost_per_call(kv[1])):
    print(f"{m['slug']:<34}{m['input_per_call']:>9,.0f}{m['output_per_call']:>9,.0f}"
          f"{m['price_in']:>9.3f}{m['price_out']:>9.2f}{cost_per_call(m):>11.6f}")
print()
print(f"[MEASURED] calls per backtest: {CALLS_PER_BACKTEST['value']:,.1f} "
      f"(range {CALLS_PER_BACKTEST['min']}-{CALLS_PER_BACKTEST['max']}, "
      f"{CALLS_PER_BACKTEST['n_runs']} runs)")
print(f"           {CALLS_PER_BACKTEST['caveat']}")
"""))

    cells.append(_md("""
## 2. Unmeasured inputs — **fill these in**

Each is `None` on purpose. The comment above each names what would resolve it;
most are a single query against the Render database.

`model_mix` is a dict of `{db_model: fraction}` — the fractions are normalised,
so `{"nemotron_3_nano_30b": 0.9, "gpt_5_5": 0.1}` is fine. Model choice spans
**""" + f"{model_span:,.0f}x" + """** in cost per call, so this is usually the
parameter that decides the answer.
"""))

    param_src = ["# ---- UNMEASURED: every one of these is blank on purpose ----", ""]
    for p in UNMEASURED_PARAMS:
        param_src.append(_wrap(p))
        default = "{}" if p == "model_mix" else "None"
        param_src.append(f"{p} = None" if default == "None" else f"{p} = None")
        param_src.append("")
    param_src.append("PARAMS = {")
    for p in UNMEASURED_PARAMS:
        param_src.append(f"    {p!r}: {p},")
    param_src.append("}")
    param_src.append("")
    param_src.append("blank = [k for k, v in PARAMS.items() if v is None or v == {}]")
    param_src.append("print(f'{len(blank)} of {len(PARAMS)} parameters still blank:')")
    param_src.append("for k in blank:")
    param_src.append("    print(f'  - {k}')")
    cells.append(_code("\n".join(param_src)))

    cells.append(_md("""
## 3. Monthly cost — live trading vs backtesting vs infrastructure

The split matters because the two workloads scale on different variables:

* **Live trading** scales with agents x decision rate. The bar interval bounds
  it — the engine requests hourly bars, so an agent cannot decide faster than
  its bars arrive.
* **Backtesting** scales with users x how often they press the button. **Nothing
  bounds it.** One backtest replays a whole window on demand: the seed runs cost
  ~161 calls each. This is the term most likely to dominate, and it is entirely
  unmeasured.
* **Infrastructure** is hosting, GPU, and database spend. Not measurable from
  this repository — no hosted-API run has ever been executed and no self-hosted
  deployment exists to price.
"""))

    cells.append(_code("""
try:
    breakdown = monthly_breakdown(MEASURED_DATA, PARAMS)
except MissingInput as exc:
    print("CANNOT COMPUTE\\n")
    print(exc)
else:
    b = breakdown
    print(f"blended cost per call : ${b['blended_cost_per_call']:.6f}  [DERIVED]")
    print(f"calls per backtest    : {b['calls_per_backtest']:,.1f}  [MEASURED]")
    print()
    print(f"{'component':<18}{'calls/month':>16}{'USD/month':>14}{'share':>9}")
    print("-" * 57)
    for label, key in (("live trading", "live_trading"),
                       ("backtesting", "backtesting"),
                       ("infrastructure", "infrastructure")):
        row = b[key]
        calls = f"{row['calls']:,.0f}" if "calls" in row else "—"
        print(f"{label:<18}{calls:>16}{row['usd']:>14,.2f}"
              f"{(row['share'] or 0) * 100:>8.1f}%")
    print("-" * 57)
    print(f"{'TOTAL':<18}{b['total_calls_per_month']:>16,.0f}"
          f"{b['total_usd_per_month']:>14,.2f}")
"""))

    cells.append(_md("""
## 4. Per-model comparison

Whole-platform monthly cost if 100% of calls used each model in turn, at the
load you specified above. Each model uses **its own measured output length** —
output tokens span 5.8x across these models under an identical prompt, so a
single shared output length would understate the spread badly.
"""))

    cells.append(_code("""
try:
    rows = per_model_comparison(MEASURED_DATA, PARAMS)
except MissingInput as exc:
    print("CANNOT COMPUTE\\n")
    print(exc)
else:
    print(f"{'model':<34}{'out tok':>9}{'$/call':>10}"
          f"{'live $/mo':>13}{'backtest $/mo':>15}{'total $/mo':>14}")
    print("-" * 95)
    for r in rows:
        print(f"{r['slug']:<34}{r['output_tokens_per_call']:>9,.0f}"
              f"{r['cost_per_call']:>10.6f}{r['live_usd']:>13,.0f}"
              f"{r['backtest_usd']:>15,.0f}{r['total_usd_per_month']:>14,.0f}")
    print()
    lo, hi = rows[0], rows[-1]
    print(f"[DERIVED] span across models: "
          f"{hi['total_usd_per_month'] / lo['total_usd_per_month']:,.0f}x "
          f"({lo['slug']} -> {hi['slug']})")
"""))

    cells.append(_md("""
## 5. Sensitivity — which lever actually moves the answer

Ordered by how many dollars each lever swings, so the dominant one is visible
rather than argued for. Numeric levers are scaled by 2x either way; model choice
is swept across the measured models instead, since it is not a numeric knob.
"""))

    cells.append(_code("""
try:
    sens = sensitivity(MEASURED_DATA, PARAMS, factor=2.0)
except MissingInput as exc:
    print("CANNOT COMPUTE\\n")
    print(exc)
else:
    print(f"base total: ${sens[0]['base_usd']:,.2f}/month  [DERIVED]\\n")
    print(f"{'lever':<34}{'low $/mo':>13}{'high $/mo':>13}"
          f"{'swing $':>13}{'span':>8}")
    print("-" * 81)
    for r in sens:
        span = f"{r['span_factor']:,.1f}x" if r["span_factor"] else "—"
        print(f"{r['lever']:<34}{r['low_usd']:>13,.0f}{r['high_usd']:>13,.0f}"
              f"{r['swing_usd']:>13,.0f}{span:>8}")
    print()
    print(f"Dominant lever: {sens[0]['lever']} ({sens[0]['kind']})")
"""))

    cells.append(_md("""
## 6. How much load would $50K/month take?

Independent of the parameters above — this only needs the measured per-call
costs. It is the honest shape of an answer to *"would $50K be a problem?"*:
not one figure, but the load each model would have to carry to get there.
"""))

    cells.append(_code("""
BUDGET = 50_000.0    # [NOT MEASURED] the budget as posed, not a measurement

print(f"{'model':<34}{'$/call':>10}{'calls/month for $50K':>24}")
print("-" * 68)
for r in calls_to_reach_budget(MEASURED_DATA, BUDGET):
    print(f"{r['slug']:<34}{r['cost_per_call']:>10.6f}"
          f"{r['calls_for_budget']:>24,.0f}")
print()
print("[MEASURED] one backtest of the seed window ~"
      f"{CALLS_PER_BACKTEST['value']:,.0f} calls, so with 100 users over 30 days:")
print()
denom = 100 * CALLS_PER_BACKTEST['value'] * 30
print(f"{'model':<34}{'backtests/user/day for $50K':>30}")
print("-" * 64)
for r in calls_to_reach_budget(MEASURED_DATA, BUDGET):
    print(f"{r['slug']:<34}{r['calls_for_budget'] / denom:>30,.2f}")
"""))

    cells.append(_md("""
## What this notebook does not model

Named so none of it is mistaken for a gap in the arithmetic:

* **Retry inflation** — never observed in a real run.
* **Multi-step pipelines in production** — all seed runs used the single-call
  path. A pipeline issues one call per step (verified 3→3 and 5→5 against the
  real runner), so `calls_per_decision` above multiplies everything, but no
  production run has ever recorded a step count.
* **Prefix caching** — the ablation was never run; vLLM reported
  `stats_source: null`, so no hit rate was ever observed.
* **Rate limits** — the model assumes purchasable throughput is unbounded.
* **Engineering and on-call cost** of any self-hosted option.
* **Backtest window length** — `calls_per_backtest` is measured for a one-month
  hourly replay and scales linearly with the window. A user backtesting a year
  costs roughly twelve times as much per run.
"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build cost_model.ipynb")
    ap.add_argument("--out", default=os.path.join(_HERE, "cost_model.ipynb"))
    args = ap.parse_args(argv)
    nb = build_notebook()
    with open(args.out, "w") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"[notebook] {args.out} ({len(nb['cells'])} cells, {n_code} code)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
