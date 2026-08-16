"""Generate ADVISOR_REPORT.md from the run artifacts and the seed database.

    python -m analysis.make_advisor_report                     # writes the report
    python -m analysis.make_advisor_report --check             # verify, write nothing

Nothing in the output is hand transcribed. Token counts come from the seed DB,
serving figures from ``results/*_summary.json``, provenance from
``results/*_manifest.json``, and every code finding is re-read from its cited
``file:line`` before it is printed.

EVERY NUMBER CARRIES A TIER
---------------------------
``MEASURED`` / ``DERIVED`` / ``NOT MEASURED``. Tables carry a Tier column;
prose carries an inline tag. ``tests/test_advisor_report.py`` re-parses the
generated file and fails if any numeric line lacks one — so the discipline is
enforced by a test rather than by care.

DELIBERATELY EXCLUDED
---------------------
Crossover agent counts, prefix-cache lever estimates, and every Arm A figure.
No hosted-API run has ever been executed, the cache ablation was never run, and
the crossover rests on both. They are model output, not measurement, and this
report is for measurements.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, NOT_MEASURED, UNMEASURED_INPUTS,
    calls_to_reach_budget, cost_per_call, load_measured,
    measured_calls_per_backtest, verify_code_findings,
)
from analysis.tier_check import check_report, format_violations  # noqa: E402

__all__ = ["build_report", "main"]

BUDGET = 50_000.0

# Conventions used only to translate a call count into a load scenario. They are
# calendar arithmetic, not measurements, and are tagged as such everywhere they
# appear.
TRADING_DAYS = 21
CALENDAR_DAYS = 30

# Dr. Liu's scenario, as posed. These are HIS numbers, not measurements of the
# platform, and the report says so.
SCENARIO_USERS = 100
SCENARIO_AGENTS = 1000

EXCLUDED = [
    ("crossover agent counts",
     "rests on an Arm A cost per request that has never been measured"),
    ("prefix-cache benefit estimates",
     f"the ablation was never run; vLLM reported stats_source: null "
     f"[{NOT_MEASURED}], so no hit rate was observed even with caching "
     f"enabled"),
    ("every Arm A figure",
     "no hosted-API run has ever been executed"),
    ("attribution of the speedup across batching / caching / engine",
     "never decomposed; the arms differ in more than one variable"),
]


def _t(tier: str) -> str:
    return f"[{tier}]"


def build_report(measured: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append

    models = measured["models"]
    serving = measured["serving"]
    trace = measured["trace"]
    findings = verify_code_findings()
    cpb = measured_calls_per_backtest(measured)
    budget_rows = calls_to_reach_budget(measured, BUDGET)
    cheapest, dearest = budget_rows[0], budget_rows[-1]

    A("# ATL cost and serving: what has been measured")
    A("")
    A(f"Prepared for Dr. Liu, in answer to: *{SCENARIO_USERS} users, "
      f"{SCENARIO_AGENTS} agents — would $50K/month be a real problem?* "
      f"Those three figures are the question as posed {_t(NOT_MEASURED)}, not "
      f"measurements of the platform.")
    A("")
    A("**Generated file — do not edit by hand.** Regenerate with:")
    A("")
    A("```bash")
    A("cd benchmarks && python -m analysis.make_advisor_report")
    A("```")
    A("")
    A("## How to read this")
    A("")
    A("Every number carries one of three tags. A number without one is a bug, "
      "and a test enforces it.")
    A("")
    A("| Tag | Meaning |")
    A("|---|---|")
    A(f"| `{MEASURED}` | Read from a run artifact or the seed database. "
      f"Traceable to a `run_id`, or to `file:line` for a code finding. |")
    A(f"| `{DERIVED}` | Arithmetic on measured values, with the arithmetic "
      f"shown. |")
    A(f"| `{NOT_MEASURED}` | Named, with what it would take to measure it. "
      f"Never silently defaulted. |")
    A("")
    A("The headline answer is a **conditional**, not a figure. The inputs that "
      "would decide it are precisely the ones no artifact in this repository "
      "contains — so they are listed rather than guessed.")
    A("")

    # ---- 1. the $50K question -------------------------------------------
    A("## 1. The $50K question")
    A("")
    A("### The short answer")
    A("")
    A(f"A $50K/month budget {_t(NOT_MEASURED)} is reachable or unreachable "
      f"depending almost entirely on **which model serves production** and "
      f"**how much backtesting users do**. Neither is measured here. The "
      f"spread between them is not a detail — it is the whole answer.")
    A("")
    A(f"At each model's measured cost per call, the monthly call volume "
      f"needed to spend the ${BUDGET:,.0f} {_t(NOT_MEASURED)} budget is:")
    A("")
    A("| Model | $/call | Calls/month for $50K | Tier | Source run_id |")
    A("|---|---:|---:|---|---|")
    for r in budget_rows:
        A(f"| `{r['slug']}` | {r['cost_per_call']:.6f} | "
          f"{r['calls_for_budget']:,.0f} | {DERIVED} | `{r['run_id']}` |")
    A("")
    A(f"Cost per call is {_t(DERIVED)}: it is "
      f"`(input_tokens/1e6 x price_in) + (output_tokens/1e6 x price_out)`, "
      f"where both token counts are {_t(MEASURED)} from the seed DB and both "
      f"prices are {_t(MEASURED)} — verified below.")
    A("")
    A(f"The span between cheapest and dearest model is "
      f"{dearest['cost_per_call'] / cheapest['cost_per_call']:,.0f}x "
      f"{_t(DERIVED)}. The same $50K buys "
      f"{cheapest['calls_for_budget']:,.0f} calls on "
      f"`{cheapest['slug']}` or {dearest['calls_for_budget']:,.0f} on "
      f"`{dearest['slug']}` {_t(DERIVED)}.")
    A("")
    A("### What that load would have to look like")
    A("")
    A(f"Translating call volume into Dr. Liu's scenario "
      f"({SCENARIO_USERS} users, {SCENARIO_AGENTS} agents "
      f"{_t(NOT_MEASURED)} — these are the posed scenario, not a measurement "
      f"of the platform):")
    A("")
    A("**Live trading.** One live agent making `D` decisions per day, at one "
      "call per decision:")
    A("")
    A(f"> {SCENARIO_AGENTS} agents x `D` x {TRADING_DAYS} trading days/month "
      f"= {SCENARIO_AGENTS * TRADING_DAYS:,} x `D` calls/month "
      f"{_t(DERIVED)}; the trading-day count is a calendar convention "
      f"{_t(NOT_MEASURED)}, and `D` is {_t(NOT_MEASURED)}.")
    A("")
    live_denom = SCENARIO_AGENTS * TRADING_DAYS
    A("| Model | Decisions/agent/day needed for $50K | Tier | Plausible? |")
    A("|---|---:|---|---|")
    for r in (cheapest, dearest):
        d_needed = r["calls_for_budget"] / live_denom
        verdict = ("far above any bar-driven rate"
                   if d_needed > 100 else "within reach")
        A(f"| `{r['slug']}` | {d_needed:,.1f} | {DERIVED} | {verdict} |")
    A("")
    A(f"For context: the seed runs recorded {cpb['bars']:,} bars over a "
      f"one-month window {_t(MEASURED)}, and the engine requests hourly bars "
      f"filtered to market sessions "
      f"(`dashboard/backend/infrastructure/market_data/alpaca_bars.py:114`) "
      f"{_t(MEASURED)}. Over {TRADING_DAYS} trading days "
      f"{_t(NOT_MEASURED)} that is about "
      f"{cpb['bars'] / TRADING_DAYS:.1f} decisions per agent per trading day "
      f"{_t(DERIVED)} — three orders of magnitude below the rate the default "
      f"model would need. So on that model, **live trading alone cannot "
      f"approach $50K/month at this scenario's agent count** {_t(DERIVED)}.")
    A("")
    A("**Backtesting — and this is where the risk actually is.** One backtest "
      "of the seed runs' window costs:")
    A("")
    A(f"> {cpb['value']:,.1f} LLM calls per backtest {_t(MEASURED)} "
      f"(range {cpb['min']:,}-{cpb['max']:,} across {cpb['n_runs']} runs), "
      f"one call per hourly bar over a one-month replay.")
    A("")
    A(f"> {SCENARIO_USERS} users x `B` backtests/day x "
      f"{cpb['value']:,.1f} calls x {CALENDAR_DAYS} days/month = "
      f"{SCENARIO_USERS * cpb['value'] * CALENDAR_DAYS:,.0f} x `B` "
      f"calls/month {_t(DERIVED)}; `B` is {_t(NOT_MEASURED)}.")
    A("")
    bt_denom = SCENARIO_USERS * cpb["value"] * CALENDAR_DAYS
    A("| Model | Backtests/user/day needed for $50K | Tier |")
    A("|---|---:|---|")
    for r in budget_rows:
        A(f"| `{r['slug']}` | {r['calls_for_budget'] / bt_denom:,.2f} | "
          f"{DERIVED} |")
    A("")
    A(f"**This is the finding worth acting on.** On the platform's default "
      f"model, reaching $50K would take "
      f"{cheapest['calls_for_budget'] / bt_denom:,.0f} backtests per user per "
      f"day {_t(DERIVED)} — implausible. On the most expensive measured "
      f"model it takes "
      f"{dearest['calls_for_budget'] / bt_denom:,.2f} {_t(DERIVED)} — which a "
      f"single engaged user could exceed before lunch. Backtest volume is "
      f"unbounded by design: a backtest replays a whole window on demand, "
      f"where live trading is rate-limited by the bar interval.")
    A("")
    A("### So the conditional")
    A("")
    A(f"> $50K/month becomes a real problem **only if** production runs a "
      f"frontier model **and** backtest volume reaches roughly single-digit "
      f"runs per user per day {_t(DERIVED)}. At the platform's current "
      f"default model and call pattern, the same load costs on the order of "
      f"{BUDGET * (cheapest['cost_per_call'] / dearest['cost_per_call']):,.0f} "
      f"dollars/month {_t(DERIVED)} — a "
      f"{dearest['cost_per_call'] / cheapest['cost_per_call']:,.0f}x "
      f"difference driven by model choice alone {_t(DERIVED)}.")
    A("")
    A("Two inputs decide it and neither is in this repository: the production "
      "**model mix** and **backtests per user per day**. Both are one SQL "
      "query away on the Render database — see `QUESTIONS_FOR_ADVISOR.md`.")
    A("")

    # ---- 2. per-model cost ----------------------------------------------
    A("## 2. Measured cost per call, by model")
    A("")
    A(f"Seven leaderboard runs in the committed seed database, one per model, "
      f"each replaying the same window under the same prompt "
      f"{_t(MEASURED)}. Because the prompt is identical, the output-token "
      f"spread is the model's verbosity rather than the workload's.")
    A("")
    A("| Model | In tok/call | Out tok/call | $/M in | $/M out | $/call | "
      "Calls | Tier | run_id |")
    A("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for name, m in sorted(models.items(), key=lambda kv: cost_per_call(kv[1])):
        A(f"| `{m['slug']}` | {m['input_per_call']:,.0f} | "
          f"{m['output_per_call']:,.0f} | {m['price_in']:.3f} | "
          f"{m['price_out']:.2f} | {cost_per_call(m):.6f} | "
          f"{m['llm_calls']:,} | {MEASURED} | `{m['run_id']}` |")
    A("")
    outs = [m["output_per_call"] for m in models.values()]
    ins = [m["input_per_call"] for m in models.values()]
    A(f"Input tokens span {max(ins) / min(ins):,.1f}x across models and output "
      f"tokens span {max(outs) / min(outs):,.1f}x {_t(DERIVED)}. Input is a "
      f"property of the prompt and carries across models; output is a property "
      f"of the model and does not. **An output-token figure must never be "
      f"substituted from one model to another.**")
    A("")
    A("### Pricing verification")
    A("")
    pv = measured["price_verification"]
    A(f"The seed DB stores model names in an underscored form that does not "
      f"substring-match the priced slugs in "
      f"`dashboard/backend/infrastructure/llm/token_cost.py`, so the mapping "
      f"is asserted rather than inferred. Each pairing was checked by "
      f"recomputing the run's cost from its own stored token totals and "
      f"comparing against the `est_cost_usd` the run itself stored:")
    A("")
    A(f"- runs checked: {pv['checked']} {_t(MEASURED)}")
    A(f"- largest disagreement: {pv['max_delta_usd']:.2e} USD, against a "
      f"tolerance of {pv['tolerance']:.0e} {_t(DERIVED)}")
    A("")
    A("The loader raises rather than reporting if any run disagrees, so a "
      "drifted price cannot reach this document.")
    A("")

    # ---- 3. serving ------------------------------------------------------
    A("## 3. Measured serving results")
    A("")
    arm_b = serving["arms"].get("armB_shared", {})
    arm_c = serving["arms"].get("armC_shared", {})
    man = arm_b.get("manifest", {})
    A(f"One GPU, one fixture, one pip freeze. Arm B is a plain HuggingFace "
      f"generate loop; Arm C is vLLM.")
    A("")
    A("| Property | Value | Tier |")
    A("|---|---|---|")
    for label, key in (("model", "model"), ("GPU", "gpu_name"),
                       ("GPU UUID", "gpu_uuid"), ("fixture", "fixture_name"),
                       ("context tokens", "context_tokens"),
                       ("max new tokens", "max_new_tokens"),
                       ("CUDA", "cuda_version"), ("driver", "driver_version")):
        A(f"| {label} | `{man.get(key)}` | {MEASURED} |")
    A(f"| fixture sha256 | `{str(man.get('fixture_sha256'))[:16]}...` | "
      f"{MEASURED} |")
    A(f"| pip freeze sha256 | `{str(man.get('pip_freeze_sha256'))[:16]}...` | "
      f"{MEASURED} |")
    A(f"| Arm C engine | `{arm_c.get('engine_kind')}` | {MEASURED} |")
    A("")
    A("### Matched levels")
    A("")
    A("| Concurrency | Arm B req/s | Arm C req/s | Arm C / Arm B | "
      "Arm B e2e p50 s | Arm C e2e p50 s | Tier |")
    A("|---:|---:|---:|---:|---:|---:|---|")
    for lv in serving["matched_levels"]:
        A(f"| {lv['concurrency']} | {lv['arm_b_rps']:.4f} | "
          f"{lv['arm_c_rps']:.4f} | {lv['speedup']:,.1f}x | "
          f"{lv['arm_b_e2e_p50']:.2f} | {lv['arm_c_e2e_p50']:.2f} | "
          f"{MEASURED} (ratio {DERIVED}) |")
    A("")
    b_levels = arm_b.get("levels", {})
    if 1 in b_levels and 32 in b_levels:
        r1 = b_levels[1]["completed_requests_per_s"]
        r32 = b_levels[32]["completed_requests_per_s"]
        A(f"**Arm B throughput inverts.** It falls from {r1:.4f} req/s at "
          f"concurrency one to {r32:.4f} at thirty-two {_t(MEASURED)} — a "
          f"factor of {r1 / r32:,.2f} the wrong way {_t(DERIVED)}. Adding "
          f"concurrent load to the naive loop makes it slower in absolute "
          f"terms, not merely sub-linear.")
        A("")

    if trace.get("available"):
        A("### Where Arm B's time goes")
        A("")
        A(f"From a streamed PyTorch profiler trace at concurrency eight "
          f"(`armB_L3_trace.json`):")
        A("")
        A("| Metric | Value | Tier |")
        A("|---|---:|---|")
        A(f"| kernels launched | {trace['kernel_count']:,} | {MEASURED} |")
        A(f"| GPU busy fraction | {trace['gpu_busy_fraction']:.3f} | "
          f"{MEASURED} |")
        A(f"| GPU idle fraction | {trace['gpu_idle_fraction']:.3f} | "
          f"{MEASURED} |")
        A(f"| distinct CUDA streams | {trace['distinct_stream_count']} | "
          f"{MEASURED} |")
        A(f"| cudaLaunchKernel calls | {trace['launch_calls']:,} | "
          f"{MEASURED} |")
        A(f"| cudaLaunchKernel CPU time (us) | {trace['launch_cpu_us']:,.0f} | "
          f"{MEASURED} |")
        A(f"| GPU busy time (us) | {trace['gpu_busy_us']:,.0f} | {MEASURED} |")
        A("")
        A(f"Launch bookkeeping on the CPU costs "
          f"{trace['launch_cpu_us'] / trace['gpu_busy_us']:,.1f}x the time the "
          f"GPU spends executing {_t(DERIVED)}, across a single stream "
          f"{_t(MEASURED)}. The bottleneck is dispatch, not arithmetic.")
        A("")
        A(f"Carried verbatim from the trace artifact's own caveats "
          f"({len(trace.get('caveats', []))} of them) {_t(MEASURED)}:")
        A("")
        A("```text")
        for c in trace.get("caveats", []):
            A(f"- {c}")
        A("```")
        A("")

    # ---- 4. code findings -----------------------------------------------
    A("## 4. Code findings")
    A("")
    A(f"Static reading of `dashboard/backend`; nothing executed. Every "
      f"citation below is re-read from the file at generation time and this "
      f"document fails to build if one has drifted "
      f"({sum(1 for f in findings if f['verified'])} of {len(findings)} "
      f"verified {_t(MEASURED)}).")
    A("")
    for f in findings:
        A(f"### `{f['file']}:{f['line']}` {_t(MEASURED)}")
        A("")
        A(f"```python")
        A(f"{f['actual']}")
        A("```")
        A("")
        A(f"{f['finding']} {_t(MEASURED)}")
        A("")
        A(f"*Consequence:* {f['consequence']} {_t(DERIVED)}")
        A("")
    A(f"One database fact belongs with these: `backtest_decisions` holds "
      f"{measured['backtest_decisions_rows']} rows across all "
      f"{measured['total_runs']} runs in the seed database {_t(MEASURED)}, "
      f"which is the schema behaving as written rather than a broken run.")
    A("")

    # ---- 5. not measured -------------------------------------------------
    A("## 5. What is NOT measured")
    A("")
    A("Every item here has been kept out of the findings above. Each is "
      "listed with what would resolve it.")
    A("")
    A("### Inputs the cost model needs and this repository does not contain")
    A("")
    A("| Input | Tier | What would resolve it |")
    A("|---|---|---|")
    for key, why in UNMEASURED_INPUTS.items():
        A(f"| `{key}` | {NOT_MEASURED} | {why} |")
    A("")
    A("### Measurements never taken")
    A("")
    A("| Item | Tier | What it would take |")
    A("|---|---|---|")
    for item, need in [
        ("Calls per decision for multi-step pipelines in production",
         "All seed runs used the single-call path. A production run with "
         "metadata.initial_pipeline populated, or the Render DB."),
        ("Retry inflation",
         "Observed calls exceeding configured steps in a real run. The stub "
         "probe cannot produce it — it always parses."),
        ("Real cross-agent prompt overlap",
         "Tokenising real production prompts. The shared-prefix fixture is a "
         "deliberate upper bound and the low-overlap fixture is CONSTRUCTED, "
         "not sampled from production."),
        ("Prefix-cache benefit",
         "The ablation was never run. vLLM reported stats_source: null, so no "
         "hit rate was observed even in the run that had caching enabled."),
        ("Attribution of the Arm C speedup",
         "The arms differ in batching, caching, and engine at once. Decomposing "
         "it needs one-variable-at-a-time runs."),
        ("Occupancy, Tensor Core utilisation, memory bandwidth, SM activity",
         "ncu and nsys were never run. Kernel residency is not occupancy."),
        ("Saturation point",
         "Neither arm was pushed to out-of-memory, so no ceiling was found."),
        ("Every Arm A number",
         "No hosted-API run has ever been executed."),
        ("Decision latency's effect on trading P&L",
         "The engine cannot express it — see finding on execution.py. A "
         "harness-side replay with shifted fills, plus real market data."),
    ]:
        A(f"| {item} | {NOT_MEASURED} | {need} |")
    A("")
    A("### Deliberately excluded from this report")
    A("")
    for item, why in EXCLUDED:
        A(f"- **{item}** — {why}.")
    A("")

    # ---- 6. caveats ------------------------------------------------------
    A("## 6. Caveats")
    A("")
    A("### Serving results are provisional")
    A("")
    A(f"Both serving runs carry a `-dirty` branch SHA "
      f"{_t(MEASURED)}:")
    A("")
    A("| Run | branch_sha | Tier |")
    A("|---|---|---|")
    for arm_key in ("armB_shared", "armC_shared"):
        a = serving["arms"].get(arm_key, {})
        A(f"| `{arm_key}` | `{a.get('branch_sha')}` | {MEASURED} |")
    A("")
    A("`RUN_MANIFEST_SCHEMA.md` treats a dirty result as non-citable: the tree "
      "had uncommitted changes, so the exact code that produced these numbers "
      "cannot be reconstructed from the SHA alone. **Treat every serving "
      "figure in section three as provisional pending a clean-tree re-run.** "
      "The cost figures in section two are unaffected — they come from the "
      "committed database, not from these runs.")
    A("")
    A("### Scope of the serving measurement")
    A("")
    A(f"One GPU, one model, one fixture, one prompt shape {_t(MEASURED)}. "
      f"Nothing here establishes how the result varies across GPUs, model "
      f"sizes, or prompt distributions.")
    A("")
    A("### The seed runs are single-call")
    A("")
    obs = [m["observed_calls_per_decision"] for m in models.values()
           if m["observed_calls_per_decision"]]
    if obs:
        A(f"Observed calls per decision across the seed runs ranges "
          f"{min(obs):.3f} to {max(obs):.3f} {_t(DERIVED)}, using bar count as "
          f"the decision denominator. Every one used the single-call path. A "
          f"multi-step pipeline multiplies this — verified as three steps to "
          f"three calls and five to five against the real runner with a stub "
          f"client {_t(MEASURED)} — but no production multi-step run has ever "
          f"been recorded.")
        A("")
    A("### The dashboard is not orchestration/FinAgents")
    A("")
    A("Every code finding above concerns `dashboard/`, which is what ships. "
      "`orchestration/FinAgents` is a separate tree containing the paper "
      "artifact; it holds transaction-cost and market-impact machinery that "
      "the dashboard does not. Neither imports the other — verified in both "
      "directions. Conflating them has been a recurring error in this project "
      "and no figure here draws on `orchestration/`.")
    A("")

    # ---- 7. questions ----------------------------------------------------
    A("## 7. Questions only you can answer")
    A("")
    A("Set out in full in `QUESTIONS_FOR_ADVISOR.md`. In brief:")
    A("")
    A(f"1. **Production backtest volume and model mix** — the two inputs "
      f"that decide the ${BUDGET:,.0f} {_t(NOT_MEASURED)} answer. Both are "
      f"single queries against the Render database.")
    A("2. **Typical pipeline depth in production** — multiplies every cost "
      "figure by the step count.")
    A("3. **Does paper trading run continuously?** — decides whether live "
      "decisions are a rounding error or a second cost centre.")
    A("4. **Dashboard or orchestration/FinAgents?** — the scoping question "
      "open since the start. They are different systems and the answer "
      "changes what should be measured next.")
    A("")
    A("A companion notebook, `cost_model.ipynb`, takes these inputs and "
      "produces the monthly figure. It refuses to compute while any of them "
      "is blank rather than substituting a default.")
    A("")
    A("---")
    A("")
    A(f"Sources: `{os.path.relpath(measured['db_path'], _BENCH_ROOT)}` "
      f"({measured['total_runs']} runs, {measured['runs_with_llm']} with LLM "
      f"usage) {_t(MEASURED)}; `results/armB_shared_*`, "
      f"`results/armC_shared_*`, `results/armB_L3_trace.json` {_t(MEASURED)}.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate ADVISOR_REPORT.md")
    ap.add_argument("--out", default=os.path.join(_HERE, "ADVISOR_REPORT.md"))
    ap.add_argument("--check", action="store_true",
                    help="build and validate, write nothing")
    args = ap.parse_args(argv)

    measured = load_measured()
    findings = verify_code_findings()
    bad = [f for f in findings if not f["verified"]]
    if bad:
        for f in bad:
            print(f"DRIFTED CITATION {f['file']}:{f['line']} — expected "
                  f"{f['expect']!r}, found {f['actual']!r}", file=sys.stderr)
        return 2

    text = build_report(measured)

    # Refuse to emit a report containing an untagged number. The whole point of
    # the document is that a reader can tell evidence from arithmetic from
    # guesswork, and one untagged figure undermines that for all of them.
    result = check_report(text)
    if not result["ok"]:
        print(f"UNTAGGED NUMERIC LINES ({result['n_violations']}):",
              file=sys.stderr)
        print(format_violations(result["violations"]), file=sys.stderr)
        print(f"rule: {result['rule']}", file=sys.stderr)
        return 3

    if args.check:
        print(f"[check] report builds, {len(text.splitlines())} lines, "
              f"{len(findings)} citations verified, every numeric line tagged")
        return 0
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"[report] {args.out} ({len(text.splitlines())} lines, "
          f"every numeric line tagged)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
