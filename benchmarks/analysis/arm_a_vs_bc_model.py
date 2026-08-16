"""At what agent count does self-hosting beat the hosted API?

    python -m analysis.arm_a_vs_bc_model \
        --arm-b results/armB_shared_summary.json \
        --arm-c results/armC_shared_summary.json \
        --api-cost-per-request 0.000183 \
        --csv-out analysis/cost_model.csv

THE MODEL
---------
A fleet of ``N`` agents each makes ``R`` decisions per day. One decision is one
request at the measured context and output length.

* **Self-hosted (arms B and C)** rents whole GPUs. Throughput per GPU comes from
  the measured ``completed_requests_per_s`` at the highest matched concurrency.
  A GPU costs the same whether it is saturated or idle, so cost is a **step
  function**: ``ceil(demand / capacity)`` GPUs, billed 24h/day.
* **Hosted API (arm A)** has no floor and no ceiling in this model — cost is
  purely ``N x R x cost_per_request``, a straight line through the origin.

A line through the origin and a step function starting above it cross exactly
once, and that crossing is the answer. Below it the API is cheaper because a
rented GPU sits mostly idle; above it the GPU wins because its marginal cost is
zero until the next one is needed.

WHAT THIS MODEL DELIBERATELY OMITS
-----------------------------------
Named because a cost model that hides its exclusions is worse than no model:

* **Engineering and operational cost.** Self-hosting means owning deploys,
  failures, upgrades and on-call. That is usually the dominant term at small N
  and it is not priced here.
* **Rate limits.** The API line assumes you can buy unbounded throughput. You
  cannot — see ``rate_limit_probe.py``. Above the measured ceiling the API
  option is not merely expensive, it is unavailable at any price without a
  contract change.
* **Latency.** Cost parity is not user-experience parity. A cheaper option that
  misses the market window is not cheaper.
* **Reliability and cold starts.** No availability or restart cost is modelled.

The crossover is therefore a *lower bound* on the agent count that justifies
self-hosting: the true break-even sits higher once operational cost is included.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.loader import load_run  # noqa: E402

__all__ = ["throughput_per_gpu", "model_costs", "find_crossover",
           "sensitivity", "lever_sensitivity", "cost_per_decision",
           "MEASURED_OUTPUT_TOKENS", "per_model_cost_table",
           "to_csv", "format_report"]

# USD per million tokens, for the lever table. Model choice spans ~100x, which
# is why it is listed first: it dominates every other lever combined.
MODEL_PRICES = {
    "nvidia/nemotron-3-nano-30b-a3b": (0.05, 0.20),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "qwen/qwen3.7-plus": (0.40, 1.60),
    "google/gemini-3.1-pro": (2.0, 12.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "openai/gpt-5.5": (5.0, 30.0),
}

# Mean output tokens per call, measured per model from the seed DB's seven
# leaderboard runs (~161 calls each) under an IDENTICAL prompt. Input is
# recorded beside it for the contrast: input spans 1.4x across these models and
# output spans 5.8x, which is why one shared output length understates the
# per-model cost spread badly.
#
# The DB records the model as an underscored name (`nemotron_3_nano_30b`) that
# does NOT substring-match token_cost.py's priced slugs, so five of the seven
# fall through to that module's default (1.0, 5.0) if re-derived from the name
# today. The mapping below was therefore not read off the name: each pairing was
# verified by recomputing the run's cost from its stored token totals and
# checking it against the stored `est_cost_usd` — all seven agree to <1e-6 USD.
MEASURED_OUTPUT_TOKENS = {
    "nvidia/nemotron-3-nano-30b-a3b": {
        "db_model": "nemotron_3_nano_30b", "output": 860.2, "input": 5501.8,
        "run_id": "lb_nemotron_3_nano_30b_20260415_20260515"},
    "anthropic/claude-sonnet-4-6": {
        "db_model": "claude_sonnet_4_6", "output": 1042.6, "input": 5016.9,
        "run_id": "lb_claude_sonnet_4_6_20260415_20260515"},
    "anthropic/claude-haiku-4-5": {
        "db_model": "claude_haiku_4_5", "output": 1157.0, "input": 4933.7,
        "run_id": "lb_claude_haiku_4_5_20260415_20260515"},
    "openai/gpt-5.5": {
        "db_model": "gpt_5_5", "output": 2226.2, "input": 3895.1,
        "run_id": "lb_gpt_5_5_20260415_20260515"},
    "deepseek/deepseek-v4-pro": {
        "db_model": "deepseek_v4_pro", "output": 3395.6, "input": 4005.1,
        "run_id": "lb_deepseek_v4_pro_20260415_20260515"},
    "qwen/qwen3.7-plus": {
        "db_model": "qwen3_7_plus", "output": 4879.2, "input": 5224.2,
        "run_id": "lb_qwen3_7_plus_20260415_20260515"},
    "google/gemini-3.1-pro": {
        "db_model": "gemini_3_1_pro_preview", "output": 5004.5, "input": 4950.3,
        "run_id": "lb_gemini_3_1_pro_preview_20260415_20260515"},
}

# ATL's production default. Named rather than inferred, because the interesting
# result below is that the default is the cheapest model on BOTH axes at once.
PRODUCTION_DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

# The calls-per-decision figure used when nothing has been measured. Kept as a
# named constant so it can never be mistaken for a measurement in the output.
ILLUSTRATIVE_CALLS_PER_DECISION = 3.0


def cost_per_decision(
    calls_per_decision: float,
    input_tokens_per_call: float,
    output_tokens_per_call: float,
    price_in_per_m: float,
    price_out_per_m: float,
    prefix_cache_hit_rate: float = 0.0,
    cached_input_discount: float = 0.1,
) -> float:
    """USD for one agent decision.

    A decision is ``calls_per_decision`` LLM calls, not one — ATL's pipeline
    issues one call per configured step, sequentially. Getting this wrong scales
    the entire cost model linearly.

    ``prefix_cache_hit_rate`` discounts only the INPUT side, and only the cached
    fraction: output tokens are generated fresh every time and never cached.
    """
    hit = max(0.0, min(1.0, prefix_cache_hit_rate))
    effective_in = input_tokens_per_call * (
        (1.0 - hit) + hit * cached_input_discount
    )
    per_call = (effective_in / 1e6) * price_in_per_m + (
        output_tokens_per_call / 1e6) * price_out_per_m
    return per_call * calls_per_decision


def per_model_cost_table(
    *,
    calls_per_decision: float,
    calls_are_measured: bool,
    input_tokens_per_call: Optional[float] = None,
    prefix_cache_hit_rate: float = 0.0,
    shared_output_tokens: Optional[float] = None,
) -> Dict[str, Any]:
    """Cost per decision per model, using each model's OWN measured output length.

    Model choice moves cost through two effects that MULTIPLY:

    * **price** — $0.05/$0.20 per M for Nemotron against $5/$30 for GPT-5.5,
      about 134x on the output side.
    * **verbosity** — 860 output tokens against 5,005 under the *same* prompt,
      about 5.8x.

    Applying one shared output length across every model — which is what the
    lever table does — captures only the first and understates the true spread.
    Both are priced here from the same measurement.

    ``input_tokens_per_call`` defaults per-model to that model's own measured
    input, which varies only 1.4x and is a property of the prompt rather than
    the model. Passing a value overrides all of them with a single figure, which
    is the right choice when modelling one fixed prompt across candidate models.

    ``shared_output_tokens``, when given, is also costed for every model so the
    understatement is visible as a ratio rather than asserted.
    """
    rows: List[Dict[str, Any]] = []
    for slug, m in MEASURED_OUTPUT_TOKENS.items():
        price = MODEL_PRICES.get(slug)
        if not price:
            continue
        pin, pout = price
        in_tok = input_tokens_per_call if input_tokens_per_call is not None else m["input"]
        own = cost_per_decision(calls_per_decision, in_tok, m["output"],
                                pin, pout,
                                prefix_cache_hit_rate=prefix_cache_hit_rate)
        row: Dict[str, Any] = {
            "model": slug,
            "db_model": m["db_model"],
            "price_in_per_m": pin,
            "price_out_per_m": pout,
            "input_tokens_per_call": in_tok,
            "output_tokens_per_call": m["output"],
            "output_tokens_source": f"measured, {m['run_id']}",
            "cost_per_decision": own,
        }
        if shared_output_tokens is not None:
            shared = cost_per_decision(calls_per_decision, in_tok,
                                       shared_output_tokens, pin, pout,
                                       prefix_cache_hit_rate=prefix_cache_hit_rate)
            row["cost_per_decision_shared_output"] = shared
            # >1: the shared figure UNDERstates this model (it is more verbose
            # than the shared length). <1: it OVERstates. Naming it for one
            # direction only would mislabel half the table.
            row["own_over_shared_factor"] = (own / shared) if shared else None
        rows.append(row)
    rows.sort(key=lambda r: r["cost_per_decision"])

    costs = [r["cost_per_decision"] for r in rows]
    span = (max(costs) / min(costs)) if costs and min(costs) else None
    prices_out = [r["price_out_per_m"] for r in rows]
    outputs = [r["output_tokens_per_call"] for r in rows]
    price_span = (max(prices_out) / min(prices_out)) if prices_out and min(prices_out) else None
    verbosity_span = (max(outputs) / min(outputs)) if outputs and min(outputs) else None

    default_row = next(
        (r for r in rows if r["model"] == PRODUCTION_DEFAULT_MODEL), None)

    return {
        "rows": rows,
        "calls_per_decision": calls_per_decision,
        "calls_per_decision_basis": (
            "MEASURED" if calls_are_measured else "ILLUSTRATIVE — not measured"),
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
        "cost_span_factor": span,
        "output_price_span_factor": price_span,
        "verbosity_span_factor": verbosity_span,
        "compounding_note": (
            f"Output price spans {price_span:.0f}x and verbosity spans "
            f"{verbosity_span:.1f}x. They multiply, so cost per decision spans "
            f"{span:.0f}x — wider than either effect alone."
            if (span and price_span and verbosity_span) else None
        ),
        "production_default": (
            {
                "model": default_row["model"],
                "cost_per_decision": default_row["cost_per_decision"],
                "rank": rows.index(default_row) + 1,
                "of": len(rows),
                "note": (
                    "ATL's default is both the cheapest per token AND the least "
                    "verbose, so the two effects compound in its favour rather "
                    "than cancelling. Any move off it pays twice."
                ),
            } if default_row else None
        ),
        "caveats": [
            "Output lengths are per-model measurements from ONE run each "
            "(~161 calls). They are means; the per-call spread was destroyed "
            "by summation before storage (see atl_token_extract.py).",
            "Verbosity was measured under ATL's prompt. A different prompt, or "
            "a lower llm_max_output_tokens cap, moves these numbers.",
            "Prices are list rates and change without notice.",
        ],
    }


def lever_sensitivity(
    *,
    baseline_calls: float,
    baseline_input: float,
    baseline_output: float,
    price_in: float,
    price_out: float,
    calls_range=(1, 3, 6),
    input_range=(2620, 4790, 20000),
    hit_rates=(0.0, 0.5, 0.9),
) -> Dict[str, Any]:
    """Rank the levers by how far each moves cost per decision.

    Ordered by measured effect size rather than by intuition. Model choice
    spanning ~100x while prefix caching saves at most the input side is itself
    the finding: optimising the cache before checking the model is optimising
    the smaller term.
    """
    base = cost_per_decision(baseline_calls, baseline_input, baseline_output,
                             price_in, price_out)

    def _span(values):
        lo, hi = min(values), max(values)
        return (hi / lo) if lo else None

    model_costs_ = {
        name: cost_per_decision(baseline_calls, baseline_input, baseline_output,
                                pin, pout)
        for name, (pin, pout) in MODEL_PRICES.items()
    }
    call_costs = {
        n: cost_per_decision(n, baseline_input, baseline_output, price_in, price_out)
        for n in calls_range
    }
    input_costs = {
        n: cost_per_decision(baseline_calls, n, baseline_output, price_in, price_out)
        for n in input_range
    }
    cache_costs = {
        h: cost_per_decision(baseline_calls, baseline_input, baseline_output,
                             price_in, price_out, prefix_cache_hit_rate=h)
        for h in hit_rates
    }

    levers = [
        {"lever": "model choice", "values": model_costs_,
         "span_factor": _span(list(model_costs_.values()))},
        {"lever": "calls per decision", "values": call_costs,
         "span_factor": _span(list(call_costs.values()))},
        {"lever": "input tokens per call", "values": input_costs,
         "span_factor": _span(list(input_costs.values()))},
        {"lever": "prefix cache hit rate", "values": cache_costs,
         "span_factor": _span(list(cache_costs.values()))},
    ]
    levers.sort(key=lambda l: l["span_factor"] or 0, reverse=True)
    return {
        "baseline_cost_per_decision": base,
        "levers": levers,
        "note": (
            "Span factor is max/min cost across the values tried for that lever "
            "alone. Model choice dominating the list means a cache or a shorter "
            "prompt cannot rescue an expensive model — pick the model first."
        ),
    }

DEFAULT_AGENT_COUNTS = (1, 10, 100, 500, 1000)
DEFAULT_GPU_USD_PER_HOUR = 2.0
DEFAULT_PRICE_IN = 0.05
DEFAULT_PRICE_OUT = 0.20
DEFAULT_DECISIONS_PER_AGENT_PER_DAY = 390  # one per minute over a 6.5h session
SECONDS_PER_DAY = 86400.0


def throughput_per_gpu(run, concurrency: Optional[int] = None) -> Dict[str, Any]:
    """Requests/sec for one GPU, from the measured sweep.

    Taken at the highest measured concurrency, which is where a self-hosted
    deployment would actually be run — quoting a C=1 figure would understate a
    batching runtime by more than an order of magnitude.
    """
    levels = run.concurrencies()
    if not levels:
        return {"available": False, "reason": f"{run.run_id} has no levels"}
    c = concurrency if concurrency in levels else levels[-1]
    lv = run.level(c) or {}
    rps = lv.get("completed_requests_per_s")
    if not rps:
        return {"available": False,
                "reason": f"{run.run_id} c={c} has no completed_requests_per_s"}
    return {
        "available": True,
        "run_id": run.run_id,
        "arm": run.arm,
        "concurrency": c,
        "requests_per_s": rps,
        "requests_per_day": rps * SECONDS_PER_DAY,
        "wall_s_for_level": lv.get("wall_time"),
    }


def model_costs(
    agent_counts: Sequence[int],
    *,
    api_cost_per_request: float,
    decisions_per_agent_per_day: int,
    gpu_usd_per_hour: float,
    arm_b_rps: Optional[float],
    arm_c_rps: Optional[float],
    context_sync_cost_per_day: float = 0.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in agent_counts:
        demand = n * decisions_per_agent_per_day  # decisions/day
        row: Dict[str, Any] = {
            "agents": n,
            "decisions_per_day": demand,
            "api_cost_per_day": demand * api_cost_per_request
            + context_sync_cost_per_day,
            "context_sync_cost_per_day": context_sync_cost_per_day,
        }
        for label, rps in (("arm_b", arm_b_rps), ("arm_c", arm_c_rps)):
            if not rps:
                row[f"{label}_gpus"] = None
                row[f"{label}_cost_per_day"] = None
                continue
            capacity = rps * SECONDS_PER_DAY
            # A GPU is indivisible and bills whether busy or not: the step is
            # the whole point of the model.
            gpus = max(1, math.ceil(demand / capacity))
            row[f"{label}_gpus"] = gpus
            row[f"{label}_cost_per_day"] = gpus * 24.0 * gpu_usd_per_hour
            row[f"{label}_utilisation"] = demand / (gpus * capacity)
        for key in ("api", "arm_b", "arm_c"):
            cost = row.get(f"{key}_cost_per_day")
            row[f"{key}_cost_per_agent_per_day"] = (cost / n) if cost and n else None
        rows.append(row)
    return rows


def find_crossover(
    *,
    api_cost_per_request: float,
    decisions_per_agent_per_day: int,
    gpu_usd_per_hour: float,
    rps: Optional[float],
    context_sync_cost_per_day: float = 0.0,
    max_agents: int = 10_000_000,
) -> Dict[str, Any]:
    """Smallest N where the self-hosted option becomes cheaper.

    Solved directly for the first GPU rather than searched: while one GPU
    suffices, self-hosting costs a constant ``24 * rate`` and the API costs
    ``N * R * price``, so

        N* = (24 * rate - sync) / (R * price)

    If that N* exceeds what one GPU can serve, the answer is checked again at
    each successive step; the closed form is only valid within a step.
    """
    if not rps:
        return {"available": False, "reason": "no measured throughput"}
    per_agent_per_day = decisions_per_agent_per_day * api_cost_per_request
    if per_agent_per_day <= 0:
        return {"available": False, "reason": "api cost per agent-day is zero"}

    capacity = rps * SECONDS_PER_DAY
    gpus = 1
    while gpus * capacity <= max_agents * decisions_per_agent_per_day:
        gpu_cost = gpus * 24.0 * gpu_usd_per_hour
        n_star = (gpu_cost - context_sync_cost_per_day) / per_agent_per_day
        n_max_for_step = (gpus * capacity) / decisions_per_agent_per_day
        n_min_for_step = ((gpus - 1) * capacity) / decisions_per_agent_per_day
        if n_min_for_step <= n_star <= n_max_for_step:
            return {
                "available": True,
                "crossover_agents": math.ceil(n_star),
                "crossover_agents_exact": n_star,
                "gpus_at_crossover": gpus,
                "gpu_capacity_decisions_per_day": capacity,
                "api_cost_per_agent_per_day": per_agent_per_day,
                "interpretation": (
                    f"Below ~{math.ceil(n_star)} agents the API is cheaper — a "
                    f"rented GPU would sit mostly idle. Above it, self-hosting "
                    f"wins because the GPU's marginal cost is zero until the "
                    f"next one is needed."
                ),
            }
        gpus += 1
        if gpus > 10000:
            break

    # No step contained a crossing. Two very different reasons for that, and
    # collapsing them into "not found" hides the more interesting one: a stack
    # whose per-agent cost never falls below the API's is never worth
    # self-hosting at ANY scale, which is a finding rather than a gap.
    probe_n = 1_000_000
    gpus_needed = max(1, math.ceil(
        probe_n * decisions_per_agent_per_day / capacity))
    self_hosted_per_agent = (gpus_needed * 24.0 * gpu_usd_per_hour) / probe_n
    if self_hosted_per_agent > per_agent_per_day:
        return {
            "available": False,
            "never_cheaper": True,
            "reason": "self-hosting is never cheaper at any agent count",
            "self_hosted_cost_per_agent_per_day": self_hosted_per_agent,
            "api_cost_per_agent_per_day": per_agent_per_day,
            "cost_ratio": self_hosted_per_agent / per_agent_per_day,
            "interpretation": (
                f"Throughput is low enough that serving one agent-day costs "
                f"${self_hosted_per_agent:.4f} on this stack against "
                f"${per_agent_per_day:.4f} through the API — a factor of "
                f"{self_hosted_per_agent / per_agent_per_day:.0f}. Adding "
                f"agents adds GPUs at roughly the same rate, so the lines "
                f"never cross. This stack does not justify self-hosting at any "
                f"scale; the batching runtime is what changes that."
            ),
        }
    return {"available": False, "never_cheaper": False,
            "reason": "no crossover within the searched range"}


def sensitivity(
    base: Dict[str, Any],
    *,
    api_cost_per_request: float,
    decisions_per_agent_per_day: int,
    gpu_usd_per_hour: float,
    rps: Optional[float],
) -> List[Dict[str, Any]]:
    """How the crossover moves when the two prices are wrong.

    Both inputs are volatile — provider pricing changes without notice and GPU
    spot rates swing — so a single crossover number is a false precision. These
    are the scenarios that change the decision.
    """
    scenarios = [
        ("baseline", api_cost_per_request, gpu_usd_per_hour),
        ("API price x2", api_cost_per_request * 2, gpu_usd_per_hour),
        ("API price x0.5", api_cost_per_request * 0.5, gpu_usd_per_hour),
        ("GPU $3/hr", api_cost_per_request, 3.0),
        ("GPU $1/hr (spot)", api_cost_per_request, 1.0),
        ("API x2 + GPU $3/hr", api_cost_per_request * 2, 3.0),
    ]
    out = []
    for name, api_price, gpu_rate in scenarios:
        cross = find_crossover(
            api_cost_per_request=api_price,
            decisions_per_agent_per_day=decisions_per_agent_per_day,
            gpu_usd_per_hour=gpu_rate, rps=rps,
        )
        out.append({
            "scenario": name,
            "api_cost_per_request": api_price,
            "gpu_usd_per_hour": gpu_rate,
            "crossover_agents": cross.get("crossover_agents"),
            "available": cross.get("available", False),
        })
    return out


def to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    keys = list(rows[0].keys())
    w = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return out.getvalue()


def format_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    A = lines.append
    A("=" * 88)
    A("COST MODEL — hosted API vs self-hosted, per day")
    A("=" * 88)

    inp = result["inputs"]
    A(f"  API cost/request        ${inp['api_cost_per_request']:.6f}")
    A(f"  decisions/agent/day     {inp['decisions_per_agent_per_day']}")
    A(f"  GPU rate                ${inp['gpu_usd_per_hour']:.2f}/hr")
    for label in ("arm_b", "arm_c"):
        t = result["throughput"].get(label) or {}
        if t.get("available"):
            A(f"  {label} throughput        {t['requests_per_s']:.3f} req/s "
              f"(C={t['concurrency']}, {t['requests_per_day']:,.0f}/day/GPU)")
        else:
            A(f"  {label} throughput        unavailable ({t.get('reason')})")

    A("")
    A(f"  {'agents':>7} {'decisions/day':>14} {'API $/day':>12} "
      f"{'armB $/day':>12} {'armB GPUs':>10} {'armC $/day':>12} {'armC GPUs':>10}")
    A("  " + "-" * 84)
    for r in result["rows"]:
        A(f"  {r['agents']:>7} {r['decisions_per_day']:>14,} "
          f"{r['api_cost_per_day']:>12.2f} "
          f"{_num(r.get('arm_b_cost_per_day')):>12} "
          f"{_num(r.get('arm_b_gpus'), 0):>10} "
          f"{_num(r.get('arm_c_cost_per_day')):>12} "
          f"{_num(r.get('arm_c_gpus'), 0):>10}")

    A("")
    for label in ("arm_b", "arm_c"):
        cross = result["crossover"].get(label) or {}
        if cross.get("available"):
            A(f"  CROSSOVER vs {label}: ~{cross['crossover_agents']:,} agents "
              f"({cross['gpus_at_crossover']} GPU(s))")
            A(f"    {cross['interpretation']}")
        elif cross.get("never_cheaper"):
            A(f"  CROSSOVER vs {label}: NEVER — self-hosting on this stack is "
              f"{cross['cost_ratio']:.0f}x the API cost per agent-day, at any scale")
            A(f"    {cross['interpretation']}")
        else:
            A(f"  CROSSOVER vs {label}: none found ({cross.get('reason')})")

    A("")
    A("  Sensitivity (crossover agent count):")
    A(f"    {'scenario':<22} {'API $/req':>12} {'GPU $/hr':>10} {'crossover':>12}")
    for s in result["sensitivity"]:
        A(f"    {s['scenario']:<22} {s['api_cost_per_request']:>12.6f} "
          f"{s['gpu_usd_per_hour']:>10.2f} "
          f"{(str(s['crossover_agents']) if s['crossover_agents'] else '—'):>12}")

    lv = result.get("levers")
    if lv:
        A("")
        A(f"  Levers, by effect size (baseline "
          f"${lv['baseline_cost_per_decision']:.6f}/decision):")
        for l in lv["levers"]:
            span = l["span_factor"]
            A(f"    {l['lever']:<24} span {span:>7.1f}x" if span
              else f"    {l['lever']:<24} span      —")
        A(f"    {lv['note']}")

    pm = result.get("per_model")
    if pm and pm.get("rows"):
        A("")
        A(f"  COST PER DECISION BY MODEL — each model's OWN measured output length")
        A(f"    calls/decision: {pm['calls_per_decision']:g} "
          f"[{pm['calls_per_decision_basis']}]")
        shared = any("cost_per_decision_shared_output" in r for r in pm["rows"])
        head = (f"    {'model':<32}{'$/M in':>8}{'$/M out':>9}"
                f"{'out tok':>9}{'$/decision':>13}")
        if shared:
            head += f"{'vs shared':>11}"
        A(head)
        A("    " + "-" * (len(head) - 4))
        for r in pm["rows"]:
            line = (f"    {r['model']:<32}{r['price_in_per_m']:>8.3f}"
                    f"{r['price_out_per_m']:>9.2f}"
                    f"{r['output_tokens_per_call']:>9,.0f}"
                    f"{r['cost_per_decision']:>13.6f}")
            if shared and r.get("own_over_shared_factor"):
                line += f"{r['own_over_shared_factor']:>10.2f}x"
            A(line)
        if pm.get("compounding_note"):
            A(f"    {pm['compounding_note']}")
        pd_ = pm.get("production_default")
        if pd_:
            A(f"    ATL default {pd_['model']} ranks {pd_['rank']} of "
              f"{pd_['of']} at ${pd_['cost_per_decision']:.6f}/decision.")
            A(f"    {pd_['note']}")
        if shared:
            A("    'vs shared' = own measured output length over one shared "
              "length: >1 the shared figure UNDERstates this model, <1 it "
              "OVERstates. A single shared length is wrong in both directions "
              "at once, which is why the span it reports is too narrow.")
        for c in pm["caveats"]:
            A(f"    - {c}")

    A("")
    A("  OMITTED — and these move the answer:")
    for note in result["omissions"]:
        A(f"    - {note}")
    return "\n".join(lines)


def _num(value, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}" if decimals else f"{value:,}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Arm A vs B/C cost model.")
    ap.add_argument("--arm-b", default=None, help="arm B *_summary.json")
    ap.add_argument("--arm-c", default=None, help="arm C *_summary.json")
    ap.add_argument("--arm-a", default=None,
                    help="arm A *_summary.json; its measured cost/request is "
                         "used in preference to --api-cost-per-request")
    ap.add_argument("--api-cost-per-request", type=float, default=None)
    ap.add_argument("--calls-per-decision", type=float, default=1.0,
                    help="LLM calls per agent decision. ATL's pipeline issues "
                         "one per configured step — see atl_pipeline_audit.py. "
                         "The default of 1 is the ORIGINAL ASSUMPTION.")
    ap.add_argument("--calls-per-decision-measured", action="store_true",
                    help="assert that --calls-per-decision came from a "
                         "measurement (atl_token_extract.py). Without this "
                         "flag the per-model table labels the value "
                         "ILLUSTRATIVE, so an assumed number can never be read "
                         "as a measured one.")
    ap.add_argument("--per-model-calls-per-decision", type=float, default=None,
                    help="calls/decision for the per-model table only. "
                         f"Defaults to the illustrative "
                         f"{ILLUSTRATIVE_CALLS_PER_DECISION:g}.")
    ap.add_argument("--per-model-shared-output", type=float, default=None,
                    help="also cost every model at this one shared output "
                         "length, to show how much a single shared figure "
                         "understates the per-model spread.")
    ap.add_argument("--input-tokens", type=float, default=None,
                    help="measured input tokens per call (atl_token_extract.py)")
    ap.add_argument("--output-tokens", type=float, default=None,
                    help="measured output tokens per call")
    ap.add_argument("--prefix-cache-hit-rate", type=float, default=0.0)
    ap.add_argument("--decisions-per-agent-per-day", type=int,
                    default=DEFAULT_DECISIONS_PER_AGENT_PER_DAY)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=DEFAULT_GPU_USD_PER_HOUR)
    ap.add_argument("--context-sync-cost-per-day", type=float, default=0.0)
    ap.add_argument("--agents", type=int, nargs="*", default=list(DEFAULT_AGENT_COUNTS))
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    tput: Dict[str, Any] = {}
    for label, path in (("arm_b", args.arm_b), ("arm_c", args.arm_c)):
        tput[label] = (throughput_per_gpu(load_run(path), args.concurrency)
                       if path else {"available": False, "reason": "not supplied"})

    api_cost = args.api_cost_per_request
    api_source = "--api-cost-per-request"
    if args.arm_a:
        run_a = load_run(args.arm_a)
        levels = run_a.concurrencies()
        for c in reversed(levels):
            note = (run_a.level(c) or {}).get("notes") or {}
            if note.get("api_cost_per_request"):
                api_cost = note["api_cost_per_request"]
                api_source = f"measured, {run_a.run_id} C={c}"
                break
    # Measured tokens override a flat per-request price, and the call count
    # multiplies whichever is used.
    assumed_cost = api_cost
    if args.input_tokens is not None and args.output_tokens is not None:
        api_cost = cost_per_decision(
            args.calls_per_decision, args.input_tokens, args.output_tokens,
            DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT,
            prefix_cache_hit_rate=args.prefix_cache_hit_rate)
        api_source = (f"measured tokens x {args.calls_per_decision} calls/decision")
    elif api_cost is not None and args.calls_per_decision != 1.0:
        api_cost = api_cost * args.calls_per_decision
        api_source += f" x {args.calls_per_decision} calls/decision"

    if api_cost is None:
        print("ERROR: supply --api-cost-per-request, --arm-a, or "
              "--input-tokens/--output-tokens. The model will not invent a "
              "price.", file=sys.stderr)
        return 2

    rows = model_costs(
        args.agents,
        api_cost_per_request=api_cost,
        decisions_per_agent_per_day=args.decisions_per_agent_per_day,
        gpu_usd_per_hour=args.gpu_usd_per_hour,
        arm_b_rps=(tput["arm_b"] or {}).get("requests_per_s"),
        arm_c_rps=(tput["arm_c"] or {}).get("requests_per_s"),
        context_sync_cost_per_day=args.context_sync_cost_per_day,
    )

    crossover = {
        label: find_crossover(
            api_cost_per_request=api_cost,
            decisions_per_agent_per_day=args.decisions_per_agent_per_day,
            gpu_usd_per_hour=args.gpu_usd_per_hour,
            rps=(tput[label] or {}).get("requests_per_s"),
            context_sync_cost_per_day=args.context_sync_cost_per_day,
        )
        for label in ("arm_b", "arm_c")
    }

    result = {
        "inputs": {
            "api_cost_per_request": api_cost,
            "api_cost_source": api_source,
            "decisions_per_agent_per_day": args.decisions_per_agent_per_day,
            "gpu_usd_per_hour": args.gpu_usd_per_hour,
            "context_sync_cost_per_day": args.context_sync_cost_per_day,
        },
        "throughput": tput,
        "rows": rows,
        "crossover": crossover,
        "assumed_cost_per_request_one_call": assumed_cost,
        "per_model": per_model_cost_table(
            # The per-model table takes its call count from D2 when one was
            # measured; otherwise it uses the illustrative 3 and SAYS SO in the
            # basis field, which is printed beside the number.
            calls_per_decision=(
                args.per_model_calls_per_decision
                if args.per_model_calls_per_decision is not None
                else (args.calls_per_decision if args.calls_per_decision_measured
                      else ILLUSTRATIVE_CALLS_PER_DECISION)),
            calls_are_measured=(
                args.calls_per_decision_measured
                and args.per_model_calls_per_decision is None),
            input_tokens_per_call=args.input_tokens,
            prefix_cache_hit_rate=args.prefix_cache_hit_rate,
            shared_output_tokens=args.per_model_shared_output,
        ),
        "levers": lever_sensitivity(
            baseline_calls=args.calls_per_decision,
            baseline_input=args.input_tokens or 2620.0,
            baseline_output=args.output_tokens or 256.0,
            price_in=DEFAULT_PRICE_IN, price_out=DEFAULT_PRICE_OUT,
        ),
        "sensitivity": sensitivity(
            {}, api_cost_per_request=api_cost,
            decisions_per_agent_per_day=args.decisions_per_agent_per_day,
            gpu_usd_per_hour=args.gpu_usd_per_hour,
            rps=(tput["arm_c"] or {}).get("requests_per_s"),
        ),
        "omissions": [
            "Engineering and on-call cost of self-hosting — usually dominant "
            "at small N, not priced here. The crossover is a LOWER bound.",
            "Rate limits: the API line assumes unbounded purchasable "
            "throughput. Above the measured ceiling it is unavailable at any "
            "price without a contract change (see rate_limit_probe.py).",
            "Latency: cost parity is not user-experience parity.",
            "Reliability, cold starts, and failover.",
        ],
    }

    print(format_report(result))
    if args.csv_out:
        with open(args.csv_out, "w") as fh:
            fh.write(to_csv(rows))
        print(f"\n[model] {args.csv_out}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[model] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
