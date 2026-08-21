"""How far the hosted API bill can be pushed, without changing the decision model.

Four levers, each costed from measured inputs, each reported with what it does
to behaviour as well as to cost. Then the STACKED ceiling, and what that still
leaves.

  2a  per-step routing        mechanical steps to a cheap model
  2b  output token caps       output is priced ~4x input
  2c  eliminated calls        backtest caching, leaderboard governance
  2d  dropdown composition    models users cannot select cost nothing

THE ORDERING OF THE ANSWER IS THE POINT
-----------------------------------------
2a-2c are engineering. Stacked, they reach some fraction of the bill. 2d is not
an optimisation at all — it is a product decision — and it is the only one that
reaches the right order of magnitude, because the per-call span across the
roster is 193x and every engineering lever put together is well under 3x.

WHAT IS DELIBERATELY ABSENT
-----------------------------
* **Prefix caching.** Measured at zero cached tokens over 160 requests against
  a 99.4% shared-prefix fixture. No saving modelled.
* **Sampling / staggering / skipping bars.** Experiment-only; on a user's agent
  it is indistinguishable from the product being broken.
* **Self-hosting.** Different experiment, different units.
* **Any fallback-rate adjustment.** Unrecoverable from stored data.

REFUSAL
--------
Every projection here takes measured inputs or refuses. In particular, routing
savings are computed from measured token counts and measured parse rates, and
the completion-rate effect is withheld where the measurement does not exist —
which, for the decision step, it does not.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, MissingInput, NOT_MEASURED, Fig, cost_per_call,
    load_measured,
)

__all__ = [
    "PHASE16_RESULTS", "load_step_measurements", "rule_of_three_upper_bound",
    "routing_projection", "cap_projection", "eliminated_calls",
    "dropdown_composition", "stacked_ceiling", "format_optimisation_report",
]

PHASE16_RESULTS = os.path.join(_BENCH_ROOT, "results",
                               "step_reliability_phase16.json")


# --------------------------------------------------------------------------
# measurement loading
# --------------------------------------------------------------------------


def load_step_measurements(path: str = PHASE16_RESULTS) -> Dict[str, Any]:
    """Per-model per-step token counts and parse rates from the Phase 16 run.

    Step 3 is EXCLUDED from the usable set. Both models scored 0/15 there, but
    that is the empty-orders defect firing against a content-free reference
    context, not a statement about either model. Using it would attribute a
    pipeline defect to a model and would sink any routing projection for the
    wrong reason.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    usable, excluded = {}, {}
    for r in raw["results"]:
        key = (r["model"], r["step_index"])
        rec = {"model": r["model"], "step_index": r["step_index"],
               "label": r["step_label"], "n": r["n_attempts"],
               "n_passed": r["n_passed"], "parse_rate": r["parse_rate"],
               "output_tokens": r["mean_output_tokens"],
               "input_tokens": r["mean_input_tokens"],
               "categories": r["categories"]}
        (excluded if r["is_last"] else usable)[key] = rec
    return {"usable": usable, "excluded": excluded,
            "provenance": raw.get("_provenance", {}),
            "attempts_per_cell": raw["attempts"]}


def rule_of_three_upper_bound(n: int) -> float:
    """95% upper bound on a failure rate after n trials with zero failures.

    3/n. This is the honest reading of "15/15 parsed": it bounds the failure
    rate at ~18.5%, not at zero. Reporting 100% from n=15 and then deploying at
    156 decisions/day would be the whole error this project exists to avoid.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


# --------------------------------------------------------------------------
# 2a — per-step routing
# --------------------------------------------------------------------------


def routing_projection(baseline_model: str, cheap_model: str,
                       measurements: Optional[Dict[str, Any]] = None,
                       measured: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """Cost of routing the mechanical steps to a cheap model.

    Uses MEASURED per-step output tokens for both models. The compound effect
    is computed rather than assumed, because the measurement contradicts the
    obvious direction: Nemotron emitted MORE tokens at step 1 than deepseek
    (1458 vs 1197), so routing step 1 to the cheaper model INFLATES the input
    of every later step.
    """
    measurements = measurements or load_step_measurements()
    measured = measured or load_measured()
    usable = measurements["usable"]

    prices = {}
    for db_name, m in measured["models"].items():
        prices[m["slug"]] = (m["price_in"], m["price_out"])
    for slug in (baseline_model, cheap_model):
        if slug not in prices:
            raise MissingInput(
                f"no measured price for {slug}. Known: {sorted(prices)}")

    steps = sorted({k[1] for k in usable})
    if not steps:
        raise MissingInput("no usable step measurements")

    def _series(model):
        out = []
        for i in steps:
            rec = usable.get((model, i))
            if rec is None:
                raise MissingInput(
                    f"no measured output tokens for {model} at step {i + 1}. "
                    f"A routing saving cannot be projected from an unmeasured "
                    f"step.")
            out.append(rec)
        return out

    base_series, cheap_series = _series(baseline_model), _series(cheap_model)

    # json.dumps(prior_outputs, indent=2) wrapper overhead per upstream entry.
    entry_overhead = 25.0

    def _cost(series, model_by_step):
        upstream, total, rows = 0.0, 0.0, []
        for i, rec in enumerate(series):
            slug = model_by_step[i]
            p_in, p_out = prices[slug]
            in_tok = rec["input_tokens"] + upstream
            out_tok = rec["output_tokens"]
            c = (in_tok / 1e6) * p_in + (out_tok / 1e6) * p_out
            rows.append({"step": i + 1, "label": rec["label"], "model": slug,
                         "input_tokens": in_tok, "output_tokens": out_tok,
                         "cost_usd": c})
            upstream += out_tok + entry_overhead
            total += c
        return total, rows

    n_steps = len(steps)
    base_total, base_rows = _cost(base_series, [baseline_model] * n_steps)
    # Routed: cheap model answers, so its measured token counts apply.
    routed_series = [dict(cheap_series[i]) for i in range(n_steps)]
    routed_total, routed_rows = _cost(routed_series, [cheap_model] * n_steps)

    # Parse rates. Both measured 15/15 on these steps; the honest reading is a
    # bound, not 100%.
    n_cell = measurements["attempts_per_cell"]
    bound = rule_of_three_upper_bound(n_cell)
    base_rates = [r["parse_rate"] for r in base_series]
    cheap_rates = [r["parse_rate"] for r in cheap_series]

    compound_direction = (
        "INFLATES" if cheap_series[0]["output_tokens"] > base_series[0]["output_tokens"]
        else "reduces")

    return {
        "lever": "2a per-step routing",
        "baseline_model": baseline_model,
        "cheap_model": cheap_model,
        "steps_projected": [s + 1 for s in steps],
        "steps_excluded": sorted({k[1] + 1 for k in measurements["excluded"]}),
        "baseline_cost_usd": Fig(base_total, DERIVED,
                                 "measured tokens x verified prices"),
        "routed_cost_usd": Fig(routed_total, DERIVED,
                               "measured tokens x verified prices"),
        "saving_fraction": Fig(
            1.0 - routed_total / base_total if base_total else None, DERIVED,
            "on the mechanical steps only"),
        "baseline_rows": base_rows,
        "routed_rows": routed_rows,
        "compound_effect": {
            "direction": compound_direction,
            "step1_output_baseline": base_series[0]["output_tokens"],
            "step1_output_cheap": cheap_series[0]["output_tokens"],
            "note": (
                f"the cheap model {compound_direction} downstream input: it "
                f"emits {cheap_series[0]['output_tokens']:.0f} tokens at step 1 "
                f"against {base_series[0]['output_tokens']:.0f}, and every "
                f"later prompt embeds that verbatim. Measured, not assumed — "
                f"the assumed direction is the opposite."),
        },
        "parse_rates": {
            "baseline": base_rates, "routed": cheap_rates,
            "n_per_cell": n_cell,
            "upper_bound_failure_rate_95pct": bound,
            "reading": (
                f"every measured cell was {n_cell}/{n_cell}. By the rule of "
                f"three that bounds the per-step failure rate at {bound:.1%} "
                f"at 95% confidence — NOT at zero. At 156 decisions/day a "
                f"{bound:.1%} ceiling is far too loose to deploy on; it needs "
                f"hundreds of attempts per cell, not fifteen."),
        },
        "completion_effect_withheld": True,
        "why_withheld": (
            "the decision step's parse rate is unmeasured — both models scored "
            "0/15 there against a content-free fixture because of the "
            "empty-orders defect, which measures the pipeline rather than the "
            "model. Without it, no end-to-end completion rate and therefore no "
            "effective saving can be stated."),
        "behaviour_change": (
            "the decision step keeps the expensive model, so trading output is "
            "unchanged by construction. What changes is the quality of the "
            "intermediate summaries the decision step reads."),
    }


# --------------------------------------------------------------------------
# 2b — output token caps
# --------------------------------------------------------------------------


def cap_projection(sweep_path: Optional[str] = None,
                   sweep: Optional[Dict[str, Any]] = None,
                   measured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Token reduction and parse effect at each cap, from a measured sweep.

    Refuses without a sweep. The saving from a cap is entirely a function of
    where the parse rate breaks, and that cannot be reasoned about.
    """
    if sweep is None:
        if not sweep_path or not os.path.exists(sweep_path):
            raise MissingInput(
                "cap_projection needs a measured sweep "
                "(analysis/output_cap_sweep.py). A cap's saving depends on "
                "where truncation starts breaking the parse, which is not "
                "derivable from prices.")
        with open(sweep_path, "r", encoding="utf-8") as fh:
            sweep = json.load(fh)

    measured = measured or load_measured()
    prices = {m["slug"]: (m["price_in"], m["price_out"])
              for m in measured["models"].values()}

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep["rows"]:
        by_model.setdefault(row["model"], []).append(row)

    results = []
    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: (r["step_index"], -r["max_tokens"]))
        uncapped = max(r["max_tokens"] for r in rows)
        per_step: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for r in rows:
            per_step.setdefault(r["step_index"], {})[r["max_tokens"]] = r

        caps_out = []
        for cap in sorted({r["max_tokens"] for r in rows}, reverse=True):
            tok_base = sum(per_step[s][uncapped]["mean_output_tokens"] or 0
                           for s in per_step)
            tok_cap = sum(per_step[s][cap]["mean_output_tokens"] or 0
                          for s in per_step)
            rates = [per_step[s][cap]["parse_rate"] for s in per_step]
            p_in, p_out = prices.get(model, (None, None))
            caps_out.append({
                "max_tokens": cap,
                "mean_output_tokens": tok_cap,
                "token_reduction_fraction": (
                    1.0 - tok_cap / tok_base) if tok_base else None,
                "min_parse_rate": min(rates) if rates else None,
                "parse_rates_by_step": rates,
                "output_cost_reduction_usd_per_call": (
                    ((tok_base - tok_cap) / 1e6) * p_out
                    if p_out is not None else None),
                "safe": all(r == 1.0 for r in rates) if rates else None,
            })
        safe = [c for c in caps_out if c["safe"]]
        results.append({
            "model": model,
            "uncapped_reference": uncapped,
            "caps": caps_out,
            "lowest_safe_cap": min(
                (c["max_tokens"] for c in safe), default=None),
            "max_safe_token_reduction": max(
                (c["token_reduction_fraction"] or 0 for c in safe), default=0.0),
        })

    return {
        "lever": "2b output token caps",
        "results": results,
        "attempts_per_cell": sweep["attempts"],
        "upper_bound_failure_rate_95pct": rule_of_three_upper_bound(
            sweep["attempts"]),
        "behaviour_change": (
            "a cap TRUNCATES; it does not persuade a model to be concise. "
            "Below the break point the output stops parsing and the decision "
            "aborts, so the safe cap is the measurement, not the saving."),
        "why_it_targets_the_right_term": (
            "output is priced ~4x input and the structure inverts by tier — "
            "input is 61.5% of Nemotron's per-call cost, output is 77.4% of "
            "GPT-5.5's. The driver is output VOLUME: input spans 3895-5502 "
            "tokens across the roster while output spans 860-5005."),
    }


# --------------------------------------------------------------------------
# 2c — eliminated calls
# --------------------------------------------------------------------------


def eliminated_calls(measured: Optional[Dict[str, Any]] = None, *,
                     repeat_backtest_fraction: Optional[float] = None,
                     leaderboard_scheduled: bool = False) -> Dict[str, Any]:
    """Calls removed entirely by caching and leaderboard governance.

    A cached backtest costs nothing at all, which makes this the only lever
    with no behaviour cost — the user sees an identical result faster. Its SIZE
    depends on what fraction of backtests are repeats, which is not
    instrumented, so it is refused rather than assumed.
    """
    out: Dict[str, Any] = {
        "lever": "2c eliminated calls",
        "behaviour_change": (
            "none. A cache hit returns the identical stored result; "
            "leaderboard governance stops re-running work nobody requested."),
        "status": "built, unmerged (feature/serving-cost-reduction)",
    }

    if repeat_backtest_fraction is None:
        out["backtest_caching"] = {
            "saving_fraction": None,
            "refused": (
                "repeat_backtest_fraction is blank. A cache saves exactly the "
                "repeat rate and nothing else, and no telemetry records how "
                "often users re-run an identical backtest. Assuming a rate "
                "here would put an invented number at the centre of the "
                "saving."),
        }
    else:
        if not 0.0 <= repeat_backtest_fraction <= 1.0:
            raise ValueError("repeat_backtest_fraction must be in [0, 1]")
        out["backtest_caching"] = {
            "saving_fraction": Fig(
                repeat_backtest_fraction, NOT_MEASURED,
                "supplied repeat rate; no telemetry exists"),
            "note": "a cache hit removes 100% of a repeat run's calls",
        }

    measured = measured or load_measured()
    per_call = {k: cost_per_call(v) for k, v in measured["models"].items()}
    contest = sum(per_call[k] * measured["models"][k]["llm_calls"]
                  for k in measured["models"])
    out["leaderboard_governance"] = {
        "currently_scheduled": leaderboard_scheduled,
        "recurring_saving_usd_per_month": Fig(
            0.0, MEASURED,
            "nothing schedules the refresh today, so there is no recurring "
            "spend to eliminate"),
        "per_manual_deploy_usd": Fig(
            contest, MEASURED,
            "7 models x measured call counts over the 161-bar contest window"),
        "note": (
            "governance prevents FUTURE recurring cost if a scheduler is "
            "added. Claiming a saving against a schedule that does not exist "
            "would be counting money that is not being spent."),
    }
    return out


# --------------------------------------------------------------------------
# 2d — dropdown composition
# --------------------------------------------------------------------------


def dropdown_composition(measured: Optional[Dict[str, Any]] = None, *,
                         calls_per_decision: int = 1) -> Dict[str, Any]:
    """Cost per user-decision by model. Not an optimisation — a product call.

    Models users cannot select cost nothing. This is the largest lever by an
    order of magnitude and it is the advisor's decision, not an engineering
    change.
    """
    measured = measured or load_measured()
    models = measured["models"]
    rows = []
    for name, m in sorted(models.items(), key=lambda kv: cost_per_call(kv[1])):
        c = cost_per_call(m)
        rows.append({
            "model": name, "slug": m["slug"],
            "cost_per_call_usd": c,
            "cost_per_decision_usd": c * calls_per_decision,
            "input_tokens_per_call": m["input_per_call"],
            "output_tokens_per_call": m["output_per_call"],
            "output_share_of_cost": (
                (m["output_per_call"] / 1e6 * m["price_out"]) / c) if c else None,
        })
    cheapest, dearest = rows[0], rows[-1]
    return {
        "lever": "2d dropdown composition",
        "is_engineering_change": False,
        "decision_owner": "advisor / product",
        "calls_per_decision": calls_per_decision,
        "rows": rows,
        "span": Fig(
            dearest["cost_per_call_usd"] / cheapest["cost_per_call_usd"],
            MEASURED, "measured per-call costs, verified to <1e-5"),
        "behaviour_change": (
            "users lose access to models they can currently pick. Whether that "
            "matters depends on whether the expensive models trade better — "
            "which one evaluation window could not establish (rho +0.071, "
            "intervals overlapping)."),
        "note": (
            f"span from {cheapest['model']} at ${cheapest['cost_per_call_usd']:.6f} "
            f"to {dearest['model']} at ${dearest['cost_per_call_usd']:.6f} per "
            f"call. No engineering lever in 2a-2c approaches this."),
    }


# --------------------------------------------------------------------------
# stacked ceiling
# --------------------------------------------------------------------------


def stacked_ceiling(routing: Dict[str, Any], caps: Dict[str, Any],
                    *, monthly_bill_usd: Optional[float] = None,
                    routing_applicable_fraction: Optional[float] = None
                    ) -> Dict[str, Any]:
    """What 2a-2c together reach, and what that leaves.

    Multiplicative on the fraction of spend each lever touches, which is the
    generous reading — the levers overlap (a cap on a routed step saves less
    than a cap on an unrouted one) so the true figure is lower.
    """
    notes = []
    routing_saving = routing["saving_fraction"].value
    cap_saving = max((r["max_safe_token_reduction"] for r in caps["results"]),
                     default=0.0)

    if routing_applicable_fraction is None:
        notes.append(
            "routing_applicable_fraction is blank [NOT MEASURED], so the "
            "stacked figure assumes routing applies to ALL spend. That is its "
            "most generous reading and it overstates in two ways. (1) Routing "
            "applies only to multi-step pipelines and no telemetry records "
            "what fraction of user agents run more than one step. (2) The "
            "routing saving is measured on the MECHANICAL STEPS ONLY — the "
            "decision step keeps the expensive model by design — so the share "
            "of a pipeline's cost it can touch is strictly less than 100%, and "
            "that share cannot be computed without the decision step's "
            "measurement, which does not exist.")
        routing_applicable_fraction = 1.0

    # Caps reduce OUTPUT tokens only; treat the reduction as applying to the
    # output share of the bill, which varies by model. Use the measured range.
    remaining = (1.0 - routing_saving * routing_applicable_fraction)
    remaining *= (1.0 - cap_saving)
    stacked = 1.0 - remaining

    if cap_saving == 0.0:
        notes.append(
            "output caps contribute NOTHING to the stack: every cap that "
            "reduces tokens breaks the parse, and the only safe cap is one "
            "that does not bind. The stacked figure is routing alone.")

    out: Dict[str, Any] = {
        "levers_included": ["2a routing", "2b caps"],
        "levers_excluded": [
            "2c eliminated calls — refuses without a measured repeat rate",
            "2d dropdown composition — a product decision, not an engineering "
            "saving, and it does not stack with the others (it changes the "
            "per-call price the others are computed against)"],
        "routing_saving_fraction": routing_saving,
        "cap_saving_fraction": cap_saving,
        "stacked_saving_fraction": Fig(
            stacked, DERIVED,
            "multiplicative on measured per-lever savings; generous"),
        "overlap_warning": (
            "the levers overlap. A cap applied to a step already routed to a "
            "cheap model saves cheap-model dollars, not expensive ones, so "
            "multiplying the fractions overstates the combined effect."),
        "notes": notes,
    }

    if monthly_bill_usd is None:
        out["residual_usd_per_month"] = None
        out["refused"] = (
            "monthly_bill_usd is blank. The residual is the number that "
            "decides whether these levers matter, and it depends on the "
            "workload split — run workload_split first and supply the figure "
            "for the workload in question.")
    else:
        residual = monthly_bill_usd * remaining
        out["monthly_bill_usd"] = monthly_bill_usd
        out["residual_usd_per_month"] = Fig(
            residual, DERIVED, "bill x (1 - stacked saving)")
        out["verdict"] = (
            f"a {stacked:.0%} reduction on ${monthly_bill_usd:,.0f}/month "
            f"leaves ${residual:,.0f}/month. Engineering levers change the "
            f"coefficient; only model choice changes the order of magnitude, "
            f"and that is the same decision as open-weight substitution.")
    return out


def format_optimisation_report(routing, caps, elim, dropdown, stacked) -> str:
    L = ["=" * 78, "API OPTIMISATION CEILING", "=" * 78]

    L.append(f"\n2a  {routing['lever']}")
    L.append(f"    steps projected {routing['steps_projected']}, "
             f"excluded {routing['steps_excluded']}")
    L.append(f"    baseline {routing['baseline_cost_usd'].tagged('.6f')} -> "
             f"routed {routing['routed_cost_usd'].tagged('.6f')}")
    L.append(f"    saving   {routing['saving_fraction'].tagged('.1%')}")
    L.append(f"    compound: {routing['compound_effect']['direction']} "
             f"downstream input "
             f"({routing['compound_effect']['step1_output_cheap']:.0f} vs "
             f"{routing['compound_effect']['step1_output_baseline']:.0f} tok)")
    L.append(f"    parse:    {routing['parse_rates']['reading'][:150]}")

    L.append(f"\n2b  {caps['lever']}")
    for r in caps["results"]:
        L.append(f"    {r['model'][:44]}")
        for c in r["caps"]:
            flag = "safe" if c["safe"] else "BREAKS"
            L.append(f"      cap {c['max_tokens']:<5} out={c['mean_output_tokens']:>6.0f}  "
                     f"reduction={c['token_reduction_fraction'] or 0:>6.1%}  "
                     f"min parse={c['min_parse_rate']:.0%}  {flag}")
        L.append(f"      lowest safe cap: {r['lowest_safe_cap']}  "
                 f"(max safe token reduction {r['max_safe_token_reduction']:.1%})")

    L.append(f"\n2c  {elim['lever']}")
    bc = elim["backtest_caching"]
    L.append(f"    backtest caching: "
             + (f"REFUSED — {bc['refused'][:90]}" if bc.get("refused")
                else bc["saving_fraction"].tagged(".1%")))
    lg = elim["leaderboard_governance"]
    L.append(f"    leaderboard recurring saving: "
             f"{lg['recurring_saving_usd_per_month'].tagged(',.2f')} "
             f"(nothing is scheduled)")
    L.append(f"    per manual deploy: {lg['per_manual_deploy_usd'].tagged(',.2f')}")

    L.append(f"\n2d  {dropdown['lever']}  — NOT an engineering change")
    L.append(f"    {'model':28s}{'$/call':>12}{'out share':>11}")
    for r in dropdown["rows"]:
        L.append(f"    {r['model'][:26]:28s}{r['cost_per_call_usd']:>12.6f}"
                 f"{r['output_share_of_cost']:>11.1%}")
    L.append(f"    span: {dropdown['span'].tagged(',.0f')}x")

    L.append("\nSTACKED")
    L.append(f"    2a+2b: {stacked['stacked_saving_fraction'].tagged('.1%')}")
    if stacked.get("residual_usd_per_month"):
        L.append(f"    {stacked['verdict']}")
    else:
        L.append(f"    residual REFUSED — {stacked['refused'][:100]}")
    L.append(f"    {stacked['overlap_warning'][:150]}")
    return "\n".join(L)
