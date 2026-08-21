"""The end-to-end routing projection Phase 16 withheld, now that step 3 is measured.

Phase 16 measured steps 1-2 at 30/30 on cheap models and correctly refused an
end-to-end number, because the decision step had only been exercised in
isolated mode against placeholder upstream. ``decision_step_probe`` closes that
gap with real in-pipeline context. This module does the arithmetic and keeps
the refusals.

WHAT ROUTING CAN AND CANNOT TOUCH
-----------------------------------
Routing sends the MECHANICAL steps to a cheap model and leaves the decision
step on the expensive one by design. So the fraction of a bill it can reach is
the mechanical steps' share of pipeline cost — computed here from measured
tokens, never assumed to be 100%. On a single-step agent it is zero: there are
no mechanical steps to route.

THE COMPOUND EFFECT RUNS BACKWARDS
------------------------------------
Later prompts embed every prior step's output verbatim
(``json.dumps(prior_outputs, indent=2)``), so a terser step-1 model shrinks
steps 2..N's input. The obvious assumption is that the cheap model is the
terser one. Measured, it is not: Nemotron emitted 1458 output tokens at step 1
against deepseek's 1197, so routing step 1 to the cheaper model INFLATES
downstream input. The direction is taken from the measurement each time rather
than asserted.

COMPLETION IS A PRODUCT, AND IT IS BOUNDED NOT KNOWN
------------------------------------------------------
``pipeline_runner`` aborts the whole decision on any unparseable step with no
retry, so end-to-end completion is the product of the per-step rates. Where an
observed rate is 100% the reported figure carries a rule-of-three bound: 16/16
bounds the failure rate at 18.8%, not at zero. A completion rate built from
three such observations is an upper estimate with a wide floor, and this module
reports both.

THE EMPTY-DECISION RATE IS SEPARATE FROM THE PARSE RATE
---------------------------------------------------------
A well-formed ``{"orders": []}`` is a valid "no trades" that production treats
as a failure and silently replaces with a rule-based decision. It is counted
apart from parse failures throughout: it is a pipeline defect, not a model one,
and folding it into a model's parse rate would blame the wrong component.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.api_optimisation import (  # noqa: E402
    load_step_measurements, rule_of_three_upper_bound,
)
from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, MissingInput, NOT_MEASURED, Fig, load_measured,
)

__all__ = [
    "DECISION_PROBE_RESULTS", "ENTRY_OVERHEAD_TOKENS",
    "load_decision_measurements", "endtoend_projection",
    "saving_against_cadence", "format_endtoend",
]

DECISION_PROBE_RESULTS = os.path.join(_BENCH_ROOT, "results",
                                      "decision_step_probe.json")

# json.dumps(prior_outputs, indent=2) wrapper per upstream entry: the
# step/label/presetKey/id keys plus indentation, before the payload itself.
ENTRY_OVERHEAD_TOKENS = 25.0


def load_decision_measurements(path: str = DECISION_PROBE_RESULTS
                               ) -> Dict[str, Any]:
    """Step-3 reliability per model, from the in-pipeline probe.

    Refuses rather than falling back to the Phase 16 isolated-mode numbers.
    Those measured a content-free fixture and using them here would reintroduce
    exactly the error this phase exists to remove.
    """
    if not os.path.exists(path):
        raise MissingInput(
            f"no decision-step measurements at {path}. Run "
            f"analysis/decision_step_probe.py. The Phase 16 isolated-mode "
            f"figures are NOT a substitute: they measured a placeholder "
            f"upstream context, which is why the end-to-end projection was "
            f"withheld in the first place.")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {"by_model": {s["model"]: s for s in raw["summary"]},
            "rows": raw["rows"],
            "upstream_model": raw["upstream_model"],
            "regimes": raw["regimes"],
            "attempts_per_regime": raw["attempts_per_regime"]}


def _prices(measured: Dict[str, Any]) -> Dict[str, Any]:
    return {m["slug"]: (m["price_in"], m["price_out"])
            for m in measured["models"].values()}


def endtoend_projection(baseline_model: str, cheap_model: str,
                        *, decision_model: Optional[str] = None,
                        steps_measurements: Optional[Dict[str, Any]] = None,
                        decision_measurements: Optional[Dict[str, Any]] = None,
                        measured: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Full-pipeline cost and completion, baseline vs routed.

    ``decision_model`` defaults to ``baseline_model``: routing keeps the
    decision step on the expensive model, which is the whole point of the
    design. Passing a different one prices routing the decision step too.
    """
    measured = measured or load_measured()
    steps_m = steps_measurements or load_step_measurements()
    dec_m = decision_measurements or load_decision_measurements()
    decision_model = decision_model or baseline_model

    prices = _prices(measured)
    for slug in {baseline_model, cheap_model, decision_model}:
        if slug not in prices:
            raise MissingInput(
                f"no measured price for {slug}. Known: {sorted(prices)}")

    usable = steps_m["usable"]
    mech_steps = sorted({k[1] for k in usable})
    if not mech_steps:
        raise MissingInput("no mechanical-step measurements")

    def _mech(model, i):
        rec = usable.get((model, i))
        if rec is None:
            raise MissingInput(
                f"no measured tokens for {model} at step {i + 1}")
        return rec

    def _decision(model):
        rec = dec_m["by_model"].get(model)
        if rec is None:
            raise MissingInput(
                f"no measured decision-step reliability for {model}. "
                f"Measured: {sorted(dec_m['by_model'])}. An end-to-end "
                f"completion rate cannot be produced without it.")
        return rec

    def _assign(mech_model: str, dec_model: str):
        """Cost every step, carrying upstream tokens forward as the runner does."""
        upstream = 0.0
        rows: List[Dict[str, Any]] = []
        for i in mech_steps:
            rec = _mech(mech_model, i)
            p_in, p_out = prices[mech_model]
            in_tok = rec["input_tokens"] + upstream
            out_tok = rec["output_tokens"]
            rows.append({
                "step": i + 1, "label": rec["label"], "model": mech_model,
                "kind": "mechanical", "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": (in_tok / 1e6) * p_in + (out_tok / 1e6) * p_out,
                "parse_rate": rec["parse_rate"], "n": rec["n"]})
            upstream += out_tok + ENTRY_OVERHEAD_TOKENS

        d = _decision(dec_model)
        p_in, p_out = prices[dec_model]
        # The probe measured step 3's real input directly (it embeds the real
        # upstream), so that figure is used rather than the carried estimate.
        in_tok = d.get("mean_input_tokens") or 0.0
        if not in_tok:
            in_tok = upstream
        out_tok = d["mean_output_tokens"] or 0.0
        rows.append({
            "step": len(mech_steps) + 1, "label": "decision",
            "model": dec_model, "kind": "decision",
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd": (in_tok / 1e6) * p_in + (out_tok / 1e6) * p_out,
            "parse_rate": d["parse_rate"], "n": d["n_total"],
            "empty_orders_rate": d["empty_orders_rate"]})
        return rows

    base_rows = _assign(baseline_model, decision_model)
    routed_rows = _assign(cheap_model, decision_model)

    def _total(rows):
        return sum(r["cost_usd"] for r in rows)

    def _mech_share(rows):
        t = _total(rows)
        m = sum(r["cost_usd"] for r in rows if r["kind"] == "mechanical")
        return (m / t) if t else None

    def _completion(rows):
        """Product of per-step rates, with a floor from the rule of three."""
        point, floor = 1.0, 1.0
        for r in rows:
            rate = r["parse_rate"]
            if rate is None:
                return None, None
            point *= rate
            floor *= (rate if rate < 1.0
                      else 1.0 - rule_of_three_upper_bound(r["n"]))
        return point, floor

    base_total, routed_total = _total(base_rows), _total(routed_rows)
    base_comp, base_floor = _completion(base_rows)
    routed_comp, routed_floor = _completion(routed_rows)

    warnings: List[str] = []

    # Compound direction, measured each time rather than asserted.
    b0, r0 = _mech(baseline_model, mech_steps[0]), _mech(cheap_model, mech_steps[0])
    direction = ("INFLATES" if r0["output_tokens"] > b0["output_tokens"]
                 else "reduces")
    if direction == "INFLATES":
        warnings.append(
            f"the cheap model INFLATES downstream input: {r0['output_tokens']:.0f} "
            f"output tokens at step 1 against {b0['output_tokens']:.0f}, and "
            f"every later prompt embeds that verbatim. The saving below is net "
            f"of that penalty, not before it.")

    dec = _decision(decision_model)
    if dec["empty_orders_rate"]:
        warnings.append(
            f"the decision step returned a well-formed empty decision in "
            f"{dec['n_empty_orders']}/{dec['n_total']} attempts "
            f"({dec['empty_orders_rate']:.0%}). Production treats that as a "
            f"failure and silently substitutes a rule-based decision, so it is "
            f"counted OUTSIDE the parse rate — it is a pipeline defect, not a "
            f"model one.")

    if base_comp is not None and base_comp == 1.0:
        warnings.append(
            "every observed parse rate was 100%. The floor column applies the "
            "rule of three to each, which is the honest reading: no observation "
            "of n trials with zero failures can establish a rate above "
            f"{1 - rule_of_three_upper_bound(dec['n_total']):.0%}.")

    return {
        "baseline_model": baseline_model,
        "cheap_model": cheap_model,
        "decision_model": decision_model,
        "upstream_model_in_probe": dec_m["upstream_model"],
        "regimes": dec_m["regimes"],
        "baseline_rows": base_rows,
        "routed_rows": routed_rows,
        "baseline_cost_usd": Fig(base_total, DERIVED,
                                 "measured tokens x verified prices"),
        "routed_cost_usd": Fig(routed_total, DERIVED,
                               "measured tokens x verified prices"),
        "saving_fraction": Fig(
            (1.0 - routed_total / base_total) if base_total else None, DERIVED,
            "END TO END, decision step included and unchanged"),
        "mechanical_share_of_baseline": Fig(
            _mech_share(base_rows), DERIVED,
            "the fraction of pipeline cost routing can touch at all"),
        "baseline_completion": Fig(base_comp, MEASURED if base_comp else NOT_MEASURED,
                                   "product of measured per-step parse rates"),
        "baseline_completion_floor": Fig(
            base_floor, DERIVED, "rule-of-three floor on the same product"),
        "routed_completion": Fig(routed_comp, MEASURED if routed_comp else NOT_MEASURED,
                                 "product of measured per-step parse rates"),
        "routed_completion_floor": Fig(
            routed_floor, DERIVED, "rule-of-three floor on the same product"),
        "empty_decision_rate": Fig(
            dec["empty_orders_rate"], MEASURED,
            f"{dec['n_empty_orders']}/{dec['n_total']} in-pipeline attempts"),
        "compound_direction": direction,
        "warnings": warnings,
        "not_measured": {
            "decision_equivalence": (
                "whether routing changes WHICH decisions are made. Needs "
                "multiple evaluation windows and a definition of 'same'; four "
                "regimes in one run cannot support it. Out of scope."),
        },
    }


def saving_against_cadence(projection: Dict[str, Any],
                           cadence_monthly_usd: Dict[str, float],
                           *, pipeline_fraction: Optional[float] = None
                           ) -> Dict[str, Any]:
    """Express the saving against each cadence's bill.

    ``pipeline_fraction`` is the share of the bill run through MULTI-STEP
    pipelines. Routing cannot touch single-step agents, and no telemetry
    records the split, so it is refused rather than defaulted.
    """
    saving = projection["saving_fraction"].value
    out: Dict[str, Any] = {"saving_fraction": saving, "cadences": {}}

    if pipeline_fraction is None:
        out["refused"] = (
            "pipeline_fraction is blank. Routing applies only to multi-step "
            "pipelines — the seed leaderboard runs measured 1.000 calls per "
            "decision, i.e. single-prompt agents with no mechanical steps to "
            "route — and no telemetry records what share of spend runs through "
            "multi-step pipelines. Applying the saving to a whole bill would "
            "assume that share is 100%.")
        for name, bill in cadence_monthly_usd.items():
            out["cadences"][name] = {
                "monthly_usd": bill,
                "saving_usd": None,
                "note": "withheld pending pipeline_fraction",
            }
        return out

    if not 0.0 <= pipeline_fraction <= 1.0:
        raise ValueError("pipeline_fraction must be in [0, 1]")

    for name, bill in cadence_monthly_usd.items():
        reachable = bill * pipeline_fraction
        out["cadences"][name] = {
            "monthly_usd": bill,
            "reachable_usd": reachable,
            "saving_usd": reachable * saving,
            "residual_usd": bill - reachable * saving,
        }
    out["pipeline_fraction"] = pipeline_fraction
    return out


def format_endtoend(p: Dict[str, Any]) -> str:
    L = ["=" * 78, "END-TO-END ROUTING PROJECTION", "=" * 78,
         f"decision step stays on {p['decision_model']}; "
         f"mechanical steps {p['baseline_model']} -> {p['cheap_model']}",
         f"probe upstream: {p['upstream_model_in_probe']}  "
         f"regimes: {', '.join(p['regimes'])}", ""]
    for title, rows in (("baseline", p["baseline_rows"]),
                        ("routed", p["routed_rows"])):
        L.append(f"{title}:")
        L.append(f"  {'step':<14}{'model':<32}{'in':>8}{'out':>7}"
                 f"{'$':>11}{'parse':>8}")
        for r in rows:
            L.append(f"  {r['label'][:12]:<14}{r['model'][:30]:<32}"
                     f"{r['input_tokens']:>8.0f}{r['output_tokens']:>7.0f}"
                     f"{r['cost_usd']:>11.6f}"
                     f"{(r['parse_rate'] or 0):>8.0%}")
        L.append(f"  {'TOTAL':<54}{sum(r['cost_usd'] for r in rows):>11.6f}")
        L.append("")
    L.append(f"  saving (end to end)      {p['saving_fraction'].tagged('.1%')}")
    L.append(f"  mechanical share         "
             f"{p['mechanical_share_of_baseline'].tagged('.1%')}")
    L.append(f"  completion baseline      "
             f"{p['baseline_completion'].tagged('.1%')}  "
             f"floor {p['baseline_completion_floor'].tagged('.1%')}")
    L.append(f"  completion routed        "
             f"{p['routed_completion'].tagged('.1%')}  "
             f"floor {p['routed_completion_floor'].tagged('.1%')}")
    L.append(f"  empty-decision rate      {p['empty_decision_rate'].tagged('.1%')}")
    L.append(f"  compound direction       {p['compound_direction']}")
    if p["warnings"]:
        L.append("\n  warnings:")
        for w in p["warnings"]:
            L.append(f"    - {w}")
    L.append(f"\n  NOT measured: {p['not_measured']['decision_equivalence']}")
    return "\n".join(L)
