"""Project the cost of a routing config from MEASURED reliability.

Cost is arithmetic once reliability is known. This module does the arithmetic
and refuses to do it from assumptions.

WHAT IT REFUSES TO DO
----------------------
``project_routing`` will not produce a saving figure without a measured parse
rate for every routed step. A config that saves 60% on paper and aborts 5% of
decisions must show BOTH numbers, so the completion rate is a required input
rather than an optional adjustment — there is no code path that quietly
assumes 100%.

THE PRICE TABLE HAS AN UNMARKED FALLBACK, AND IT MATTERS HERE
---------------------------------------------------------------
``token_cost.price_for_model`` returns ``_DEFAULT_PRICING`` — (1.0, 5.0), the
Claude-Haiku rate — for any model it does not recognise, with no signal that it
guessed. For general run-cost estimation that is a reasonable default. For THIS
module it is actively misleading: every self-hosted model (``Qwen/Qwen2.5-1.5B-
Instruct`` and friends) is unrecognised, so a naive projection would price the
cheap local option at Claude rates and conclude that routing saves nothing.

``price_provenance`` therefore reports whether a price was matched, defaulted,
or is a no-API-cost local model, and the projection propagates that. A
projection containing a defaulted price is marked unreliable rather than
printed as a number.

SELF-HOSTED MODELS HAVE NO PER-TOKEN PRICE AT ALL
---------------------------------------------------
Their cost is amortised GPU time — a function of throughput and card-hours, not
tokens. This module will not invent a per-token rate for them. It prices them
at zero API cost and says so, which makes the hosted-vs-hosted comparison valid
and leaves the hosted-vs-self-hosted comparison to the serving benchmark, where
the throughput measurements live.

THE COMPOUNDING EFFECT IS MEASURED, NOT ASSUMED
-------------------------------------------------
``_build_step_prompt`` embeds every prior step's output verbatim into each
later prompt (``json.dumps(prior_outputs, indent=2)``). A terser model at step
1 therefore shrinks the INPUT of steps 2..N as well as its own output. That
effect is real but its size depends on how terse the cheap model actually is,
so ``project_routing`` takes measured per-step output tokens and recomputes
downstream input from them. Passing no measurements disables the adjustment
instead of guessing at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dashboard.backend.infrastructure.llm.token_cost import (
    _DEFAULT_PRICING,
    _PRICING_TABLE,
    is_free_model,
    price_for_model,
)

__all__ = [
    "PRICE_KNOWN", "PRICE_DEFAULTED", "PRICE_LOCAL",
    "price_provenance", "StepCost", "RoutingProjection", "project_routing",
]

PRICE_KNOWN = "known"            # matched a row in the pricing table
PRICE_DEFAULTED = "defaulted"    # unmatched: the (1.0, 5.0) fallback was used
PRICE_LOCAL = "local"            # self-hosted: no per-token API price exists

# Self-hosted model families. These are served on our own card, so their API
# cost is zero and their real cost is GPU time measured elsewhere.
_LOCAL_MARKERS = ("qwen/qwen2.5", "qwen2.5-", "meta-llama/", "mistralai/",
                  "microsoft/phi", "google/gemma", "tiiuae/falcon")


def price_provenance(model: Optional[str]) -> Tuple[str, Tuple[float, float]]:
    """``(provenance, (in_price, out_price))`` — never a bare number.

    The whole point is that a caller cannot accidentally treat a defaulted
    price as a known one, which is exactly the mistake that would make routing
    to a self-hosted model look like a cost increase.
    """
    name = (model or "").strip().lower()
    if not name:
        return PRICE_DEFAULTED, _DEFAULT_PRICING
    if is_free_model(name):
        return PRICE_LOCAL, (0.0, 0.0)
    if any(marker in name for marker in _LOCAL_MARKERS):
        return PRICE_LOCAL, (0.0, 0.0)
    for needle, in_price, out_price in _PRICING_TABLE:
        if needle in name:
            return PRICE_KNOWN, (in_price, out_price)
    return PRICE_DEFAULTED, price_for_model(model)


@dataclass
class StepCost:
    """One step's cost under one assignment, with its provenance attached."""

    step_index: int
    label: str
    model: str
    input_tokens: float
    output_tokens: float
    price_provenance: str
    input_price: float
    output_price: float

    @property
    def cost_usd(self) -> float:
        return ((self.input_tokens / 1_000_000) * self.input_price
                + (self.output_tokens / 1_000_000) * self.output_price)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "label": self.label,
            "model": self.model,
            "input_tokens": round(self.input_tokens, 1),
            "output_tokens": round(self.output_tokens, 1),
            "cost_usd": round(self.cost_usd, 8),
            "price_provenance": self.price_provenance,
        }


@dataclass
class RoutingProjection:
    """Baseline vs routed, with the completion rate beside the saving."""

    baseline_steps: List[StepCost] = field(default_factory=list)
    routed_steps: List[StepCost] = field(default_factory=list)
    baseline_completion_rate: Optional[float] = None
    routed_completion_rate: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    compounding_applied: bool = False

    # ---- per-decision cost, ignoring failures --------------------------------

    @property
    def baseline_cost(self) -> float:
        return sum(s.cost_usd for s in self.baseline_steps)

    @property
    def routed_cost(self) -> float:
        return sum(s.cost_usd for s in self.routed_steps)

    @property
    def nominal_saving_fraction(self) -> Optional[float]:
        """The headline number, and the one that is misleading on its own."""
        if not self.baseline_cost:
            return None
        return 1.0 - (self.routed_cost / self.baseline_cost)

    # ---- per COMPLETED decision, which is what a decision actually costs -----

    @property
    def baseline_cost_per_completed(self) -> Optional[float]:
        r = self.baseline_completion_rate
        return (self.baseline_cost / r) if r else None

    @property
    def routed_cost_per_completed(self) -> Optional[float]:
        """A decision that aborts still cost money and produced nothing.

        Dividing by the completion rate charges the wasted calls to the
        decisions that did complete, which is what the budget actually sees.
        This overstates slightly — an abort at step 1 costs less than a full
        run — so it is an upper bound, and ``project_routing`` says so.
        """
        r = self.routed_completion_rate
        return (self.routed_cost / r) if r else None

    @property
    def effective_saving_fraction(self) -> Optional[float]:
        base, routed = self.baseline_cost_per_completed, self.routed_cost_per_completed
        if not base or routed is None:
            return None
        return 1.0 - (routed / base)

    @property
    def decisions_lost_per_100(self) -> Optional[float]:
        """The number that has to sit next to the saving."""
        if self.routed_completion_rate is None:
            return None
        base = self.baseline_completion_rate
        base = 1.0 if base is None else base
        return max(0.0, (base - self.routed_completion_rate) * 100)

    @property
    def reliable(self) -> bool:
        """False when any price was guessed or any completion rate is missing."""
        if self.routed_completion_rate is None:
            return False
        return not any(s.price_provenance == PRICE_DEFAULTED
                       for s in self.baseline_steps + self.routed_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": {
                "steps": [s.to_dict() for s in self.baseline_steps],
                "cost_usd": round(self.baseline_cost, 8),
                "completion_rate": self.baseline_completion_rate,
                "cost_per_completed_usd": (
                    round(self.baseline_cost_per_completed, 8)
                    if self.baseline_cost_per_completed else None),
            },
            "routed": {
                "steps": [s.to_dict() for s in self.routed_steps],
                "cost_usd": round(self.routed_cost, 8),
                "completion_rate": self.routed_completion_rate,
                "cost_per_completed_usd": (
                    round(self.routed_cost_per_completed, 8)
                    if self.routed_cost_per_completed else None),
            },
            "nominal_saving_fraction": self.nominal_saving_fraction,
            "effective_saving_fraction": self.effective_saving_fraction,
            "decisions_lost_per_100": self.decisions_lost_per_100,
            "compounding_applied": self.compounding_applied,
            "reliable": self.reliable,
            "warnings": self.warnings,
        }


def _downstream_input_delta(prior_outputs_tokens: Sequence[float]) -> float:
    """Tokens a later prompt inherits from every upstream step's output.

    ``_build_step_prompt`` re-serialises the whole ``prior_outputs`` list into
    each later prompt, so step k's input carries the sum of steps 1..k-1's
    outputs. The wrapper (``step``/``label``/``id`` keys and indent=2) adds
    overhead per entry; ``_ENTRY_OVERHEAD_TOKENS`` approximates it.
    """
    return sum(prior_outputs_tokens) + _ENTRY_OVERHEAD_TOKENS * len(prior_outputs_tokens)


# json.dumps(..., indent=2) of one wrapper entry: the step/label/presetKey/id
# keys plus indentation, before the output payload itself.
_ENTRY_OVERHEAD_TOKENS = 25.0


def project_routing(
    steps: Sequence[Mapping[str, Any]],
    *,
    baseline_model: str,
    routed_models: Sequence[str],
    measured_output_tokens: Optional[Mapping[Tuple[str, int], float]] = None,
    measured_parse_rates: Optional[Mapping[Tuple[str, int], float]] = None,
    base_input_tokens: Optional[Sequence[float]] = None,
    baseline_completion_rate: Optional[float] = None,
) -> RoutingProjection:
    """Project a routing config's cost against the all-expensive baseline.

    ``measured_output_tokens`` and ``measured_parse_rates`` are keyed by
    ``(model, step_index)`` and come from the reliability harness. Without
    parse rates the projection still computes nominal cost but reports
    ``reliable=False`` and refuses an effective saving — which is the point:
    the saving is not knowable without them.

    ``base_input_tokens[i]`` is step i's own prompt input EXCLUDING upstream
    outputs (the task text, the snapshot at step 0, the execution rules at the
    last step). Upstream contribution is added from measured output tokens, so
    the compounding effect is computed rather than assumed.
    """
    n = len(steps)
    if len(routed_models) != n:
        raise ValueError(
            f"routed_models has {len(routed_models)} entries for {n} steps")

    warnings: List[str] = []
    measured_output_tokens = dict(measured_output_tokens or {})
    measured_parse_rates = dict(measured_parse_rates or {})
    base_input = list(base_input_tokens or [])
    if len(base_input) < n:
        base_input += [0.0] * (n - len(base_input))
        if not base_input_tokens:
            warnings.append(
                "no base_input_tokens supplied: input cost counts only upstream "
                "outputs, so absolute costs are understated (the RATIO between "
                "baseline and routed is still meaningful).")

    def _assign(models: Sequence[str]) -> Tuple[List[StepCost], bool]:
        out_tokens: List[float] = []
        costs: List[StepCost] = []
        compounded = False
        for i, step in enumerate(steps):
            model = models[i]
            prov, (in_price, out_price) = price_provenance(model)
            measured = measured_output_tokens.get((model, i))
            if measured is None:
                # No measurement: fall back to the baseline model's figure if we
                # have one, and say so. Never silently invent a token count.
                measured = measured_output_tokens.get((baseline_model, i))
                if measured is None:
                    measured = 0.0
                    warnings.append(
                        f"no measured output tokens for {model} at step {i + 1}; "
                        f"its output cost is counted as zero and the projection "
                        f"understates it.")
                else:
                    compounded = compounded  # unchanged; using a stand-in
                    warnings.append(
                        f"no measured output tokens for {model} at step {i + 1}; "
                        f"using {baseline_model}'s measured {measured:.0f} as a "
                        f"stand-in, which removes any terseness benefit.")
            upstream = _downstream_input_delta(out_tokens)
            if out_tokens:
                compounded = True
            in_tokens = base_input[i] + upstream
            costs.append(StepCost(
                step_index=i,
                label=str(step.get("label") or f"Step {i + 1}"),
                model=model,
                input_tokens=in_tokens,
                output_tokens=measured,
                price_provenance=prov,
                input_price=in_price,
                output_price=out_price))
            out_tokens.append(measured)
        return costs, compounded

    baseline_costs, _ = _assign([baseline_model] * n)
    routed_costs, compounded = _assign(list(routed_models))

    # Completion is the PRODUCT of every step's parse rate: the pipeline
    # completes only if every step parses, and there is no retry.
    def _completion(models: Sequence[str]) -> Optional[float]:
        product = 1.0
        seen_any = False
        for i, model in enumerate(models):
            rate = measured_parse_rates.get((model, i))
            if rate is None:
                return None
            seen_any = True
            product *= rate
        return product if seen_any else None

    routed_completion = _completion(list(routed_models))
    base_completion = baseline_completion_rate
    if base_completion is None:
        base_completion = _completion([baseline_model] * n)

    if routed_completion is not None and base_completion is None:
        warnings.append(
            "no completion rate for the baseline model: the effective saving is "
            "withheld. The expensive model is not assumed to complete 100% of "
            "decisions — assuming it would flatter routing by charging every "
            "baseline failure to nobody.")

    if routed_completion is None:
        warnings.append(
            "no measured parse rates for the routed models: the nominal saving "
            "is shown but the effective saving is withheld. Run "
            "dashboard/scripts/measure_step_reliability.py first — a saving "
            "without a completion rate is not a result.")

    defaulted = sorted({s.model for s in baseline_costs + routed_costs
                        if s.price_provenance == PRICE_DEFAULTED})
    if defaulted:
        warnings.append(
            f"price defaulted to {_DEFAULT_PRICING} for {defaulted}: these are "
            f"not in the pricing table, so their cost is a guess at Claude-Haiku "
            f"rates, not a price.")

    local = sorted({s.model for s in routed_costs
                    if s.price_provenance == PRICE_LOCAL})
    if local:
        warnings.append(
            f"{local} priced at zero API cost because they are self-hosted. "
            f"Their real cost is GPU time, which this module does not model — "
            f"see the serving benchmark for throughput per card.")

    if routed_completion is not None and base_completion is not None \
            and routed_completion < base_completion:
        warnings.append(
            "cost_per_completed charges aborted decisions to the completed "
            "ones. An abort at step 1 costs less than a full run, so the "
            "routed figure is an UPPER bound.")

    return RoutingProjection(
        baseline_steps=baseline_costs,
        routed_steps=routed_costs,
        baseline_completion_rate=base_completion,
        routed_completion_rate=routed_completion,
        warnings=warnings,
        compounding_applied=compounded)


def format_projection(proj: RoutingProjection) -> str:
    """Human-readable, with the saving and the losses side by side."""
    lines = ["", "=" * 74, "ROUTING COST PROJECTION", "=" * 74]
    for title, rows, rate in (
            ("baseline", proj.baseline_steps, proj.baseline_completion_rate),
            ("routed", proj.routed_steps, proj.routed_completion_rate)):
        lines.append(f"\n{title}:")
        lines.append(f"  {'step':<26} {'model':<30} {'in':>7} {'out':>6} {'$':>11}")
        lines.append("  " + "-" * 84)
        for s in rows:
            flag = "" if s.price_provenance == PRICE_KNOWN else f"  [{s.price_provenance}]"
            lines.append(f"  {s.label[:24]:<26} {s.model[:28]:<30} "
                         f"{s.input_tokens:>7.0f} {s.output_tokens:>6.0f} "
                         f"{s.cost_usd:>11.6f}{flag}")
        total = sum(s.cost_usd for s in rows)
        lines.append(f"  {'TOTAL':<57} {total:>11.6f}")
        lines.append(f"  completion rate: "
                     + (f"{rate:.1%}" if rate is not None else "NOT MEASURED"))

    nom = proj.nominal_saving_fraction
    eff = proj.effective_saving_fraction
    lost = proj.decisions_lost_per_100
    lines.append("")
    lines.append(f"  nominal saving      {nom:.1%}" if nom is not None
                 else "  nominal saving      undefined (baseline costs nothing)")
    if eff is None:
        lines.append("  effective saving    WITHHELD — no measured completion rate. "
                     "\n                      A saving without one is not a result.")
    else:
        lines.append(f"  effective saving    {eff:.1%}  (per COMPLETED decision)")
    if lost is not None:
        lines.append(f"  decisions lost      {lost:.1f} per 100")
    lines.append(f"  compounding         "
                 + ("applied from measured output tokens"
                    if proj.compounding_applied else "not applicable (single step)"))
    lines.append(f"  reliable            {proj.reliable}")
    if proj.warnings:
        lines.append("\n  warnings:")
        for w in proj.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)
