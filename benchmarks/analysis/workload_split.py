"""Two cost workloads, separately parameterised, because they differ in control.

Every prior model mixed them, which is why the $50K figure never pinned down.

EXPERIMENT (Experiment C in the NVIDIA note)
    The lab chooses agent count, model set, cadence and pipeline depth. Cost is
    bounded by design and every parameter is a lever.

USER PLATFORM
    Users sign up, pick a model from the dropdown, configure their own pipeline
    depth, and run backtests when they like. Cost scales with signups and
    almost nothing is a lever — an agent that skips bars to save money looks
    broken to the person who configured it.

The distinction is not cosmetic. A saving that works on one is often
unavailable on the other, and a number that averages the two describes neither.

CADENCE IS THE 22x TERM AND IT CHANGED
----------------------------------------
Target cadence is Nof1's 2-3 minutes — 156 decisions/agent/day on equity hours
— not the hourly 7 that every earlier figure assumed. Both are carried side by
side throughout, because the ratio between them is itself the finding: nothing
else in either model moves cost by 22x.

REFUSAL IS THE DEFAULT
-----------------------
Both models raise ``MissingInput`` naming every blank parameter rather than
substituting a default. A silent default here becomes a finding three documents
later, which has already happened once in this project.

WHAT IS NOT MODELLED, AND WHY
------------------------------
* **Prefix caching.** Measured at zero cached tokens across 160 requests
  against a 99.4% shared-prefix fixture, with ``stats_source`` null on vLLM
  0.26 and 0.27.1. No saving is modelled for it.
* **Sampling / staggering / skipping bars.** Applies only to the experiment
  workload, where the lab owns the population. Doing it to a user's agent
  produces a product that looks broken.
* **Self-hosting.** A separate experiment with its own throughput
  measurements; nothing here converts GPU time to a per-call price.
* **Fallback rate.** Unrecoverable from stored data in either direction.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cadence_model import (  # noqa: E402
    ASSUMED, CADENCES, TRADING_DAYS_PER_MONTH,
)
from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, MissingInput, NOT_MEASURED, Fig, cost_per_call,
    load_measured,
)

__all__ = [
    "LAB_CONTROLLED", "USER_DETERMINED",
    "EXPERIMENT_PARAMS", "PLATFORM_PARAMS",
    "experiment_cost", "platform_cost", "leaderboard_recurring",
    "experiment_sensitivity", "platform_sensitivity",
    "power_question_options", "format_workload_report",
]

LAB_CONTROLLED = "LAB-CONTROLLED"
USER_DETERMINED = "USER-DETERMINED"

# Trading-day count is shared with cadence_model rather than redeclared.
_DAYS = TRADING_DAYS_PER_MONTH


@dataclass(frozen=True)
class Param:
    """One model input: who controls it, and whether it may be optimised."""

    name: str
    control: str
    description: str
    # Experiment C's independent variable cannot be optimised away without
    # destroying the experiment. Flagged so a sensitivity ranking cannot be
    # read as a to-do list.
    required_by_experiment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "control": self.control,
                "description": self.description,
                "required_by_experiment": self.required_by_experiment}


EXPERIMENT_PARAMS: Dict[str, Param] = {
    "n_agents": Param(
        "n_agents", LAB_CONTROLLED,
        "How many agents the experiment runs. A power/design choice — see "
        "power_question_options(); nothing in the NVIDIA note fixes it."),
    "n_distinct_models": Param(
        "n_distinct_models", LAB_CONTROLLED,
        "How many distinct models the population spans. THIS IS EXPERIMENT C'S "
        "INDEPENDENT VARIABLE: correlated failure across models cannot be "
        "measured with one model. Reducing it does not make the experiment "
        "cheaper, it makes it a different experiment.",
        required_by_experiment=True),
    "cadence_seconds": Param(
        "cadence_seconds", LAB_CONTROLLED,
        "Seconds between decisions. Free: the lab sets it. Drives cost "
        "linearly and is the largest free lever."),
    "calls_per_decision": Param(
        "calls_per_decision", LAB_CONTROLLED,
        "Pipeline depth. Measured at 1.000 for the seed leaderboard runs; "
        "3-step and 5-step pipelines measured 3.000 and 5.000."),
    "experiment_duration_days": Param(
        "experiment_duration_days", LAB_CONTROLLED,
        "How long the experiment runs. Free, and multiplies everything."),
}

PLATFORM_PARAMS: Dict[str, Param] = {
    "n_users": Param(
        "n_users", USER_DETERMINED,
        "Follows signups. Not a lever."),
    "agents_per_user": Param(
        "agents_per_user", USER_DETERMINED,
        "Users create agents as they like. A quota is the only lever and it is "
        "a product decision, not an engineering one."),
    "backtests_per_user_per_day": Param(
        "backtests_per_user_per_day", USER_DETERMINED,
        "Users run backtests when they like. Caching makes repeats free; it "
        "does not reduce distinct runs."),
    "model_mix": Param(
        "model_mix", USER_DETERMINED,
        "Users pick from the dropdown. LAB-CONTROLLED only in which models the "
        "dropdown OFFERS and which is the default — see dropdown composition."),
    "pipeline_depth_distribution": Param(
        "pipeline_depth_distribution", USER_DETERMINED,
        "Users configure their own pipeline depth. Per-step routing reduces "
        "the cost of each step without changing the count, and is invisible."),
    "platform_paid_fraction": Param(
        "platform_paid_fraction", LAB_CONTROLLED,
        "The share of user activity the PLATFORM actually pays for. Not every "
        "user LLM call is an operator cost: token_cost.py states that external "
        "agents run their own client, so 'the backend never sees the real "
        "token counts' and pays for none of them; the credits module meters "
        "only POST /backtest/run with decision_source='llm', explicitly "
        "excluding the protocol surfaces for that reason. A BYOK vault is "
        "merged upstream (credential storage only so far), which points the "
        "same way. Pricing every user call as an operator cost overstates the "
        "bill by whatever this fraction is not."),
    "live_trading_on": Param(
        "live_trading_on", USER_DETERMINED,
        "Whether a user's agent trades live, i.e. runs continuously at cadence "
        "rather than only on demand. The dominant term when it is on."),
}

# The platform levers that ARE the lab's. Named explicitly so the report can
# separate "what we can do" from "what we must absorb".
PLATFORM_LEVERS = (
    "which models the dropdown offers",
    "the default model",
    "per-step routing (invisible to users)",
    "backtest result caching (invisible to users)",
    "leaderboard cadence",
    "per-user quota (a product decision)",
)


def _require(params: Dict[str, Any], needed: Sequence[str],
             catalogue: Dict[str, Param]) -> None:
    """Refuse, naming each blank input and who controls it."""
    missing = [k for k in needed
               if params.get(k) is None or params.get(k) == {}]
    if missing:
        lines = [f"cannot compute: {len(missing)} input(s) are blank"]
        for k in missing:
            p = catalogue.get(k)
            lines.append(f"  - {k} [{p.control if p else '?'}]: "
                         f"{p.description if p else 'no description'}")
        lines.append("These are deliberately blank. A default here would "
                     "become a finding.")
        raise MissingInput("\n".join(lines))


def _decisions_per_day(cadence_seconds: float, session_hours: float = 6.5) -> float:
    """Decisions one agent makes in a session. The 22x term."""
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")
    return (session_hours * 3600.0) / cadence_seconds


def _blended_cost_per_call(measured: Dict[str, Any],
                           model_mix: Dict[str, float]) -> float:
    """Weighted per-call cost. Weights must be a distribution over known models."""
    models = measured["models"]
    unknown = set(model_mix) - set(models)
    if unknown:
        raise MissingInput(
            f"no measured cost for {sorted(unknown)}. Known models: "
            f"{sorted(models)}. A model without a measured cost cannot be "
            f"priced by analogy.")
    total = sum(model_mix.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"model_mix must sum to 1.0, got {total}")
    return sum(cost_per_call(models[k]) * w for k, w in model_mix.items())


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------


def experiment_cost(params: Dict[str, Any],
                    measured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cost of the lab-controlled experiment workload.

    Every parameter is a lever, so the output carries a sensitivity ranking and
    a flag for the one parameter the experiment requires to be varied.
    """
    measured = measured or load_measured()
    needed = list(EXPERIMENT_PARAMS)
    _require(params, needed, EXPERIMENT_PARAMS)

    n_agents = params["n_agents"]
    n_models = params["n_distinct_models"]
    cadence_s = params["cadence_seconds"]
    cpd = params["calls_per_decision"]
    days = params["experiment_duration_days"]
    session_hours = params.get("session_hours", 6.5)

    # The population spans n_distinct_models; absent an explicit mix, an even
    # split across the cheapest n models is NOT assumed — the caller supplies
    # the mix or names the models.
    mix = params.get("model_mix")
    if mix is None:
        models = params.get("models")
        if not models:
            raise MissingInput(
                "experiment_cost needs either model_mix or models: which "
                "models the population spans determines cost across a 193x "
                "span, so an even split over an unnamed set would be an "
                "invented figure.")
        if len(models) != n_models:
            raise ValueError(
                f"models lists {len(models)} entries but n_distinct_models is "
                f"{n_models}")
        mix = {m: 1.0 / len(models) for m in models}

    per_call = _blended_cost_per_call(measured, mix)
    decisions_per_agent_day = _decisions_per_day(cadence_s, session_hours)
    calls_per_day = n_agents * decisions_per_agent_day * cpd
    cost_per_day = calls_per_day * per_call

    return {
        "workload": "experiment",
        "params": {k: params.get(k) for k in needed},
        "model_mix": mix,
        "blended_cost_per_call_usd": Fig(
            per_call, MEASURED,
            "load_measured(); per-model costs verified against stored "
            "est_cost_usd to <1e-5"),
        "decisions_per_agent_day": Fig(
            decisions_per_agent_day, DERIVED,
            f"{session_hours}h session / {cadence_s}s"),
        "calls_per_day": Fig(calls_per_day, DERIVED, "agents x decisions x depth"),
        "cost_per_day_usd": Fig(cost_per_day, DERIVED, "calls/day x $/call"),
        "cost_per_month_usd": Fig(
            cost_per_day * _DAYS, DERIVED, f"{_DAYS} trading days"),
        "total_cost_usd": Fig(
            cost_per_day * days, DERIVED, f"{days}-day experiment"),
        "all_parameters_are_levers": True,
        "required_by_experiment": [
            p.name for p in EXPERIMENT_PARAMS.values() if p.required_by_experiment],
        "free_parameters": [
            p.name for p in EXPERIMENT_PARAMS.values()
            if not p.required_by_experiment],
    }


def experiment_sensitivity(params: Dict[str, Any],
                           measured: Optional[Dict[str, Any]] = None,
                           factor: float = 2.0) -> List[Dict[str, Any]]:
    """Rank parameters by cost impact, marking which may not be optimised.

    Cost is multiplicative in agents, cadence, depth and duration, so doubling
    any of them doubles cost — the ranking is only interesting because the
    MODEL MIX is not multiplicative and because one parameter is off-limits.
    """
    measured = measured or load_measured()
    base = experiment_cost(params, measured)["total_cost_usd"].value

    rows = []
    for name, p in EXPERIMENT_PARAMS.items():
        if name == "n_distinct_models":
            # Varying it means varying the mix, which is a different question
            # from scaling a scalar. Handled below by swapping the mix.
            continue
        bumped = dict(params)
        bumped[name] = params[name] * factor
        if name == "cadence_seconds":
            # Cadence is inverse: a longer interval means FEWER decisions.
            bumped[name] = params[name] * factor
        cost = experiment_cost(bumped, measured)["total_cost_usd"].value
        rows.append({
            "parameter": name,
            "control": p.control,
            "required_by_experiment": p.required_by_experiment,
            "factor_applied": factor,
            "cost_ratio": cost / base if base else None,
            "note": ("inverse: a longer interval reduces cost"
                     if name == "cadence_seconds" else "linear"),
        })

    # Model mix, varied by swapping the whole population onto each single model.
    models = measured["models"]
    costs = {k: cost_per_call(v) for k, v in models.items()}
    cheapest, dearest = min(costs, key=costs.get), max(costs, key=costs.get)
    rows.append({
        "parameter": "n_distinct_models / model_mix",
        "control": LAB_CONTROLLED,
        "required_by_experiment": True,
        "factor_applied": None,
        "cost_ratio": costs[dearest] / costs[cheapest],
        "note": (f"span from all-{cheapest} to all-{dearest}. THIS IS THE "
                 f"EXPERIMENT'S INDEPENDENT VARIABLE — the ranking shows its "
                 f"cost, not a saving that may be taken."),
    })

    rows.sort(key=lambda r: -(r["cost_ratio"] or 0))
    return rows


def power_question_options(measured: Optional[Dict[str, Any]] = None,
                           *, duration_days: int = 30,
                           models: Optional[Sequence[str]] = None
                           ) -> Dict[str, Any]:
    """The question nobody has asked, with the cost of each answer attached.

    Does measuring correlated failure need 1,000 agents at 2-minute cadence
    across 7 models, or would a smaller population, a slower cadence, or fewer
    models give adequate statistical power?

    This function does NOT answer it — the required power depends on the effect
    size being sought, which nobody has stated. It prices the options so the
    question can be asked with numbers attached.

    REDUCING MODEL COUNT IS PRICED AS A RANGE, NOT A NUMBER. Which models are
    dropped matters more than how many: the per-call span is 193x, so "3
    models" costs anywhere between the three cheapest and the three dearest.
    An arbitrary slice of the roster would produce a precise-looking figure
    that is an artifact of dict ordering.
    """
    measured = measured or load_measured()
    names = list(models or measured["models"].keys())
    by_cost = sorted(names, key=lambda k: cost_per_call(measured["models"][k]))

    def _cost(n_agents, cadence_s, chosen):
        return experiment_cost({
            "n_agents": n_agents,
            "n_distinct_models": len(chosen),
            "cadence_seconds": cadence_s,
            "calls_per_decision": 1,
            "experiment_duration_days": duration_days,
            "models": list(chosen),
        }, measured)["total_cost_usd"].value

    full = _cost(1000, 120, by_cost)
    options = []
    for label, n_agents, cadence_s in (
            ("as described: 1000 agents, 2 min, 7 models", 1000, 120),
            ("300 agents, 2 min, 7 models", 300, 120),
            ("1000 agents, 10 min, 7 models", 1000, 600),
            ("300 agents, 10 min, 7 models", 300, 600),
    ):
        c = _cost(n_agents, cadence_s, by_cost)
        options.append({
            "option": label, "n_agents": n_agents,
            "cadence_seconds": cadence_s, "n_models": len(by_cost),
            "cost_usd": c, "cost_usd_high": None,
            "ratio_to_full": c / full if full else None,
            "destroys_experiment": False,
            "note": "model set unchanged; agent count and cadence are free choices",
        })

    # Fewer models: a RANGE, because which three dominates how many.
    cheap3, dear3 = by_cost[:3], by_cost[-3:]
    lo, hi = _cost(1000, 120, cheap3), _cost(1000, 120, dear3)
    options.append({
        "option": "1000 agents, 2 min, 3 models",
        "n_agents": 1000, "cadence_seconds": 120, "n_models": 3,
        "cost_usd": lo, "cost_usd_high": hi,
        "ratio_to_full": lo / full if full else None,
        "ratio_to_full_high": hi / full if full else None,
        "destroys_experiment": True,
        "note": (f"RANGE, not a figure: {lo:,.0f} for the three cheapest "
                 f"({', '.join(cheap3)}) to {hi:,.0f} for the three dearest "
                 f"({', '.join(dear3)}) — a {hi / lo:.0f}x span from the same "
                 f"'3 models'. And it changes the experiment: model diversity "
                 f"is the independent variable."),
    })

    return {
        "question": (
            "Does measuring correlated failure across models need 1,000 agents "
            "at 2-minute cadence across 7 models? Agent count, cadence and "
            "duration are power/design choices, not requirements. Model count "
            "is the independent variable and reducing it changes the question "
            "being asked."),
        "answerable_here": False,
        "why_not": (
            "Required power depends on the effect size sought — how correlated "
            "is 'correlated', and at what confidence. Nobody has stated it, so "
            "this prices the options rather than choosing among them."),
        "duration_days": duration_days,
        "models_by_cost": by_cost,
        "options": options,
        "tier": NOT_MEASURED,
    }


# --------------------------------------------------------------------------
# platform
# --------------------------------------------------------------------------


def platform_cost(params: Dict[str, Any],
                  measured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cost of the user workload, split into what follows signups and what does not.

    Backtests are on-demand and bounded by user behaviour. Live agents run at
    cadence and are the dominant term when enabled, which is why
    ``live_trading_on`` is separated rather than folded into a blended rate.
    """
    measured = measured or load_measured()
    needed = ["n_users", "agents_per_user", "backtests_per_user_per_day",
              "model_mix", "pipeline_depth_distribution", "live_trading_on"]
    _require(params, needed, PLATFORM_PARAMS)

    # Who pays is a separate axis from how much it costs, and conflating them
    # overstates the operator's bill. Server-side backtests are operator-paid;
    # protocol/external agents run their own client and cost the operator
    # nothing; BYOK (merged upstream as credential storage) points the same
    # way. No telemetry records the split, so it is not defaulted.
    platform_paid = params.get("platform_paid_fraction")
    if platform_paid is not None and not 0.0 <= platform_paid <= 1.0:
        raise ValueError("platform_paid_fraction must be in [0, 1]")

    per_call = _blended_cost_per_call(measured, params["model_mix"])

    depth_dist = params["pipeline_depth_distribution"]
    dtotal = sum(depth_dist.values())
    if abs(dtotal - 1.0) > 1e-6:
        raise ValueError(f"pipeline_depth_distribution must sum to 1.0, got {dtotal}")
    mean_depth = sum(float(d) * w for d, w in depth_dist.items())

    n_users = params["n_users"]
    agents = n_users * params["agents_per_user"]

    # --- on-demand backtests ---
    bars_per_backtest = params.get("bars_per_backtest")
    if bars_per_backtest is None:
        raise MissingInput(
            "bars_per_backtest is blank. The seed runs measured 161 bars for a "
            "one-month hourly window; at Nof1 cadence the same window is ~3276. "
            "It is not defaulted because the two differ by 20x.")
    backtest_calls_day = (n_users * params["backtests_per_user_per_day"]
                          * bars_per_backtest * mean_depth)

    # --- live agents at cadence ---
    live_calls_day = 0.0
    live_fraction = params.get("live_fraction", 1.0 if params["live_trading_on"] else 0.0)
    if params["live_trading_on"]:
        cadence_s = params.get("cadence_seconds")
        if cadence_s is None:
            raise MissingInput(
                "live_trading_on is True but cadence_seconds is blank. Live "
                "cost is entirely a function of cadence — hourly and Nof1 "
                "differ by 22x — so it cannot be defaulted.")
        live_calls_day = (agents * live_fraction
                          * _decisions_per_day(cadence_s,
                                               params.get("session_hours", 6.5))
                          * mean_depth)

    total_calls_day = backtest_calls_day + live_calls_day
    gross_cost_day = total_calls_day * per_call
    cost_day = gross_cost_day
    who_pays: Dict[str, Any] = {
        "platform_paid_fraction": platform_paid,
        "note": (
            "gross cost assumes the operator pays for EVERY call. External "
            "agents run their own LLM client (token_cost.py: 'the backend "
            "never sees the real token counts'), and the credits module "
            "meters only server-side llm backtests, so the operator's true "
            "bill is a subset."),
    }
    if platform_paid is None:
        who_pays["operator_cost_per_month_usd"] = None
        who_pays["refused"] = (
            "platform_paid_fraction is blank, so only the GROSS figure is "
            "reported. What the operator actually pays needs the share of "
            "activity that runs on the operator's own key, and nothing "
            "records it.")
    else:
        cost_day = gross_cost_day * platform_paid
        who_pays["operator_cost_per_month_usd"] = Fig(
            cost_day * _DAYS, DERIVED, "gross x platform_paid_fraction")

    return {
        "workload": "platform",
        "gross_cost_per_month_usd": Fig(
            gross_cost_day * _DAYS, DERIVED,
            "every call priced as operator-paid — an UPPER bound"),
        "who_pays": who_pays,
        "params": {k: params.get(k) for k in needed},
        "blended_cost_per_call_usd": Fig(per_call, MEASURED, "load_measured()"),
        "mean_pipeline_depth": Fig(mean_depth, NOT_MEASURED,
                                   "supplied distribution; no telemetry exists"),
        "n_agents": Fig(agents, DERIVED, "n_users x agents_per_user"),
        "backtest_calls_per_day": Fig(backtest_calls_day, DERIVED,
                                      "users x backtests x bars x depth"),
        "live_calls_per_day": Fig(live_calls_day, DERIVED,
                                  "agents x live_fraction x decisions x depth"),
        "cost_per_day_usd": Fig(cost_day, DERIVED, "calls x $/call"),
        "cost_per_month_usd": Fig(cost_day * _DAYS, DERIVED, f"{_DAYS} trading days"),
        "cost_per_user_per_month_usd": Fig(
            (cost_day * _DAYS / n_users) if n_users else None, DERIVED,
            "the unit that decides whether a price plan covers cost"),
        "lab_controlled_levers": list(PLATFORM_LEVERS),
        "user_determined": [p.name for p in PLATFORM_PARAMS.values()
                            if p.control == USER_DETERMINED],
    }


def platform_sensitivity(params: Dict[str, Any],
                         measured: Optional[Dict[str, Any]] = None,
                         factor: float = 2.0) -> List[Dict[str, Any]]:
    """Rank platform parameters, separating levers from things that just happen."""
    measured = measured or load_measured()
    base = platform_cost(params, measured)["cost_per_month_usd"].value

    rows = []
    for name in ("n_users", "agents_per_user", "backtests_per_user_per_day",
                 "bars_per_backtest"):
        if params.get(name) is None:
            continue
        bumped = dict(params)
        bumped[name] = params[name] * factor
        cost = platform_cost(bumped, measured)["cost_per_month_usd"].value
        p = PLATFORM_PARAMS.get(name)
        rows.append({
            "parameter": name,
            "control": p.control if p else LAB_CONTROLLED,
            "factor_applied": factor,
            "cost_ratio": cost / base if base else None,
            "is_lever": False if p and p.control == USER_DETERMINED else True,
        })

    models = measured["models"]
    costs = {k: cost_per_call(v) for k, v in models.items()}
    cheapest, dearest = min(costs, key=costs.get), max(costs, key=costs.get)
    rows.append({
        "parameter": "model_mix (dropdown composition)",
        "control": f"{USER_DETERMINED} choice from a {LAB_CONTROLLED} menu",
        "factor_applied": None,
        "cost_ratio": costs[dearest] / costs[cheapest],
        "is_lever": True,
        "note": (f"users choose, the lab decides what they choose FROM. "
                 f"all-{cheapest} to all-{dearest}."),
    })

    rows.sort(key=lambda r: -(r["cost_ratio"] or 0))
    return rows


# --------------------------------------------------------------------------
# leaderboard — reported separately from both
# --------------------------------------------------------------------------


def leaderboard_recurring(measured: Optional[Dict[str, Any]] = None,
                          *, bars_per_daily_window: Optional[int] = None
                          ) -> Dict[str, Any]:
    """The leaderboard's cost, which follows neither workload.

    It runs whether or not any user is active, so it belongs in neither model.

    THE RECURRING FIGURE IS CURRENTLY ZERO, and that is a fact about the repo
    rather than an estimate — but the reason has changed. Upstream now ships
    ``.github/workflows/daily-leaderboard.yml`` with a weekday cron that is
    commented out on purpose, because the Daily board was replaced by the Live
    board and nightly deploys would bill for curves nobody can open. So the
    zero is a deliberate pause of working infrastructure, not its absence.

    That makes the counterfactual worth pricing, and ``bars_per_daily_window``
    now has a concrete meaning: the cron is ``30 22 * * 1-5`` (weekdays) and a
    scheduled run always sets ``deploy_models=true``, so re-enabling it deploys
    all seven models over a rolling one-day window every trading day.
    """
    measured = measured or load_measured()
    models = measured["models"]
    per_call = {k: cost_per_call(v) for k, v in models.items()}
    contest_calls = {k: v["llm_calls"] for k, v in models.items()}

    contest_total = sum(per_call[k] * contest_calls[k] for k in models)

    out: Dict[str, Any] = {
        "scheduled_today": False,
        "scheduled_evidence": (
            "upstream now HAS the scheduler and has deliberately switched it "
            "off: .github/workflows/daily-leaderboard.yml carries "
            "cron '30 22 * * 1-5' with the whole schedule: block commented "
            "out, and workflow_dispatch defaults deploy_models=false. Its own "
            "comment gives the reason: 'left on schedule it would keep "
            "deploying every competition LLM nightly, billable, for a board "
            "nobody can open.' Earlier phases said no scheduler existed; that "
            "was true of the branch and is no longer true of upstream."),
        "recurring_cost_usd_per_month": Fig(
            0.0, MEASURED,
            "the schedule block is commented out, so nothing fires. $0 by "
            "deliberate pause, NOT by absence of infrastructure — the "
            "difference matters because re-enabling it is one uncomment"),
        "contest_window": {
            "description": "one-off per manual deploy: 7 models over the "
                           "161-bar contest window",
            "calls_per_model": contest_calls,
            "cost_usd": Fig(contest_total, MEASURED,
                            "measured per-call costs x measured call counts"),
        },
        "daily_window": {
            "description": "rolling 1-day window (last completed US weekday), "
                           "IF a scheduler is added",
        },
    }

    if bars_per_daily_window is None:
        out["daily_window"]["cost_usd"] = None
        out["daily_window"]["refused"] = (
            "bars_per_daily_window is blank. The daily board is a rolling "
            "ONE-DAY window, so its call count is that day's bar count — ~7 at "
            "hourly cadence, ~156 at Nof1. Those differ by 22x and the figure "
            "is not defaulted.")
    else:
        daily = sum(per_call[k] * bars_per_daily_window for k in models)
        out["daily_window"]["bars"] = bars_per_daily_window
        out["daily_window"]["cost_usd_per_run"] = Fig(
            daily, DERIVED, f"7 models x {bars_per_daily_window} bars")
        out["daily_window"]["cost_usd_per_month_if_scheduled"] = Fig(
            daily * _DAYS, DERIVED,
            f"{_DAYS} trading days — HYPOTHETICAL; no scheduler exists")
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_workload_report(experiment: Dict[str, Any],
                           platform: Dict[str, Any],
                           leaderboard: Dict[str, Any]) -> str:
    lines = ["=" * 78, "TWO WORKLOADS, SEPARATELY PARAMETERISED", "=" * 78]

    lines.append("\nEXPERIMENT (every parameter is a lever)")
    for k in ("blended_cost_per_call_usd", "decisions_per_agent_day",
              "calls_per_day", "cost_per_day_usd", "cost_per_month_usd",
              "total_cost_usd"):
        f = experiment[k]
        lines.append(f"  {k:32s} {f.tagged(',.2f')}")
    lines.append(f"  required (cannot be optimised): "
                 f"{experiment['required_by_experiment']}")
    lines.append(f"  free: {experiment['free_parameters']}")

    lines.append("\nPLATFORM (cost follows signups)")
    for k in ("blended_cost_per_call_usd", "n_agents",
              "backtest_calls_per_day", "live_calls_per_day",
              "cost_per_month_usd", "cost_per_user_per_month_usd"):
        f = platform[k]
        lines.append(f"  {k:32s} {f.tagged(',.2f')}")
    lines.append("  lab-controlled levers:")
    for lever in platform["lab_controlled_levers"]:
        lines.append(f"    - {lever}")

    lines.append("\nLEADERBOARD (neither workload)")
    lines.append(f"  scheduled today: {leaderboard['scheduled_today']}")
    lines.append(f"  recurring/month: "
                 f"{leaderboard['recurring_cost_usd_per_month'].tagged(',.2f')}")
    lines.append(f"  one-off contest deploy: "
                 f"{leaderboard['contest_window']['cost_usd'].tagged(',.2f')}")
    dw = leaderboard["daily_window"]
    if dw.get("refused"):
        lines.append(f"  daily window: REFUSED — {dw['refused'][:70]}...")
    return "\n".join(lines)
