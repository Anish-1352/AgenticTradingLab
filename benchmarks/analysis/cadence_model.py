"""Cost, crossover and latency re-scoped to Nof1 decision cadence.

Every prior figure in this project assumed ATL's hourly bar: ~7 decisions per
agent per trading day. Nof1's Alpha Arena runs inference every 2-3 minutes —
~156 decisions/day on a US equity session, ~500 on 24/7 crypto. That is a
21-70x change in call volume, and it inverts three earlier conclusions:

1. **"Live trading cannot reach $50K/month."** True at 7 decisions/day. At 156
   or 500 it is false for several models.
2. **"Self-hosting never pays at this scale."** True at ~6,300 calls/day. The
   crossover is a fixed calls/day number; raising volume 70x walks straight
   through it.
3. **"Latency is harmless because the engine cannot model it."** True when a
   3,600s bar swallowed a 596s decision. At a 120-180s bar it does not.

NOTHING HERE IS RE-DERIVED
---------------------------
Unit costs, calls-per-decision, Arm A/B/C throughput and latency are all
measured and verified elsewhere. This module only recombines them at a
different cadence. The one new input is the cadence itself, and it is
**assumed, not confirmed** — every figure downstream of it carries that tag.

THE HOURLY FIGURES ARE KEPT
----------------------------
Deliberately not replaced. The comparison is the finding: the same measured
unit costs produce opposite conclusions at the two cadences, which is a
statement about scope, not about the measurements.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, NOT_MEASURED, cost_per_call, load_arm_a, load_measured,
)

__all__ = ["ASSUMED", "CADENCES", "Cadence", "cadence_table", "crossover_table",
           "latency_vs_bar_table", "gpu_ceiling", "FIXTURE_GAP"]

# A fourth tier, used only for the cadence itself. It is not MEASURED (no Nof1
# run exists here), not DERIVED (it is an input, not arithmetic), and not
# NOT_MEASURED (that tier means "absent"; this one is present and load-bearing).
# Every figure computed from it inherits the tag.
ASSUMED = "ASSUMED — not confirmed with advisor"

# Standard scenario count. 300 agents is ATL's stated fleet size.
DEFAULT_AGENTS = 300
TRADING_DAYS_PER_MONTH = 21
CALENDAR_DAYS_PER_MONTH = 30

# Cloud A100 80GB, on-demand, one card. Stated rather than derived: rates vary
# by provider, region and commitment, and a reserved card is materially cheaper.
# The crossover moves linearly with this number.
A100_USD_PER_MONTH = 1440.0


@dataclass(frozen=True)
class Cadence:
    """One decision cadence. ``decisions_per_day`` is the load-bearing input."""

    name: str
    interval_seconds: int
    session_hours: float
    days_per_month: int
    decisions_per_day: int
    tier: str
    note: str

    @property
    def is_measured(self) -> bool:
        return self.tier == MEASURED


CADENCES: Dict[str, Cadence] = {
    "hourly": Cadence(
        name="hourly (ATL today)",
        interval_seconds=3600,
        session_hours=6.5,
        days_per_month=TRADING_DAYS_PER_MONTH,
        decisions_per_day=7,
        # Measured: the seed runs recorded 161 bars over a one-month window,
        # and the engine requests TimeFrame.Hour.
        tier=MEASURED,
        note="161 bars over ~23 trading days in the seed runs; "
             "alpaca_bars.py requests TimeFrame.Hour",
    ),
    "nof1_equity": Cadence(
        name="Nof1 equity (2.5 min)",
        interval_seconds=150,
        session_hours=6.5,
        days_per_month=TRADING_DAYS_PER_MONTH,
        decisions_per_day=156,
        tier=ASSUMED,
        note="6.5h session / 150s interval. Cadence from a description of "
             "Alpha Arena, not from a run this project executed.",
    ),
    "nof1_crypto": Cadence(
        name="Nof1 crypto (2.5 min, 24/7)",
        interval_seconds=150,
        session_hours=24.0,
        days_per_month=CALENDAR_DAYS_PER_MONTH,
        decisions_per_day=500,
        tier=ASSUMED,
        note="24h / 150s interval, rounded down from 576. Crypto trades "
             "continuously, so calendar days apply rather than trading days.",
    ),
}


def cadence_table(
    measured: Optional[Dict[str, Any]] = None,
    *,
    agents: int = DEFAULT_AGENTS,
    calls_per_decision: Sequence[int] = (1, 3, 5),
    budget_usd: float = 50_000.0,
) -> Dict[str, Any]:
    """Monthly cost per model at each cadence and pipeline depth.

    Calls per decision is MEASURED (3 steps -> 3.000, 5 -> 5.000 against the
    real API); the cadence is ASSUMED. So each cell is measured unit cost times
    a measured multiplier times an assumed volume.
    """
    measured = measured or load_measured()
    rows: List[Dict[str, Any]] = []
    for cad_key, cad in CADENCES.items():
        for depth in calls_per_decision:
            calls_per_day = agents * cad.decisions_per_day * depth
            for name, m in measured["models"].items():
                cpc = cost_per_call(m)
                monthly = calls_per_day * cad.days_per_month * cpc
                rows.append({
                    "cadence": cad_key,
                    "cadence_name": cad.name,
                    "cadence_tier": cad.tier,
                    "decisions_per_agent_per_day": cad.decisions_per_day,
                    "calls_per_decision": depth,
                    "agents": agents,
                    "calls_per_day": calls_per_day,
                    "calls_per_month": calls_per_day * cad.days_per_month,
                    "db_model": name,
                    "slug": m["slug"],
                    "cost_per_call": cpc,
                    "monthly_usd": monthly,
                    "over_budget": monthly > budget_usd,
                })
    over = [r for r in rows if r["over_budget"]]
    return {
        "rows": rows,
        "agents": agents,
        "budget_usd": budget_usd,
        "cells_over_budget": len(over),
        "cells_total": len(rows),
        "cheapest_over_budget": min(
            (r for r in over), key=lambda r: r["cost_per_call"], default=None),
    }


def gpu_ceiling(arm_c_rps: float, cadence: Cadence, *,
                calls_per_decision: int = 1) -> Dict[str, Any]:
    """How many agents one GPU serves at a cadence, from measured throughput.

    Arm C measured 9.28 completed req/s at C=32. That is a hard per-card
    ceiling for this model and prompt shape: an agent needs
    ``decisions_per_day x calls_per_decision`` requests per day, so the card
    serves ``rps x seconds_per_day`` divided by that.
    """
    seconds_per_day = cadence.session_hours * 3600.0
    capacity_per_day = arm_c_rps * seconds_per_day
    per_agent_per_day = cadence.decisions_per_day * calls_per_decision
    agents_per_gpu = capacity_per_day / per_agent_per_day if per_agent_per_day else None
    return {
        "arm_c_rps": arm_c_rps,
        "session_seconds_per_day": seconds_per_day,
        "capacity_requests_per_day": capacity_per_day,
        "requests_per_agent_per_day": per_agent_per_day,
        "agents_per_gpu": agents_per_gpu,
        "gpus_for_300_agents": (
            math.ceil(DEFAULT_AGENTS / agents_per_gpu) if agents_per_gpu else None),
    }


def crossover_table(
    measured: Optional[Dict[str, Any]] = None,
    *,
    arm_c_rps: Optional[float] = None,
    gpu_usd_per_month: float = A100_USD_PER_MONTH,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Calls/day at which one self-hosted GPU undercuts the hosted API.

    A GPU bills the same whether saturated or idle, so self-hosting is a flat
    monthly cost and the API is a straight line through the origin. They cross
    at exactly one call volume per model:

        crossover_calls_per_day = gpu_usd_per_month / (cost_per_call x days)

    The earlier "self-hosting never pays" conclusion was this arithmetic at
    ~6,300 calls/day. It is not a property of the stack; it is a property of the
    volume, and the volume is what changed.
    """
    measured = measured or load_measured()
    if arm_c_rps is None:
        path = os.path.join(results_dir or os.path.join(_BENCH_ROOT, "results"),
                            "armC_shared_summary.json")
        with open(path) as fh:
            levels = json.load(fh)["levels"]
        arm_c_rps = max(lv["completed_requests_per_s"] for lv in levels)

    rows: List[Dict[str, Any]] = []
    for name, m in measured["models"].items():
        cpc = cost_per_call(m)
        per_day = gpu_usd_per_month / CALENDAR_DAYS_PER_MONTH
        crossover_calls_day = (per_day / cpc) if cpc else None
        entry: Dict[str, Any] = {
            "db_model": name, "slug": m["slug"], "cost_per_call": cpc,
            "crossover_calls_per_day": crossover_calls_day,
            "by_cadence": {},
        }
        for cad_key, cad in CADENCES.items():
            calls_day = DEFAULT_AGENTS * cad.decisions_per_day  # 1 call/decision
            entry["by_cadence"][cad_key] = {
                "calls_per_day": calls_day,
                "past_crossover": (
                    calls_day > crossover_calls_day if crossover_calls_day else None),
                "api_monthly": calls_day * cad.days_per_month * cpc,
            }
        rows.append(entry)
    rows.sort(key=lambda r: r["cost_per_call"])

    ceilings = {k: gpu_ceiling(arm_c_rps, c) for k, c in CADENCES.items()}
    return {
        "rows": rows,
        "arm_c_rps": arm_c_rps,
        "gpu_usd_per_month": gpu_usd_per_month,
        "agents": DEFAULT_AGENTS,
        "ceilings": ceilings,
        "assumptions": [
            f"A100 80GB at ${gpu_usd_per_month:,.0f}/month on-demand, one card "
            f"[NOT MEASURED] — a stated rate, not a quote. Provider, region "
            f"and commitment move every crossover here linearly.",
            "Self-hosted cost is the card alone: engineering, on-call and "
            "failover are not priced, so every crossover here is a LOWER bound "
            "on the volume that justifies self-hosting.",
            f"Throughput is Arm C's {arm_c_rps:.3f} req/s at C=32 [MEASURED] on "
            f"the shared_prefix fixture — see the fixture gap.",
        ],
    }


# Candidate bar intervals for the latency question. 3,600s is included so the
# hourly result stays visible beside the ones that bind.
BAR_INTERVALS_S = (60, 120, 150, 180, 300, 3600)


def latency_vs_bar_table(results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Measured e2e latency against candidate bar intervals.

    A decision that takes longer than its bar is not late — it is a decision
    the engine cannot place, because the next bar has already arrived. At 3,600s
    every measured configuration fits, which is exactly why the latency audit
    concluded the engine's blindness to timing was harmless. At 150s it is not.
    """
    results_dir = results_dir or os.path.join(_BENCH_ROOT, "results")
    arms: List[Dict[str, Any]] = []

    for arm_key, path, label in (
        ("B", "armB_shared_summary.json", "Arm B (HF generate loop)"),
        ("C", "armC_shared_summary.json", "Arm C (vLLM)"),
    ):
        full = os.path.join(results_dir, path)
        if not os.path.exists(full):
            continue
        for lv in json.load(open(full))["levels"]:
            arms.append({
                "arm": arm_key, "label": label, "concurrency": lv["concurrency"],
                "e2e_p50": lv["e2e_p50"], "e2e_p95": lv["e2e_p95"],
                "e2e_p99": lv["e2e_p99"],
                "output_tokens": 256, "comparable": True,
            })

    arm_a = load_arm_a(results_dir)
    if arm_a.get("available"):
        for lv in arm_a["levels"]:
            arms.append({
                "arm": "A", "label": "Arm A (hosted API)",
                "concurrency": lv["concurrency"],
                "e2e_p50": lv["e2e_p50"], "e2e_p95": lv["e2e_p95"],
                "e2e_p99": lv["e2e_p99"],
                "output_tokens": None, "comparable": False,
            })

    for row in arms:
        row["fits"] = {
            bar: {
                "p50": row["e2e_p50"] <= bar,
                "p95": row["e2e_p95"] <= bar,
                "p99": row["e2e_p99"] <= bar,
                "overrun_factor_p50": (row["e2e_p50"] / bar) if bar else None,
            }
            for bar in BAR_INTERVALS_S
        }

    return {
        "rows": arms,
        "bar_intervals_s": list(BAR_INTERVALS_S),
        "arm_a_exclusion": (
            "Arm A's completions ran 53-86 output tokens because the provider "
            "ignored min_tokens, against the 256 arms B and C forced. Its "
            "latency is real but measures less work, so it is NOT directly "
            "comparable — it is included because a hosted decision's wall time "
            "is what a cadence has to accommodate, whatever it contains."
        ),
        "note": (
            "these are ONE call. A multi-step pipeline is sequential, so a "
            "3-step decision costs roughly 3x these figures end to end — from "
            "the 3.000 / 5.000 calls per decision [MEASURED]."
        ),
    }


FIXTURE_GAP = {
    "what": (
        "Arms B and C have only ever run on the shared_prefix fixture: 2,620 "
        "tokens per request with a 99.4% common prefix, by construction a "
        "deliberate UPPER BOUND on cross-agent overlap."
    ),
    "never_run": "atl_realistic (~10% overlap, built from measured ATL prompt "
                 "structure) has never been executed on arm B or arm C.",
    "why_it_matters": (
        "Every serving number in this project therefore describes a workload "
        "we have separately established is not ATL's. A 99.4% [MEASURED] "
        "shared prefix is the best possible case for a prefix cache and for "
        "batching efficiency."
    ),
    "claims_that_would_change": [
        "The 173x arm C / arm B throughput ratio at C=32 [MEASURED] — taken "
        "under maximal prefix sharing, and expected to shrink at ~10% "
        "overlap. Anything derived from it, including the GPU ceiling and "
        "every crossover in this module, moves with it.",
        "Arm C's 9.28 req/s per card [MEASURED], and therefore agents-per-GPU "
        "and the number of A100s a fleet needs.",
        "Any statement about prefix-cache benefit, which was never separately "
        "ablated on either arm.",
    ],
    "claims_that_hold": [
        "Arm B is launch-bound: 2.57M cudaLaunchKernel calls costing 2.6x the "
        "GPU's busy time, on a single stream with 79% idle [MEASURED]. A "
        "dispatch-mechanism finding; it does not depend on prompt overlap.",
        "Arm B's throughput INVERTS with concurrency, 0.109 -> 0.054 req/s "
        "[MEASURED]. The mechanism is per-request Python dispatch, not the "
        "prompts.",
        "Pipeline steps are sequential with a real data dependency, so "
        "intra-decision latency adds regardless of fixture.",
    ],
    "priority": (
        "TOP priority for the next GPU session: re-run arms B and C on "
        "atl_realistic at the same concurrency levels. Until then every "
        "throughput-derived figure carries the upper-bound caveat."
    ),
}


# ==========================================================================
# Heterogeneous multi-model serving (Phase 13)
# ==========================================================================

# Per-GPU monthly cost, on-demand cloud. SOURCES, stated rather than implied:
# these are the order-of-magnitude rates Colab/GCP/Lambda advertise for
# on-demand single cards, converted to a 730-hour month. They are NOT quotes,
# they vary by provider/region/commitment by more than 2x, and a reserved or
# spot card is materially cheaper. Every card-count figure below is a division
# by one of these, so the whole table moves linearly with them.
GPU_MONTHLY_USD: Dict[str, Dict[str, Any]] = {
    "A100-80GB": {"usd_per_month": 1440.0, "vram_gb": 80.0,
                  "source": "~$1.97/hr on-demand x 730h"},
    "L4-24GB": {"usd_per_month": 430.0, "vram_gb": 24.0,
                "source": "~$0.59/hr on-demand x 730h"},
    "A10G-24GB": {"usd_per_month": 730.0, "vram_gb": 24.0,
                  "source": "~$1.00/hr on-demand x 730h"},
    "T4-16GB": {"usd_per_month": 260.0, "vram_gb": 16.0,
                "source": "~$0.35/hr on-demand x 730h"},
}


def hosted_baseline_monthly(measured: Optional[Dict[str, Any]] = None, *,
                            agents: int = DEFAULT_AGENTS,
                            cadence_key: str = "nof1_equity",
                            calls_per_decision: int = 1) -> Dict[str, Any]:
    """Hosted-API cost for a fleet split evenly across the seven measured models.

    Even split is an assumption — production mix is unmeasured — but it is the
    one split that needs no further input, and the per-model column shows how
    lopsided the total is regardless.
    """
    measured = measured or load_measured()
    cad = CADENCES[cadence_key]
    per_model_agents = agents / len(measured["models"])
    rows = []
    for name, m in measured["models"].items():
        cpc = cost_per_call(m)
        calls_month = (per_model_agents * cad.decisions_per_day
                       * calls_per_decision * cad.days_per_month)
        rows.append({"db_model": name, "slug": m["slug"], "cost_per_call": cpc,
                     "calls_per_month": calls_month,
                     "monthly_usd": calls_month * cpc,
                     "self_hostable": not any(
                         v in m["slug"] for v in ("openai/", "google/", "anthropic/"))})
    rows.sort(key=lambda r: -r["monthly_usd"])
    total = sum(r["monthly_usd"] for r in rows)
    unhostable = sum(r["monthly_usd"] for r in rows if not r["self_hostable"])
    return {
        "rows": rows, "total_monthly_usd": total,
        "agents": agents, "cadence": cadence_key,
        "calls_per_day": agents * cad.decisions_per_day * calls_per_decision,
        "not_self_hostable_usd": unhostable,
        "not_self_hostable_share": (unhostable / total) if total else None,
        "note": (
            "Even split across the seven measured models [NOT MEASURED — the "
            "production mix is unknown]. The not-self-hostable share is the "
            "part no amount of GPU purchasing can address."
        ),
    }


def self_hosted_cards(*, n_models: int, models_per_card: Optional[float],
                      card: str = "A100-80GB") -> Dict[str, Any]:
    """Cards needed, given a MEASURED models-per-card figure.

    ``models_per_card`` is deliberately required rather than defaulted: it is
    the output of the sweep, and inventing it would make this function produce
    a confident number from nothing. None in, None out.
    """
    spec = GPU_MONTHLY_USD.get(card)
    if spec is None:
        raise KeyError(f"unknown card {card!r}; known: {sorted(GPU_MONTHLY_USD)}")
    if not models_per_card:
        return {
            "available": False,
            "card": card,
            "reason": (
                "models_per_card is not measured. It is the primary output of "
                "the heterogeneous sweep, and every figure here divides by it. "
                "Run the sweep."
            ),
        }
    cards = math.ceil(n_models / models_per_card)
    return {
        "available": True, "card": card, "n_models": n_models,
        "models_per_card": models_per_card, "cards_required": cards,
        "monthly_usd": cards * spec["usd_per_month"],
        "card_usd_per_month": spec["usd_per_month"],
        "card_source": spec["source"],
        "excludes": (
            "engineering, on-call, failover and idle capacity. A self-hosted "
            "figure without those is a lower bound on cost, so the saving "
            "shown is an upper bound."
        ),
    }
