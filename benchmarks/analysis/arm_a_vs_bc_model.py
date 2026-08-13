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
           "sensitivity", "to_csv", "format_report"]

DEFAULT_AGENT_COUNTS = (1, 10, 100, 500, 1000)
DEFAULT_GPU_USD_PER_HOUR = 2.0
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
    if api_cost is None:
        print("ERROR: supply --api-cost-per-request or --arm-a with a measured "
              "cost. The model will not invent a price.", file=sys.stderr)
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
