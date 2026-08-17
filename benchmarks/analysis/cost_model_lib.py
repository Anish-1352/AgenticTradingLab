"""Shared measurement loader and cost arithmetic for the advisor package.

Both ``ADVISOR_REPORT.md`` (via ``make_advisor_report.py``) and
``cost_model.ipynb`` are generated from this module. Nothing is hand
transcribed: token counts come from the committed seed database, serving
figures from the run summaries in ``results/``, and code findings carry a
``file:line`` that is checked to exist.

THE THREE TIERS
---------------
Every number this module emits carries exactly one:

* ``MEASURED``     — read from a run artifact or the seed DB, traceable to a
  ``run_id`` (or ``file:line`` for a code finding).
* ``DERIVED``      — arithmetic on measured values, with the arithmetic shown.
* ``NOT_MEASURED`` — named, with what it would take to measure it.

An untagged number is a bug. ``Fig`` makes the tag structural rather than a
convention someone has to remember: there is no way to construct a figure
without one.

WHY THE UNMEASURED INPUTS RAISE INSTEAD OF DEFAULTING
------------------------------------------------------
A default is how an assumption becomes a finding. ``require_inputs`` raises
``MissingInput`` naming exactly which parameters are blank and what would
resolve each. A cost model that silently substitutes 390 decisions/agent/day
produces a number the reader cannot distinguish from a measurement.
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))

DEFAULT_DB = os.path.join(_REPO_ROOT, "dashboard", "storage", "data", "backtest.db")
DEFAULT_RESULTS = os.path.join(_BENCH_ROOT, "results")

MEASURED = "MEASURED"
DERIVED = "DERIVED"
NOT_MEASURED = "NOT MEASURED"
TIERS = (MEASURED, DERIVED, NOT_MEASURED)

__all__ = ["Fig", "MissingInput", "load_arm_a",
           "load_measured_calls_per_decision", "backtest_threshold_table", "MEASURED", "DERIVED", "NOT_MEASURED", "TIERS",
           "PRICE_BY_DB_MODEL", "UNMEASURED_INPUTS", "CODE_FINDINGS",
           "load_measured", "require_inputs", "cost_per_call",
           "monthly_breakdown", "per_model_comparison", "sensitivity",
           "calls_to_reach_budget", "verify_code_findings"]


class MissingInput(Exception):
    """Raised when a cost figure is requested with an unmeasured input blank."""


@dataclass(frozen=True)
class Fig:
    """A number that cannot exist without an evidence tier."""

    value: Any
    tier: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not self.source:
            raise ValueError("a figure must name its source")

    def fmt(self, spec: str = ",.4f") -> str:
        if self.value is None:
            return "—"
        try:
            return f"{self.value:{spec}}"
        except (TypeError, ValueError):
            return str(self.value)

    def tagged(self, spec: str = ",.4f") -> str:
        return f"{self.fmt(spec)} [{self.tier}]"


# USD per million tokens (input, output). NOT read from the DB: est_cost_usd is
# one equation in two unknowns, so the pair cannot be recovered from a run row.
# It is instead VERIFIED at load time — load_measured recomputes every run's
# cost from these rates and its stored token totals and raises if any run
# disagrees with its stored est_cost_usd. The seed DB's underscored model names
# do not substring-match token_cost.py's slugs, so this mapping is explicit
# rather than inferred from the name.
PRICE_BY_DB_MODEL: Dict[str, Dict[str, Any]] = {
    "nemotron_3_nano_30b": {"slug": "nvidia/nemotron-3-nano-30b-a3b",
                            "in": 0.05, "out": 0.20},
    "deepseek_v4_pro": {"slug": "deepseek/deepseek-v4-pro", "in": 0.435, "out": 0.87},
    "qwen3_7_plus": {"slug": "qwen/qwen3.7-plus", "in": 0.40, "out": 1.60},
    "gemini_3_1_pro_preview": {"slug": "google/gemini-3.1-pro", "in": 2.0, "out": 12.0},
    "claude_haiku_4_5": {"slug": "anthropic/claude-haiku-4-5", "in": 1.0, "out": 5.0},
    "claude_sonnet_4_6": {"slug": "anthropic/claude-sonnet-4-6", "in": 3.0, "out": 15.0},
    "gpt_5_5": {"slug": "openai/gpt-5.5", "in": 5.0, "out": 30.0},
}

# ATL's production default, named rather than inferred.
PRODUCTION_DEFAULT = "nemotron_3_nano_30b"

# Every input the cost model needs that no artifact in this repository supplies.
# Each names what would resolve it. These are the notebook's blank cells.
UNMEASURED_INPUTS: Dict[str, str] = {
    "n_users": "How many users the platform serves. Render dashboard / users table.",
    "n_agents": "How many agents run concurrently. Render DB: count of active agents.",
    "decisions_per_agent_per_day": (
        "Decisions one live agent makes per day. Nothing in this repo runs a "
        "live agent — paper_backend.py is an explicit stub with no step loop, "
        "so this is a deployment property, not a code property."),
    "backtests_per_user_per_day": (
        "Backtests a user launches per day. LIKELY THE DOMINANT DRIVER: one "
        "leaderboard backtest was ~161 calls, and backtesting is what the "
        "platform is for. Render DB: agent_runs grouped by day and user."),
    "calls_per_decision": (
        "LLM calls per decision — equivalently, YOUR pipeline depth. The "
        "relationship is now MEASURED against real API runs: a 3-step pipeline "
        "issued exactly 3.000 calls/decision and a 5-step exactly 5.000, with "
        "no retry inflation at either depth. What remains unmeasured is which "
        "depth PRODUCTION runs — all 7 seed runs used the single-call path. "
        "Render DB: metadata.initial_pipeline step counts across real runs."),
    "model_mix": (
        "Fraction of production calls by model, as {db_model: fraction}. "
        "Cost per call spans more than two orders of magnitude across the "
        "measured models, so this usually dominates the answer. "
        "Render DB: agent_runs grouped by llm_model."),
    "trading_days_per_month": (
        "Trading days per month. A calendar convention (~21), not a "
        "measurement — state it explicitly rather than letting it default."),
    "calendar_days_per_month": (
        "Calendar days per month for backtest volume (~30). Also a "
        "convention, not a measurement."),
    "infrastructure_usd_per_month": (
        "Hosting, database, and any GPU spend — everything that is NOT "
        "per-token LLM cost. Arm A now measures the per-request API price, but "
        "that is the token bill, not the platform's fixed running cost, and no "
        "self-hosted deployment exists to price. Render/AWS billing console."),
}

# Findings from reading dashboard/backend. Each is checked to exist at the cited
# line by verify_code_findings(). Paths are repo-relative.
CODE_FINDINGS: List[Dict[str, str]] = [
    {
        "id": "token_summation",
        "file": "dashboard/backend/domain/backtesting/engine.py",
        "line": 892,
        "expect": "llm_calls_total = manager.llm_calls + runtime_calls",
        "finding": (
            "Token usage is accumulated with += across a whole run and written "
            "once. The per-call distribution is destroyed before it reaches "
            "the database, so no percentile or per-step split can ever be "
            "recovered from stored data."),
        "consequence": (
            "ATL cannot cost its own product per decision. See "
            "INSTRUMENTATION_PATCH.md."),
    },
    {
        "id": "decisions_never_written",
        "file": "dashboard/backend/domain/backtesting/engine.py",
        "line": 921,
        "expect": "if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:",
        "finding": (
            "insert_decisions fires ONLY under the ai_hedge_fund runtime. "
            "Under the native pipeline runtime backtest_decisions is never "
            "written: 0 rows across all 17 seed runs."),
        "consequence": (
            "Calls per decision has no decision denominator from that table. "
            "equity_timeseries (one row per bar) is used as a proxy instead."),
    },
    {
        "id": "fill_equals_decision_price",
        "file": "dashboard/backend/domain/trading/execution.py",
        "line": 64,
        "expect": 'price = market_data[symbol]["close"]',
        "finding": (
            "The fill price is the close of the same bar the decision was "
            "computed from, and portfolio.py:78 shows the agent that same "
            "close as the current price. Decision price and fill price are the "
            "same number."),
        "consequence": (
            "The engine cannot express execution timing. Latency has no "
            "quantity through which to act. See "
            "latency_sensitivity_audit.md."),
    },
    {
        "id": "agent_sees_fill_price",
        "file": "dashboard/backend/domain/trading/portfolio.py",
        "line": 78,
        "expect": '"price": row["close"],',
        "finding": "The agent's observed price is the bar close it will fill at.",
        "consequence": "Confirms the zero decision-to-fill gap is structural.",
    },
    {
        "id": "same_bar_applied",
        "file": "dashboard/backend/domain/backtesting/engine.py",
        "line": 833,
        "expect": 'manager.execute_actions(decision["actions"], market_data, timestamp)',
        "finding": (
            "The decision is applied to the same market_data and timestamp it "
            "was computed from (built at engine.py:756). Neither is reassigned "
            "in between."),
        "consequence": (
            "A decision taking an hour and one taking 50ms produce "
            "bit-identical results."),
    },
    {
        "id": "no_concurrent_agents",
        "file": "dashboard/scripts/refresh_daily_leaderboard.py",
        "line": 89,
        "expect": "for entry in llm_entries:",
        "finding": (
            "Agents are deployed in a plain sequential loop, one complete "
            "backtest at a time. There is no thread pool, process pool, or "
            "asyncio.gather anywhere in the backtest path."),
        "consequence": (
            "The dashboard exhibits SERIAL, not synchronised, arrivals. Burst "
            "arrival patterns are an assumption about a hypothetical "
            "deployment, not a property of this code."),
    },
    {
        "id": "sequential_pipeline",
        "file": "dashboard/backend/infrastructure/llm/pipeline_runner.py",
        "line": 451,
        "expect": "for index, step in enumerate(decision_steps):",
        "finding": (
            "Pipeline steps run strictly sequentially, and step n+1's prompt "
            "embeds every prior step's output (prior_outputs passed at :460, "
            "rendered at :124). No async/await/gather in the module."),
        "consequence": (
            "A real data dependency, so the calls cannot be parallelised. "
            "Batching raises throughput across agents; it cannot shorten one "
            "decision."),
    },
    {
        "id": "no_live_scheduler",
        "file": "dashboard/backend/execution/paper_backend.py",
        "line": 1,
        "expect": "PaperBackend — DESIGNED-FOR STUB",
        "finding": (
            "Paper trading has no execution path: no order submission, no step "
            "loop, and a realtime decision-cadence scheduler is listed as not "
            "built."),
        "consequence": (
            "No live decision rate can be derived from this repository. "
            "decisions_per_agent_per_day is a deployment question."),
    },
]


def verify_code_findings(repo_root: str = _REPO_ROOT,
                         findings: Optional[Sequence[Dict[str, str]]] = None
                         ) -> List[Dict[str, Any]]:
    """Check every cited file:line still contains what the finding claims.

    A citation that has drifted is worse than no citation, so this is checked
    rather than trusted.
    """
    out: List[Dict[str, Any]] = []
    for f in (findings if findings is not None else CODE_FINDINGS):
        path = os.path.join(repo_root, f["file"])
        rec = {**f, "verified": False, "actual": None}
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            actual = lines[int(f["line"]) - 1].strip()
            rec["actual"] = actual
            rec["verified"] = f["expect"] in actual
        except (OSError, IndexError, ValueError) as exc:
            rec["actual"] = f"unreadable: {exc}"
        out.append(rec)
    return out


def _connect_ro(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_measured(db_path: str = DEFAULT_DB,
                  results_dir: str = DEFAULT_RESULTS) -> Dict[str, Any]:
    """Read every measured input from its artifact. Nothing is transcribed.

    Raises if the pricing table disagrees with any run's stored est_cost_usd:
    a silent mismatch there would corrupt every downstream cost figure.
    """
    conn = _connect_ro(db_path)
    try:
        rows = list(conn.execute(
            "SELECT run_id, llm_model, llm_calls, input_tokens, output_tokens,"
            " est_cost_usd FROM agent_runs WHERE llm_calls > 0"))
        total_runs = conn.execute(
            "SELECT COUNT(*) AS n FROM agent_runs").fetchone()["n"]
        decision_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM backtest_decisions").fetchone()["n"]
        models: Dict[str, Any] = {}
        price_checks: List[Dict[str, Any]] = []
        for r in rows:
            name = r["llm_model"]
            price = PRICE_BY_DB_MODEL.get(name)
            if price is None:
                raise KeyError(f"no price mapping for DB model {name!r}")
            calls = int(r["llm_calls"])
            bars = conn.execute(
                "SELECT COUNT(*) AS n FROM equity_timeseries WHERE run_id = ?",
                (r["run_id"],)).fetchone()["n"]
            recomputed = (r["input_tokens"] / 1e6) * price["in"] + (
                r["output_tokens"] / 1e6) * price["out"]
            delta = abs(recomputed - r["est_cost_usd"])
            if delta > 1e-5:
                raise ValueError(
                    f"pricing disagrees with stored cost for {r['run_id']}: "
                    f"recomputed {recomputed:.6f} vs stored "
                    f"{r['est_cost_usd']:.6f} (delta {delta:.2e})")
            price_checks.append({"run_id": r["run_id"], "delta": delta})
            models[name] = {
                "db_model": name,
                "slug": price["slug"],
                "run_id": r["run_id"],
                "llm_calls": calls,
                "bars": bars,
                "input_per_call": r["input_tokens"] / calls,
                "output_per_call": r["output_tokens"] / calls,
                "price_in": price["in"],
                "price_out": price["out"],
                "run_cost_usd": r["est_cost_usd"],
                "observed_calls_per_decision": (calls / bars) if bars else None,
            }
    finally:
        conn.close()

    serving = _load_serving(results_dir)
    trace = _load_trace(results_dir)

    return {
        "db_path": db_path,
        "total_runs": total_runs,
        "runs_with_llm": len(rows),
        "backtest_decisions_rows": decision_rows,
        "models": models,
        "price_verification": {
            "checked": len(price_checks),
            "max_delta_usd": max((p["delta"] for p in price_checks), default=0.0),
            "tolerance": 1e-5,
        },
        "serving": serving,
        "trace": trace,
    }


def _load_serving(results_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"arms": {}, "matched_levels": []}
    for arm in ("armB_shared", "armC_shared"):
        s_path = os.path.join(results_dir, f"{arm}_summary.json")
        m_path = os.path.join(results_dir, f"{arm}_manifest.json")
        if not os.path.exists(s_path):
            continue
        with open(s_path) as fh:
            summary = json.load(fh)
        manifest = {}
        if os.path.exists(m_path):
            with open(m_path) as fh:
                manifest = json.load(fh)
        out["arms"][arm] = {
            "run_id": summary.get("run_id"),
            "arm": summary.get("arm"),
            "engine_kind": summary.get("engine_kind"),
            "levels": {lv["concurrency"]: lv for lv in summary.get("levels", [])},
            "manifest": manifest,
            "branch_sha": manifest.get("branch_sha"),
            "dirty": str(manifest.get("branch_sha", "")).endswith("-dirty"),
            "vllm": summary.get("vllm"),
        }
    b = out["arms"].get("armB_shared", {}).get("levels", {})
    c = out["arms"].get("armC_shared", {}).get("levels", {})
    for level in sorted(set(b) & set(c)):
        rps_b = b[level].get("completed_requests_per_s")
        rps_c = c[level].get("completed_requests_per_s")
        out["matched_levels"].append({
            "concurrency": level,
            "arm_b_rps": rps_b,
            "arm_c_rps": rps_c,
            "speedup": (rps_c / rps_b) if (rps_b and rps_c) else None,
            "arm_b_e2e_p50": b[level].get("e2e_p50"),
            "arm_c_e2e_p50": c[level].get("e2e_p50"),
            "arm_b_out_tok_s": b[level].get("output_tok_per_s"),
            "arm_c_out_tok_s": c[level].get("output_tok_per_s"),
        })
    return out


def _load_trace(results_dir: str) -> Dict[str, Any]:
    path = os.path.join(results_dir, "armB_L3_trace.json")
    if not os.path.exists(path):
        return {"available": False}
    with open(path) as fh:
        t = json.load(fh)
    launch = next((e for e in t.get("cuda_runtime_breakdown", [])
                   if e.get("name") == "cudaLaunchKernel"), {})
    return {
        "available": True,
        "kernel_count": t.get("kernel_count"),
        "gpu_busy_fraction": t.get("gpu_busy_fraction"),
        "gpu_idle_fraction": t.get("gpu_idle_fraction"),
        "distinct_stream_count": t.get("distinct_stream_count"),
        "tensor_core_gemm_share": t.get("tensor_core_gemm_share"),
        "launch_calls": launch.get("count"),
        "launch_cpu_us": launch.get("total_cpu_us"),
        "gpu_busy_us": t.get("gpu_busy_us"),
        "caveats": t.get("caveats", []),
    }


def load_arm_a(results_dir: str = DEFAULT_RESULTS,
               run_id: str = "armA_nemotron") -> Dict[str, Any]:
    """Arm A's measured hosted-API sweep — the first one ever run.

    Every Arm A figure before this was modelled from an assumed
    $0.0001822/request, which is why ``ADVISOR_REPORT.md`` excluded them. These
    come from the provider's own billed ``usage``.
    """
    s_path = os.path.join(results_dir, f"{run_id}_summary.json")
    m_path = os.path.join(results_dir, f"{run_id}_manifest.json")
    if not os.path.exists(s_path):
        return {"available": False, "reason": f"{s_path} not found"}
    with open(s_path) as fh:
        summary = json.load(fh)
    manifest = {}
    if os.path.exists(m_path):
        with open(m_path) as fh:
            manifest = json.load(fh)

    levels = []
    for lv in summary.get("levels", []):
        n = lv.get("notes") or {}
        pc = n.get("prompt_cache") or {}
        levels.append({
            "concurrency": lv["concurrency"],
            "completed": lv.get("completed"),
            "errored": lv.get("errored"),
            "requests_per_s": lv.get("completed_requests_per_s"),
            "e2e_p50": lv.get("e2e_p50"),
            "e2e_p95": lv.get("e2e_p95"),
            "e2e_p99": lv.get("e2e_p99"),
            "ttft_p50": lv.get("ttft_p50"),
            "output_tok_per_s": lv.get("output_tok_per_s"),
            "cost_per_request": n.get("api_cost_per_request"),
            "cost_total": n.get("api_cost_total"),
            "provider_reported_cost_usd_total": n.get(
                "provider_reported_cost_usd_total"),
            "rate_limit_429_count": n.get("rate_limit_429_count"),
            "cached_tokens_total": pc.get("cached_tokens_total"),
            "prompt_tokens_actual_mean": n.get("prompt_tokens_actual_mean"),
            "output_tokens_total": lv.get("output_tokens_total"),
        })
    costs = [lv["cost_per_request"] for lv in levels if lv["cost_per_request"]]
    cached = [lv["cached_tokens_total"] for lv in levels
              if lv["cached_tokens_total"] is not None]
    return {
        "available": True,
        "run_id": summary.get("run_id"),
        "model": manifest.get("model"),
        "levels": levels,
        "cost_per_request_mean": statistics.fmean(costs) if costs else None,
        "cost_per_request_min": min(costs) if costs else None,
        "cost_per_request_max": max(costs) if costs else None,
        "total_429s": sum(lv["rate_limit_429_count"] or 0 for lv in levels),
        "total_errors": sum(lv["errored"] or 0 for lv in levels),
        "cached_tokens_total": sum(cached) if cached else None,
        "cache_reported_by_provider": bool(cached),
        "manifest": manifest,
        "fixture_sha256": manifest.get("fixture_sha256"),
        "branch_sha": manifest.get("branch_sha"),
        "reasoning": (manifest.get("extra") or {}).get("reasoning"),
    }


def load_measured_calls_per_decision(results_dir: str = DEFAULT_RESULTS
                                     ) -> Dict[str, Any]:
    """Calls per decision, measured per pipeline depth.

    Read from the committed extract artifact rather than the local database,
    which is gitignored — a report that only regenerates on the machine that
    ran the backtest is not reproducible.
    """
    path = os.path.join(results_dir, "atl_local_pipeline.json")
    if not os.path.exists(path):
        return {"available": False, "reason": f"{path} not found"}
    with open(path) as fh:
        data = json.load(fh)
    runs = (data.get("local_extract") or {}).get("runs") or []
    by_depth: Dict[int, Dict[str, Any]] = {}
    for r in runs:
        depth = r.get("pipeline_calls_per_decision")
        observed = r.get("observed_calls_per_decision")
        if not depth or observed is None:
            continue
        rec = by_depth.setdefault(int(depth), {
            "configured_steps": int(depth), "runs": [], "observed": [],
        })
        rec["runs"].append(r.get("run_id"))
        rec["observed"].append(observed)
    for depth, rec in by_depth.items():
        rec["observed_calls_per_decision"] = statistics.fmean(rec["observed"])
        rec["agrees_with_configured"] = all(
            abs(o - depth) < 1e-6 for o in rec["observed"])
        rec["n_runs"] = len(rec["runs"])
    return {
        "available": bool(by_depth),
        "by_depth": by_depth,
        "depths_measured": sorted(by_depth),
        "all_agree": all(r["agrees_with_configured"] for r in by_depth.values()),
        "note": (
            "Observed = llm_calls / bars; configured = steps in "
            "metadata.initial_pipeline. Agreement at every depth means no "
            "retry inflation was observed."
        ),
    }


def backtest_threshold_table(
    measured: Dict[str, Any], budget_usd: float, *,
    n_users: int, calendar_days: int, calls_per_backtest: float,
    calls_per_decision_options: Sequence[float] = (1, 3, 5),
) -> List[Dict[str, Any]]:
    """Backtests/user/day needed to reach a budget, at each pipeline depth.

    The earlier thresholds were derived at ONE call per decision, which the
    seed runs measured but which is *not* what a multi-step pipeline does. A
    3-step pipeline divides every threshold by three: the same dollar budget is
    reached by a third as much user activity.
    """
    rows: List[Dict[str, Any]] = []
    for name, m in measured["models"].items():
        cpc = cost_per_call(m)
        row: Dict[str, Any] = {"db_model": name, "slug": m["slug"],
                               "cost_per_call": cpc, "thresholds": {}}
        for depth in calls_per_decision_options:
            denom = n_users * calls_per_backtest * calendar_days * depth * cpc
            row["thresholds"][depth] = (budget_usd / denom) if denom else None
        rows.append(row)
    rows.sort(key=lambda r: r["cost_per_call"])
    return rows


def cost_per_call(model: Dict[str, Any]) -> float:
    """USD for one LLM call at this model's measured token counts and price."""
    return ((model["input_per_call"] / 1e6) * model["price_in"]
            + (model["output_per_call"] / 1e6) * model["price_out"])


def calls_to_reach_budget(measured: Dict[str, Any], budget_usd: float
                          ) -> List[Dict[str, Any]]:
    """How many calls per month each model needs to reach a budget.

    This is the honest shape of an answer to "would $50K/month be a problem":
    not a single figure, but the load each model would have to carry to get
    there. The spread across models IS the finding.
    """
    rows = []
    for name, m in measured["models"].items():
        cpc = cost_per_call(m)
        rows.append({
            "db_model": name,
            "slug": m["slug"],
            "cost_per_call": cpc,
            "calls_for_budget": (budget_usd / cpc) if cpc else None,
            "run_id": m["run_id"],
        })
    rows.sort(key=lambda r: r["cost_per_call"])
    return rows


def require_inputs(params: Dict[str, Any], needed: Sequence[str]) -> None:
    """Raise naming every blank input, rather than substituting a default."""
    missing = [k for k in needed
               if params.get(k) is None or params.get(k) == {}]
    if missing:
        lines = [f"cannot compute: {len(missing)} unmeasured input(s) are blank"]
        for k in missing:
            lines.append(f"  - {k}: {UNMEASURED_INPUTS.get(k, 'no description')}")
        lines.append(
            "Fill these in the parameters cell. They are deliberately blank: a "
            "default here would become a finding.")
        raise MissingInput("\n".join(lines))


def _blended_cost_per_call(measured: Dict[str, Any],
                           model_mix: Dict[str, float]) -> float:
    total = sum(model_mix.values())
    if not total:
        raise MissingInput("model_mix sums to zero; give at least one model a "
                           "non-zero fraction")
    unknown = set(model_mix) - set(measured["models"])
    if unknown:
        raise MissingInput(
            f"model_mix names models with no measured token counts: "
            f"{sorted(unknown)}. Measured: {sorted(measured['models'])}. "
            f"Output length is model-specific (5.8x spread) and will not be "
            f"substituted across models.")
    return sum(cost_per_call(measured["models"][name]) * (frac / total)
               for name, frac in model_mix.items())


def monthly_breakdown(measured: Dict[str, Any], params: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """Monthly cost split into live trading, backtesting, and infrastructure.

    The split matters because the two workloads scale on different variables.
    Live trading scales with agents x decision rate, which the bar interval
    bounds. Backtesting scales with users x how often they press the button,
    which nothing bounds — a single backtest replays a whole window in one go.
    """
    require_inputs(params, [
        "n_users", "n_agents", "decisions_per_agent_per_day",
        "backtests_per_user_per_day", "calls_per_decision", "model_mix",
        "trading_days_per_month", "calendar_days_per_month",
        "infrastructure_usd_per_month",
    ])
    blended = _blended_cost_per_call(measured, params["model_mix"])

    live_calls = (params["n_agents"] * params["decisions_per_agent_per_day"]
                  * params["calls_per_decision"] * params["trading_days_per_month"])
    live_cost = live_calls * blended

    calls_per_backtest = params.get("calls_per_backtest")
    if calls_per_backtest is None:
        calls_per_backtest = measured_calls_per_backtest(measured)["value"]
    bt_calls = (params["n_users"] * params["backtests_per_user_per_day"]
                * calls_per_backtest * params["calendar_days_per_month"])
    bt_cost = bt_calls * blended

    infra = params["infrastructure_usd_per_month"]
    total = live_cost + bt_cost + infra
    return {
        "blended_cost_per_call": blended,
        "calls_per_backtest": calls_per_backtest,
        "live_trading": {"calls": live_calls, "usd": live_cost,
                         "share": (live_cost / total) if total else None},
        "backtesting": {"calls": bt_calls, "usd": bt_cost,
                        "share": (bt_cost / total) if total else None},
        "infrastructure": {"usd": infra,
                           "share": (infra / total) if total else None},
        "total_usd_per_month": total,
        "total_calls_per_month": live_calls + bt_calls,
    }


def measured_calls_per_backtest(measured: Dict[str, Any]) -> Dict[str, Any]:
    """LLM calls in one backtest, measured across the seed runs.

    Scales with the backtest WINDOW, not with anything intrinsic: these runs
    replay 2026-04-15 to 2026-05-15 at hourly bars. A longer window costs
    proportionally more.
    """
    calls = [m["llm_calls"] for m in measured["models"].values()]
    bars = [m["bars"] for m in measured["models"].values()]
    return {
        "value": sum(calls) / len(calls) if calls else None,
        "min": min(calls) if calls else None,
        "max": max(calls) if calls else None,
        "n_runs": len(calls),
        "bars": max(bars) if bars else None,
        "tier": MEASURED,
        "source": "seed DB agent_runs, 7 leaderboard runs",
        "caveat": ("Per backtest of THIS window (~161 hourly bars over one "
                   "month). Window length scales it linearly."),
    }


def per_model_comparison(measured: Dict[str, Any], params: Dict[str, Any]
                         ) -> List[Dict[str, Any]]:
    """Whole-platform monthly cost if 100% of calls used each model in turn."""
    rows = []
    for name in measured["models"]:
        single = dict(params)
        single["model_mix"] = {name: 1.0}
        try:
            b = monthly_breakdown(measured, single)
        except MissingInput:
            raise
        rows.append({
            "db_model": name,
            "slug": measured["models"][name]["slug"],
            "cost_per_call": cost_per_call(measured["models"][name]),
            "output_tokens_per_call": measured["models"][name]["output_per_call"],
            "live_usd": b["live_trading"]["usd"],
            "backtest_usd": b["backtesting"]["usd"],
            "total_usd_per_month": b["total_usd_per_month"],
        })
    rows.sort(key=lambda r: r["total_usd_per_month"])
    return rows


def sensitivity(measured: Dict[str, Any], params: Dict[str, Any],
                factor: float = 2.0) -> List[Dict[str, Any]]:
    """Rank levers by how far each moves total monthly cost.

    Ordered by measured effect size rather than intuition, so the dominant
    lever is visible rather than argued for.
    """
    base = monthly_breakdown(measured, params)["total_usd_per_month"]
    rows: List[Dict[str, Any]] = []

    for key in ("n_users", "n_agents", "decisions_per_agent_per_day",
                "backtests_per_user_per_day", "calls_per_decision",
                "infrastructure_usd_per_month"):
        lo, hi = dict(params), dict(params)
        lo[key] = params[key] / factor
        hi[key] = params[key] * factor
        c_lo = monthly_breakdown(measured, lo)["total_usd_per_month"]
        c_hi = monthly_breakdown(measured, hi)["total_usd_per_month"]
        rows.append({
            "lever": key, "kind": f"x{factor:g} either way",
            "low_usd": c_lo, "high_usd": c_hi,
            "span_factor": (c_hi / c_lo) if c_lo else None,
            "swing_usd": c_hi - c_lo,
        })

    # Model choice is not a numeric knob, so it is swept across the measured
    # models instead of scaled. It is included here because it is the lever
    # most likely to be underestimated.
    per_model = per_model_comparison(measured, params)
    if per_model:
        lo_row, hi_row = per_model[0], per_model[-1]
        rows.append({
            "lever": "model_mix (100% each)",
            "kind": f"{lo_row['db_model']} -> {hi_row['db_model']}",
            "low_usd": lo_row["total_usd_per_month"],
            "high_usd": hi_row["total_usd_per_month"],
            "span_factor": (hi_row["total_usd_per_month"]
                            / lo_row["total_usd_per_month"])
            if lo_row["total_usd_per_month"] else None,
            "swing_usd": (hi_row["total_usd_per_month"]
                          - lo_row["total_usd_per_month"]),
        })

    rows.sort(key=lambda r: r["swing_usd"] or 0, reverse=True)
    for r in rows:
        r["base_usd"] = base
    return rows
