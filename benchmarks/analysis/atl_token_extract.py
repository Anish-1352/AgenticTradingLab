"""Extract real token distributions from ATL backtest runs.

    python -m analysis.atl_token_extract --db dashboard/storage/data/backtest.db
    python -m analysis.atl_token_extract --db … --run-id <run_id>
    python -m analysis.atl_token_extract --local-db "$DATABASE_PATH"   # a local run

THE HEADLINE FINDING: THE PER-CALL DISTRIBUTION IS NOT RECOVERABLE
------------------------------------------------------------------
``agent_runs`` stores token usage as three **run-level aggregates**:

    llm_calls INTEGER, input_tokens INTEGER, output_tokens INTEGER

They are accumulated with ``+=`` across the entire run
(``portfolio_manager.py`` and ``engine.py``) and written once at ``insert_run``.
``backtest_decisions`` records what a decision *did* — actions, source,
context_ref — and carries no token columns at all.

So a min / median / p95 / max **cannot be computed from stored data**.
Summation destroyed the distribution before it reached the database, and no
amount of querying brings it back. This module therefore reports what genuinely
survives and refuses to synthesise the rest:

* **Recoverable:** total input/output tokens, total LLM calls, and therefore the
  MEAN tokens per call; calls per decision from ``llm_calls`` divided by the
  decision count; the configured pipeline (hence steps per decision) from
  ``metadata.initial_pipeline``; ``llm_max_output_tokens``; the model.
* **NOT recoverable:** any percentile, any spread, any per-call value, and the
  split of tokens between pipeline steps.

A mean alone is a poor input to a cost model when the underlying spread is
unknown — a pipeline whose first step sends 12k tokens and whose later steps
send 800 has the same mean as one that sends 3k four times, and they behave very
differently under a prefix cache. The instrumentation needed to fix that is
spelled out in ``INSTRUMENTATION_GAP`` and printed with every report.

CALLS PER DECISION, DERIVED TWO INDEPENDENT WAYS
-------------------------------------------------
The two derivations answer different questions and are deliberately *not*
reconciled into one number:

(a) **Observed** — ``llm_calls / decisions``. What the run actually spent.
(b) **Configured** — decision steps in ``metadata.initial_pipeline``. What the
    run was *supposed* to spend, one call per step.

Disagreement is the finding, not an error. ``a > b`` means calls fired that no
configured step asked for — retries. ``a < b`` means steps did not run: an abort
partway through the pipeline, or a fallback to the rule-based path.

**The decision denominator needs care.** ``backtest_decisions`` looks like the
natural source and is the wrong one: ``engine.py`` calls ``insert_decisions``
only under the ``ai_hedge_fund`` runtime, so for the native pipeline runtime —
the path this whole measurement is about — the table is empty *by construction*,
not because the run misbehaved. The denominator used instead is
``equity_timeseries``, which the engine writes one row per simulated bar and
takes exactly one decision per bar. Every figure records which source it used;
see ``DECISION_COUNT_SOURCES``.

OUTPUT TOKENS ARE MODEL-SPECIFIC AND ARE TAGGED AS SUCH
---------------------------------------------------------
Under an identical prompt the seed runs span 860 output tokens/call (Nemotron)
to 5,005 (Gemini) — 5.8x — while input tokens span only 1.4x. Input is a
property of the prompt and transfers across models; output is a property of the
model and does not. Every output figure this module emits carries the model that
produced it, and ``output_tokens_for_model`` refuses to answer for a model it
did not measure rather than substituting a neighbour's number.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

__all__ = ["INSTRUMENTATION_GAP", "DECISION_COUNT_SOURCES", "ORIGINAL_ASSUMPTION",
           "FREE_TIER_MARKERS", "is_free_tier_model", "guarded_est_cost",
           "extract_runs", "summarise", "pipeline_steps_from_metadata",
           "reconcile_calls_per_decision", "output_tokens_by_model",
           "output_tokens_for_model", "three_way_comparison", "format_report"]

# Where a decision count may come from, best first. Order matters: the first
# source that yields a positive count wins, and the winner is recorded on every
# figure derived from it.
DECISION_COUNT_SOURCES = (
    (
        "backtest_decisions",
        "One row per logged decision. Authoritative WHEN PRESENT, but "
        "engine.py calls insert_decisions() only under the ai_hedge_fund "
        "runtime — under the native pipeline runtime this table is empty by "
        "construction, so an empty result here is not evidence of anything.",
    ),
    (
        "equity_timeseries",
        "One row per simulated bar. The engine takes exactly one decision per "
        "bar, so the bar count is the decision count. A PROXY: it is exact for "
        "the hourly backtest loop, and it would overcount if a bar were ever "
        "recorded without a decision being attempted.",
    ),
)

# What the GPU benchmark assumed before any of this was measured: one call per
# decision against the synthetic fixture (2,620 context tokens, 256 new tokens
# under ignore_eos). Kept verbatim so later figures are compared against it
# rather than quietly replacing it.
ORIGINAL_ASSUMPTION = {
    "label": "original assumption",
    "source": "benchmarks GPU fixture, arms A/B/C (see RESULTS.md)",
    "calls_per_decision": 1.0,
    "input_tokens_per_call": 2620.0,
    "output_tokens_per_call": 256.0,
    "model": None,
    "model_note": (
        "No model — the fixture pinned output length with ignore_eos, so 256 "
        "is a knob setting, not a measurement of any model's verbosity."
    ),
}

INSTRUMENTATION_GAP = {
    "what_is_missing": "per-LLM-call token records",
    "why": (
        "portfolio_manager and engine accumulate input_tokens / output_tokens / "
        "llm_calls with += over the whole run, and insert_run writes the three "
        "totals. The per-call values exist only as transient locals."
    ),
    "minimal_fix": (
        "A table `llm_call_usage(run_id, decision_index, step_index, model, "
        "input_tokens, output_tokens, created_at)` written where "
        "extract_token_usage() already returns the pair — pipeline_runner.py "
        "line ~473 and ~388, and portfolio_manager.py line ~399. Those three "
        "sites already hold everything the row needs; nothing new has to be "
        "computed."
    ),
    "why_it_is_worth_it": (
        "It converts every downstream cost figure from an assumption into a "
        "measurement, and it is the only way to see whether the first pipeline "
        "step dominates input tokens — which decides whether prefix caching "
        "helps ATL at all."
    ),
}


# Slugs whose provider bills nothing. token_cost.price_for_model() substring-
# matches these to the PAID model of the same name — "nvidia/nemotron-3-nano-
# 30b-a3b:free" matches the "nvidia/nemotron-3-nano-30b-a3b" needle — so the
# est_cost_usd a free-tier run stores is a real number describing a charge that
# never happened. Reporting it is worse than reporting nothing, because it is
# indistinguishable from a measurement.
FREE_TIER_MARKERS = (":free",)


def is_free_tier_model(model: Any) -> bool:
    name = str(model or "").strip().lower()
    return any(marker in name for marker in FREE_TIER_MARKERS)


def guarded_est_cost(model: Any, stored: Any) -> Dict[str, Any]:
    """Return the stored cost, or a refusal when the slug is free-tier."""
    if not is_free_tier_model(model):
        return {"est_cost_usd": stored, "est_cost_usd_available": True}
    return {
        "est_cost_usd": None,
        "est_cost_usd_available": False,
        "est_cost_usd_stored_but_wrong": stored,
        "reason": (
            f"{model!r} is a free-tier slug: the provider billed nothing, but "
            f"price_for_model() substring-matched it to the paid model's rate "
            f"and stored {stored}. That figure describes a charge that never "
            f"happened. Token counts from this run remain valid; the cost does "
            f"not. Re-run on the paid slug for a real cost."
        ),
    }


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")
    # Read-only URI: this tool must never write to a results database.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def pipeline_steps_from_metadata(metadata: Any) -> Dict[str, Any]:
    """Recover the configured pipeline shape — the true calls-per-decision."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            return {"available": False, "reason": "metadata is not JSON"}
    if not isinstance(metadata, dict):
        return {"available": False, "reason": "no metadata"}

    out: Dict[str, Any] = {"available": False}
    for key in ("initial_pipeline", "final_pipeline"):
        pipeline = metadata.get(key)
        if not isinstance(pipeline, list):
            continue
        decision = [s for s in pipeline if isinstance(s, dict)
                    and s.get("presetKey") != "post_trade_analysis"]
        post = [s for s in pipeline if isinstance(s, dict)
                and s.get("presetKey") == "post_trade_analysis"]
        out.update({
            "available": True,
            f"{key}_total_steps": len(pipeline),
            f"{key}_decision_steps": len(decision),
            f"{key}_post_trade_steps": len(post),
        })
        # Decision steps are the per-decision cost; post-trade steps run once
        # per trading DAY and amortise over that day's bars.
        out["calls_per_decision_from_pipeline"] = len(decision)
        out["post_trade_calls_per_day"] = len(post)
    if metadata.get("llm_max_output_tokens") is not None:
        out["llm_max_output_tokens"] = metadata["llm_max_output_tokens"]
    if metadata.get("runtime_calls") is not None:
        out["runtime_calls"] = metadata["runtime_calls"]
    if not out["available"]:
        out["reason"] = "metadata carries no initial_pipeline/final_pipeline"
    return out


def _count(conn: sqlite3.Connection, table: str, run_id: str) -> Optional[int]:
    try:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE run_id = ?", (run_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    return int(row["n"]) if row else None


def decision_count(conn: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    """Decisions in a run, plus which table the number came from.

    Tries each source in ``DECISION_COUNT_SOURCES`` order and takes the first
    positive count. Reporting the source alongside the count is not bookkeeping
    — an ``equity_timeseries`` denominator is a proxy and a ``backtest_decisions``
    one is not, and a reader has to be able to tell which they were handed.
    """
    attempts: List[Dict[str, Any]] = []
    chosen: Optional[str] = None
    count: Optional[int] = None
    for table, why in DECISION_COUNT_SOURCES:
        n = _count(conn, table, run_id)
        attempts.append({"table": table, "count": n, "rationale": why})
        if chosen is None and n:
            chosen, count = table, n
    return {
        "decisions": count,
        "source": chosen,
        "is_proxy": chosen == "equity_timeseries",
        "attempts": attempts,
    }


def reconcile_calls_per_decision(
    observed: Optional[float], from_pipeline: Optional[float],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Compare the two derivations. Disagreement is reported, never averaged.

    The two numbers measure different things — what ran and what was configured
    to run — so a gap between them carries information that a reconciled single
    figure would destroy.
    """
    out: Dict[str, Any] = {
        "observed_llm_calls_over_decisions": observed,
        "configured_pipeline_steps": from_pipeline,
    }
    if observed is None and from_pipeline is None:
        out.update(agree=None, verdict="neither derivation available",
                   interpretation="No llm_calls/decisions and no pipeline in "
                                  "metadata. Calls per decision is unmeasured.")
        return out
    if from_pipeline is None:
        out.update(agree=None, verdict="configured count unavailable",
                   interpretation=(
                       "metadata carries no initial_pipeline, so the run used "
                       "the SINGLE-CALL path — there were no configured steps "
                       "to count. The multi-step path remains unmeasured."))
        return out
    if observed is None:
        out.update(agree=None, verdict="observed count unavailable",
                   interpretation=("No decision denominator was found, so the "
                                   "configured step count stands unchecked "
                                   "against what actually ran."))
        return out

    delta = observed - from_pipeline
    out["delta"] = delta
    out["ratio"] = (observed / from_pipeline) if from_pipeline else None
    if abs(delta) <= tolerance:
        out.update(agree=True, verdict="agree", interpretation=(
            f"Both derivations give {observed:.3f} calls per decision. Every "
            f"configured step fired exactly once: no retries, no aborts."))
    elif delta > 0:
        out.update(agree=False, verdict="observed EXCEEDS configured",
                   excess_calls_per_decision=delta, interpretation=(
                       f"{observed:.3f} calls ran against {from_pipeline:.0f} "
                       f"configured steps — {delta:.3f} extra calls per "
                       f"decision that no step asked for. That is RETRY "
                       f"INFLATION, and it multiplies the cost model by "
                       f"{observed / from_pipeline:.2f}x over the configured "
                       f"figure."))
    else:
        out.update(agree=False, verdict="observed BELOW configured",
                   missing_calls_per_decision=-delta, interpretation=(
                       f"Only {observed:.3f} calls ran against "
                       f"{from_pipeline:.0f} configured steps. Steps did not "
                       f"execute — the pipeline aborted partway (an "
                       f"unparseable step output returns early) or decisions "
                       f"fell back to the rule-based path. Cost is lower than "
                       f"configured, but so is the work done."))
    return out


def extract_runs(db_path: str, run_id: Optional[str] = None,
                 limit: int = 200) -> Dict[str, Any]:
    conn = _connect(db_path)
    try:
        cols = _columns(conn, "agent_runs")
        if not cols:
            return {"available": False,
                    "reason": "agent_runs table not found in this database"}

        # The column is `llm_model`; `model` is accepted too because an export
        # or a future schema may rename it. Selecting only "model" — as this
        # did — silently dropped the model from every row, which is exactly the
        # field that output-token figures must be tagged with.
        wanted = [c for c in ("run_id", "agent_name", "llm_model", "model",
                              "mode", "llm_calls", "input_tokens",
                              "output_tokens", "est_cost_usd", "metadata",
                              "created_at")
                  if c in cols]
        sql = f"SELECT {', '.join(wanted)} FROM agent_runs"
        params: Sequence[Any] = ()
        if run_id:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        sql += f" LIMIT {int(limit)}"

        runs: List[Dict[str, Any]] = []
        for row in conn.execute(sql, params):
            rec = {k: row[k] for k in wanted}
            rec["pipeline"] = pipeline_steps_from_metadata(rec.pop("metadata", None))

            calls = int(rec.get("llm_calls") or 0)
            in_tok = int(rec.get("input_tokens") or 0)
            out_tok = int(rec.get("output_tokens") or 0)
            rec["model"] = rec.get("llm_model") or rec.get("model")
            rec.update(guarded_est_cost(rec["model"], rec.get("est_cost_usd")))
            rec["free_tier"] = is_free_tier_model(rec["model"])
            rec["mean_input_tokens_per_call"] = (in_tok / calls) if calls else None
            # Output is model-specific; it travels with the model that made it.
            rec["mean_output_tokens_per_call"] = (out_tok / calls) if calls else None
            rec["output_tokens_model_tag"] = rec["model"]

            dc = decision_count(conn, rec["run_id"])
            rec["decision_count"] = dc
            rec["decisions_recorded"] = dc["decisions"]
            n = dc["decisions"]
            # (a) observed: what the run spent.
            rec["observed_calls_per_decision"] = (calls / n) if n else None
            # (b) configured: what the pipeline asked for.
            rec["pipeline_calls_per_decision"] = rec["pipeline"].get(
                "calls_per_decision_from_pipeline")
            rec["calls_per_decision_reconciliation"] = reconcile_calls_per_decision(
                rec["observed_calls_per_decision"],
                rec["pipeline_calls_per_decision"])
            runs.append(rec)

        return {"available": True, "db_path": db_path, "runs": runs,
                "agent_runs_columns": cols,
                "has_per_call_table": "llm_call_usage" in [
                    r["name"] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")
                ]}
    finally:
        conn.close()


class CrossModelSubstitution(KeyError):
    """Raised when an output-token figure is requested for an unmeasured model.

    Returning a neighbour's number instead would be silent and wrong by up to
    5.8x, which is larger than most of the effects these benchmarks study.
    """


def output_tokens_by_model(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean output tokens/call, keyed by the model that produced them.

    Input tokens are deliberately absent from the per-model view: they are a
    property of the prompt, vary only 1.4x across these models, and are the one
    figure that *is* safe to carry between models.
    """
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        model = r.get("model")
        mean_out = r.get("mean_output_tokens_per_call")
        if not model or not mean_out:
            continue
        entry = by_model.setdefault(str(model), {
            "model": str(model), "run_ids": [], "means": [],
        })
        entry["run_ids"].append(r.get("run_id"))
        entry["means"].append(mean_out)
    for entry in by_model.values():
        entry["mean_output_tokens_per_call"] = statistics.fmean(entry["means"])
        entry["n_runs"] = len(entry["means"])
        entry["mean_input_tokens_per_call"] = None
        entry.pop("means")
        entry["substitutable_across_models"] = False
    if by_model:
        values = [e["mean_output_tokens_per_call"] for e in by_model.values()]
        spread = (max(values) / min(values)) if min(values) else None
    else:
        spread = None
    return {
        "available": bool(by_model),
        "models": by_model,
        "spread_factor": spread,
        "note": (
            "Output tokens per call under an IDENTICAL prompt. The spread is "
            "the model's verbosity, not the workload's. Never carry one "
            "model's figure to another — use output_tokens_for_model, which "
            "raises instead of substituting."
        ),
    }


def output_tokens_for_model(by_model: Dict[str, Any], model: str) -> float:
    """Look up a measured output length, or refuse.

    The refusal is the point. A cost model that silently reuses Gemini's 5,005
    tokens for Nemotron overstates Nemotron's output cost by 5.8x.
    """
    models = (by_model or {}).get("models") or {}
    if model in models:
        return models[model]["mean_output_tokens_per_call"]
    raise CrossModelSubstitution(
        f"no measured output length for {model!r}. Measured: "
        f"{sorted(models)}. Output tokens are model-specific (5.8x spread in "
        f"the seed data) and will not be substituted across models — measure "
        f"{model!r}, or state its output length as an explicit assumption."
    )


def three_way_comparison(
    seed: Optional[Dict[str, Any]] = None,
    local: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Original assumption vs seed DB vs local pipeline run, side by side.

    Each column keeps its own figures. The later measurements do not overwrite
    the assumption they corrected — the size of the correction is the result,
    and it is only visible if the original number is still on the page.
    """
    def _col(label: str, summary: Optional[Dict[str, Any]],
             source: str) -> Dict[str, Any]:
        if not summary or not summary.get("available"):
            return {
                "label": label, "source": source, "available": False,
                "reason": (summary or {}).get("reason", "not supplied"),
            }
        cpd_obs = summary.get("calls_per_decision_observed") or {}
        configured = summary.get("calls_per_decision_from_pipeline_config") or []
        in_agg = summary.get("input_tokens_per_call") or {}
        out_agg = summary.get("output_tokens_per_call") or {}
        return {
            "label": label,
            "source": source,
            "available": True,
            "runs": summary.get("runs_with_llm_calls"),
            "calls_per_decision_observed": cpd_obs.get("mean_of_run_means"),
            "calls_per_decision_observed_range": (
                [cpd_obs.get("min_run_mean"), cpd_obs.get("max_run_mean")]
                if cpd_obs else None),
            "calls_per_decision_configured": configured or None,
            "input_tokens_per_call": in_agg.get("mean_of_run_means"),
            "output_tokens_per_call": out_agg.get("mean_of_run_means"),
            "output_tokens_models": summary.get("models"),
            "output_is_model_specific": True,
        }

    assumption = dict(ORIGINAL_ASSUMPTION)
    assumption.update(available=True, source=ORIGINAL_ASSUMPTION["source"])
    cols = {
        "original_assumption": assumption,
        "seed_db": _col("seed DB", seed,
                        "dashboard/storage/data/backtest.db, 7 leaderboard runs"),
        "local_pipeline_run": _col("local pipeline run", local,
                                   "locally produced DB via --local-db"),
    }

    deltas: Dict[str, Any] = {}
    seed_col = cols["seed_db"]
    if seed_col.get("available"):
        for key in ("input_tokens_per_call", "output_tokens_per_call"):
            base, got = ORIGINAL_ASSUMPTION[key], seed_col.get(key)
            if base and got:
                deltas[f"seed_vs_assumption_{key}"] = got / base
    local_col = cols["local_pipeline_run"]
    if local_col.get("available"):
        got = local_col.get("calls_per_decision_observed")
        if got:
            deltas["local_vs_assumption_calls_per_decision"] = (
                got / ORIGINAL_ASSUMPTION["calls_per_decision"])
        seed_cpd = seed_col.get("calls_per_decision_observed")
        if got and seed_cpd:
            deltas["local_vs_seed_calls_per_decision"] = got / seed_cpd

    return {
        "columns": cols,
        "deltas": deltas,
        "note": (
            "Columns are independent measurements, not revisions. CALLS PER "
            "DECISION is the one figure comparable across all three: it is a "
            "property of pipeline depth, not of the prompt. TOKEN COUNTS ARE "
            "NOT COMPARABLE ACROSS COLUMNS unless the prompt text and the "
            "ticker universe match — a local run over 2 symbols with a short "
            "benchmark prompt will show far fewer input tokens than a "
            "production run over 30 symbols, and that difference is the "
            "workload, not a change in the platform. Output tokens are "
            "additionally model-specific."
        ),
    }


def summarise(extract: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate across runs. Reports means; refuses to report percentiles."""
    if not extract.get("available"):
        return {"available": False, "reason": extract.get("reason")}

    llm_runs = [r for r in extract["runs"] if (r.get("llm_calls") or 0) > 0]
    if not llm_runs:
        return {
            "available": False,
            "reason": (
                f"none of the {len(extract['runs'])} run(s) recorded llm_calls > 0 "
                f"— they are rule-based or external-agent runs, which never "
                f"report server-side usage"
            ),
            "runs_examined": len(extract["runs"]),
        }

    means_in = [r["mean_input_tokens_per_call"] for r in llm_runs
                if r.get("mean_input_tokens_per_call")]
    means_out = [r["mean_output_tokens_per_call"] for r in llm_runs
                 if r.get("mean_output_tokens_per_call")]
    cpd = [r["observed_calls_per_decision"] for r in llm_runs
           if r.get("observed_calls_per_decision")]
    pipeline_cpd = [r["pipeline"].get("calls_per_decision_from_pipeline")
                    for r in llm_runs
                    if r["pipeline"].get("calls_per_decision_from_pipeline")]

    def _agg(values: List[float]) -> Optional[Dict[str, Any]]:
        if not values:
            return None
        return {
            "n_runs": len(values),
            "mean_of_run_means": statistics.fmean(values),
            "min_run_mean": min(values),
            "max_run_mean": max(values),
            "note": (
                "These are statistics over RUN-LEVEL MEANS, not over calls. "
                "Each run contributes one number. A per-call percentile does "
                "not exist in the stored data."
            ),
        }

    free_runs = [r for r in llm_runs if r.get("free_tier")]
    by_model = output_tokens_by_model(llm_runs)
    reconciliations = [r["calls_per_decision_reconciliation"] for r in llm_runs
                       if r.get("calls_per_decision_reconciliation")]
    disagreements = [r for r in reconciliations if r.get("agree") is False]
    denominators = sorted({(r.get("decision_count") or {}).get("source")
                           for r in llm_runs} - {None})

    return {
        "available": True,
        "runs_with_llm_calls": len(llm_runs),
        "models": sorted({str(r.get("model")) for r in llm_runs if r.get("model")}),
        "input_tokens_per_call": _agg(means_in),
        "output_tokens_per_call": _agg(means_out),
        "output_tokens_per_call_warning": (
            "This aggregate spans MULTIPLE MODELS and is not a usable cost "
            "input. Output length is model-specific — use "
            "output_tokens_by_model."
        ) if len(by_model.get("models") or {}) > 1 else None,
        "output_tokens_by_model": by_model,
        "free_tier_runs": {
            "count": len(free_runs),
            "run_ids": [r.get("run_id") for r in free_runs],
            "note": (
                "These runs cost nothing. Any est_cost_usd stored against them "
                "is fabricated by substring price matching and is withheld; "
                "their TOKEN counts are still valid."
            ),
        } if free_runs else None,
        "calls_per_decision_observed": _agg(cpd),
        "calls_per_decision_from_pipeline_config": sorted(set(pipeline_cpd)),
        "calls_per_decision_derivations": {
            "a_observed": "llm_calls / decisions",
            "b_configured": "decision steps in metadata.initial_pipeline",
            "decision_denominator_sources_used": denominators,
            "runs_reconciled": len(reconciliations),
            "runs_in_disagreement": len(disagreements),
            "disagreements": disagreements,
            "note": (
                "Disagreement means retries fired (a > b) or steps did not run "
                "(a < b). It is reported, not reconciled."
            ),
        },
        "per_call_distribution": {
            "available": False,
            "reason": (
                "agent_runs stores summed totals; backtest_decisions carries no "
                "token columns. min/median/p95/max cannot be derived — the "
                "distribution was destroyed by summation before persistence."
            ),
            "instrumentation_gap": INSTRUMENTATION_GAP,
        },
    }


def format_report(extract: Dict[str, Any], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A("ATL TOKEN EXTRACT — what the database actually retains")
    A("=" * 78)

    if not extract.get("available"):
        A(f"  UNAVAILABLE: {extract.get('reason')}")
        return "\n".join(lines)

    A(f"  db: {extract['db_path']}")
    A(f"  runs examined: {len(extract['runs'])}")
    A(f"  per-call usage table present: "
      f"{'yes' if extract.get('has_per_call_table') else 'NO'}")

    if not summary.get("available"):
        A("")
        A(f"  NO USABLE TOKEN DATA: {summary.get('reason')}")
    else:
        A(f"  runs with llm_calls > 0: {summary['runs_with_llm_calls']}")
        A(f"  models: {summary['models']}")
        # Calls/decision gets 3 decimals: the whole question is whether it is
        # 1 or 3, and a run that fell just short of 1.000 (a step that never
        # fired) is invisible at one decimal place.
        for key, label, dp in (("input_tokens_per_call", "input tokens/call", 1),
                               ("output_tokens_per_call", "output tokens/call", 1),
                               ("calls_per_decision_observed", "calls/decision", 3)):
            agg = summary.get(key)
            A("")
            if not agg:
                A(f"  {label}: unavailable")
                continue
            A(f"  {label} (over {agg['n_runs']} run-level means)")
            A(f"    mean {agg['mean_of_run_means']:.{dp}f}   "
              f"range {agg['min_run_mean']:.{dp}f} .. {agg['max_run_mean']:.{dp}f}")
        if summary.get("calls_per_decision_from_pipeline_config"):
            A("")
            A(f"  configured decision steps per pipeline: "
              f"{summary['calls_per_decision_from_pipeline_config']}")

        der = summary.get("calls_per_decision_derivations") or {}
        if der:
            A("")
            A("  CALLS PER DECISION — two independent derivations")
            A(f"    (a) observed   {der['a_observed']}")
            A(f"    (b) configured {der['b_configured']}")
            A(f"    decision denominator from: "
              f"{der.get('decision_denominator_sources_used') or '— none found'}")
            A(f"    runs reconciled: {der.get('runs_reconciled', 0)}   "
              f"in disagreement: {der.get('runs_in_disagreement', 0)}")
            for d in (der.get("disagreements") or [])[:5]:
                A(f"    ! {d['verdict']}: {d['interpretation']}")

        bym = summary.get("output_tokens_by_model") or {}
        if bym.get("available"):
            A("")
            A("  OUTPUT TOKENS/CALL BY MODEL — tagged, not substitutable")
            for name, e in sorted(
                    bym["models"].items(),
                    key=lambda kv: kv[1]["mean_output_tokens_per_call"]):
                A(f"    {name:<26} {e['mean_output_tokens_per_call']:>8.0f}"
                  f"   ({e['n_runs']} run(s))")
            if bym.get("spread_factor"):
                A(f"    spread {bym['spread_factor']:.1f}x under the SAME "
                  f"prompt — this is model verbosity, not workload.")
        if summary.get("output_tokens_per_call_warning"):
            A(f"    ! {summary['output_tokens_per_call_warning']}")

    A("")
    A("  " + "!" * 70)
    A("  PER-CALL DISTRIBUTION IS NOT RECOVERABLE")
    A("  " + "!" * 70)
    A("  agent_runs stores llm_calls / input_tokens / output_tokens as run-level")
    A("  SUMS. backtest_decisions has no token columns. No percentile, spread,")
    A("  or per-step split can be computed from what is stored — the numbers")
    A("  were added together before they were saved.")
    A("")
    A(f"  Minimal fix: {INSTRUMENTATION_GAP['minimal_fix']}")
    A("")
    A(f"  Why bother: {INSTRUMENTATION_GAP['why_it_is_worth_it']}")
    return "\n".join(lines)


def format_comparison(cmp: Dict[str, Any]) -> str:
    """Three columns, side by side, none overwriting another."""
    lines: List[str] = []
    A = lines.append
    A("")
    A("=" * 78)
    A("THREE-WAY COMPARISON — assumption vs seed DB vs local pipeline run")
    A("=" * 78)

    cols = cmp["columns"]
    order = ["original_assumption", "seed_db", "local_pipeline_run"]
    A(f"  {'':<30}{'assumption':>15}{'seed DB':>15}{'local run':>15}")
    A("  " + "-" * 74)

    def _cell(col: Dict[str, Any], key: str, fmt: str = ",.0f") -> str:
        if not col.get("available"):
            return "—"
        v = col.get(key)
        if v is None:
            return "—"
        return f"{v:{fmt}}"

    for key, label, fmt in (
        ("calls_per_decision", "calls per decision", ".3f"),
        ("calls_per_decision_observed", "  observed (a)", ".3f"),
        ("calls_per_decision_configured", "  configured (b)", ""),
        ("input_tokens_per_call", "input tokens/call", ",.0f"),
        ("output_tokens_per_call", "output tokens/call *", ",.0f"),
    ):
        cells = []
        for name in order:
            col = cols[name]
            if not col.get("available"):
                cells.append("—")
            elif key == "calls_per_decision_configured":
                v = col.get(key)
                cells.append(str(v) if v else "—")
            else:
                cells.append(_cell(col, key, fmt or ".3f"))
        A(f"  {label:<30}{cells[0]:>15}{cells[1]:>15}{cells[2]:>15}")

    A("")
    for name in order:
        col = cols[name]
        if col.get("available"):
            models = col.get("output_tokens_models")
            A(f"  {col['label']}: {col['source']}")
            if isinstance(models, list) and models:
                A(f"    * output tokens above average over models {models} — "
                  f"NOT a usable per-model figure; see the by-model table.")
            elif col.get("model_note"):
                A(f"    * {col['model_note']}")
        else:
            A(f"  {col['label']}: NOT AVAILABLE — {col.get('reason')}")

    if cmp.get("deltas"):
        A("")
        A("  Deltas:")
        for k, v in cmp["deltas"].items():
            A(f"    {k:<52} {v:>8.2f}x")
    A("")
    A(f"  {cmp['note']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract token data from ATL runs.")
    ap.add_argument("--db", default=os.path.join(
        _REPO_ROOT, "dashboard", "storage", "data", "backtest.db"),
        help="the SEED database (committed). Read-only.")
    ap.add_argument("--local-db", default=None,
                    help="a LOCALLY produced database — the output of a local "
                         "backtest against a real pipeline. See "
                         "analysis/local_atl_setup.md. Never point this at a "
                         "deployed instance.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.local_db and os.path.abspath(args.local_db) == os.path.abspath(args.db):
        print("ERROR: --local-db points at the seed database. The three-way "
              "comparison needs them to be different runs.", file=sys.stderr)
        return 2

    try:
        extract = extract_runs(args.db, args.run_id, args.limit)
    except FileNotFoundError as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        print("Supply --db pointing at a backtest database, or an export.",
              file=sys.stderr)
        return 2

    summary = summarise(extract)
    print(format_report(extract, summary))

    local_extract: Optional[Dict[str, Any]] = None
    local_summary: Optional[Dict[str, Any]] = None
    if args.local_db:
        try:
            local_extract = extract_runs(args.local_db, args.run_id, args.limit)
        except FileNotFoundError as exc:
            # Not fatal: the seed figures still stand, and a missing local DB
            # is the expected state until someone runs the pipeline locally.
            local_extract = {"available": False, "reason": str(exc)}
        local_summary = summarise(local_extract)
        print("")
        print(format_report(local_extract, local_summary))

    comparison = three_way_comparison(summary, local_summary)
    print(format_comparison(comparison))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        payload = {"extract": extract, "summary": summary,
                   "comparison": comparison}
        if args.local_db:
            payload["local_db_path"] = args.local_db
            payload["local_extract"] = local_extract
            payload["local_summary"] = local_summary
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\n[extract] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
