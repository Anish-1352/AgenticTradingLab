"""Extract real token distributions from ATL backtest runs.

    python -m analysis.atl_token_extract --db dashboard/storage/data/backtest.db
    python -m analysis.atl_token_extract --db … --run-id <run_id>

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

__all__ = ["INSTRUMENTATION_GAP", "extract_runs", "summarise",
           "pipeline_steps_from_metadata", "format_report"]

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


def extract_runs(db_path: str, run_id: Optional[str] = None,
                 limit: int = 200) -> Dict[str, Any]:
    conn = _connect(db_path)
    try:
        cols = _columns(conn, "agent_runs")
        if not cols:
            return {"available": False,
                    "reason": "agent_runs table not found in this database"}

        wanted = [c for c in ("run_id", "agent_name", "model", "mode",
                              "llm_calls", "input_tokens", "output_tokens",
                              "est_cost_usd", "metadata", "created_at")
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
            rec["mean_input_tokens_per_call"] = (in_tok / calls) if calls else None
            rec["mean_output_tokens_per_call"] = (out_tok / calls) if calls else None

            # Decisions come from backtest_decisions, which has a row per
            # decision but no tokens. Joining the two is what yields observed
            # calls-per-decision.
            try:
                n = conn.execute(
                    "SELECT COUNT(*) AS n FROM backtest_decisions WHERE run_id = ?",
                    (rec["run_id"],)).fetchone()["n"]
            except sqlite3.Error:
                n = None
            rec["decisions_recorded"] = n
            rec["observed_calls_per_decision"] = (calls / n) if n else None
            runs.append(rec)

        return {"available": True, "db_path": db_path, "runs": runs,
                "agent_runs_columns": cols,
                "has_per_call_table": "llm_call_usage" in [
                    r["name"] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")
                ]}
    finally:
        conn.close()


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

    return {
        "available": True,
        "runs_with_llm_calls": len(llm_runs),
        "models": sorted({str(r.get("model")) for r in llm_runs if r.get("model")}),
        "input_tokens_per_call": _agg(means_in),
        "output_tokens_per_call": _agg(means_out),
        "calls_per_decision_observed": _agg(cpd),
        "calls_per_decision_from_pipeline_config": sorted(set(pipeline_cpd)),
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
        for key, label in (("input_tokens_per_call", "input tokens/call"),
                           ("output_tokens_per_call", "output tokens/call"),
                           ("calls_per_decision_observed", "calls/decision")):
            agg = summary.get(key)
            A("")
            if not agg:
                A(f"  {label}: unavailable")
                continue
            A(f"  {label} (over {agg['n_runs']} run-level means)")
            A(f"    mean {agg['mean_of_run_means']:.1f}   "
              f"range {agg['min_run_mean']:.1f} .. {agg['max_run_mean']:.1f}")
        if summary.get("calls_per_decision_from_pipeline_config"):
            A("")
            A(f"  configured decision steps per pipeline: "
              f"{summary['calls_per_decision_from_pipeline_config']}")

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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract token data from ATL runs.")
    ap.add_argument("--db", default=os.path.join(
        _REPO_ROOT, "dashboard", "storage", "data", "backtest.db"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    try:
        extract = extract_runs(args.db, args.run_id, args.limit)
    except FileNotFoundError as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        print("Supply --db pointing at a backtest database, or an export.",
              file=sys.stderr)
        return 2

    summary = summarise(extract)
    print(format_report(extract, summary))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({"extract": extract, "summary": summary}, fh, indent=2,
                      default=str)
        print(f"\n[extract] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
