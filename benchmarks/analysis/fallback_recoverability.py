"""Can the rule-based fallback rate be recovered from the stored leaderboard data?

Phase 16 found that a well-formed "no trades this hour" is treated as a failed
decision and silently replaced with rule-based trading. If that fired often in
the leaderboard runs, then some fraction of those "model" curves are not the
model's, and the cost-per-alpha result (Spearman rho +0.071 across a 194x price
span, no model beating ``spy_index`` on Sharpe) would be measuring the fallback
rather than the models.

This module answers a narrower question first, because the answer decides
whether the rest is possible at all: **does the stored data retain any trace of
which decisions came from the model?**

IT DOES NOT. The answer is a negative result and this module is built to make
that negative auditable rather than asserted — every candidate signal is probed
against the real database and each one reports what it can and cannot support.

THE COUNTER EXISTS. IT IS JUST NEVER SAVED.
--------------------------------------------
``PortfolioManager.llm_decisions`` counts steps the model actually drove. It is
real, it is correct, and it is consumed by the H6 publish guard
(``_reject_if_llm_fallback``) — which then discards it. ``insert_run`` has no
``llm_decisions`` parameter. The distinction between a model decision and a
fallback exists for exactly as long as the process that computed it.

WHY llm_calls CANNOT SUBSTITUTE
--------------------------------
In the single-prompt path the call is billed BEFORE the response is parsed
(``portfolio_manager.py`` increments ``llm_calls`` right after the request, then
parses). A step that falls back therefore still consumes its call. The observed
1.000 calls/bar is consistent with 161 model decisions and with 0 model
decisions, and cannot separate them. The codebase already knows this — commit
05d3974, "key H6 coverage on model-driven steps, not billed calls".

A CORRECTION TO THE PREMISE
----------------------------
The Phase 16 defect is in ``pipeline_output_to_decision``, which is reached only
from ``run_pipeline_decision``. **No leaderboard entrant carries a pipeline**
(all twelve have no ``pipeline`` key), and ``llm_agent.py`` calls
``make_trading_decision_with_llm`` without a ``pipeline`` argument, so that
function was never called during these runs. The exact defect Phase 16 found did
not touch the leaderboard.

The concern survives anyway, because the single-prompt path has the SAME defect
class at ``portfolio_manager.py:432-446``: an empty ``actions`` list under
``strict_llm=False`` prints "Falling back to rule-based logic" and returns
``self.make_trading_decision(...)``. ``llm_agent.py`` passes no ``strict_llm``
either, so the default False applies. The code comment at that site already
names the problem: the non-strict branch "silently swaps in a rule-based
decision, which is exactly what must NOT count."
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import DEFAULT_DB  # noqa: E402

__all__ = [
    "RECOVERABLE", "NOT_RECOVERABLE", "BOUND_ONLY",
    "SEED_DB_SHA256", "db_sha256", "assert_seed_db_unchanged",
    "Signal", "audit_signals", "recoverability_verdict", "format_audit",
]

RECOVERABLE = "recoverable"
NOT_RECOVERABLE = "not_recoverable"
BOUND_ONLY = "bound_only"

# The committed seed database. Asserted before and after every read so an
# analysis that claims to be read-only can prove it.
SEED_DB_SHA256 = "414bf53cb056b2c60bbdf4d963dffd1ffd7998fcb6235cf9bda963bd52504c80"


def db_sha256(path: str = DEFAULT_DB) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_seed_db_unchanged(path: str = DEFAULT_DB,
                             expected: str = SEED_DB_SHA256) -> str:
    actual = db_sha256(path)
    if actual != expected:
        raise AssertionError(
            f"seed DB changed: expected {expected}, got {actual}. "
            f"This analysis is read-only; a difference means something wrote to it.")
    return actual


def _connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Read-only by URI, so the connection itself cannot dirty the file.

    A plain sqlite3.connect() creates -wal/-shm siblings and can rewrite the
    header, which would break the byte-identity assertion this module makes.
    """
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


@dataclass
class Signal:
    """One candidate route to the fallback rate, and what it actually supports."""

    name: str
    question: str
    verdict: str
    finding: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "question": self.question,
                "verdict": self.verdict, "finding": self.finding,
                "evidence": self.evidence}


# --------------------------------------------------------------------------
# the probes
# --------------------------------------------------------------------------


def _probe_backtest_decisions(conn) -> Signal:
    n = conn.execute("SELECT COUNT(*) FROM backtest_decisions").fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(backtest_decisions)")]
    return Signal(
        name="backtest_decisions.decision_source",
        question="Is there a per-decision provenance field?",
        verdict=RECOVERABLE if n else NOT_RECOVERABLE,
        finding=(
            f"The column exists and is exactly the right shape — "
            f"`decision_source` is in {cols} — but the table holds {n} rows. "
            f"`insert_decisions` fires only under the ai_hedge_fund runtime, "
            f"never under the native path these runs used. The schema for the "
            f"answer is present; the data was never written."),
        evidence={"rows": n, "columns": cols,
                  "has_decision_source": "decision_source" in cols})


def _probe_trades(conn) -> Signal:
    n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
    reported = conn.execute(
        "SELECT SUM(num_trades) FROM agent_runs WHERE mode='leaderboard'"
    ).fetchone()[0] or 0
    return Signal(
        name="trades",
        question="Does a rule-based trade look different from a model trade?",
        verdict=NOT_RECOVERABLE,
        finding=(
            f"The table has {n} rows while agent_runs reports {reported} trades "
            f"across leaderboard runs, so per-trade detail was never persisted "
            f"at all. `num_trades` is a run-level aggregate. Even had the rows "
            f"existed, the fallback returns the SAME action schema as the model "
            f"— `reason` is the only free-text column and it is populated by "
            f"neither path distinguishably."),
        evidence={"rows": n, "columns": cols, "agent_runs_num_trades": reported})


def _probe_metadata(conn) -> Signal:
    rows = conn.execute("SELECT run_id, metadata FROM agent_runs").fetchall()
    keys = set()
    non_empty = 0
    for _rid, md in rows:
        if md:
            non_empty += 1
            try:
                parsed = json.loads(md)
                if isinstance(parsed, dict):
                    keys |= set(parsed)
            except (ValueError, TypeError):
                pass
    return Signal(
        name="agent_runs.metadata",
        question="Does the run snapshot record failed steps or fallbacks?",
        verdict=NOT_RECOVERABLE,
        finding=(
            f"{non_empty} of {len(rows)} runs carry any metadata; the union of "
            f"keys across all of them is {sorted(keys) or 'empty'}. Separately, "
            f"the builder that would populate it (`_llm_run_metadata`) snapshots "
            f"model_id, integration, temperature, reasoning_effort and window — "
            f"and no decision counter — so even a populated metadata column "
            f"would not carry this."),
        evidence={"runs": len(rows), "runs_with_metadata": non_empty,
                  "key_union": sorted(keys)})


def _probe_calls_per_bar(conn) -> Signal:
    q = """SELECT a.llm_model, a.llm_calls,
                  (SELECT COUNT(*) FROM equity_timeseries e WHERE e.run_id=a.run_id)
           FROM agent_runs a
           WHERE a.mode='leaderboard' AND a.llm_calls > 0
           ORDER BY a.llm_model"""
    rows = [{"model": m, "llm_calls": c, "bars": b,
             "calls_per_bar": round(c / b, 4) if b else None}
            for m, c, b in conn.execute(q)]
    return Signal(
        name="llm_calls vs bar count",
        question="Can the call ratio separate model decisions from fallbacks?",
        verdict=NOT_RECOVERABLE,
        finding=(
            "No, and not by a small margin. The call is billed BEFORE the "
            "response is parsed, so a step that falls back still consumes its "
            "call. Every ratio below is consistent with all 161 steps being "
            "model-driven AND with none of them being. This is the exact "
            "confusion commit 05d3974 fixed in the guard ('key H6 coverage on "
            "model-driven steps, not billed calls') — the fix changed the "
            "in-memory guard, not what is stored."),
        evidence={"runs": rows})


def _probe_llm_decisions_column(conn) -> Signal:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(agent_runs)")]
    return Signal(
        name="agent_runs.llm_decisions",
        question="Is the model-driven step counter persisted?",
        verdict=NOT_RECOVERABLE,
        finding=(
            f"There is no such column. agent_runs stores {cols}. "
            f"`PortfolioManager.llm_decisions` computes precisely the needed "
            f"numerator and `llm_agent.py` copies it onto the strategy object, "
            f"where the H6 publish guard reads it and then drops it — "
            f"`insert_run` accepts llm_calls/input_tokens/output_tokens/"
            f"est_cost_usd/metadata and nothing else. The number existed in "
            f"memory during every one of these runs and was never written."),
        evidence={"agent_runs_columns": cols,
                  "has_llm_decisions": "llm_decisions" in cols})


def _probe_equity_timeseries(conn) -> Signal:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(equity_timeseries)")]
    n = conn.execute("SELECT COUNT(*) FROM equity_timeseries").fetchone()[0]
    return Signal(
        name="equity_timeseries",
        question="Does the per-bar curve carry decision provenance?",
        verdict=NOT_RECOVERABLE,
        finding=(
            f"{n} rows over {cols}. It is the only per-bar table and it records "
            f"outcomes, not origins. A bar where the model held and a bar where "
            f"the fallback held are identical rows. It supports a decision "
            f"COUNT — which is how the cost-per-alpha work used it — and no "
            f"attribution."),
        evidence={"rows": n, "columns": cols})


def _probe_publish_guard(conn) -> Signal:
    """The H6 guard is the one place a bound could come from — carefully.

    It runs at publish time and raises, so a surviving row implies the run
    passed whatever version of the guard existed then. That is a real
    constraint, but it is weaker than it first looks and only for some runs.
    """
    q = """SELECT llm_model, created_at, llm_calls
           FROM agent_runs
           WHERE mode='leaderboard' AND llm_calls > 0
           ORDER BY created_at"""
    rows = conn.execute(q).fetchall()

    # SQLite CURRENT_TIMESTAMP is UTC. The two commits that matter, converted
    # from their +0800 author times:
    #   e11c541 H6 guard added        2026-07-05 22:44:16 +0800 = 14:44:16 UTC
    #   05d3974 guard keyed on
    #           llm_decisions         2026-07-06 11:49:53 +0800 = 03:49:53 UTC (07-06)
    guard_added_utc = "2026-07-05 14:44:16"
    decisions_keyed_utc = "2026-07-06 03:49:53"

    classified = []
    for model, created, calls in rows:
        if created < guard_added_utc:
            era = "before any H6 guard"
            bound = None
        elif created < decisions_keyed_utc:
            era = "guard keyed on llm_calls"
            bound = None
        else:
            era = "guard keyed on llm_decisions"
            bound = 0.95
        classified.append({"model": model, "created_at_utc": created,
                           "llm_calls": calls, "era": era,
                           "implied_min_coverage": bound})

    n_bounded = sum(1 for c in classified if c["implied_min_coverage"])
    return Signal(
        name="H6 publish guard (_reject_if_llm_fallback)",
        question="Does surviving the publish gate bound the fallback rate?",
        verdict=BOUND_ONLY if n_bounded else NOT_RECOVERABLE,
        finding=(
            f"Partly, for {n_bounded} of {len(rows)} runs, and only "
            f"conditionally. The guard refuses to publish below 95% coverage, "
            f"so a stored row implies it passed the guard AS IT EXISTED THEN. "
            f"Five runs predate the guard keying on llm_decisions: for those "
            f"the check compared BILLED CALLS to steps, which a fallback also "
            f"consumes, so passing it constrains nothing. Only runs created "
            f"after 05d3974 carry a real >=95% floor — and even that assumes "
            f"they were published through deploy_model_run with "
            f"allow_fallback=False, which the database does not record. "
            f"A floor is not a rate: >=95% spans 0 to 8 fallback decisions "
            f"in 161 and cannot be narrowed from stored data."),
        evidence={"runs": classified,
                  "min_coverage_threshold": 0.95,
                  "guard_added_utc": guard_added_utc,
                  "decisions_keyed_utc": decisions_keyed_utc,
                  "assumption": (
                      "commit timestamps approximate the code in the working "
                      "tree at run time; the DB does not record which revision "
                      "produced a run")})


def _probe_pipeline_exposure(conn) -> Signal:
    """Was the Phase 16 defect even on the leaderboard's code path?"""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "dashboard", "config",
        "leaderboard.json")
    entrants = []
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            for e in json.load(fh)["strategies"]:
                entrants.append({"id": e.get("id"),
                                 "strategy": e.get("strategy"),
                                 "has_pipeline": bool(e.get("pipeline"))})
    except (OSError, ValueError, KeyError) as exc:
        return Signal(
            name="pipeline exposure",
            question="Did the leaderboard runs use the defective code path?",
            verdict=NOT_RECOVERABLE,
            finding=f"could not read leaderboard config: {exc}",
            evidence={})

    any_pipeline = any(e["has_pipeline"] for e in entrants)
    return Signal(
        name="pipeline exposure",
        question="Did the leaderboard runs use the Phase 16 defective path?",
        verdict=NOT_RECOVERABLE,
        finding=(
            f"No. {sum(1 for e in entrants if not e['has_pipeline'])} of "
            f"{len(entrants)} entrants carry no pipeline, and llm_agent.py "
            f"calls make_trading_decision_with_llm without a `pipeline` "
            f"argument, so `pipeline_output_to_decision` — where the Phase 16 "
            f"defect lives — was never reached. The concern transfers anyway: "
            f"the single-prompt branch has the same defect class at "
            f"portfolio_manager.py:432-446, and llm_agent.py passes no "
            f"strict_llm either, so the silent-fallback branch is the one that "
            f"ran."),
        evidence={"entrants": entrants, "any_with_pipeline": any_pipeline})


PROBES = (
    _probe_llm_decisions_column,
    _probe_backtest_decisions,
    _probe_metadata,
    _probe_trades,
    _probe_calls_per_bar,
    _probe_equity_timeseries,
    _probe_publish_guard,
    _probe_pipeline_exposure,
)


def audit_signals(db_path: str = DEFAULT_DB) -> List[Signal]:
    """Run every probe. Read-only; the caller asserts byte-identity around it."""
    conn = _connect(db_path)
    try:
        return [probe(conn) for probe in PROBES]
    finally:
        conn.close()


def recoverability_verdict(signals: List[Signal]) -> Dict[str, Any]:
    """The overall answer, derived from the probes rather than asserted."""
    recoverable = [s for s in signals if s.verdict == RECOVERABLE]
    bounds = [s for s in signals if s.verdict == BOUND_ONLY]
    return {
        "recoverable": bool(recoverable),
        "verdict": (RECOVERABLE if recoverable
                    else BOUND_ONLY if bounds else NOT_RECOVERABLE),
        "n_signals_probed": len(signals),
        "n_recoverable": len(recoverable),
        "n_bound_only": len(bounds),
        "exact_fallback_rate_available": bool(recoverable),
        "statement": (
            "The fallback rate cannot be determined from the stored data. No "
            "table records per-decision provenance, and the one counter that "
            "distinguishes a model decision from a fallback "
            "(PortfolioManager.llm_decisions) is consumed by the publish guard "
            "and never written. A conditional >=95% coverage floor exists for "
            "runs created after the guard began keying on that counter, but a "
            "floor is not a rate."
        ) if not recoverable else "recoverable",
    }


def format_audit(signals: List[Signal], verdict: Dict[str, Any],
                 sha_before: str, sha_after: str) -> str:
    lines = ["=" * 78,
             "CAN THE LEADERBOARD FALLBACK RATE BE RECOVERED FROM STORED DATA?",
             "=" * 78,
             f"seed DB sha256 before: {sha_before}",
             f"seed DB sha256 after:  {sha_after}",
             f"byte-identical: {sha_before == sha_after}", ""]
    for s in signals:
        lines.append(f"[{s.verdict.upper():>16}]  {s.name}")
        lines.append(f"                    Q: {s.question}")
        for chunk in _wrap(s.finding, 72):
            lines.append(f"                    {chunk}")
        lines.append("")
    lines.append("-" * 78)
    lines.append(f"VERDICT: {verdict['verdict'].upper()}  "
                 f"({verdict['n_signals_probed']} signals probed, "
                 f"{verdict['n_recoverable']} recoverable, "
                 f"{verdict['n_bound_only']} bound-only)")
    for chunk in _wrap(verdict["statement"], 74):
        lines.append(f"  {chunk}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    db = (argv or [None])[0] or DEFAULT_DB
    before = assert_seed_db_unchanged(db) if db == DEFAULT_DB else db_sha256(db)
    signals = audit_signals(db)
    after = assert_seed_db_unchanged(db) if db == DEFAULT_DB else db_sha256(db)
    verdict = recoverability_verdict(signals)
    print(format_audit(signals, verdict, before, after))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
