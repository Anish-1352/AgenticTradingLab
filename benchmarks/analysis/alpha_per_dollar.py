"""Cost per unit of alpha — does the 193x model price spread buy performance?

The leaderboard runs seven models on the SAME task over the SAME window and
records return, Sharpe and max drawdown. Cost per call is separately measured
and verified. Joining them asks the question the whole serving-cost effort has
been circling: **is the expensive capability earning its price?**

WHY THIS IS MOSTLY A STATISTICS PROBLEM
----------------------------------------
There is exactly ONE evaluation window. Seven models on one month of one
universe cannot separate model skill from luck, and a ranking produced from it
would be noise presented as a finding. This module therefore computes the
uncertainty on every Sharpe it reports, and the headline output is the
CONFIDENCE INTERVAL, not the point estimate.

The standard error of a Sharpe estimate over n observations is approximately

    SE(S) ~= sqrt((1 + S^2 / 2) / n)

(Lo 2002, for iid returns.) At n=161 hourly bars that is large enough that
almost every pair of models overlaps, which is the actual result: the data
cannot rank them.

WHAT ONE WINDOW *CAN* SUPPORT
------------------------------
Two claims survive the small sample, because neither requires ranking:

1. **Cost does not track performance.** If price bought skill, the ordering by
   Sharpe would correlate with the ordering by cost. Whether it does is a
   property of this window, and a rank correlation near zero — or negative —
   is informative even when the individual ranks are not trustworthy.
2. **Comparison against passive baselines.** The leaderboard also ran index and
   buy-and-hold strategies over the identical window at zero LLM cost. An LLM
   that does not beat a free baseline has a cost problem that no serving
   optimisation addresses.

NEGATIVE RESULTS ARE THE POINT
-------------------------------
"The expensive model does not outperform" and "no model beats the index" are
the most valuable outcomes available here, because they would redirect the
project away from infrastructure work entirely.
"""

from __future__ import annotations

import math
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cost_model_lib import (  # noqa: E402
    DEFAULT_DB, PRICE_BY_DB_MODEL, cost_per_call, load_measured,
)

__all__ = [
    "load_leaderboard", "sharpe_standard_error", "sharpe_confidence_interval",
    "pairwise_overlap", "rank_correlation", "windows_needed",
    "cost_per_alpha_table", "BASELINE_STRATEGIES",
]

# Strategies with no LLM cost. They are the comparison that matters: an agent
# that cannot beat a free index has a problem no serving work can fix.
BASELINE_STRATEGIES = (
    "lb_spy_index", "lb_djia_index", "lb_buy_hold_djia",
    "lb_equal_weight_djia", "lb_mean_variance_djia",
)

TRADING_HOURS_PER_YEAR = 1638.0   # 6.5h x 252 sessions


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _returns_from_equity(conn: sqlite3.Connection, run_id: str) -> List[float]:
    """Per-bar simple returns from the stored equity curve."""
    rows = [r["equity"] for r in conn.execute(
        "SELECT equity FROM equity_timeseries WHERE run_id = ? ORDER BY timestamp",
        (run_id,))]
    out = []
    for prev, cur in zip(rows, rows[1:]):
        if prev:
            out.append((cur - prev) / prev)
    return out


def sharpe_standard_error(sharpe: float, n_obs: int) -> Optional[float]:
    """Approximate SE of a Sharpe estimate (Lo 2002, iid case).

    Assumes iid returns, which hourly equity-curve returns are not — they are
    autocorrelated and heteroskedastic, both of which make the true SE LARGER.
    So this is an optimistic bound on precision, and the conclusion it supports
    (that the intervals overlap) only gets stronger with a better estimator.
    """
    if n_obs is None or n_obs < 2:
        return None
    return math.sqrt((1.0 + 0.5 * sharpe * sharpe) / n_obs)


def sharpe_confidence_interval(sharpe: float, n_obs: int, z: float = 1.96
                               ) -> Optional[Tuple[float, float]]:
    se = sharpe_standard_error(sharpe, n_obs)
    if se is None:
        return None
    return (sharpe - z * se, sharpe + z * se)


def pairwise_overlap(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How many model pairs have overlapping Sharpe confidence intervals.

    Overlapping intervals do not prove equality, but non-overlap is the weakest
    evidence anyone would accept for "A beats B". If nearly every pair
    overlaps, the data cannot order the models and saying so is the finding.
    """
    pairs, overlapping = [], 0
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            ci_a, ci_b = a.get("sharpe_ci"), b.get("sharpe_ci")
            if not ci_a or not ci_b:
                continue
            over = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
            overlapping += int(over)
            pairs.append({"a": a["label"], "b": b["label"], "overlap": over})
    return {
        "n_pairs": len(pairs),
        "n_overlapping": overlapping,
        "fraction_overlapping": (overlapping / len(pairs)) if pairs else None,
        "separable_pairs": [p for p in pairs if not p["overlap"]],
        "pairs": pairs,
    }


def rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho, computed directly to avoid a scipy dependency.

    Used for cost-vs-performance. Near zero means price buys nothing on this
    window; negative means the expensive models did worse.
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return (num / den) if den else None


def windows_needed(observed_gap: float, sharpe: float, n_obs: int,
                   n_windows: int = 1) -> Dict[str, Any]:
    """How many independent windows would be needed — and why it cannot be said.

    A within-window standard error answers "how precisely did we measure THIS
    month", which is not the question. The question is whether the ordering
    GENERALISES, and that depends on across-window variance: how much a model's
    Sharpe moves from one month to the next.

    With one window that variance is not merely imprecise, it is
    **unestimable** — a single sample has no dispersion. So this returns a
    refusal rather than a number. Computing a sample size from within-window
    precision would produce a small, confident, wrong answer, which is exactly
    the failure this whole analysis is built to avoid.
    """
    within_se = sharpe_standard_error(sharpe, n_obs)
    if n_windows < 2:
        return {
            "available": False,
            "within_window_se": within_se,
            "reason": (
                "Across-window variance cannot be estimated from ONE window. "
                "Sizing a study needs to know how much a model's Sharpe moves "
                "between months, and a single sample carries no such "
                "information. The within-window SE below describes only how "
                "precisely this one month was measured, which is a different "
                "and much easier question."
            ),
            "what_would_resolve_it": (
                "Three to five non-overlapping windows would give a first "
                "estimate of across-window dispersion, from which a real "
                "sample size could be computed. Rolling or seasonally varied "
                "windows are better than consecutive ones, since adjacent "
                "months share regime."
            ),
        }
    return {"available": True, "within_window_se": within_se,
            "note": "across-window sizing requires the observed dispersion"}


def load_leaderboard(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    """Every leaderboard run, with cost joined and uncertainty computed."""
    conn = _connect(db_path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT run_id, llm_model, start_date, end_date, total_return, "
            "sharpe_ratio, max_drawdown, num_trades, llm_calls, "
            "input_tokens, output_tokens, est_cost_usd "
            "FROM agent_runs WHERE mode = 'leaderboard'")]
        windows = sorted({(r["start_date"], r["end_date"]) for r in rows})
        entries: List[Dict[str, Any]] = []
        for r in rows:
            rets = _returns_from_equity(conn, r["run_id"])
            n_obs = len(rets)
            per_period = None
            if n_obs >= 2:
                sd = statistics.pstdev(rets)
                per_period = (statistics.fmean(rets) / sd) if sd else None
            annualised = (per_period * math.sqrt(TRADING_HOURS_PER_YEAR)
                          if per_period is not None else None)
            stored = r["sharpe_ratio"]
            basis = annualised if annualised is not None else stored
            ci = (sharpe_confidence_interval(basis, n_obs)
                  if basis is not None else None)
            is_llm = bool(r["llm_calls"])
            entries.append({
                "run_id": r["run_id"],
                "label": (r["llm_model"] or r["run_id"].replace("lb_", "")),
                "is_llm": is_llm,
                "is_baseline": any(r["run_id"].startswith(b)
                                   for b in BASELINE_STRATEGIES),
                "window": (r["start_date"], r["end_date"]),
                "total_return": r["total_return"],
                "sharpe_stored": stored,
                "sharpe_recomputed": annualised,
                "sharpe_ci": ci,
                "n_observations": n_obs,
                "max_drawdown": r["max_drawdown"],
                "num_trades": r["num_trades"],
                "llm_calls": r["llm_calls"],
                "run_cost_usd": r["est_cost_usd"] if is_llm else 0.0,
            })
    finally:
        conn.close()

    return {
        "entries": entries,
        "windows": windows,
        "n_windows": len(windows),
        "n_llm_models": sum(1 for e in entries if e["is_llm"]),
        "n_baselines": sum(1 for e in entries if e["is_baseline"]),
        "single_window": len(windows) == 1,
    }


def cost_per_alpha_table(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    """The join: cost against risk-adjusted performance, with uncertainty.

    ``cost_per_sharpe`` is reported ONLY where Sharpe is positive. Dividing a
    cost by a negative Sharpe produces a negative "efficiency" that sorts as if
    it were excellent, which is worse than reporting nothing.
    """
    board = load_leaderboard(db_path)
    llm = [e for e in board["entries"] if e["is_llm"]]
    base = [e for e in board["entries"] if e["is_baseline"]]

    for e in llm:
        s = e["sharpe_recomputed"]
        e["cost_per_sharpe"] = (
            (e["run_cost_usd"] / s) if (s and s > 0) else None)
        e["cost_per_sharpe_note"] = (
            None if (s and s > 0)
            else "Sharpe <= 0: a cost-per-unit ratio is undefined, not zero")

    llm.sort(key=lambda e: -(e["sharpe_recomputed"] or float("-inf")))
    base.sort(key=lambda e: -(e["sharpe_recomputed"] or float("-inf")))

    costs = [e["run_cost_usd"] for e in llm]
    sharpes = [e["sharpe_recomputed"] or 0.0 for e in llm]
    returns = [e["total_return"] or 0.0 for e in llm]

    best_base = base[0] if base else None
    beat_baseline = [
        e for e in llm
        if best_base and (e["sharpe_recomputed"] or -9e9)
        > (best_base["sharpe_recomputed"] or 9e9)
    ]

    overlap = pairwise_overlap(llm)
    # How much more data to resolve the observed best-vs-worst LLM gap.
    gap = ((llm[0]["sharpe_recomputed"] or 0)
           - (llm[-1]["sharpe_recomputed"] or 0)) if len(llm) >= 2 else 0.0
    need = windows_needed(gap, llm[0]["sharpe_recomputed"] or 0,
                          llm[0]["n_observations"],
                          n_windows=board["n_windows"])

    return {
        "llm": llm,
        "baselines": base,
        "board": board,
        "cost_sharpe_rank_correlation": rank_correlation(costs, sharpes),
        "cost_return_rank_correlation": rank_correlation(costs, returns),
        "cost_span": (max(costs) / min(costs)) if costs and min(costs) else None,
        "best_baseline": best_base,
        "llm_beating_best_baseline": beat_baseline,
        "overlap": overlap,
        "observed_best_worst_gap": gap,
        "windows_needed_for_that_gap": need,
        "can_rank": (
            board["n_windows"] > 1
            and overlap["fraction_overlapping"] is not None
            and overlap["fraction_overlapping"] < 0.5
        ),
    }
