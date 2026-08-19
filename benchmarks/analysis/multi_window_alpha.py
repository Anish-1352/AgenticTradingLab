"""Cost per unit of alpha, across N evaluation windows.

The single-window analysis in ``alpha_per_dollar`` established what one draw
can support: cost does not track performance, and the sample cannot rank the
models. This module is the consumer for the data that fixes the second half —
it aggregates across windows and produces the across-window spread that a
single window cannot.

**This EXTENDS the single-window result rather than replacing it.** The
one-window figures stay reachable and stay true of that window; what changes
with more data is whether an ordering can be trusted, not what happened in
April.

WHAT ARRIVES WITH THE SECOND WINDOW
------------------------------------
Across-window variance. It is the quantity that decides whether a model's
Sharpe advantage is skill or the month, and with one window it is not merely
imprecise — a single sample has no dispersion, so it is unestimable. Every gate
in this module keys off that.

THE GATES ARE REUSED, NOT REDERIVED
------------------------------------
``alpha_per_dollar`` already owns them and they are imported, not copied:

* ``sharpe_standard_error`` — Lo (2002), and it assumes **iid returns**.
  Hourly equity returns are autocorrelated and heteroskedastic, both of which
  make the true interval WIDER than shown. Non-overlapping pairs are therefore
  the least trustworthy part of any table this produces.
* ``pairwise_overlap`` — how many model pairs are separable at all.
* ``rank_correlation`` — cost against performance.
* ``windows_needed`` — refuses to answer from a single draw.

WHY THIS IS THE LARGEST COST LEVER
-----------------------------------
Model choice spans 134x in the sensitivity ranking, more than every other lever
combined. If the price premium buys no performance across several windows, then
substituting open-weight models is justified on evidence rather than on budget
— and that is a configuration change, not an infrastructure programme.

The harness is built now so that it consumes the windows the moment they exist,
rather than being written under time pressure once they do.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.alpha_per_dollar import (  # noqa: E402
    BASELINE_STRATEGIES, load_leaderboard, pairwise_overlap, rank_correlation,
    sharpe_standard_error, windows_needed,
)
from analysis.cost_model_lib import DEFAULT_DB  # noqa: E402

__all__ = [
    "MIN_WINDOWS_TO_RANK", "group_by_model", "across_window_spread",
    "multi_window_table", "readiness",
]

# Two is the arithmetic minimum for a dispersion estimate and nowhere near
# enough to trust one. The gate is set at 2 because below it the quantity does
# not exist; a caller wanting confidence should read n_windows and the spread,
# not this flag alone.
MIN_WINDOWS_TO_RANK = 2

# Below this, an across-window standard deviation is reported but flagged as
# too thin to interpret. Three points is the usual floor for a dispersion
# anyone will act on.
MIN_WINDOWS_FOR_STABLE_SPREAD = 3


def group_by_model(entries: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Collect each model's runs across windows, keyed by model label."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        out.setdefault(e["label"], []).append(e)
    for runs in out.values():
        runs.sort(key=lambda r: r["window"])
    return out


def across_window_spread(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Dispersion of one model's metric across windows.

    ``available`` is False for a single window — not because the computation
    fails, but because a one-element sample has no dispersion and reporting
    0.0 would read as "perfectly stable", which is the opposite of the truth.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"available": False, "n": 0, "reason": "no values"}
    if n < 2:
        return {
            "available": False, "n": n, "mean": clean[0],
            "reason": (
                "one window: dispersion is unestimable, not zero. A single "
                "sample cannot say whether this figure is stable."
            ),
        }
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    return {
        "available": True,
        "n": n,
        "mean": mean,
        "stdev": sd,
        "min": min(clean),
        "max": max(clean),
        "range": max(clean) - min(clean),
        # Standard error of the mean across windows — the quantity that
        # actually bounds a claim about a model's typical performance.
        "sem": sd / math.sqrt(n),
        "coefficient_of_variation": (sd / abs(mean)) if mean else None,
        "thin": n < MIN_WINDOWS_FOR_STABLE_SPREAD,
        "thin_note": (
            f"fewer than {MIN_WINDOWS_FOR_STABLE_SPREAD} windows: the spread "
            f"exists but is itself estimated from very few points"
        ) if n < MIN_WINDOWS_FOR_STABLE_SPREAD else None,
    }


def multi_window_table(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    """Per-model cost and performance aggregated across every window present.

    Works today against one window — and says so, rather than presenting a
    one-window mean as if it were a multi-window one.
    """
    board = load_leaderboard(db_path)
    llm = [e for e in board["entries"] if e["is_llm"]]
    base = [e for e in board["entries"] if e["is_baseline"]]

    by_model = group_by_model(llm)
    rows: List[Dict[str, Any]] = []
    for label, runs in by_model.items():
        sharpes = [r["sharpe_recomputed"] for r in runs]
        returns = [r["total_return"] for r in runs]
        drawdowns = [r["max_drawdown"] for r in runs]
        costs = [r["run_cost_usd"] for r in runs]
        total_cost = sum(c for c in costs if c is not None)
        sharpe_spread = across_window_spread(sharpes)
        mean_sharpe = sharpe_spread.get("mean")

        rows.append({
            "label": label,
            "n_windows": len(runs),
            "windows": [r["window"] for r in runs],
            "total_cost_usd": total_cost,
            "mean_cost_per_window": total_cost / len(runs) if runs else None,
            "return": across_window_spread(returns),
            "sharpe": sharpe_spread,
            "max_drawdown": across_window_spread(drawdowns),
            # Cost per unit of risk-adjusted return, withheld for a
            # non-positive denominator: a negative would sort as excellent.
            "cost_per_sharpe": (
                (total_cost / len(runs)) / mean_sharpe
                if (mean_sharpe and mean_sharpe > 0) else None),
            "cost_per_sharpe_note": (
                None if (mean_sharpe and mean_sharpe > 0)
                else "mean Sharpe <= 0: the ratio is undefined, not zero"),
            "n_observations_total": sum(r["n_observations"] for r in runs),
        })

    rows.sort(key=lambda r: -(r["sharpe"].get("mean") or float("-inf")))

    # Baselines aggregated the same way, so the comparison survives more
    # windows too.
    base_rows = []
    for label, runs in group_by_model(base).items():
        base_rows.append({
            "label": label,
            "n_windows": len(runs),
            "return": across_window_spread([r["total_return"] for r in runs]),
            "sharpe": across_window_spread(
                [r["sharpe_recomputed"] for r in runs]),
            "total_cost_usd": 0.0,
        })
    base_rows.sort(key=lambda r: -(r["sharpe"].get("mean") or float("-inf")))

    mean_costs = [r["mean_cost_per_window"] or 0.0 for r in rows]
    mean_sharpes = [r["sharpe"].get("mean") or 0.0 for r in rows]
    mean_returns = [r["return"].get("mean") or 0.0 for r in rows]

    best_base = base_rows[0] if base_rows else None
    beat_on_sharpe, beat_on_return = [], []
    if best_base:
        bs = best_base["sharpe"].get("mean")
        br = best_base["return"].get("mean")
        for r in rows:
            if bs is not None and (r["sharpe"].get("mean") or -9e9) > bs:
                beat_on_sharpe.append(r["label"])
            if br is not None and (r["return"].get("mean") or -9e9) > br:
                beat_on_return.append(r["label"])

    overlap = pairwise_overlap([
        {"label": r["label"],
         "sharpe_ci": _mean_sharpe_ci(r)} for r in rows])

    gap = ((rows[0]["sharpe"].get("mean") or 0)
           - (rows[-1]["sharpe"].get("mean") or 0)) if len(rows) >= 2 else 0.0

    return {
        "rows": rows,
        "baselines": base_rows,
        "board": board,
        "n_windows": board["n_windows"],
        "windows": board["windows"],
        "cost_sharpe_rank_correlation": rank_correlation(mean_costs, mean_sharpes),
        "cost_return_rank_correlation": rank_correlation(mean_costs, mean_returns),
        "cost_span": (max(mean_costs) / min(mean_costs))
        if mean_costs and min(mean_costs) else None,
        "best_baseline": best_base,
        "beat_baseline_on_sharpe": beat_on_sharpe,
        "beat_baseline_on_return": beat_on_return,
        "overlap": overlap,
        "observed_best_worst_gap": gap,
        "windows_needed": windows_needed(
            gap, rows[0]["sharpe"].get("mean") or 0,
            rows[0]["n_observations_total"] if rows else 0,
            n_windows=board["n_windows"]) if rows else None,
        # THE GATE. Reused, not re-derived.
        "can_rank": (
            board["n_windows"] >= MIN_WINDOWS_TO_RANK
            and overlap["fraction_overlapping"] is not None
            and overlap["fraction_overlapping"] < 0.5
        ),
        "readiness": readiness(board["n_windows"]),
    }


def _mean_sharpe_ci(row: Dict[str, Any]) -> Optional[tuple]:
    """Interval around a model's mean Sharpe.

    With several windows this uses the across-window SEM, which is the honest
    bound on "this model's typical Sharpe". With one it falls back to the
    within-window Lo error, which answers a narrower question and is labelled
    as such by the caller.
    """
    s = row["sharpe"]
    mean = s.get("mean")
    if mean is None:
        return None
    if s.get("available") and s.get("sem"):
        return (mean - 1.96 * s["sem"], mean + 1.96 * s["sem"])
    se = sharpe_standard_error(mean, row["n_observations_total"])
    return (mean - 1.96 * se, mean + 1.96 * se) if se else None


def readiness(n_windows: int) -> Dict[str, Any]:
    """What this many windows can and cannot support. The harness's status line."""
    if n_windows < MIN_WINDOWS_TO_RANK:
        return {
            "state": "insufficient",
            "n_windows": n_windows,
            "can_rank": False,
            "can_estimate_spread": False,
            "supports": [
                "cost-vs-performance rank correlation for the window observed",
                "comparison against zero-cost baselines on that window",
            ],
            "does_not_support": [
                "ranking models",
                "any claim that a model's advantage generalises",
                "sizing how many further windows are needed",
            ],
            "next": (
                f"Run the leaderboard over at least "
                f"{MIN_WINDOWS_TO_RANK} non-overlapping windows; "
                f"{MIN_WINDOWS_FOR_STABLE_SPREAD} before acting on a spread. "
                f"Non-adjacent windows are better than consecutive ones, since "
                f"neighbouring months share regime."
            ),
        }
    if n_windows < MIN_WINDOWS_FOR_STABLE_SPREAD:
        return {
            "state": "thin",
            "n_windows": n_windows,
            "can_rank": True,
            "can_estimate_spread": True,
            "supports": ["a dispersion estimate, and a provisional ordering"],
            "does_not_support": ["confidence in that dispersion"],
            "next": (f"A {MIN_WINDOWS_FOR_STABLE_SPREAD}rd window would make "
                     f"the spread itself worth quoting."),
        }
    return {
        "state": "ready",
        "n_windows": n_windows,
        "can_rank": True,
        "can_estimate_spread": True,
        "supports": ["ranking with an across-window interval",
                     "cost per unit of risk-adjusted return, with spread"],
        "does_not_support": [
            "treating the interval as exact — the Lo error assumes iid returns "
            "and hourly equity returns are not, so true intervals are wider"
        ],
        "next": "None blocking.",
    }
