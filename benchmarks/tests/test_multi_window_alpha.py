"""Multi-window cost-per-alpha harness.

Built before the data exists, so the tests carry most of the assurance. They
fall into two groups:

* it behaves correctly on the ONE window that exists today — refusing, and
  refusing for the right reason;
* it behaves correctly on N windows, exercised against a synthetic multi-window
  database so the aggregation path is not first executed on real data under
  time pressure.

The synthetic data validates the CODE. It is not a finding and no number
derived from it appears in any report.
"""
import os
import random
import shutil
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import multi_window_alpha as mw     # noqa: E402
from analysis.cost_model_lib import DEFAULT_DB    # noqa: E402


@pytest.fixture(scope="module")
def one_window():
    return mw.multi_window_table(DEFAULT_DB)


@pytest.fixture(scope="module")
def three_windows(tmp_path_factory):
    """Clone the real leaderboard runs into two extra windows, perturbed.

    Synthetic on purpose: the point is to drive the aggregation path, not to
    learn anything about the models.
    """
    d = tmp_path_factory.mktemp("mw")
    p = str(d / "multi.db")
    shutil.copy(DEFAULT_DB, p)
    conn = sqlite3.connect(p)
    random.seed(7)
    rows = conn.execute(
        "SELECT * FROM agent_runs WHERE mode='leaderboard' AND llm_calls>0"
    ).fetchall()
    cols = [c[1] for c in conn.execute("PRAGMA table_info(agent_runs)")]
    for w, (s, e) in enumerate(
            [("2026-05-16", "2026-06-15"), ("2026-06-16", "2026-07-15")], 1):
        for r in rows:
            rec = dict(zip(cols, r))
            old, new = rec["run_id"], f"{rec['run_id']}_w{w}"
            rec.update(run_id=new, start_date=s, end_date=e)
            conn.execute(
                f"INSERT INTO agent_runs ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [rec[k] for k in cols])
            eq = conn.execute(
                "SELECT timestamp, equity, cash, positions_value "
                "FROM equity_timeseries WHERE run_id=? ORDER BY timestamp",
                (old,)).fetchall()
            drift = random.uniform(-0.2, 0.2)
            for i, (ts, q, cash, pv) in enumerate(eq):
                conn.execute(
                    "INSERT INTO equity_timeseries "
                    "(run_id, timestamp, equity, cash, positions_value) "
                    "VALUES (?,?,?,?,?)",
                    (new, ts, q * (1 + drift * i / max(len(eq), 1)), cash, pv))
    conn.commit()
    conn.close()
    return mw.multi_window_table(p)


# ------------------------------------------- one window: refuse correctly ---

def test_one_window_cannot_rank(one_window):
    assert one_window["n_windows"] == 1
    assert one_window["can_rank"] is False


def test_one_window_readiness_is_insufficient(one_window):
    r = one_window["readiness"]
    assert r["state"] == "insufficient"
    assert r["can_estimate_spread"] is False
    assert "ranking models" in r["does_not_support"]
    assert r["next"]


def test_one_window_spread_is_unestimable_not_zero(one_window):
    """0.0 would read as 'perfectly stable', the opposite of the truth."""
    for row in one_window["rows"]:
        s = row["sharpe"]
        assert s["available"] is False
        assert "stdev" not in s
        assert "unestimable" in s["reason"]


def test_one_window_still_supports_the_correlation(one_window):
    """The claim that survives a single draw stays available."""
    assert one_window["cost_sharpe_rank_correlation"] is not None
    assert "cost-vs-performance rank correlation" in \
        " ".join(one_window["readiness"]["supports"])


def test_windows_needed_still_refuses_on_one_window(one_window):
    assert one_window["windows_needed"]["available"] is False


# ------------------------------------------------ N windows: aggregate -----

def test_three_windows_are_detected(three_windows):
    assert three_windows["n_windows"] == 3
    assert all(r["n_windows"] == 3 for r in three_windows["rows"])


def test_three_windows_produce_a_spread(three_windows):
    for row in three_windows["rows"]:
        s = row["sharpe"]
        assert s["available"] is True
        assert s["stdev"] >= 0 and s["min"] <= s["mean"] <= s["max"]
        assert s["sem"] == pytest.approx(s["stdev"] / (3 ** 0.5))


def test_three_windows_readiness_is_ready(three_windows):
    r = three_windows["readiness"]
    assert r["state"] == "ready"
    assert r["can_estimate_spread"] is True
    # Even 'ready' must keep the iid caveat attached.
    assert any("iid" in d for d in r["does_not_support"])


def test_windows_needed_becomes_answerable(three_windows):
    assert three_windows["windows_needed"]["available"] is True


def test_can_rank_needs_separability_not_just_windows(three_windows):
    """Enough windows is necessary, not sufficient.

    The synthetic runs have large across-window dispersion, so the intervals
    overlap and the gate still refuses. That is the correct behaviour and the
    reason can_rank is not simply `n_windows >= 2`.
    """
    assert three_windows["n_windows"] >= mw.MIN_WINDOWS_TO_RANK
    assert three_windows["overlap"]["fraction_overlapping"] >= 0.5
    assert three_windows["can_rank"] is False


def test_two_windows_flags_a_thin_spread():
    assert mw.readiness(2)["state"] == "thin"
    assert mw.readiness(2)["can_rank"] is True
    assert mw.readiness(1)["can_rank"] is False
    assert mw.readiness(5)["state"] == "ready"


# ------------------------------------------------------- the metric set -----

def test_every_required_metric_is_present(three_windows):
    """Brief: cost, return, Sharpe, max drawdown, cost per risk-adjusted unit."""
    for row in three_windows["rows"]:
        assert row["total_cost_usd"] is not None
        for key in ("return", "sharpe", "max_drawdown"):
            assert key in row and "mean" in row[key]
        assert "cost_per_sharpe" in row


def test_cost_per_sharpe_undefined_for_nonpositive_mean(three_windows):
    for row in three_windows["rows"]:
        mean = row["sharpe"]["mean"]
        if mean is not None and mean <= 0:
            assert row["cost_per_sharpe"] is None
            assert "undefined" in row["cost_per_sharpe_note"]


def test_both_framings_are_reported_separately(three_windows):
    """Sharpe and raw return disagree; the harness must not collapse them."""
    assert "beat_baseline_on_sharpe" in three_windows
    assert "beat_baseline_on_return" in three_windows
    assert isinstance(three_windows["beat_baseline_on_return"], list)


def test_baselines_are_aggregated_across_windows_too(three_windows):
    assert three_windows["baselines"]
    assert all(b["total_cost_usd"] == 0.0 for b in three_windows["baselines"])


# ------------------------------------------------------------- spread -------

def test_spread_of_empty_is_unavailable():
    assert mw.across_window_spread([])["available"] is False


def test_spread_ignores_none_values():
    s = mw.across_window_spread([1.0, None, 3.0])
    assert s["n"] == 2 and s["mean"] == pytest.approx(2.0)


def test_spread_flags_thin_samples():
    assert mw.across_window_spread([1.0, 2.0])["thin"] is True
    assert mw.across_window_spread([1.0, 2.0, 3.0, 4.0])["thin"] is False


def test_spread_reports_range_and_cv():
    s = mw.across_window_spread([2.0, 4.0, 6.0])
    assert s["range"] == pytest.approx(4.0)
    assert s["coefficient_of_variation"] == pytest.approx(s["stdev"] / 4.0)


# --------------------------------------------------------- gate reuse -------

def test_gates_are_imported_not_reimplemented():
    """A second copy of a statistical gate is a second place to get it wrong."""
    import inspect
    from analysis import alpha_per_dollar as ap
    src = inspect.getsource(mw)
    assert "def sharpe_standard_error" not in src
    assert "def pairwise_overlap" not in src
    assert "def rank_correlation" not in src
    assert "def windows_needed" not in src
    assert mw.windows_needed is ap.windows_needed
    assert mw.pairwise_overlap is ap.pairwise_overlap


def test_grouping_is_stable_and_window_ordered(three_windows):
    for row in three_windows["rows"]:
        assert row["windows"] == sorted(row["windows"])
