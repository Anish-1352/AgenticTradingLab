"""Backtest latency trace, its guards, and the generated report.

The most important test here is the seed-DB guard. Importing
``dashboard.backend.database`` runs migrations at import time against whatever
DATABASE_PATH resolves to, and with it unset that is the committed seed
database — which this script rewrote once during development. The guard makes
that failure mode impossible rather than remembered.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.tier_check import check_report          # noqa: E402

_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
REPORT = os.path.join(_ANALYSIS, "BACKTEST_LATENCY.md")

_spec = importlib.util.spec_from_file_location(
    "_blt", os.path.join(_ANALYSIS, "backtest_latency_trace.py"))
blt = importlib.util.module_from_spec(_spec)
sys.modules["_blt"] = blt
_spec.loader.exec_module(blt)


@pytest.fixture(scope="module")
def traces():
    out = {}
    for k in ("A", "B"):
        with open(os.path.join(_RESULTS, f"backtest_trace_{k}.json"),
                  "r", encoding="utf-8") as fh:
            out[k] = json.load(fh)
    return out


# ------------------------------------------------------- the seed-DB guard --

def test_guard_refuses_when_database_path_is_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    with pytest.raises(SystemExit, match="DATABASE_PATH is not set"):
        blt.guard_seed_db()


def test_guard_refuses_when_pointed_at_the_seed_db(monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", blt.SEED_DB)
    with pytest.raises(SystemExit, match="committed seed DB"):
        blt.guard_seed_db()


def test_guard_allows_any_other_path(monkeypatch, tmp_path):
    p = str(tmp_path / "scratch.db")
    monkeypatch.setenv("DATABASE_PATH", p)
    assert blt.guard_seed_db() == p


def test_the_seed_db_is_currently_intact():
    assert blt.seed_db_sha256() == blt.SEED_DB_SHA256
    blt.assert_seed_db_untouched()


def test_both_traced_runs_left_the_seed_db_untouched(traces):
    for t in traces.values():
        assert t["seed_db_sha256_before"] == blt.SEED_DB_SHA256
        assert t["seed_db_sha256_after"] == blt.SEED_DB_SHA256


# ------------------------------------------------------------ the timers ----

def test_every_timer_attached_in_both_runs(traces):
    """An unattached timer would surface as unattributed time and be read as
    overhead that does not exist."""
    for k, t in traces.items():
        missing = [n for n, ok in t["timers_attached"].items() if not ok]
        assert not missing, f"run {k} missing timers: {missing}"


def test_wrappers_are_removed_after_instrumenting():
    from dashboard.backend.domain.backtesting import portfolio_manager as pm
    before = pm.PortfolioManager.get_portfolio_state
    timings = blt.Timings()
    blt.instrument(timings)
    assert pm.PortfolioManager.get_portfolio_state is not before
    blt.restore()
    assert pm.PortfolioManager.get_portfolio_state is before


def test_a_wrapper_records_time_without_changing_the_return():
    timings = blt.Timings()

    class Owner:
        @staticmethod
        def work(x):
            return x * 2

    assert blt._wrap(Owner, "work", "phase", timings) is True
    assert Owner.work(21) == 42
    assert timings.count["phase"] == 1
    blt.restore()


def test_wrapping_a_missing_attribute_reports_false():
    assert blt._wrap(object(), "nope", "p", blt.Timings()) is False


# ------------------------------------------------------- the measurement ----

def test_llm_dominates_the_bar_loop_in_both_runs(traces):
    for k, t in traces.items():
        share = t["timings"]["phases"]["llm"]["share_of_wall"]
        assert share > 0.99, f"run {k}: LLM share only {share}"


def test_all_non_llm_work_is_under_three_percent(traces):
    """The hypothesis allowed for a hidden 40%. There is not a hidden 3%."""
    for k, t in traces.items():
        p = t["timings"]["phases"]
        non_llm = sum(v["total_seconds"] for n, v in p.items() if n != "llm")
        assert non_llm / t["timings"]["wall_seconds"] < 0.03


def test_market_data_is_fetched_once_not_per_bar(traces):
    for t in traces.values():
        assert t["timings"]["phases"]["market_data"]["calls"] == 1
        assert t["bars"] > 1


def test_per_call_latency_exceeds_the_assumed_band(traces):
    """Assumed 1-3s; measured means are above it in both runs."""
    for t in traces.values():
        assert t["timings"]["phases"]["llm"]["mean_seconds"] > 3.0


def test_requests_per_bar_is_not_reliably_one(traces):
    """Run B measured 1.00, run A more. The variance is the finding."""
    ratios = {k: t["timings"]["phases"]["llm"]["calls"] / t["bars"]
              for k, t in traces.items()}
    assert min(ratios.values()) == pytest.approx(1.0)
    assert max(ratios.values()) > 1.5


def test_startup_is_small_and_outside_the_bar_loop(traces):
    for t in traces.values():
        startup = (t.get("construct_seconds", 0) + t.get("load_data_seconds", 0)
                   + t.get("indicators_seconds", 0))
        assert startup < 5.0
        assert startup < t["timings"]["wall_seconds"]


# ------------------------------------------------------------- the report ---

def test_report_is_not_stale():
    from analysis import make_latency_report as mk
    with open(REPORT, "r", encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "BACKTEST_LATENCY.md is stale; regenerate with "
            "`python benchmarks/analysis/make_latency_report.py`")


def test_report_is_fully_tagged():
    with open(REPORT, "r", encoding="utf-8") as fh:
        assert check_report(fh.read())["ok"]


def test_report_says_bars_cannot_be_decoupled():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "answer (b)" in t
    assert "changes what the backtest computes" in t
    assert "portfolio_manager.py:615" in t
    assert "validator.py:719-725" in t


def test_report_rejects_ziplime_on_fit_and_licence():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "GPL-3.0" in t and "OpenMDW" in t
    assert "authoring time" in t
    assert "different problem" in t


def test_report_flags_which_options_change_results():
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "changes RESULTS?" in t
    assert "product decision" in t


def test_report_reconciles_with_the_reported_five_minutes():
    """My traces imply LONGER than 5 min, which must be said plainly."""
    with open(REPORT, "r", encoding="utf-8") as fh:
        t = fh.read()
    assert "longer* than the reported" in t or "longer than the reported" in t


def test_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_latency_report.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------- the conftest-level guard --

def test_conftest_redirects_database_path_away_from_the_seed_db():
    """Meta-test. Without this redirect, importing any dashboard module from a
    test rewrites the seed DB via import-time migrations — which happened twice
    while this phase was written."""
    target = os.environ.get("DATABASE_PATH")
    assert target, "conftest did not set DATABASE_PATH"
    assert os.path.abspath(target) != os.path.abspath(blt.SEED_DB)


def test_importing_a_dashboard_module_leaves_the_seed_db_intact():
    before = blt.seed_db_sha256()
    from dashboard.backend import database  # noqa: F401
    from dashboard.backend.domain.backtesting import portfolio_manager  # noqa: F401
    assert blt.seed_db_sha256() == before == blt.SEED_DB_SHA256
