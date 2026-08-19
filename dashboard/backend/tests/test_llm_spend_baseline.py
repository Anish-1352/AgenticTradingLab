"""Baseline LLM spend reporting.

The property that carries the weight: **on an empty table it reports nothing
and estimates nothing.** An estimated baseline would silently propagate into
every later "cost fell X%" claim, which is precisely what the per-call table
was built to prevent.
"""
import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_spec = importlib.util.spec_from_file_location(
    "llm_spend_baseline", os.path.join(_SCRIPTS, "llm_spend_baseline.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    from dashboard.backend.database import BacktestDatabase
    return BacktestDatabase(str(tmp_path / "t.db"))


def _run(db, run_id, mode="backtest"):
    db.insert_run(run_id=run_id, session_id="s", agent_name="A", mode=mode,
                  start_date="2026-04-15", end_date="2026-05-15",
                  initial_equity=1000.0, final_equity=1010.0, total_return=0.01)


def _call(idx, model, inp, out, step="pipeline:1:A", err=None, cached=None):
    return {"call_index": idx, "step_label": step, "model": model,
            "input_tokens": inp, "output_tokens": out,
            "cached_input_tokens": cached, "latency_ms": 100.0,
            "timestamp": "2026-08-15T00:00:00Z", "error": err}


# ------------------------------------------------ refusing to guess ---------

def test_missing_database_reports_absence(tmp_path):
    calls, state = baseline.load_calls(str(tmp_path / "nope.db"), None, None)
    assert calls == [] and state["db_exists"] is False


def test_missing_table_reports_absence(db):
    calls, state = baseline.load_calls(str(db.db_path), None, None)
    assert calls == []
    assert state["db_exists"] is True and state["table_exists"] is False


def test_empty_report_estimates_nothing(db):
    calls, state = baseline.load_calls(str(db.db_path), None, None)
    text = baseline.render(baseline.summarise(calls), db_path=str(db.db_path),
                           since=None, until=None, state=state)
    assert "No baseline yet" in text
    assert "none is estimated" in text
    # And it must not quietly substitute the run-level aggregate.
    assert "Deliberately NOT used as a substitute" in text
    assert "$0.0000" not in text


def test_empty_report_names_which_absence_it_is(db):
    """'no table' and 'table but no rows' are different operational states."""
    calls, state = baseline.load_calls(str(db.db_path), None, None)
    text = baseline.render(baseline.summarise(calls), db_path=str(db.db_path),
                           since=None, until=None, state=state)
    assert "does not exist in this database" in text

    _run(db, "r1")
    # insert_llm_call_usage returns early on an empty list and does NOT create
    # the table; a read is what materialises it. Using the read here keeps the
    # test honest about which call has that side effect.
    db.get_llm_call_usage("r1")
    calls2, state2 = baseline.load_calls(str(db.db_path), None, None)
    text2 = baseline.render(baseline.summarise(calls2), db_path=str(db.db_path),
                            since=None, until=None, state=state2)
    assert state2["table_exists"] is True
    assert "holds no rows" in text2


def test_dry_run_writes_nothing(db, tmp_path, capsys):
    out = tmp_path / "should_not_exist.md"
    code = baseline.main(["--db", str(db.db_path), "--dry-run",
                          "--out", str(out)])
    assert code == 0
    assert not out.exists()
    assert "nothing written" in capsys.readouterr().out


# ------------------------------------------------------- the breakdowns -----

def test_workload_split(db):
    _run(db, "bt1", "backtest")
    _run(db, "lb1", "leaderboard")
    _run(db, "pp1", "paper_baseline")
    db.insert_llm_call_usage("bt1", [_call(0, "nvidia/nemotron-3-nano-30b-a3b", 100, 10)])
    db.insert_llm_call_usage("lb1", [_call(0, "openai/gpt-5.5", 100, 10)])
    db.insert_llm_call_usage("pp1", [_call(0, "openai/gpt-5.5", 100, 10)])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    s = baseline.summarise(calls)
    assert set(s["by_workload"]) == {"backtest", "leaderboard", "live_paper"}


def test_unknown_mode_is_surfaced_not_folded(db):
    """An unrecognised mode must not be absorbed into a bucket it may not be in."""
    _run(db, "x1", "some_new_mode")
    db.insert_llm_call_usage("x1", [_call(0, "openai/gpt-5.5", 100, 10)])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    assert "some_new_mode" in baseline.summarise(calls)["by_workload"]


def test_step_kind_is_the_axis(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [
        _call(0, "openai/gpt-5.5", 100, 10, step="pipeline:1:Technical"),
        _call(1, "openai/gpt-5.5", 100, 10, step="pipeline:2:Risk"),
        _call(2, "openai/gpt-5.5", 100, 10, step="post_trade:daily"),
    ])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    by_step = baseline.summarise(calls)["by_step"]
    assert by_step["pipeline"]["calls"] == 2
    assert by_step["post_trade"]["calls"] == 1


def test_model_split_and_cost(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [
        _call(0, "nvidia/nemotron-3-nano-30b-a3b", 1_000_000, 1_000_000),
    ])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    s = baseline.summarise(calls)
    # 1M input at $0.05 + 1M output at $0.20
    assert s["total_cost_usd"] == pytest.approx(0.25)


# ------------------------------------------------------------- pricing -----

def test_unpriced_model_counted_in_tokens_excluded_from_dollars(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [_call(0, "some/unlisted", 1_000_000, 1_000_000)])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    s = baseline.summarise(calls)
    assert s["by_model"]["some/unlisted"]["input"] == 1_000_000
    assert s["total_cost_usd"] == 0.0
    assert "some/unlisted" in s["unpriced_models"]


def test_free_slug_prices_at_zero_not_at_the_paid_rate():
    """Substring-matching ':free' to the paid model invents a charge."""
    assert baseline.price_for("nvidia/nemotron-3-nano-30b-a3b:free") == (0.0, 0.0)
    assert baseline.price_for("nvidia/nemotron-3-nano-30b-a3b") == (0.05, 0.20)


def test_unknown_model_has_no_price():
    assert baseline.price_for("who/knows") is None
    assert baseline.price_for(None) is None


# ---------------------------------------------------------------- cache -----

def test_absent_cache_field_is_not_a_zero_hit_rate(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [_call(0, "openai/gpt-5.5", 100, 10)])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    s = baseline.summarise(calls)
    assert s["cache_reporting_calls"] == 0
    text = baseline.render(s, db_path=str(db.db_path), since=None, until=None,
                           state={"db_exists": True, "table_exists": True})
    assert "not** a measured zero" in text


def test_reported_cache_is_counted(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [_call(0, "openai/gpt-5.5", 100, 10, cached=64)])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    s = baseline.summarise(calls)
    assert s["cache_reporting_calls"] == 1 and s["cached_tokens_total"] == 64


# ------------------------------------------------------------- windowing ----

def test_date_filter_bounds_the_window(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [
        {**_call(0, "openai/gpt-5.5", 100, 10), "timestamp": "2026-08-01T00:00:00Z"},
        {**_call(1, "openai/gpt-5.5", 100, 10), "timestamp": "2026-08-20T00:00:00Z"},
    ])
    calls, _ = baseline.load_calls(str(db.db_path), "2026-08-10", None)
    assert len(calls) == 1


def test_errors_are_counted(db):
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [_call(0, "openai/gpt-5.5", 0, 0, err="boom")])
    calls, _ = baseline.load_calls(str(db.db_path), None, None)
    assert baseline.summarise(calls)["by_model"]["openai/gpt-5.5"]["errors"] == 1


def test_report_warns_that_windows_must_be_comparable(db):
    """A quieter week is not a saving, and the script says so."""
    _run(db, "r1")
    db.insert_llm_call_usage("r1", [_call(0, "openai/gpt-5.5", 100, 10)])
    calls, state = baseline.load_calls(str(db.db_path), None, None)
    text = baseline.render(baseline.summarise(calls), db_path=str(db.db_path),
                           since=None, until=None, state=state)
    assert "not a saving" in text
