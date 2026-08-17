"""Per-LLM-call usage logging.

The load-bearing property is the last test in the first section: **the sum of
per-call rows equals the aggregate ``agent_runs`` already stores.** Those two
numbers are produced by completely independent code — the aggregate by ``+=``
in the portfolio manager, the rows by the recorder — so if they ever disagree,
one of them is wrong and the per-call table cannot be trusted for attribution.
That is the whole reason the aggregate was kept rather than replaced.
"""
import os
import sqlite3
import sys
import types

import pytest

from dashboard.backend.infrastructure.llm import usage_recorder as ur


# ---------------------------------------------------------------- recorder ---

def test_record_call_is_a_noop_without_a_recorder():
    """Every path outside a backtest must behave exactly as before."""
    ur.record_call(input_tokens=10, output_tokens=5)   # must not raise
    assert ur.active_recorder() is None


def test_recorder_captures_calls_in_order():
    rec = ur.UsageRecorder(run_id="r1")
    with ur.recording(rec):
        ur.record_call(input_tokens=100, output_tokens=10, step_label="a")
        ur.record_call(input_tokens=200, output_tokens=20, step_label="b")
    assert [c.call_index for c in rec.calls] == [0, 1]
    assert [c.step_label for c in rec.calls] == ["a", "b"]


def test_recorder_is_reset_after_the_block():
    rec = ur.UsageRecorder()
    with ur.recording(rec):
        assert ur.active_recorder() is rec
    assert ur.active_recorder() is None


def test_recorder_is_reset_even_when_the_block_raises():
    """A failed run must not leak its recorder into the next one."""
    rec = ur.UsageRecorder()
    with pytest.raises(ValueError):
        with ur.recording(rec):
            raise ValueError("step blew up")
    assert ur.active_recorder() is None


def test_totals_match_the_sum_of_rows():
    rec = ur.UsageRecorder()
    with ur.recording(rec):
        ur.record_call(input_tokens=100, output_tokens=10)
        ur.record_call(input_tokens=250, output_tokens=30)
    t = rec.totals()
    assert t == {"llm_calls": 2, "input_tokens": 350,
                 "output_tokens": 40, "cached_input_tokens": 0}


def test_failed_calls_are_recorded_so_spend_is_not_understated():
    rec = ur.UsageRecorder()
    with ur.recording(rec):
        ur.record_call(input_tokens=0, output_tokens=0, error="boom")
    assert len(rec) == 1
    assert rec.calls[0].error == "boom"


# ------------------------------------------------------------- kill switch ---

def test_usage_logging_is_on_by_default(monkeypatch):
    monkeypatch.delenv(ur._ENV_FLAG, raising=False)
    assert ur.usage_logging_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "DISABLED"])
def test_kill_switch_disables(monkeypatch, value):
    monkeypatch.setenv(ur._ENV_FLAG, value)
    assert ur.usage_logging_enabled() is False


def test_recording_none_suppresses_capture():
    """The disabled path installs None, so call sites need no branch."""
    with ur.recording(None):
        ur.record_call(input_tokens=1, output_tokens=1)
        assert ur.active_recorder() is None


# ----------------------------------------------------- cached-token capture ---

def _resp(**usage_fields):
    return types.SimpleNamespace(usage=types.SimpleNamespace(**usage_fields))


def test_cached_tokens_none_when_provider_says_nothing():
    """None and 0 are different facts and must not be collapsed.

    'the provider does not report caching' vs 'the cache was offered and
    returned nothing' — deliverable 3 has to tell those apart.
    """
    assert ur.extract_cached_input_tokens(_resp(input_tokens=10)) is None
    assert ur.extract_cached_input_tokens(types.SimpleNamespace()) is None


def test_cached_tokens_anthropic_spelling():
    assert ur.extract_cached_input_tokens(
        _resp(cache_read_input_tokens=1234)) == 1234


def test_cached_tokens_openai_spelling():
    resp = _resp(prompt_tokens_details=types.SimpleNamespace(cached_tokens=99))
    assert ur.extract_cached_input_tokens(resp) == 99


def test_cached_tokens_zero_is_preserved_not_treated_as_missing():
    assert ur.extract_cached_input_tokens(_resp(cache_read_input_tokens=0)) == 0


# ------------------------------------------------------------------ storage ---

@pytest.fixture()
def db(tmp_path, monkeypatch):
    # The path is passed explicitly, so there is no need to reimport the module
    # — and popping it from sys.modules would break the shared-singleton
    # identity that test_module_identity asserts for every later test.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    from dashboard.backend.database import BacktestDatabase
    return BacktestDatabase(str(tmp_path / "t.db"))


def _seed_run(db, run_id="run1", **totals):
    db.insert_run(
        run_id=run_id, session_id="s", agent_name="Agent", mode="backtest",
        start_date="2026-04-15", end_date="2026-04-16", initial_equity=1000.0,
        final_equity=1010.0, total_return=0.01, **totals)


def test_rows_round_trip(db):
    _seed_run(db)
    db.insert_llm_call_usage("run1", [
        {"call_index": 0, "step_label": "pipeline:1:A", "model": "m",
         "input_tokens": 100, "output_tokens": 10, "cached_input_tokens": None,
         "latency_ms": 12.5, "timestamp": "2026-04-15T00:00:00Z", "error": None},
        {"call_index": 1, "step_label": "pipeline:2:B", "model": "m",
         "input_tokens": 200, "output_tokens": 20, "cached_input_tokens": 64,
         "latency_ms": 8.0, "timestamp": "2026-04-15T00:00:01Z", "error": None},
    ])
    rows = db.get_llm_call_usage("run1")
    assert [r["call_index"] for r in rows] == [0, 1]
    assert rows[0]["cached_input_tokens"] is None
    assert rows[1]["cached_input_tokens"] == 64
    assert rows[0]["step_label"] == "pipeline:1:A"


def test_empty_insert_is_a_noop(db):
    _seed_run(db)
    db.insert_llm_call_usage("run1", [])
    assert db.get_llm_call_usage("run1") == []


def test_unknown_run_returns_empty_not_error(db):
    assert db.get_llm_call_usage("never-ran") == []


def test_per_call_sum_equals_the_agent_runs_aggregate(db):
    """THE INVARIANT. Two independent accumulators must agree.

    agent_runs totals come from `+=` in the portfolio manager; the rows come
    from the recorder. Disagreement means the per-call table cannot be trusted
    for attribution, which is the only reason it exists.
    """
    rec = ur.UsageRecorder()
    with ur.recording(rec):
        ur.record_call(input_tokens=4790, output_tokens=860, step_label="s1")
        ur.record_call(input_tokens=5100, output_tokens=910, step_label="s2")
        ur.record_call(input_tokens=4300, output_tokens=780, step_label="s3")
    totals = rec.totals()

    _seed_run(db, llm_calls=totals["llm_calls"],
              input_tokens=totals["input_tokens"],
              output_tokens=totals["output_tokens"])
    db.insert_llm_call_usage("run1", rec.rows())

    rows = db.get_llm_call_usage("run1")
    conn = sqlite3.connect(db.db_path)
    agg = conn.execute(
        "SELECT llm_calls, input_tokens, output_tokens FROM agent_runs "
        "WHERE run_id = 'run1'").fetchone()
    conn.close()

    assert len(rows) == agg[0]
    assert sum(r["input_tokens"] for r in rows) == agg[1]
    assert sum(r["output_tokens"] for r in rows) == agg[2]


def test_table_creation_is_idempotent(db):
    """Created on first write; a second write must not fail or duplicate."""
    _seed_run(db)
    row = {"call_index": 0, "step_label": "x", "model": "m", "input_tokens": 1,
           "output_tokens": 1, "cached_input_tokens": None, "latency_ms": None,
           "timestamp": "t", "error": None}
    db.insert_llm_call_usage("run1", [row])
    db.insert_llm_call_usage("run1", [{**row, "call_index": 1}])
    assert len(db.get_llm_call_usage("run1")) == 2
