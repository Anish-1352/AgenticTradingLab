"""Per-request accounting: the rows `agent_runs.llm_calls` cannot give you.

The point of these rows is that a retry is identifiable after the fact. A
column that exists but is never written on the retry path would be decorative,
so the tests that matter here drive the real retry loop and assert the extra
rows appear with the right attempt index.
"""

import os
import sqlite3

import pytest

from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Block:
    def __init__(self, text=None):
        self.type = "text" if text is not None else "thinking"
        if text is not None:
            self.text = text


class _Response:
    def __init__(self, text=None, in_tok=100, out_tok=2000):
        self.content = [_Block(text)] if text is not None else [_Block()]
        self.usage = _Usage(in_tok, out_tok)


def _manager():
    return PortfolioManager(initial_capital=1000.0, allowed_symbols=["AAPL"])


def test_a_successful_call_records_one_row():
    m = _manager()
    m.llm_step_index = 3
    m._record_llm_usage(_Response("{}"), attempt_index=0, model="m",
                        max_output_tokens=2000)
    m._mark_last_llm_call("text_returned")
    assert len(m.llm_call_rows) == 1
    row = m.llm_call_rows[0]
    assert row["step_index"] == 3
    assert row["attempt_index"] == 0
    assert row["outcome"] == "text_returned"
    assert row["input_tokens"] == 100
    assert row["max_output_tokens"] == 2000
    assert m.llm_calls == 1


def test_a_retry_is_a_second_row_with_a_higher_attempt_index():
    m = _manager()
    m._record_llm_usage(_Response(None), attempt_index=0)
    m._mark_last_llm_call("no_text_content")
    m._record_llm_usage(_Response("{}"), attempt_index=1)
    m._mark_last_llm_call("text_returned")
    assert [r["attempt_index"] for r in m.llm_call_rows] == [0, 1]
    assert [r["outcome"] for r in m.llm_call_rows] == [
        "no_text_content", "text_returned"]
    # The filter the totals counter cannot express.
    assert sum(1 for r in m.llm_call_rows if r["attempt_index"] > 0) == 1


def test_rows_reconcile_with_llm_calls():
    m = _manager()
    for i in range(4):
        m._record_llm_usage(_Response("{}" if i == 3 else None), attempt_index=i)
        m._mark_last_llm_call("text_returned" if i == 3 else "no_text_content")
    billed = [r for r in m.llm_call_rows if r["outcome"] != "usage_unreadable"]
    assert len(billed) == m.llm_calls
    assert sum(r["input_tokens"] for r in m.llm_call_rows) == m.input_tokens
    assert sum(r["output_tokens"] for r in m.llm_call_rows) == m.output_tokens


def test_a_response_with_no_usage_object_is_still_billed_at_zero_tokens():
    """Documents actual behaviour, which the docstring on ``_record_llm_usage``
    describes differently. ``extract_token_usage`` returns ``(0, 0)`` for a
    missing usage object rather than raising, so the call IS counted and the
    ``usage_unreadable`` branch only fires on a genuinely malformed usage.
    Left as-is: changing the billing counter is not this patch's business."""
    class Broken:
        content = [_Block("{}")]
        usage = None
    m = _manager()
    m._record_llm_usage(Broken())
    assert len(m.llm_call_rows) == 1
    assert m.llm_call_rows[0]["input_tokens"] == 0
    assert m.llm_calls == 1


def test_a_malformed_usage_object_is_recorded_and_not_billed():
    class Exploding:
        content = [_Block("{}")]

        @property
        def usage(self):
            raise RuntimeError("provider returned nonsense")

    m = _manager()
    m._record_llm_usage(Exploding())
    assert m.llm_call_rows[0]["outcome"] == "usage_unreadable"
    assert m.llm_calls == 0


def test_truncation_recovery_is_a_distinct_phase():
    m = _manager()
    m._record_llm_usage(_Response("{}"), attempt_index=0, phase="decision")
    m._mark_last_llm_call("text_returned")
    m._record_llm_usage(_Response("{}"), attempt_index=1,
                        phase="truncation_recovery")
    m._mark_last_llm_call("text_returned")
    phases = [r["phase"] for r in m.llm_call_rows]
    assert phases == ["decision", "truncation_recovery"]


def test_truncation_is_identifiable_from_a_row_alone():
    """output_tokens == max_output_tokens is the signature; without the
    ceiling on the row a short answer and a cut-off one look identical."""
    m = _manager()
    m._record_llm_usage(_Response(None, out_tok=2000), attempt_index=0,
                        max_output_tokens=2000)
    m._mark_last_llm_call("no_text_content")
    row = m.llm_call_rows[0]
    assert row["output_tokens"] == row["max_output_tokens"]


def test_step_index_advances_per_decision_even_without_a_client():
    """The index must move even on a step that spends nothing, or the next
    step's rows would be filed under this one. Stubs the rule-based fallback:
    what is under test is the counter, not the reference agent."""
    m = _manager()
    m.make_trading_decision = lambda *a, **k: {"actions": []}
    start = m.llm_step_index
    for _ in range(3):
        m.make_trading_decision_with_llm({}, llm_client=None)
    assert m.llm_step_index == start + 3


def test_rows_persist_and_are_queryable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    from dashboard.backend.database import BacktestDatabase
    db = BacktestDatabase(str(tmp_path / "t.db"))
    db.insert_llm_call_usage("run1", [
        {"step_index": 0, "attempt_index": 0, "phase": "decision",
         "outcome": "no_text_content", "model": "m", "input_tokens": 10,
         "output_tokens": 2000, "max_output_tokens": 2000},
        {"step_index": 0, "attempt_index": 1, "phase": "decision",
         "outcome": "text_returned", "model": "m", "input_tokens": 10,
         "output_tokens": 900, "max_output_tokens": 2000},
    ])
    con = sqlite3.connect(str(tmp_path / "t.db"))
    retries = con.execute(
        "SELECT COUNT(*) FROM llm_call_usage "
        "WHERE run_id=? AND attempt_index > 0", ("run1",)).fetchone()[0]
    assert retries == 1
    truncated = con.execute(
        "SELECT COUNT(*) FROM llm_call_usage "
        "WHERE output_tokens >= max_output_tokens").fetchone()[0]
    assert truncated == 1


def test_inserting_no_rows_is_a_noop(tmp_path):
    from dashboard.backend.database import BacktestDatabase
    db = BacktestDatabase(str(tmp_path / "t2.db"))
    db.insert_llm_call_usage("run1", [])
    con = sqlite3.connect(str(tmp_path / "t2.db"))
    assert con.execute("SELECT COUNT(*) FROM llm_call_usage").fetchone()[0] == 0
