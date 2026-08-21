"""The fallback-recoverability audit.

Two groups. The first checks each probe reports what the database actually
holds. The second is the one that matters: that the module never produces a
fallback rate. The brief is explicit — "an invented fraction here would corrupt
exactly the result it is meant to strengthen" — so the absence of a number is a
tested property, not a stylistic choice.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import fallback_recoverability as fr    # noqa: E402
from analysis.cost_model_lib import DEFAULT_DB        # noqa: E402


@pytest.fixture(scope="module")
def signals():
    return fr.audit_signals(DEFAULT_DB)


@pytest.fixture(scope="module")
def by_name(signals):
    return {s.name: s for s in signals}


# ------------------------------------------------------- the DB is untouched -

def test_seed_db_matches_the_committed_hash():
    assert fr.assert_seed_db_unchanged(DEFAULT_DB) == fr.SEED_DB_SHA256


def test_the_audit_does_not_modify_the_database():
    before = fr.db_sha256(DEFAULT_DB)
    fr.audit_signals(DEFAULT_DB)
    assert fr.db_sha256(DEFAULT_DB) == before == fr.SEED_DB_SHA256


def test_the_audit_writes_nothing_to_the_db_file_itself():
    """Bytes AND mtime, so a rewrite-with-identical-content would still show.

    Not asserted: the absence of -wal/-shm siblings. The seed DB is in WAL
    mode and a legitimate read-only connection maps the -shm; a stale empty
    -wal also predates this analysis. The invariant that matters is that
    backtest.db is not written.
    """
    mtime_before = os.path.getmtime(DEFAULT_DB)
    before = fr.db_sha256(DEFAULT_DB)
    fr.audit_signals(DEFAULT_DB)
    assert fr.db_sha256(DEFAULT_DB) == before == fr.SEED_DB_SHA256
    assert os.path.getmtime(DEFAULT_DB) == mtime_before


def test_the_audit_leaves_no_pending_writes_in_the_wal():
    """A non-empty -wal after a read would mean the connection wrote."""
    fr.audit_signals(DEFAULT_DB)
    wal = DEFAULT_DB + "-wal"
    if os.path.exists(wal):
        assert os.path.getsize(wal) == 0, "the audit left pending writes"


def test_a_changed_db_is_refused(tmp_path):
    p = tmp_path / "fake.db"
    p.write_bytes(b"not the seed db")
    with pytest.raises(AssertionError, match="seed DB changed"):
        fr.assert_seed_db_unchanged(str(p))


# ---------------------------------------------------------------- the probes -

def test_every_probe_reports_a_known_verdict(signals):
    assert len(signals) == len(fr.PROBES)
    for s in signals:
        assert s.verdict in (fr.RECOVERABLE, fr.NOT_RECOVERABLE, fr.BOUND_ONLY)
        assert s.question and s.finding


def test_the_provenance_column_exists_but_is_empty(by_name):
    """backtest_decisions.decision_source is exactly the right field."""
    s = by_name["backtest_decisions.decision_source"]
    assert s.evidence["has_decision_source"] is True
    assert s.evidence["rows"] == 0
    assert s.verdict == fr.NOT_RECOVERABLE


def test_llm_decisions_is_computed_but_never_stored(by_name):
    s = by_name["agent_runs.llm_decisions"]
    assert s.evidence["has_llm_decisions"] is False
    assert s.verdict == fr.NOT_RECOVERABLE


def test_trades_table_is_empty_despite_a_nonzero_run_total(by_name):
    s = by_name["trades"]
    assert s.evidence["rows"] == 0
    assert s.evidence["agent_runs_num_trades"] > 0


def test_metadata_is_empty_across_every_run(by_name):
    s = by_name["agent_runs.metadata"]
    assert s.evidence["runs_with_metadata"] == 0
    assert s.evidence["key_union"] == []


def test_calls_per_bar_is_one_and_therefore_uninformative(by_name):
    """1.000 calls/bar is consistent with 161 model decisions and with 0."""
    s = by_name["llm_calls vs bar count"]
    ratios = [r["calls_per_bar"] for r in s.evidence["runs"]]
    assert len(ratios) == 7
    assert all(0.99 <= r <= 1.0 for r in ratios)
    assert s.verdict == fr.NOT_RECOVERABLE


def test_leaderboard_never_used_the_phase16_code_path(by_name):
    """The defect is in pipeline_output_to_decision; no entrant has a pipeline."""
    s = by_name["pipeline exposure"]
    assert s.evidence["any_with_pipeline"] is False
    assert len(s.evidence["entrants"]) == 12
    assert "same defect class" in s.finding


# ------------------------------------------------------------- the H6 bound --

def test_the_guard_bound_applies_only_to_runs_after_the_counter_landed(by_name):
    s = by_name["H6 publish guard (_reject_if_llm_fallback)"]
    runs = s.evidence["runs"]
    assert len(runs) == 7
    bounded = [r for r in runs if r["implied_min_coverage"]]
    unbounded = [r for r in runs if not r["implied_min_coverage"]]
    assert len(bounded) == 2 and len(unbounded) == 5
    assert all(r["implied_min_coverage"] == 0.95 for r in bounded)


def test_runs_predating_the_counter_get_no_bound_because_calls_are_blind(by_name):
    """The guard then compared BILLED calls, which a fallback also consumes."""
    s = by_name["H6 publish guard (_reject_if_llm_fallback)"]
    early = [r for r in s.evidence["runs"] if not r["implied_min_coverage"]]
    assert {r["era"] for r in early} == {
        "before any H6 guard", "guard keyed on llm_calls"}


def test_the_bound_carries_its_assumption(by_name):
    """It rests on commit dates approximating the code that ran."""
    s = by_name["H6 publish guard (_reject_if_llm_fallback)"]
    assert "does not record which revision" in s.evidence["assumption"]
    assert "allow_fallback=False" in s.finding


def test_a_floor_is_not_a_rate(by_name):
    s = by_name["H6 publish guard (_reject_if_llm_fallback)"]
    assert s.verdict == fr.BOUND_ONLY
    assert "A floor is not a rate" in s.finding


# ================================================== the anti-fabrication test =

def test_the_verdict_is_that_the_rate_is_not_recoverable(signals):
    v = fr.recoverability_verdict(signals)
    assert v["recoverable"] is False
    assert v["exact_fallback_rate_available"] is False
    assert v["n_recoverable"] == 0
    assert "cannot be determined from the stored data" in v["statement"]


def test_no_probe_reports_a_fallback_count_or_fraction(signals):
    """THE point of this module.

    A number here would look like a measurement and be an invention. The only
    numbers any probe may carry are things actually stored: row counts, column
    lists, call counts, timestamps, and the guard's own 0.95 threshold.
    """
    banned = ("fallback_rate", "fallback_fraction", "fallback_count",
              "n_fallbacks", "estimated_fallbacks", "llm_decisions_estimate")
    for s in signals:
        for key in s.evidence:
            assert key not in banned, f"{s.name} invented {key}"
        blob = str(s.evidence)
        assert "fallback_rate" not in blob


def test_the_verdict_exposes_no_rate_field(signals):
    v = fr.recoverability_verdict(signals)
    for key in v:
        assert "rate" not in key or key == "exact_fallback_rate_available"


def test_the_report_renders_and_states_the_negative(signals):
    v = fr.recoverability_verdict(signals)
    text = fr.format_audit(signals, v, fr.SEED_DB_SHA256, fr.SEED_DB_SHA256)
    assert "byte-identical: True" in text
    assert "cannot be determined" in text
    assert "VERDICT:" in text


def test_main_runs_clean_and_leaves_the_db_alone(capsys):
    assert fr.main([]) == 0
    assert fr.db_sha256(DEFAULT_DB) == fr.SEED_DB_SHA256
    assert "byte-identical: True" in capsys.readouterr().out
