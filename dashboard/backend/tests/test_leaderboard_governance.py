"""Leaderboard refresh governance.

The first section is the one that protects production: **with no configuration,
every entry still refreshes.** A leaderboard that silently stops updating looks
exactly like one that is updating with unchanged numbers, so the default must
be the current, expensive, correct behaviour.
"""
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.leaderboard import governance as gov

WINDOW = {"start_date": "2026-04-15", "end_date": "2026-05-15",
          "initial_capital": 1000.0, "session_id": "lb"}
ENTRY = {"id": "gpt_5_5", "strategy": "llm_agent", "model": "GPT-5.5",
         "prompt": "trade well"}
NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


# ------------------------------------------------- defaults = today's behaviour

def test_no_config_means_refresh_everything(monkeypatch):
    for var in ("SKIP_UNCHANGED", "CADENCE_DAYS", "WINDOW_DAYS"):
        monkeypatch.delenv(gov._ENV_PREFIX + var, raising=False)
    g = gov.load_governance({})
    assert g.skip_unchanged is False
    assert g.cadence_days is None
    assert g.window_days is None
    assert g.active is False


def test_default_policy_refreshes_even_when_nothing_changed(monkeypatch):
    """Change-detection is opt-in, not implicit."""
    monkeypatch.delenv(gov._ENV_PREFIX + "SKIP_UNCHANGED", raising=False)
    g = gov.load_governance({})
    fp = gov.config_fingerprint(ENTRY, WINDOW)
    d = gov.should_refresh(ENTRY, WINDOW, g, last_fingerprint=fp,
                           last_refreshed_at=NOW.isoformat(), now=NOW)
    assert d.refresh is True


def test_default_window_is_untouched():
    g = gov.GovernanceConfig()
    assert gov.resolve_window(WINDOW, g) == dict(WINDOW)


def test_plan_refreshes_all_entries_by_default():
    entries = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    plan = gov.plan_refresh(entries, WINDOW, gov.GovernanceConfig())
    assert all(d.refresh for d in plan)


# ------------------------------------------------------------ fingerprinting

def test_fingerprint_is_stable():
    assert gov.config_fingerprint(ENTRY, WINDOW) == \
        gov.config_fingerprint(dict(ENTRY), dict(WINDOW))


def test_fingerprint_changes_with_the_window():
    other = {**WINDOW, "end_date": "2026-05-16"}
    assert gov.config_fingerprint(ENTRY, WINDOW) != \
        gov.config_fingerprint(ENTRY, other)


def test_fingerprint_changes_with_any_entry_field():
    """Whole-entry hashing: a new config key must not slip past."""
    assert gov.config_fingerprint(ENTRY, WINDOW) != \
        gov.config_fingerprint({**ENTRY, "prompt": "trade differently"}, WINDOW)
    assert gov.config_fingerprint(ENTRY, WINDOW) != \
        gov.config_fingerprint({**ENTRY, "brand_new_key": 1}, WINDOW)


# -------------------------------------------------------- change detection

def test_skip_unchanged_skips_when_identical():
    g = gov.GovernanceConfig(skip_unchanged=True)
    fp = gov.config_fingerprint(ENTRY, WINDOW)
    d = gov.should_refresh(ENTRY, WINDOW, g, last_fingerprint=fp, now=NOW)
    assert d.refresh is False and "unchanged" in d.reason


def test_skip_unchanged_still_refreshes_when_config_changed():
    g = gov.GovernanceConfig(skip_unchanged=True)
    stale = gov.config_fingerprint({**ENTRY, "prompt": "old"}, WINDOW)
    d = gov.should_refresh(ENTRY, WINDOW, g, last_fingerprint=stale, now=NOW)
    assert d.refresh is True


def test_skip_unchanged_refreshes_when_there_is_no_history():
    """First run must always execute — nothing to compare against."""
    g = gov.GovernanceConfig(skip_unchanged=True)
    d = gov.should_refresh(ENTRY, WINDOW, g, last_fingerprint=None, now=NOW)
    assert d.refresh is True


# ---------------------------------------------------------------- cadence

def test_cadence_skips_inside_the_interval():
    g = gov.GovernanceConfig(cadence_days={"gpt_5_5": 7})
    recent = (NOW - timedelta(days=2)).isoformat()
    d = gov.should_refresh(ENTRY, WINDOW, g, last_refreshed_at=recent, now=NOW)
    assert d.refresh is False and "cadence" in d.reason


def test_cadence_allows_after_the_interval():
    g = gov.GovernanceConfig(cadence_days={"gpt_5_5": 7})
    old = (NOW - timedelta(days=8)).isoformat()
    d = gov.should_refresh(ENTRY, WINDOW, g, last_refreshed_at=old, now=NOW)
    assert d.refresh is True


def test_cadence_may_key_on_model_name():
    g = gov.GovernanceConfig(cadence_days={"GPT-5.5": 7})
    recent = (NOW - timedelta(days=1)).isoformat()
    d = gov.should_refresh(ENTRY, WINDOW, g, last_refreshed_at=recent, now=NOW)
    assert d.refresh is False


def test_cadence_with_no_history_refreshes():
    g = gov.GovernanceConfig(cadence_days={"gpt_5_5": 7})
    d = gov.should_refresh(ENTRY, WINDOW, g, last_refreshed_at=None, now=NOW)
    assert d.refresh is True


def test_entry_without_cadence_is_unaffected():
    """Expensive weekly must not make cheap models weekly too."""
    g = gov.GovernanceConfig(cadence_days={"gpt_5_5": 7})
    cheap = {"id": "nemotron", "model": "Nemotron"}
    recent = (NOW - timedelta(days=1)).isoformat()
    d = gov.should_refresh(cheap, WINDOW, g, last_refreshed_at=recent, now=NOW)
    assert d.refresh is True


def test_cadence_beats_change_detection():
    """A trivial config edit must not silently restore daily spend."""
    g = gov.GovernanceConfig(skip_unchanged=True, cadence_days={"gpt_5_5": 7})
    changed = gov.config_fingerprint({**ENTRY, "prompt": "edited"}, WINDOW)
    recent = (NOW - timedelta(days=1)).isoformat()
    d = gov.should_refresh(ENTRY, WINDOW, g, last_fingerprint=changed,
                           last_refreshed_at=recent, now=NOW)
    assert d.refresh is False and "cadence" in d.reason


# ----------------------------------------------------------------- window

def test_window_days_shortens_the_range():
    g = gov.GovernanceConfig(window_days=7)
    out = gov.resolve_window(WINDOW, g)
    assert out["end_date"] == "2026-05-15"
    assert out["start_date"] == "2026-05-08"


def test_invalid_window_leaves_config_alone():
    g = gov.GovernanceConfig(window_days=7)
    assert gov.resolve_window({"end_date": "not-a-date"}, g)["end_date"] == "not-a-date"


def test_nonpositive_window_is_ignored(monkeypatch):
    monkeypatch.setenv(gov._ENV_PREFIX + "WINDOW_DAYS", "0")
    assert gov.load_governance({}).window_days is None


# ----------------------------------------------------------------- config

def test_config_file_supplies_policy(monkeypatch):
    for var in ("SKIP_UNCHANGED", "CADENCE_DAYS", "WINDOW_DAYS"):
        monkeypatch.delenv(gov._ENV_PREFIX + var, raising=False)
    g = gov.load_governance({"governance": {
        "skip_unchanged": True, "cadence_days": {"gpt_5_5": 7}, "window_days": 30}})
    assert g.skip_unchanged is True
    assert g.cadence_days == {"gpt_5_5": 7}
    assert g.window_days == 30


def test_env_overrides_the_file(monkeypatch):
    monkeypatch.setenv(gov._ENV_PREFIX + "SKIP_UNCHANGED", "1")
    monkeypatch.setenv(gov._ENV_PREFIX + "CADENCE_DAYS", '{"gpt_5_5": 14}')
    g = gov.load_governance({"governance": {"skip_unchanged": False}})
    assert g.skip_unchanged is True
    assert g.cadence_days == {"gpt_5_5": 14}


def test_malformed_env_cadence_is_ignored(monkeypatch):
    monkeypatch.setenv(gov._ENV_PREFIX + "CADENCE_DAYS", "not json")
    g = gov.load_governance({"governance": {"cadence_days": {"a": 3}}})
    assert g.cadence_days == {"a": 3}
