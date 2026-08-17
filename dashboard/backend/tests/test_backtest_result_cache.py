"""Backtest result cache — key correctness and storage.

The tests that matter most are the MISS tests. A missed hit costs one backtest;
a wrong hit shows a user results that do not match the configuration in front
of them, and is very hard to notice. So every field that can move a number gets
a test proving that changing it changes the key.
"""
import sys

import pytest

from dashboard.backend.domain.backtesting import result_cache as rc


def _inputs(**over):
    base = dict(
        session_id="user-1",
        model="nvidia/nemotron-3-nano-30b-a3b",
        start_date="2026-04-15",
        end_date="2026-05-15",
        symbols=["AAPL", "MSFT"],
        bar_interval="1h",
        initial_capital=1000.0,
        data_source="alpaca",
        mode="safe_trading",
        strategy_prompt="Buy low, sell high.",
        pipeline=[{"label": "A", "prompt": "p", "outputFormat": "{}"}],
        runtime_type="pipeline",
        runtime_config={},
        max_output_tokens=2000,
    )
    base.update(over)
    return rc.CacheKeyInputs(**base)


def key(**over) -> str:
    return rc.compute_cache_key(_inputs(**over))


# ------------------------------------------------------------- determinism ---

def test_same_config_same_key():
    assert key() == key()


def test_dict_ordering_does_not_change_the_key():
    """Otherwise the same config hashes differently run to run and never hits."""
    a = key(runtime_config={"a": 1, "b": 2})
    b = key(runtime_config={"b": 2, "a": 1})
    assert a == b


def test_symbol_order_does_not_change_the_key():
    """The same universe in a different order is the same universe."""
    assert key(symbols=["AAPL", "MSFT"]) == key(symbols=["MSFT", "AAPL"])


def test_integral_float_capital_matches_int():
    assert key(initial_capital=1000.0) == key(initial_capital=1000)


# -------------------------------------------------- every input must matter ---

@pytest.mark.parametrize("field,changed", [
    ("model", "openai/gpt-5.5"),
    ("start_date", "2026-04-16"),
    ("end_date", "2026-05-16"),
    ("symbols", ["AAPL"]),
    ("bar_interval", "1d"),
    ("initial_capital", 2000.0),
    ("data_source", "vnpy_simulation"),
    ("mode", "buy_and_hold"),
    ("runtime_type", "ai_hedge_fund"),
    ("runtime_config", {"depth": 3}),
    ("max_output_tokens", 4096),
    ("extra", {"anything": "new"}),
])
def test_changing_any_input_changes_the_key(field, changed):
    assert key(**{field: changed}) != key()


def test_a_changed_prompt_misses():
    """The headline case: users iterate on prompts, and an edited prompt is a
    different experiment. One character must be enough."""
    assert key(strategy_prompt="Buy low, sell high.") != \
        key(strategy_prompt="Buy low, sell high!")


def test_a_changed_pipeline_step_misses():
    assert key(pipeline=[{"label": "A", "prompt": "p", "outputFormat": "{}"}]) != \
        key(pipeline=[{"label": "A", "prompt": "CHANGED", "outputFormat": "{}"}])


def test_adding_a_pipeline_step_misses():
    two = [{"label": "A", "prompt": "p", "outputFormat": "{}"},
           {"label": "B", "prompt": "q", "outputFormat": "{}"}]
    assert key(pipeline=two) != key()


def test_pipeline_step_order_matters():
    """Steps are sequential and each feeds the next, so order changes results."""
    a = [{"label": "A"}, {"label": "B"}]
    b = [{"label": "B"}, {"label": "A"}]
    assert key(pipeline=a) != key(pipeline=b)


def test_none_prompt_differs_from_empty_prompt():
    assert key(strategy_prompt=None) != key(strategy_prompt="")


def test_key_version_busts_every_entry():
    before = key()
    original = rc.CACHE_KEY_VERSION
    try:
        rc.CACHE_KEY_VERSION = original + 1
        assert key() != before
    finally:
        rc.CACHE_KEY_VERSION = original


# ------------------------------------------------------------------- scope ---

def test_user_scope_separates_users():
    a = rc.compute_cache_key(_inputs(session_id="user-1"), scope="user")
    b = rc.compute_cache_key(_inputs(session_id="user-2"), scope="user")
    assert a != b


def test_global_scope_lets_identical_configs_share():
    a = rc.compute_cache_key(_inputs(session_id="user-1"), scope="global")
    b = rc.compute_cache_key(_inputs(session_id="user-2"), scope="global")
    assert a == b


def test_default_scope_is_per_user(monkeypatch):
    monkeypatch.delenv(rc._ENV_SCOPE, raising=False)
    assert rc.cache_scope() == "user"


def test_unknown_scope_falls_back_to_user(monkeypatch):
    monkeypatch.setenv(rc._ENV_SCOPE, "everyone")
    assert rc.cache_scope() == "user"


# ------------------------------------------------------------------- flags ---

def test_cache_is_off_by_default(monkeypatch):
    """Unset must mean current behaviour."""
    monkeypatch.delenv(rc._ENV_ENABLED, raising=False)
    assert rc.cache_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "on", "YES", "enabled"])
def test_cache_can_be_enabled(monkeypatch, value):
    monkeypatch.setenv(rc._ENV_ENABLED, value)
    assert rc.cache_enabled() is True


def test_ttl_default_and_override(monkeypatch):
    monkeypatch.delenv(rc._ENV_TTL_HOURS, raising=False)
    assert rc.cache_ttl_hours() == float(rc.DEFAULT_TTL_HOURS)
    monkeypatch.setenv(rc._ENV_TTL_HOURS, "48")
    assert rc.cache_ttl_hours() == 48.0
    monkeypatch.setenv(rc._ENV_TTL_HOURS, "nonsense")
    assert rc.cache_ttl_hours() == float(rc.DEFAULT_TTL_HOURS)


# --------------------------------------------------------------------- TTL ---

def test_expiry():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 17, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat()
    stale = (now - timedelta(hours=200)).isoformat()
    assert rc.is_expired(fresh, ttl_hours=24, now=now) is False
    assert rc.is_expired(stale, ttl_hours=24, now=now) is True


def test_ttl_zero_never_expires():
    assert rc.is_expired("2020-01-01T00:00:00+00:00", ttl_hours=0) is False


def test_unknown_age_is_treated_as_expired():
    """Re-running costs money; serving a result of unknown vintage costs trust."""
    assert rc.is_expired(None, ttl_hours=24) is True
    assert rc.is_expired("not-a-date", ttl_hours=24) is True


# ----------------------------------------------------------------- lookup ----

def test_cache_lookup_surfaces_provenance():
    hit = rc.CacheLookup(hit=True, run_id="r1", cached_at="2026-08-01")
    assert "cached result from 2026-08-01" == hit.cached_from
    assert rc.CacheLookup(hit=False).cached_from is None


# ---------------------------------------------------------------- storage ----

@pytest.fixture()
def db(tmp_path, monkeypatch):
    # The path is passed explicitly, so there is no need to reimport the module
    # — and popping it from sys.modules would break the shared-singleton
    # identity that test_module_identity asserts for every later test.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    from dashboard.backend.database import BacktestDatabase
    return BacktestDatabase(str(tmp_path / "t.db"))


def _seed_run(db, run_id):
    db.insert_run(run_id=run_id, session_id="s", agent_name="Agent",
                  mode="backtest", start_date="2026-04-15", end_date="2026-04-16",
                  initial_equity=1000.0, final_equity=1010.0, total_return=0.01)


def test_put_then_get(db):
    _seed_run(db, "run1")
    db.put_cached_run_id("k1", "run1", session_id="s")
    got = db.get_cached_run_id("k1")
    assert got is not None and got[0] == "run1"


def test_miss_returns_none(db):
    assert db.get_cached_run_id("nope") is None


def test_entry_pointing_at_a_deleted_run_reads_as_a_miss(db):
    """A dangling pointer must not be served as a hit."""
    _seed_run(db, "run1")
    db.put_cached_run_id("k1", "run1")
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.execute("DELETE FROM agent_runs WHERE run_id='run1'")
    conn.commit(); conn.close()
    assert db.get_cached_run_id("k1") is None


def test_invalidate_by_key(db):
    _seed_run(db, "run1")
    db.put_cached_run_id("k1", "run1")
    assert db.invalidate_cached_run(cache_key="k1") == 1
    assert db.get_cached_run_id("k1") is None


def test_invalidate_by_run_id_drops_every_key_for_it(db):
    _seed_run(db, "run1")
    db.put_cached_run_id("k1", "run1")
    db.put_cached_run_id("k2", "run1")
    assert db.invalidate_cached_run(run_id="run1") == 2
    assert db.get_cached_run_id("k1") is None
    assert db.get_cached_run_id("k2") is None


def test_invalidate_with_no_argument_is_a_noop(db):
    assert db.invalidate_cached_run() == 0


def test_reinsert_replaces(db):
    _seed_run(db, "run1")
    _seed_run(db, "run2")
    db.put_cached_run_id("k1", "run1")
    db.put_cached_run_id("k1", "run2")
    assert db.get_cached_run_id("k1")[0] == "run2"
