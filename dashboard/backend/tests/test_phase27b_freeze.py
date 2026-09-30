"""Phase 27B freeze guards: the calendar is checked against the tape, and every
symbol must be tradable one share at a time under ATL's position limit."""
import pandas as pd
import pytest

from dashboard.backend.tests.test_phase27b_dataset import tape, small_spec
from dashboard.scripts.phase27b_freeze import verify_calendar, price_feasibility, symbol_data_gaps


def _auction(frame):
    """Give every session a market-on-close spike in its 15:55 bar, as real tapes have."""
    frame = frame.copy()
    local = frame.index.tz_convert("America/New_York")
    frame.loc[local.strftime("%H:%M") == "15:55", "volume"] *= 8
    return frame


def _early_close_with_after_hours(frame, day):
    """A 13:00 close whose afternoon slots are all filled by thin after-hours prints.

    A liquid ETF prints every five minutes after an early close, so its bar
    count looks exactly like a full session. Only the missing auction tells.
    """
    frame = frame.copy()
    local = frame.index.tz_convert("America/New_York")
    after = (local.date == pd.Timestamp(day).date()) & (local.strftime("%H:%M") >= "13:00")
    frame.loc[after, "volume"] = 5.0
    return frame


def _truncate(frame, day, last="12:55"):
    local = frame.index.tz_convert("America/New_York")
    keep = ~((local.date == pd.Timestamp(day).date()) & (local.strftime("%H:%M") > last))
    return frame.loc[keep]


def _drop_day(frame, day):
    local = frame.index.tz_convert("America/New_York")
    return frame.loc[local.date != pd.Timestamp(day).date()]


def test_a_correct_calendar_verifies():
    s = small_spec(excluded_sessions=["2024-01-15", "2024-01-17"])
    raw = _truncate(_drop_day(_auction(tape(sessions=20)), "2024-01-15"), "2024-01-17")
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-29")
    assert r["ok"]
    assert r["excluded"]["2024-01-15"] == "closed" and r["excluded"]["2024-01-17"] == "early_close"


def test_a_full_session_on_an_excluded_date_fails():
    s = small_spec(excluded_sessions=["2024-01-16"])
    r = verify_calendar(_auction(tape(sessions=20)), s, "2024-01-02", "2024-01-29")
    assert not r["ok"] and r["excluded"]["2024-01-16"] == "UNEXPECTED_FULL_SESSION"


def test_a_closure_the_calendar_missed_fails():
    s = small_spec(excluded_sessions=[])
    raw = _drop_day(_auction(tape(sessions=20)), "2024-01-15")
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-29")
    assert not r["ok"] and "2024-01-15" in r["unexpected_closures"]


def test_a_partial_ordinary_session_is_a_reported_gap_not_a_calendar_error():
    s = small_spec(excluded_sessions=["2024-01-15"])
    raw = _drop_day(_auction(tape(sessions=20)), "2024-01-15")
    raw = raw.drop(raw.index[100])
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-29")
    assert r["ok"] and len(r["incomplete_sessions"]) == 1


def test_an_early_close_with_after_hours_prints_is_still_an_early_close():
    s = small_spec(excluded_sessions=["2024-01-17"])
    raw = _early_close_with_after_hours(_auction(tape(sessions=20)), "2024-01-17")
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-29")
    assert r["ok"] and r["excluded"]["2024-01-17"] == "early_close"


def test_an_early_close_the_calendar_missed_fails():
    s = small_spec(excluded_sessions=[])
    raw = _early_close_with_after_hours(_auction(tape(sessions=20)), "2024-01-17")
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-29")
    assert not r["ok"] and "2024-01-17" in r["unexpected_early_closes"]


def test_the_window_end_is_exclusive_like_the_provider_request():
    """Alpaca returns nothing on the end date; that is not a market closure."""
    s = small_spec(excluded_sessions=[])
    raw = _auction(tape(sessions=20))           # last session 2024-01-29
    r = verify_calendar(raw, s, "2024-01-02", "2024-01-30")
    assert r["ok"] and not r["unexpected_closures"]


def test_price_feasibility_uses_the_atl_position_limit():
    s = small_spec()
    cheap, dear = tape(sessions=3), tape(sessions=3) * 1
    dear[["open", "high", "low", "close"]] *= 8        # ~$800 a share
    r = price_feasibility({"A": cheap, "B": dear}, s, max_position_weight=0.25)
    assert r["limit"] == pytest.approx(0.25 * s["initial_cash"])
    assert r["symbols"]["A"]["feasible"] and not r["symbols"]["B"]["feasible"]
    assert r["infeasible"] == ["B"]


def test_symbol_data_gaps_report_every_anomaly_not_only_short_sessions():
    """Against a verified calendar, a symbol's truncated or missing session is a
    data gap. Counting only one category hid eight real provider gaps."""
    s = small_spec(excluded_sessions=["2024-01-15"])
    raw = _drop_day(_auction(tape(sessions=20)), "2024-01-15")
    raw = _drop_day(raw, "2024-01-10")                              # missing
    raw = _truncate(raw, "2024-01-11", last="10:40")               # stops mid-morning
    raw = raw.drop(raw.index[5])                                    # one missing bar
    gaps = symbol_data_gaps(raw, s, "2024-01-02", "2024-01-30")
    kinds = {g["session"]: g["kind"] for g in gaps}
    assert kinds["2024-01-10"] == "missing_session"
    assert kinds["2024-01-11"] == "truncated_session"
    assert kinds["2024-01-02"] == "incomplete_session"
    assert "2024-01-15" not in kinds                                # a real holiday is not a gap
