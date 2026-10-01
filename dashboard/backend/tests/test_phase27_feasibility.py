"""Phase 27 stop-gate probes. Failing assertions deliberately remain unmasked.

Only market acquisition and background comparison jobs are replaced. HTTP,
protocol validation, indicators, execution, accounting and SQLite are real.
"""
import pandas as pd
import pytest

from dashboard.backend.tests.test_protocol_api import (
    client, _new_agent, _new_version, _create_run, _wait_for_step,
)
from dashboard.backend.tests.test_external_minute_source import _MinuteLoader, _minute_bars
from dashboard.backend.domain.backtesting import external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.domain.runs import service as runs


@pytest.fixture(autouse=True)
def frozen_minute_source(client, monkeypatch):
    mds._reset_for_tests()
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _MinuteLoader)
    yield
    mds._reset_for_tests()


def start(client):
    agent, key, _ = _new_agent(client)
    version = _new_version(client, agent, key)
    run = _create_run(client, key, version, end="2026-04-15")
    step = _wait_for_step(client, run, key)
    return run, key, step


def submit(client, run, key, step, orders):
    response = client.post(
        f"/api/v1/runs/{run}/steps/{step['step_id']}/decision",
        headers={"X-API-Key": key},
        json={"idempotency_key": f"phase27-{step['sequence']}",
              "orders": orders, "confidence": 1.0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["accepted"] is True
    return response.json()


def test_hold_lifecycle_is_reproducible_and_persisted(client):
    histories = []
    identifiers = []
    for _ in range(2):
        run, key, step = start(client)
        identifiers.append(run)
        history = []
        for sequence in range(20):
            if step["status"] == "completed":
                break
            assert step["sequence"] == sequence
            if history:
                assert pd.Timestamp(step["timestamp"]) > pd.Timestamp(history[-1][0])
            for bar in step["observation"]["market"]["bars"].values():
                assert pd.Timestamp(bar["timestamp"]) <= pd.Timestamp(step["timestamp"])
            result = submit(client, run, key, step, [])
            assert result["fills"] == []
            assert result["portfolio_after"] == {"cash": 1000.0, "equity": 1000.0, "positions": []}
            history.append((step["timestamp"], step["observation"], result["portfolio_after"]))
            step = _wait_for_step(client, run, key)
        assert step["status"] == "completed"
        # Since #529, include the 16:00 ET closing decision, filled at the
        # final regular-hours source bar's close rather than an after-hours open.
        assert len(history) == 7
        headers = {"X-API-Key": key}
        before = client.get(f"/api/v1/runs/{run}/result", headers=headers)
        assert before.status_code == 200, before.text
        # Evict process-local objects: the next read must use persisted records.
        runs._runs.clear()
        ebs._sessions.clear()
        after = client.get(f"/api/v1/runs/{run}/result", headers=headers)
        assert after.status_code == 200, after.text
        assert after.json() == before.json()
        histories.append(history)
    assert identifiers[0] != identifiers[1]
    assert histories[0] == histories[1]


def test_first_http_observation_is_invariant_to_future_prices(client, monkeypatch):
    _, _, first = start(client)

    class FutureChanged(_MinuteLoader):
        def fetch_bars(self, symbols, start, end):
            frames = _minute_bars(symbols, start, end)
            for frame in frames.values():
                future = frame.index >= pd.Timestamp("2026-04-15 15:30:00+00:00")
                frame.loc[future, ["open", "high", "low", "close"]] += 50
            return frames

    mds._reset_for_tests()
    monkeypatch.setattr(ebs, "AlpacaDataLoader", FutureChanged)
    _, _, changed = start(client)
    assert first["timestamp"] == changed["timestamp"] == "2026-04-15T14:30:00+00:00"
    assert first["observation"]["market"]["bars"] == changed["observation"]["market"]["bars"]
    assert first["observation"] == changed["observation"], (
        "Changing prices strictly after t changed the model's first HTTP observation",
        first["observation"]["market"]["features"], changed["observation"]["market"]["features"],
    )


def buy_first(client):
    run, key, step = start(client)
    result = submit(client, run, key, step, [
        {"symbol": "AAPL", "side": "buy", "quantity_type": "shares", "quantity": 1, "order_type": "market"}
    ])
    assert len(result["fills"]) == 1
    assert result["fills"][0]["fill_price"] == pytest.approx(100.37)
    session = ebs.get_session(runs._runs[run].backtest_id)
    return session


def test_first_buy_cannot_rewrite_pre_fill_equity(client):
    session = buy_first(client)
    fill_time = pd.Timestamp(session.manager.trades[0]["timestamp"])
    earlier = [point for point in session.manager.equity_history if pd.Timestamp(point["timestamp"]) < fill_time]
    assert earlier
    assert all(point["cash"] == 1000 and point["positions_value"] == 0 and point["equity"] == 1000 for point in earlier), earlier


def test_open_fill_cannot_be_marked_using_same_bar_future_close(client):
    session = buy_first(client)
    fill_time = pd.Timestamp(session.manager.trades[0]["timestamp"])
    at_fill = [point for point in session.manager.equity_history if pd.Timestamp(point["timestamp"]) == fill_time]
    assert len(at_fill) == 1
    # Zero-cost existing path: one share bought at the open has no instant P&L.
    assert at_fill[0]["equity"] == pytest.approx(1000.0), at_fill[0]


def test_later_sale_preserves_earlier_holdings_and_realizes_only_at_fill(client):
    import copy
    run, key, step = start(client)
    bought = submit(client, run, key, step, [
        {"symbol": "AAPL", "side": "buy", "quantity": 1}
    ])
    session = ebs.get_session(runs._runs[run].backtest_id)
    # The protocol response must mark the same contemporaneous price as the ledger.
    assert bought["portfolio_after"]["positions"][0]["current_price"] == 100.37
    before = copy.deepcopy(session.manager.equity_history)
    step = _wait_for_step(client, run, key)
    sold = submit(client, run, key, step, [
        {"symbol": "AAPL", "side": "sell", "quantity": 1}
    ])
    assert sold["fills"][0]["fill_price"] == 100.49
    assert sold["portfolio_after"]["positions"] == []
    assert sold["portfolio_after"]["cash"] == pytest.approx(1000.12)
    assert session.manager.equity_history[:len(before)] == before
    between = [p for p in session.manager.equity_history
               if pd.Timestamp("2026-04-15 14:30Z") < p["timestamp"] < pd.Timestamp("2026-04-15 15:30Z")]
    assert between
    assert all(p["cash"] == pytest.approx(899.63) and p["positions_value"] > 0 for p in between)
    close = next(p for p in between if p["timestamp"] == pd.Timestamp("2026-04-15 14:35Z"))
    assert close["positions_value"] == pytest.approx(100.12)


def test_fill_event_ignores_unfinished_bar_ohlcv(client, monkeypatch):
    first = buy_first(client)
    history = [dict(p) for p in first.manager.equity_history]
    trades = [dict(t) for t in first.manager.trades]

    class UnfinishedChanged(_MinuteLoader):
        def fetch_bars(self, symbols, start, end):
            frames = _minute_bars(symbols, start, end)
            for frame in frames.values():
                t = pd.Timestamp("2026-04-15 14:30Z")
                frame.loc[t, "close"] += 50
                frame.loc[t, "high"] += 100
                frame.loc[t, "low"] -= 50
                frame.loc[t, "volume"] *= 100
            return frames
    mds._reset_for_tests()
    monkeypatch.setattr(ebs, "AlpacaDataLoader", UnfinishedChanged)
    second = buy_first(client)
    assert second.manager.trades == trades
    assert second.manager.equity_history == history


def test_terminal_fill_response_does_not_borrow_finalization_prices(client):
    run, key, step = start(client)
    session = ebs.get_session(runs._runs[run].backtest_id)
    # A final executable decision can precede the final source close (e.g.
    # incomplete later decision buckets). Final metrics still drain the tape.
    session.timestamps = session.timestamps[:1]
    session.execution_fills = session.execution_fills[:1]
    session.total_steps = 1
    result = submit(client, run, key, step, [
        {"symbol": "AAPL", "side": "buy", "quantity": 1}
    ])
    assert session.status == "completed"
    assert result["portfolio_after"]["positions"][0]["current_price"] == 100.37
    assert session.manager.equity_history[-1]["positions_value"] == pytest.approx(100.77)
