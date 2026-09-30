"""The dashboard engine must obey the same event clock as external runs."""
import pandas as pd
import pytest
from dashboard.backend.tests.backtesting.test_engine_minute_source import _MinuteLoader, _make_minute_bars, _DB
from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager


def test_dashboard_first_fill_has_no_retroactive_or_future_close_mark(monkeypatch):
    loader = _MinuteLoader(_make_minute_bars())
    def factory(*args, source_timeframe=None, **kwargs):
        loader.configure_source_timeframe(source_timeframe)
        return loader
    db = _DB()
    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    monkeypatch.setattr(engine_mod, "db", db)
    def buy_once(self, state):
        return {"actions": [] if self.positions else [{"symbol": "AAPL", "action": "buy", "shares": 1}]}
    monkeypatch.setattr(PortfolioManager, "make_trading_decision", buy_once)
    engine = engine_mod.HourlyBacktester("2026-03-02", "2026-03-13", use_llm=False, symbols=["AAPL"])
    engine.load_data()
    engine.calculate_indicators()
    _, curve = engine.run_agent_backtest()
    fill = pd.Timestamp(db.trades[0][1][0]["timestamp"])
    before = [p for p in curve if pd.Timestamp(p["timestamp"]) < fill]
    assert before
    assert all(p["positions_value"] == 0 and p["cash"] == engine.initial_capital for p in before)
    at_fill = [p for p in curve if pd.Timestamp(p["timestamp"]) == fill]
    assert len(at_fill) == 1
    assert at_fill[0]["equity"] == pytest.approx(engine.initial_capital)
