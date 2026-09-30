"""Incremental event-time marks for the existing PortfolioManager ledger.

Source closes become available at the END of their bar. Consumers must drain
completed bars before mutating holdings, then mark the fill's contemporaneous
price. This is shared accounting, not a separate execution/backtesting engine.
"""
from datetime import timedelta


class EventTimeValuator:
    def __init__(self, manager, timestamps, market_data_at, *, source_minutes=0,
                 transform=None):
        self.manager = manager
        self.timestamps = list(timestamps)
        self.market_data_at = market_data_at
        self.span = timedelta(minutes=source_minutes)
        self.transform = transform
        self.cursor = 0
        self.prices = {}
        self.started = False

    @property
    def market_data(self):
        # Values are already-known marks. No unfinished OHLCV enters accounting.
        return {symbol: {"close": price} for symbol, price in self.prices.items()}

    def mark(self, timestamp, rows, price_field):
        self.prices.update({s: row[price_field] for s, row in rows.items()
                            if price_field in row})
        history = self.manager.equity_history
        if history and timestamp < history[-1]["timestamp"]:
            raise ValueError("Cannot value a portfolio before its last event")
        same_event = bool(history and history[-1]["timestamp"] == timestamp)
        self.manager.update_equity(self.market_data, timestamp=timestamp)
        if self.transform:
            history[-1] = self.transform(history[-1])
        if same_event:
            # A close and following open/fill can share a boundary. Publish the
            # final state at that instant without modifying any earlier record.
            history[-2:] = [history[-1]]

    def through(self, timestamp=None):
        if not self.started and self.timestamps:
            first = self.timestamps[0]
            if timestamp is not None and first > timestamp:
                return
            self.started = True
            if self.span:
                self.mark(first, self.market_data_at(first), "open")
        while self.cursor < len(self.timestamps):
            bar = self.timestamps[self.cursor]
            available_at = bar + self.span
            if timestamp is not None and available_at > timestamp:
                break
            self.mark(available_at, self.market_data_at(bar), "close")
            self.cursor += 1

    def after_fill(self, fill):
        self.mark(fill.filled_at, self.market_data_at(fill.bar), fill.price_field)
