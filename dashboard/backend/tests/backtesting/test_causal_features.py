"""Prefix and future-perturbation invariants for the shared feature layer."""
import numpy as np
import pandas as pd
import pytest
from dashboard.backend.domain.backtesting import features


def frame(n=100):
    return pd.DataFrame({"close": 100 + np.sin(np.arange(n)) + np.arange(n) / 10},
                        index=pd.date_range("2026-01-01", periods=n, freq="h"))


@pytest.mark.parametrize("n", [1, 13, 14, 19, 20, 25, 26, 33, 34, 49, 50, 70])
def test_features_equal_legal_prefix(n):
    raw = frame()
    full = features.TechnicalIndicators.calculate_indicators(raw)
    prefix = features.TechnicalIndicators.calculate_indicators(raw.iloc[:n])
    pd.testing.assert_frame_equal(full.iloc[:n], prefix)


@pytest.mark.parametrize("indicator", ["rsi", "macd", "bbands", "sma"])
@pytest.mark.parametrize("failure", ["none", "raise"])
def test_indicator_fallback_is_causal(monkeypatch, indicator, failure):
    def unavailable(*args, **kwargs):
        if failure == "raise":
            raise RuntimeError("forced indicator failure")
        return None
    monkeypatch.setattr(features.ta, indicator, unavailable)
    raw = frame()
    changed = raw.copy()
    changed.iloc[60:, 0] += 1000
    before = features.TechnicalIndicators.calculate_indicators(raw)
    after = features.TechnicalIndicators.calculate_indicators(changed)
    pd.testing.assert_frame_equal(before.iloc[:60], after.iloc[:60])
