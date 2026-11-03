"""Availability checks cover mature indicators as well as cold HTTP windows."""
import numpy as np
import pandas as pd
import pytest
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting import features


@pytest.mark.parametrize('length',[1,14,15,19,20,26,33,34,49,50,70,100])
def test_legal_prefix_is_identical_to_full_window_at_every_indicator_gate(length):
    t=np.arange(110,dtype=float)
    frame=pd.DataFrame({'close':100+.02*t+np.sin(t/3),'volume':1000+t},index=pd.date_range('2026-01-01',periods=110,freq='h',tz='UTC'))
    columns=['rsi_14','macd','macd_signal','bb_upper','bb_lower','sma20','sma50']
    full=TechnicalIndicators.calculate_indicators(frame)
    prefix=TechnicalIndicators.calculate_indicators(frame.iloc[:length])
    pd.testing.assert_frame_equal(full.iloc[:length][columns],prefix[columns],check_exact=False,rtol=1e-12,atol=1e-12)


def test_future_price_mutation_cannot_change_mature_features_or_fallbacks(monkeypatch):
    t=np.arange(100,dtype=float)
    frame=pd.DataFrame({'close':100+t/100+np.sin(t),'volume':1000+t})
    changed=frame.copy();changed.loc[70:,'close']+=1000;changed.loc[70:,'volume']*=100
    a=TechnicalIndicators.calculate_indicators(frame);b=TechnicalIndicators.calculate_indicators(changed)
    pd.testing.assert_frame_equal(a.iloc[:70],b.iloc[:70])
    def fail(*a,**k):raise RuntimeError('forced indicator outage')
    monkeypatch.setattr(features,'_library_indicators',fail)
    pd.testing.assert_frame_equal(TechnicalIndicators.calculate_indicators(frame).iloc[:70],
                                  TechnicalIndicators.calculate_indicators(changed).iloc[:70])
