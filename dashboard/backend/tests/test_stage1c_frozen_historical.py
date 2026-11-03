"""Frozen-source runner contracts; fixtures are synthetic and explicitly labeled."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dashboard.backend.tests.test_protocol_api import client
from dashboard.examples.vllm_agent.stage1c_historical import load_spec, load_frozen_source, evaluate


@pytest.fixture(autouse=True)
def local_sdk(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'packaging/agentictrading/src'))


def make_source(directory):
    times=[t for d in pd.bdate_range('2026-02-09','2026-03-03') for t in pd.date_range(str(d.date())+' 09:30',periods=78,freq='5min',tz='America/New_York')]
    meta={'feed':'sip','adjustment':'split','synthetic_fixture':True,'frame_attrs':{'bar_open_stamped_minutes':5},'files':{}}
    for i,symbol in enumerate(['AAPL','KO']):
        p=100+10*i+np.sin(np.arange(len(times))/15)
        frame=pd.DataFrame({'timestamp':pd.DatetimeIndex(times).tz_convert('UTC'),'open':p,'high':p+.4,'low':p-.4,'close':p+.05,'volume':1000})
        path=directory/(symbol+'-5m.csv');frame.to_csv(path,index=False)
        meta['files'][symbol]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path=directory/'source-manifest.json';path.write_text(json.dumps(meta));return path


def test_frozen_source_hashes_fail_closed(tmp_path):
    manifest=make_source(tmp_path)
    frames,meta=load_frozen_source(manifest,load_spec())
    assert set(frames)=={'AAPL','KO'} and meta['synthetic_fixture'] is True
    with (tmp_path/'AAPL-5m.csv').open('a') as f:f.write('corruption\n')
    with pytest.raises(ValueError,match='hash'):load_frozen_source(manifest,load_spec())


def test_two_day_two_asset_baselines_repeat_and_persist(client,tmp_path):
    source=tmp_path/'source';source.mkdir();manifest=make_source(source)
    frames,meta=load_frozen_source(manifest,load_spec())
    result=evaluate(client,tmp_path/'evaluation',frames,meta,load_spec(),include_qwen=False)
    assert result['dataset_kind']=='synthetic_fixture'
    assert result['ATL_REAL_MODEL_BACKTEST']=='NOT_RUN'
    assert set(result['baselines'])=={'hold','reference'}
    for baseline in result['baselines'].values():
        assert baseline['reproducible'] is True
        assert baseline['metrics']['accounting_reconciles'] is True
    reference=result['baselines']['reference']['metrics']
    assert reference['executed_orders']>0 and reference['fees']>0 and reference['slippage']>0
    assert reference['decision_dates']==2 and reference['asset_count']==2
    assert (tmp_path/'evaluation/runs/reference/0/result.json').is_file()
