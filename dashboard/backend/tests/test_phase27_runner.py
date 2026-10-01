"""Test first: SDK orchestration, strict model boundary, sealed replay evidence."""
import copy
from pathlib import Path
import json
import numpy as np
import pytest
from dashboard.backend.tests.test_protocol_api import client
from dashboard.backend.tests.test_phase27_dataset import tape
from dashboard.backend.domain.research.phase27 import build_dataset, load_spec, digest
from dashboard.scripts.phase27_experiment import run_window, validate_prediction, fit_gate


@pytest.fixture(autouse=True)
def local_sdk_source(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'packaging/agentictrading/src'))


@pytest.mark.parametrize('value', [
    {'trade_probability': float('nan'), 'direction': 'NONE'},
    {'trade_probability': 1.1, 'direction': 'LONG'},
    {'trade_probability': .8, 'direction': 'SHORT'},
    {'trade_probability': .8, 'direction': 'NONE'},
    {'trade_probability': .2, 'direction': 'NONE', 'future_return': 1},
])
def test_invalid_model_outputs_fail_closed(value):
    with pytest.raises(ValueError):
        validate_prediction(value)


def test_fit_uses_only_training_ids():
    raw, spec = tape(), load_spec()
    rows, outcomes, _ = build_dataset(raw, spec)
    ids = [r['record_id'] for r in rows[:20]]
    before = fit_gate(rows, outcomes, ids, spec).state()
    other = copy.deepcopy(outcomes)
    for row in other[20:]:
        row['trade_worthy'] = not row['trade_worthy']
    assert fit_gate(rows, other, ids, spec).state() == before


def test_sdk_rule_hold_and_repeat_with_persisted_evaluation(client, tmp_path):
    raw, spec = tape(), load_spec()
    inputs, outcomes, _ = build_dataset(raw, spec)
    selected = inputs[:10]
    received = []
    def rule(payload):
        assert set(payload) == {'timestamp', 'symbol', 'features'}
        assert set(payload['features']) == set(spec['feature_fields'])
        received.append(payload)
        p = float(payload['features']['return_1'] > 0)
        return {'trade_probability': p, 'direction': 'LONG' if p else 'NONE'}
    results = [run_window(client, tmp_path / str(i), raw, selected, spec,
                         '2023-01-03', '2023-01-23', rule, {'experiment_id': 'test', 'seed': 27}) for i in range(2)]
    assert results[0]['fingerprint'] == results[1]['fingerprint']
    assert len(received) == 20
    assert results[0]['trades']  # meaningful execution, not only all-HOLD
    lookup = {o['record_id']: o for o in outcomes}
    expected = spec['initial_cash'] + sum(lookup[p['record_id']]['cost_adjusted_forward_return'] *
        raw.loc[p['timestamp'], 'open'] for p in results[0]['predictions'] if p['trade_probability'] >= .5)
    assert results[0]['metrics']['final_equity'] == pytest.approx(expected)
    from dashboard.backend.database import db
    manifest = db.get_run_manifest(results[0]['result_run_id'])
    assert manifest['experiment_id'] == 'test'
    assert manifest['predictions_sha256'] == digest((tmp_path / '0' / 'predictions.jsonl').read_bytes())
    hold = run_window(client, tmp_path / 'hold', raw, selected, spec, '2023-01-03', '2023-01-23',
        lambda p: {'trade_probability': 0., 'direction': 'NONE'}, {'experiment_id': 'test'})
    assert hold['trades'] == []
    assert hold['metrics']['final_equity'] == spec['initial_cash']


def test_failed_model_records_failure_without_sending_order(client, tmp_path):
    raw, spec = tape(), load_spec()
    inputs, _, _ = build_dataset(raw, spec)
    with pytest.raises(ValueError):
        run_window(client, tmp_path, raw, inputs[:1], spec, '2023-01-03', '2023-01-23',
            lambda p: {'trade_probability': float('nan'), 'direction': 'NONE'}, {'experiment_id': 'failure'})
    assert json.loads((tmp_path / 'failure.json').read_text())['error_type'] == 'ValueError'


def test_window_retrieval_releases_engine_session(client, tmp_path):
    from dashboard.backend.domain.backtesting import external_run_service as ebs
    raw, spec = tape(), load_spec()
    inputs, _, _ = build_dataset(raw, spec)
    run_window(client, tmp_path, raw, inputs[:1], spec, '2023-01-03', '2023-01-23',
               lambda p: {'trade_probability': 0., 'direction': 'NONE'}, {})
    assert not ebs._sessions


def test_three_window_evaluator_includes_actual_trading_smoke(client, tmp_path):
    from dashboard.scripts.phase27_experiment import evaluate
    raw, spec = tape(), load_spec()
    spec['embargo_hours'] = 0
    spec['folds'] = [
        {'id': 'a', 'train': ['2023-01-03', '2023-01-06'], 'validation': ['2023-01-06', '2023-01-09'], 'test': ['2023-01-09', '2023-01-11']},
        {'id': 'b', 'train': ['2023-01-03', '2023-01-11'], 'validation': ['2023-01-11', '2023-01-12'], 'test': ['2023-01-12', '2023-01-14']},
        {'id': 'c', 'train': ['2023-01-03', '2023-01-17'], 'validation': ['2023-01-17', '2023-01-18'], 'test': ['2023-01-18', '2023-01-20']},
    ]
    result = evaluate(client, tmp_path, raw, {'sha256': 'synthetic'}, spec, 'test-code')
    assert len(result['windows']) == 9
    smoke = [w for w in result['windows'] if w['baseline'] == 'momentum_smoke']
    assert len(smoke) == 3 and all(w['financial']['fills'] > 0 for w in smoke)
    assert all(w['reproducible'] and w['cash_matches_outcomes'] for w in result['windows'])
    assert all(w['classification']['n'] == 10 for w in result['windows'])


def test_offline_launcher_cannot_inherit_remote_databases(monkeypatch, tmp_path):
    from dashboard.scripts.phase27_experiment import configure_offline
    names = ['AGENT_RUNS_DATABASE_URL', 'CONTENT_DATABASE_URL', 'USERS_DATABASE_URL', 'DATABASE_URL']
    for name in names:
        monkeypatch.setenv(name, 'postgresql://example.invalid/never-connect')
    monkeypatch.setenv('DATABASE_PATH', '/unused/database.db')
    monkeypatch.setenv('ATL_BAR_CACHE', '1')
    configure_offline(tmp_path)
    import os
    assert all(os.environ[name] == '' for name in names)
    assert os.environ['DATABASE_PATH'] == str(tmp_path / 'atl-research.db')
    assert os.environ['ATL_BAR_CACHE'] == '0'
