"""Frozen Phase 27 evaluation through the public API and unmodified SDK loop.

Run as a module. The CLI sets an isolated SQLite path before loading the app.
No data acquisition, model training framework, or execution simulator lives here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]


def validate_prediction(value):
    import math
    if not isinstance(value, dict) or set(value) != {'trade_probability', 'direction'}:
        raise ValueError('invalid prediction schema')
    p = value['trade_probability']
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid trade probability')
    if value['direction'] != ('LONG' if p >= .5 else 'NONE'):
        raise ValueError('invalid long-only direction')
    return {'trade_probability': float(p), 'direction': value['direction']}


def fit_gate(inputs, outcomes, train_ids, spec):
    from dashboard.backend.domain.research.phase27 import LogisticGate, model_payload
    rows = {r['record_id']: r for r in inputs}
    labels = {r['record_id']: r['trade_worthy'] for r in outcomes if r['record_id'] in set(train_ids)}
    x = [[model_payload(rows[rid], spec)['features'][f] for f in spec['feature_fields']] for rid in train_ids]
    cfg = spec['classifier']
    return LogisticGate(cfg['iterations'], cfg['learning_rate'], cfg['l2']).fit(x, [labels[rid] for rid in train_ids])


def run_window(http, directory, raw, inputs, spec, start, end, predict, metadata):
    """Inference/execution only: hidden outcomes cannot be passed to this function.

    `predict` is the replacement point for a future structured-output open model.
    It sees the six allowlisted numbers and timestamp/symbol, never raw tape,
    outcomes, result metrics, or an execution response.
    """
    import pandas as pd
    from agentictrading import ATLClient, AgentRunner
    from dashboard.backend.domain.research.phase27 import canonical, digest, model_payload
    from dashboard.backend.domain.research.replay_dataset import make_replay
    from dashboard.backend.domain.backtesting.replay import register_replay
    from dashboard.backend.domain.backtesting import external_run_service as ebs
    from dashboard.backend.domain.runs import service as runs
    from dashboard.backend.database import db

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (directory / name).write_text(canonical(value) + '\n')

    class LocalClient(ATLClient):
        # Only HTTP transport changes. SDK schemas and lifecycle run unchanged.
        def _request(self, method, path, *, body=None, params=None, timeout=None):
            response = http.request(method, path, json=body, params=params, headers={'X-API-Key': self._api_key})
            response.raise_for_status()
            return response.json()

    class Policy:
        def __init__(self):
            self.rows = {pd.Timestamp(r['timestamp']): r for r in inputs}
            self.predictions, self.decisions, self.executions = [], [], []
            self.expected_fills = 0
            self.previous_timestamp = None

        def decide(self, obs):
            bar = obs.market['bars']['AAPL']
            t = pd.Timestamp(bar['timestamp'])
            if self.previous_timestamp is not None and t <= self.previous_timestamp:
                raise ValueError('non-increasing decision clock')
            self.previous_timestamp = t
            orders = []
            held = sum(p['quantity'] for p in obs.positions if p['symbol'] == 'AAPL')
            if held:
                orders.append({'symbol': 'AAPL', 'side': 'sell', 'quantity': held})
            if t in self.rows:
                row = self.rows[t]
                if abs(bar['close'] - row['features']['close']) > 1e-9:
                    raise ValueError('ATL observation and frozen model input disagree')
                prediction = validate_prediction(predict(model_payload(row, spec)))
                self.predictions.append({'record_id': row['record_id'], 'timestamp': row['timestamp'], **prediction})
                if prediction['direction'] == 'LONG':
                    orders.append({'symbol': 'AAPL', 'side': 'buy', 'quantity': spec['quantity']})
            self.expected_fills = len(orders)
            self.decisions.append({'timestamp': t.isoformat(), 'orders': orders})
            return {'orders': orders, 'confidence': 1.}

        def on_execution_result(self, result):
            if result.rejections or len(result.fills) != self.expected_fills:
                raise ValueError('rejected or missing execution')
            self.executions.append(result.raw)

    policy = Policy()
    try:
        replay_id = 'phase27-' + uuid.uuid4().hex
        register_replay(replay_id, make_replay(raw, inputs, spec, start, end))
        response = http.post('/api/v1/agents', json={'name': 'phase27-offline'}, headers={'X-Session-Id': uuid.uuid4().hex})
        response.raise_for_status()
        registered = response.json()
        key, agent_id = registered['api_key'], registered['agent']['agent_id']
        response = http.post(f'/api/v1/agents/{agent_id}/versions', headers={'X-API-Key': key},
                             json={'version': 'phase27-v0', 'model_backbones': [metadata.get('baseline', 'deterministic-gate')]})
        response.raise_for_status()
        sdk = LocalClient('http://testserver', key)
        result = AgentRunner(sdk, policy).run_backtest(response.json()['agent_version']['agent_version_id'],
            environment_id='us-equity-hourly-v1', start_date=start, end_date=end,
            symbols=spec['universe'], initial_cash=spec['initial_cash'], config={'replay_id': replay_id}, poll_interval=.001)
        if len(policy.predictions) != len(inputs) or len(policy.executions) != len(policy.decisions):
            raise ValueError('missing prediction or auto-held decision')
        if result.metrics.get('timeout_holds', 0):
            raise ValueError('timed-out decision')
        (directory / 'predictions.jsonl').write_text(''.join(canonical(p) + '\n' for p in policy.predictions))
        save('decisions.json', policy.decisions)
        save('executions.json', policy.executions)
        save('result.json', result.raw)
        # Prove persisted retrieval without the live execution/session objects.
        runs._runs.pop(result.run_id, None)
        for identifier, session in list(ebs._sessions.items()):
            if session.run_id == result.result_run_id:
                ebs._sessions.pop(identifier, None)
        persisted = sdk.get_run_result(result.run_id)
        if persisted.raw != result.raw:
            raise ValueError('persisted result differs from completed run')
        trade_fields = ('timestamp', 'symbol', 'side', 'quantity', 'price', 'total_fees', 'slippage_amount', 'reference_price')
        stable = {'predictions': policy.predictions, 'decisions': policy.decisions,
                  'trades': [{k: t.get(k) for k in trade_fields} for t in result.trades],
                  'equity_curve': result.equity_curve, 'metrics': result.metrics}
        fingerprint = digest(stable)
        manifest = {**metadata, 'spec': spec, 'run_id': result.run_id, 'result_run_id': result.result_run_id,
                    'fingerprint': fingerprint, 'predictions_sha256': digest((directory / 'predictions.jsonl').read_bytes()),
                    'files': {name: digest((directory / name).read_bytes()) for name in
                              ('result.json', 'decisions.json', 'executions.json', 'predictions.jsonl')}}
        db.insert_run_manifest(result.result_run_id, manifest)
        save('manifest.json', manifest)
        return {**stable, 'fingerprint': fingerprint, 'run_id': result.run_id, 'result_run_id': result.result_run_id}
    except Exception as error:
        # No transport exception string: it could include an authenticated URL.
        save('failure.json', {'error_type': type(error).__name__, 'metadata': metadata,
                              'prediction_count': len(policy.predictions), 'decision_count': len(policy.decisions)})
        save('partial-decisions.json', policy.decisions)
        raise


def evaluate(http, output, raw, source, spec, code_hash):
    import pandas as pd
    from dashboard.backend.domain.research.phase27 import (build_dataset, build_splits, write_dataset,
        score_predictions, canonical, digest)
    from dashboard.backend.database import db
    output = Path(output)
    inputs, outcomes, quality = build_dataset(raw, spec)
    manifest = write_dataset(output / 'dataset', inputs, outcomes, quality, spec, source, code_hash)
    folds = build_splits(inputs, outcomes, spec)
    indexed = {r['record_id']: r for r in inputs}
    hidden = {r['record_id']: r for r in outcomes}
    experiment_id = digest([manifest['dataset_id'], spec, code_hash])
    report = {'experiment_id': experiment_id, 'dataset_id': manifest['dataset_id'], 'code_hash': code_hash,
              'source_sha256': source['sha256'], 'spec_hash': digest(spec), 'windows': [], 'failures': []}
    for fold in folds:
        if any(not fold[p + '_ids'] for p in ('train', 'validation', 'test')):
            raise ValueError('empty split')
        model = fit_gate(inputs, outcomes, fold['train_ids'], spec)
        selected = [indexed[rid] for rid in fold['test_ids']]
        def quant(payload):
            p = float(model.predict([[payload['features'][f] for f in spec['feature_fields']]])[0])
            return {'trade_probability': p, 'direction': 'LONG' if p >= .5 else 'NONE'}
        def momentum(payload):
            p = float(payload['features']['return_1'] > 0)
            return {'trade_probability': p, 'direction': 'LONG' if p else 'NONE'}
        for name, gate in (('hold', lambda payload: {'trade_probability': 0., 'direction': 'NONE'}),
                           ('logistic', quant), ('momentum_smoke', momentum)):
            root = output / 'runs' / fold['id'] / name
            runs = []
            for repeat in range(2):
                metadata = {'experiment_id': experiment_id, 'dataset_id': manifest['dataset_id'], 'code_hash': code_hash,
                            'fold': fold, 'split_sha256': manifest['files']['splits.json'], 'baseline': name,
                            'model_state': model.state() if name == 'logistic' else ({'constant_probability': 0} if name == 'hold' else {'rule': 'return_1 > 0'}),
                            'seed': spec['seed'], 'repeat': repeat}
                result = run_window(http, root / str(repeat), raw, selected, spec, fold['bounds']['test'][0],
                                    (pd.Timestamp(fold['bounds']['test'][1]) - pd.Timedelta(days=1)).date().isoformat(), gate, metadata)
                runs.append(result)
            if runs[0]['fingerprint'] != runs[1]['fingerprint']:
                raise ValueError('non-reproducible replay')
            # Test outcomes are joined only AFTER both prediction files are sealed.
            predictions = runs[0]['predictions']
            labels = [hidden[p['record_id']] for p in predictions]
            metrics = score_predictions([r['trade_worthy'] for r in labels], [p['trade_probability'] for p in predictions],
                [r['cost_adjusted_forward_return'] for r in labels], [r['direction'] for r in labels])
            expected_cash = spec['initial_cash'] + sum(o['cost_adjusted_forward_return'] * raw.loc[p['timestamp'], 'open'] * spec['quantity']
                for p,o in zip(predictions, labels) if p['trade_probability'] >= .5)
            actual = runs[0]['metrics']['final_equity']
            if abs(actual - expected_cash) > 1e-7:
                raise ValueError('ATL executed P&L disagrees with hidden outcomes')
            financial = {'initial_cash': spec['initial_cash'], 'final_equity': actual,
                         'cumulative_return': actual / spec['initial_cash'] - 1,
                         'fills': len(runs[0]['trades']),
                         'fees': sum(t['total_fees'] for t in runs[0]['trades']),
                         'slippage': sum(t['slippage_amount'] for t in runs[0]['trades']),
                         'turnover': sum(t['quantity'] * t['price'] for t in runs[0]['trades']) / spec['initial_cash'],
                         'atl_metrics': runs[0]['metrics']}
            entry = {'fold': fold['id'], 'baseline': name, 'classification': metrics, 'financial': financial,
                     'reproducible': True, 'cash_matches_outcomes': True, 'fingerprint': runs[0]['fingerprint'],
                     'run_ids': [r['run_id'] for r in runs], 'result_run_ids': [r['result_run_id'] for r in runs],
                     'train_rows': len(fold['train_ids']), 'validation_rows': len(fold['validation_ids']),
                     'test_sessions': len({r['timestamp'][:10] for r in selected})}
            for r in runs:
                stored = db.get_run_manifest(r['result_run_id'])
                stored['evaluation'] = entry
                db.insert_run_manifest(r['result_run_id'], stored)
            report['windows'].append(entry)
            (root / 'evaluation.json').write_text(canonical(entry) + '\n')
            (output / 'experiment.json').write_text(canonical(report) + '\n')
            print(f"Completed {fold['id']} {name}: {metrics['n']} predictions, {financial['fills']} fills; repeat identical", flush=True)
    return report


def configure_offline(output):
    """Select local storage before importing any ATL store or app module."""
    os.environ['DATABASE_PATH'] = str(Path(output) / 'atl-research.db')
    # Empty values also prevent dotenv from repopulating remote configuration.
    for name in ('CONTENT_DATABASE_URL', 'USERS_DATABASE_URL', 'AGENT_RUNS_DATABASE_URL', 'DATABASE_URL'):
        os.environ[name] = ''
    os.environ['ATL_BAR_CACHE'] = '0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_offline(output)
    from dashboard.backend.domain.research.phase27 import read_source, load_spec, digest, canonical
    from dashboard.backend.app import app
    from fastapi.testclient import TestClient
    paths = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard'], cwd=REPO, text=True).splitlines()
    files = {p: digest((REPO / p).read_bytes()) for p in sorted(set(paths)) if (REPO / p).is_file()}
    provenance = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'files': files, 'source_tree_hash': digest(files), 'python': sys.version}
    (output / 'code-manifest.json').write_text(canonical(provenance) + '\n')
    raw, source = read_source(args.source_manifest)
    try:
        evaluate(TestClient(app), output, raw, source, load_spec(), provenance['source_tree_hash'])
    except Exception as error:
        (output / 'failure.json').write_text(canonical({'error_type': type(error).__name__}) + '\n')
        raise


if __name__ == '__main__':
    main()
