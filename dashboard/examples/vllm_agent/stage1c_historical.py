"""Frozen two-day ATL baseline/model validation; uses the existing engine and SDK.

Run as a module with SDK installed or on PYTHONPATH. No brokerage orders. The
only offline substitutions are a verified CSV market-data loader and omission
of unrelated asynchronous comparison jobs. Real-model mode has no mock option.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=str)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def load_spec():
    return json.loads((HERE / 'stage1c-spec.json').read_text())


def load_frozen_source(manifest_path, spec):
    import pandas as pd
    import numpy as np
    from dashboard.backend.infrastructure.market_data.provider import warmup_fetch_start
    path = Path(manifest_path).resolve()
    source = json.loads(path.read_text())
    frames = {}
    begin = pd.Timestamp(warmup_fetch_start(spec['start_date']), tz='UTC')
    end = pd.Timestamp(spec['end_date'], tz='UTC') + pd.Timedelta(days=1)
    for symbol in spec['symbols']:
        entry = source['files'][symbol]
        file = path.parent / entry['file']
        if file.resolve().parent != path.parent or digest(file.read_bytes()) != entry['sha256']:
            raise ValueError('frozen source hash/path mismatch for ' + symbol)
        frame = pd.read_csv(file, index_col='timestamp', float_precision='round_trip')
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.loc[(frame.index >= begin) & (frame.index < end)].copy()
        if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('invalid frozen source clock for ' + symbol)
        if not np.isfinite(frame[['open','high','low','close','volume']].to_numpy(float)).all():
            raise ValueError('non-finite source data')
        frame.attrs.update(source.get('frame_attrs', {}))
        frame.attrs.update(bar_open_stamped_minutes=5, alpaca_feed=source.get('feed'),
                           alpaca_sip_fallback=False, alpaca_end_clamped=False)
        frames[symbol] = frame
    return frames, {**source, 'source_manifest_sha256': digest(path.read_bytes())}


def configure_offline(output):
    os.environ['DATABASE_PATH'] = str(Path(output) / 'atl-research.db')
    for name in ('AGENT_RUNS_DATABASE_URL','CONTENT_DATABASE_URL','USERS_DATABASE_URL','DATABASE_URL',
                 'US_EQUITY_DATASET_PATH','US_EQUITY_DATA_ROOT','ALPACA_API_KEY','ALPACA_SECRET_KEY',
                 'ANTHROPIC_API_KEY','OPENAI_API_KEY','DEEPSEEK_API_KEY','COMMONSTACK_API_KEY','OPENROUTER_API_KEY'):
        os.environ[name] = ''
    os.environ['ATL_BAR_CACHE'] = '0'
    os.environ['ROBINHOOD_EXECUTE'] = 'false'


def evaluate(http, output, frames, source, spec, *, include_qwen=False):
    from contextlib import ExitStack
    from unittest.mock import patch
    import pandas as pd
    from agentictrading import ATLClient, AgentRunner, Decision
    from agentictrading.integrations.vllm import VLLMConfig, VLLMChatClient, VLLMTradingAgent, summarize_records
    from agentictrading.integrations._vllm_provenance import capture_source_provenance, write_artifact_checksums
    from dashboard.backend.domain.backtesting import market_data_store as mds
    from dashboard.backend.domain.backtesting import external_run_service as ebs
    from dashboard.backend.domain.runs import service as runs
    from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
    from dashboard.backend.domain.trading.execution import calculate_transaction_costs
    from dashboard.backend.database import db
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    costs = TransactionCostProfile(**spec['costs'])

    class FrozenLoader:
        source_timeframe = '5m'
        def configure_source_timeframe(self, value):
            if value != '5m': raise ValueError('unsupported frozen frequency')
        def fetch_bars(self, symbols, start, end):
            return {s: f.loc[(f.index >= pd.Timestamp(start, tz='UTC')) & (f.index < pd.Timestamp(end, tz='UTC'))].copy()
                    for s,f in frames.items() if s in symbols}

    # Prewarm via the existing loader-factory seam. Protocol creation subsequently
    # adopts this same native dataset; no alternate fill/valuation loop exists.
    dataset = mds.get_dataset(ebs.DJIA_30, spec['start_date'], spec['end_date'], loader_factory=FrozenLoader,
                             source_timeframe='5m', decision_timeframe='60m', market='US')
    if set(dataset.all_data) != set(spec['symbols']):
        raise ValueError('unexpected cached universe; use a fresh isolated process')
    frozen_fingerprint = digest({s: f.to_json(orient='split', date_format='iso') for s,f in dataset.source_data.items()})
    for symbol in spec['symbols']:
        if len({pd.Timestamp(t).date() for t in dataset.all_data[symbol].index}) != 2:
            raise ValueError('expected two historical trading dates')
    report = {'spec': spec, 'data_sha256': frozen_fingerprint, 'source': source,
              'native_market_data_quality':dataset.data_quality,
              'indicator_warmup':dataset.indicator_warmup,
              'dataset_kind': 'synthetic_fixture' if source.get('synthetic_fixture') else 'frozen_historical_provider_snapshot',
              'source_provenance': capture_source_provenance(HERE.parents[2]), 'baselines': {},
              'ATL_REAL_MODEL_BACKTEST': 'NOT_RUN', 'qwen': None,
              'comparison_jobs': 'omitted; matched HOLD and reference policies run through the same protocol'}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(value) + '\n')
    save(output / 'spec.json', spec)
    save(output / 'source-manifest.json', source)
    # Export a small, real-source slice for the Colab backtest, preserving origin hashes.
    pack = output / 'source-pack'; pack.mkdir(exist_ok=True)
    packed = {**source, 'files': {}, 'parent_source_manifest_sha256': source.get('source_manifest_sha256')}
    for symbol, frame in frames.items():
        path = pack / (symbol + '-5m.csv'); frame.to_csv(path, index_label='timestamp')
        packed['files'][symbol] = {'file': path.name, 'sha256': digest(path.read_bytes()),
                                   'parent_sha256': source['files'][symbol]['sha256'], 'rows': len(frame)}
    save(pack / 'source-manifest.json', packed)

    class LocalClient(ATLClient):
        def _request(self, method, path, *, body=None, params=None, timeout=None):
            response = http.request(method, path, json=body, params=params, headers={'X-API-Key': self._api_key})
            response.raise_for_status()
            value = response.json()
            if method == 'POST' and path == '/api/v1/runs':
                session = ebs.get_session(runs._runs[value['run_id']].backtest_id)
                if session.total_steps != dataset.total_steps or session.status != 'waiting_decision':
                    raise ValueError('frozen dataset was not adopted synchronously')
                session.manager.transaction_cost_profile = costs
            return value

    class Reference:
        def __init__(self, hold=False): self.hold, self.sequence = hold, 0
        def decide(self, obs):
            orders = []
            if not self.hold:
                held = {p['symbol']:p['quantity'] for p in obs.positions}
                for symbol in spec['symbols']:
                    if held.get(symbol): orders.append({'symbol':symbol,'side':'sell','quantity':held[symbol]})
                    elif self.sequence % 2 == 0:
                        ts = pd.Timestamp(obs.market['bars'][symbol]['timestamp']).tz_convert('America/New_York')
                        if ts.hour < 16: orders.append({'symbol':symbol,'side':'buy','quantity':1})
            self.sequence += 1
            return Decision(orders=orders, confidence=1, rationale='fixed Stage 1C reference')

    class Recorder:
        def __init__(self, agent): self.agent, self.steps, self.pending = agent, [], None
        def on_step(self, step):
            self.step = step
            if hasattr(self.agent, 'on_step'): self.agent.on_step(step)
        def decide(self, obs):
            client = getattr(self.agent, 'client', None)
            if hasattr(client, 'last_proposal'): client.last_proposal = None
            result = self.agent.decide(obs)
            self.pending = {'timestamp':self.step.timestamp, 'sequence':self.step.sequence,
                'model_generated_reply':getattr(client,'last_proposal',None),
                'parsed_decision':result.to_dict(), 'constraints':self.step.constraints,
                'portfolio_before':obs.portfolio}
            return result
        def on_execution_result(self, result):
            session = ebs.get_session(runs._runs[result.run_id].backtest_id)
            self.steps.append({**self.pending, 'execution':result.raw,
                               'engine_audit':session.decision_log[-1]})

    def run_one(label, repeat, agent):
        directory = output / 'runs' / label / str(repeat)
        directory.mkdir(parents=True, exist_ok=False)
        response = http.post('/api/v1/agents', json={'name':'stage1c-'+label},
                             headers={'X-Session-Id':uuid.uuid4().hex})
        response.raise_for_status(); registered = response.json()
        sdk = LocalClient('http://testserver', registered['api_key'])
        version = sdk.create_agent_version(registered['agent']['agent_id'], version='stage1c-v1',
                                          model_backbones=[spec['model'] if label=='qwen' else 'deterministic'])
        wrapper = Recorder(agent)
        result = AgentRunner(sdk, wrapper).run_backtest(version.id, environment_id=spec['environment_id'],
            start_date=spec['start_date'], end_date=spec['end_date'], symbols=spec['symbols'],
            initial_cash=spec['initial_cash'], poll_interval=.001)
        if result.status != 'completed' or len(wrapper.steps) != dataset.total_steps or result.metrics.get('timeout_holds'):
            raise ValueError('run incomplete or a step auto-held')
        session = ebs.get_session(runs._runs[result.run_id].backtest_id)
        trades, curve = result.trades, result.equity_curve
        fee = sum(t.get('total_fees',0) for t in trades)
        slip = sum(t.get('slippage_amount',0) for t in trades)
        expected_cash = spec['initial_cash'] + sum(t['net_cash_impact'] for t in trades)
        quantities = {}
        completed_quantity = 0
        completed_round_trips = 0
        for t in trades:
            timestamp = pd.Timestamp(t['timestamp'])
            fill = next((f for f in dataset.execution_fills if pd.Timestamp(f.filled_at) == timestamp), None)
            if fill is None:
                raise ValueError('trade is not attributed to a planned fill event')
            reference = float(dataset.source_data[t['symbol']].loc[fill.bar, fill.price_field])
            quote = calculate_transaction_costs(side=t['side'].lower(), reference_price=reference,
                                               shares=t['quantity'], transaction_cost_profile=costs)
            if any(abs(t[k] - quote[k]) > 1e-8 for k in ('price','total_fees','slippage_amount','net_cash_impact')):
                raise ValueError('persisted trade quote/cost differs from frozen execution policy')
            q = t['quantity'] if t['side'].lower()=='buy' else -t['quantity']
            if q < 0: completed_quantity += -q
            quantities[t['symbol']] = quantities.get(t['symbol'],0) + q
            if q < 0 and quantities[t['symbol']] == 0: completed_round_trips += 1
        quantities = {s:q for s,q in quantities.items() if q}
        if abs(session.manager.cash - expected_cash) > 1e-7 or quantities != {s:q for s,q in session.manager.positions.items() if q}:
            raise ValueError('cash or quantity accounting does not reconcile')
        final_time = pd.Timestamp(curve[-1]['timestamp'])
        expected_equity = expected_cash
        for symbol, quantity in quantities.items():
            frame = dataset.source_data[symbol]
            known = frame.loc[frame.index + pd.Timedelta(minutes=5) <= final_time]
            if known.empty:
                raise ValueError('no completed price is available for terminal holdings')
            expected_equity += quantity * float(known.iloc[-1]['close'])
        if abs(expected_equity - result.metrics['final_equity']) > 1e-7:
            raise ValueError('terminal equity does not reconcile at available closing marks')
        net = result.metrics['final_equity'] - spec['initial_cash']
        proposals = [s['parsed_decision'] for s in wrapper.steps]
        metrics = {'starting_equity':spec['initial_cash'], 'ending_equity':result.metrics['final_equity'],
            'ending_cash':session.manager.cash, 'gross_pnl_before_recorded_costs':net+fee+slip,
            'fees':fee, 'slippage':slip, 'net_pnl':net, 'max_drawdown':result.metrics['max_drawdown'],
            'turnover':sum(t['quantity']*t['price'] for t in trades)/spec['initial_cash'],
            'proposed_orders':sum(len(p['orders']) for p in proposals),
            'protocol_risk_approved_orders':sum(len(s['engine_audit']['actions_submitted']) for s in wrapper.steps),
            'execution_batches_accepted':sum(bool(s['execution']['accepted']) for s in wrapper.steps),
            'rejected_orders':sum(len(s['execution']['validation']['rejections']) for s in wrapper.steps),
            'executed_orders':len(trades), 'completed_sell_quantity':completed_quantity,
            'completed_round_trips':completed_round_trips,
            'genuine_hold_count':sum(not p['orders'] and (p.get('trace') or {}).get('status','valid')=='valid' for p in proposals),
            'error_fallback_hold_count':sum(not p['orders'] and (p.get('trace') or {}).get('status') in ('failed','rejected') for p in proposals),
            'buy_count':sum(any(o['side']=='buy' for o in p['orders']) for p in proposals),
            'sell_count':sum(any(o['side']=='sell' for o in p['orders']) for p in proposals),
            'decision_dates':len({s['timestamp'][:10] for s in wrapper.steps}), 'asset_count':len(spec['symbols']),
            'accounting_reconciles':True, 'ending_positions':quantities,
            'exposure_over_time':[{'timestamp':p['timestamp'],'exposure':p['positions_value']/p['equity'] if p['equity'] else 0} for p in curve]}
        stable = {'metrics':metrics,'trades':trades,'equity_curve':curve,
                  'decisions':[{'timestamp':s['timestamp'],'orders':s['parsed_decision']['orders'],
                                'rejections':s['execution']['validation']['rejections']} for s in wrapper.steps]}
        fingerprint = digest(stable)
        save(directory/'result.json',result.raw); save(directory/'steps.json',wrapper.steps)
        records = [r.to_dict() for r in getattr(agent,'records',[])]
        save(directory/'inference-records.json',records)
        manifest = {'label':label,'repeat':repeat,'run_id':result.run_id,'result_run_id':result.result_run_id,
                    'data_sha256':frozen_fingerprint,'spec':spec,'source_provenance':report['source_provenance'],
                    'fingerprint':fingerprint,'metrics':metrics}
        db.insert_run_manifest(result.result_run_id,manifest)
        save(directory/'manifest.json',manifest)
        # Retrieval must work after eviction of the live protocol/session objects.
        runs._runs.pop(result.run_id,None)
        for key,s in list(ebs._sessions.items()):
            if s.run_id == result.result_run_id:ebs._sessions.pop(key,None)
        if sdk.get_run_result(result.run_id).raw != result.raw:
            raise ValueError('persisted result differs')
        return manifest

    with ExitStack() as stack:
        stack.enter_context(patch.object(ebs.baseline_worker,'submit',lambda job:None))
        # A miss must use this verified frozen loader, never live market data.
        stack.enter_context(patch.object(ebs,'AlpacaDataLoader',FrozenLoader))
        try:
            for label in ('hold','reference'):
                results = [run_one(label,i,Reference(label=='hold')) for i in range(2)]
                if results[0]['fingerprint'] != results[1]['fingerprint']:
                    raise ValueError('non-reproducible deterministic baseline')
                report['baselines'][label] = {**results[0], 'reproducible':True,
                                               'run_ids':[r['run_id'] for r in results]}
                save(output/'evaluation.json',report)
            if include_qwen:
                import urllib.request
                from dashboard.examples.vllm_agent.vllm_inference_smoke import GpuSampler, scrape_metrics, mean_delta
                config = VLLMConfig.from_env()
                if (config.model_id,config.model_revision,config.max_output_tokens,config.temperature,config.top_p,config.seed) != (spec['model'],spec['model_revision'],256,0,1,0):
                    raise ValueError('model/inference configuration differs from frozen specification')
                class AuditedModelClient(VLLMChatClient):
                    # Research artifacts only: preserve actual model text so the
                    # proposed response can be distinguished from parser output.
                    # No request headers, keys, or executable model code is saved.
                    last_proposal = None
                    def complete(self, messages, *, timeout):
                        answer = super().complete(messages, timeout=timeout)
                        self.last_proposal = {'model':answer.model,'content':answer.content,
                                              'content_sha256':answer.content_sha256}
                        return answer
                model = AuditedModelClient(config)
                if not model.health(timeout=5) or config.model_id not in model.list_models(timeout=5):
                    raise ValueError('configured model is not served')
                server_root = config.base_url.removesuffix('/v1')
                with urllib.request.urlopen(server_root + '/version', timeout=5) as response:
                    server_version = json.loads(response.read()).get('version')
                if server_version != '0.31.0':
                    raise ValueError('server vLLM version differs from frozen specification')
                agent = VLLMTradingAgent(model)
                gpu = GpuSampler()
                gpu.start()
                before = scrape_metrics(config.base_url)
                try:
                    report['qwen'] = run_one('qwen',0,agent)
                finally:
                    gpu.stop()
                report['qwen']['inference_summary'] = summarize_records(agent.records)
                report['qwen']['config'] = config.public_dict()
                report['qwen']['infrastructure'] = {'vllm_version':server_version,'gpu':gpu.info,
                    'gpu_available_at_client':gpu.available,'gpu_peak_memory_used_mib':gpu.peak_mib or None,
                    'ttft_mean_seconds':mean_delta(before,scrape_metrics(config.base_url),'vllm:time_to_first_token_seconds'),
                    'server_model_revision':'requested pin; verify against separately captured serving log'}
                report['qwen']['acceptance_checks'] = {'all_inferences_valid':all(r.status=='valid' for r in agent.records),
                    'model_orders_reached_fills':report['qwen']['metrics']['executed_orders'] > 0,
                    'accounting_reconciles':report['qwen']['metrics']['accounting_reconciles']}
                report['ATL_REAL_MODEL_BACKTEST'] = 'PASS' if all(report['qwen']['acceptance_checks'].values()) else 'FAIL'
        except Exception as error:
            save(output/'failure.json',{'error_type':type(error).__name__,'completed_baselines':list(report['baselines'])})
            raise
        finally:
            report['source_provenance_at_finish'] = capture_source_provenance(HERE.parents[2])
            save(output/'evaluation.json',report)
            write_artifact_checksums(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--qwen',action='store_true',help='use the real pinned Qwen endpoint after deterministic baselines')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('output must be a fresh empty directory')
    args.out.mkdir(parents=True,exist_ok=True)
    configure_offline(args.out.resolve())
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    frames,source = load_frozen_source(args.source_manifest,load_spec())
    result = evaluate(TestClient(app),args.out,frames,source,load_spec(),include_qwen=args.qwen)
    print(json.dumps({'baselines':{k:v['metrics'] for k,v in result['baselines'].items()},
                      'ATL_REAL_MODEL_BACKTEST':result['ATL_REAL_MODEL_BACKTEST']},indent=2))
    return 0 if result['ATL_REAL_MODEL_BACKTEST'] != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
