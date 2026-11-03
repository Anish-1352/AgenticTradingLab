"""Synthetic integration probes: parser/risk/engine stages, never GPU evidence."""
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from dashboard.backend.tests.test_stage1c_causal_integration import client, frozen_minute_source, start, submit
from dashboard.backend.tests.test_protocol_api import _wait_for_step
from dashboard.backend.domain.backtesting import external_run_service as ebs
from dashboard.backend.domain.runs import service as runs
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
from dashboard.backend.domain.trading.execution import calculate_transaction_costs


@pytest.fixture(autouse=True)
def sdk_source(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'packaging/agentictrading/src'))


def cost_profile():
    return TransactionCostProfile(version='stage1c-fixed-v1',currency='USD',commission_rate=.0001,
        minimum_commission=0,stamp_duty_sell_rate=0,transfer_fee_rate=0,buy_slippage_rate=.0005,
        sell_slippage_rate=.0005,price_tick=.01)


def model_decision(step, reply):
    from agentictrading import Step
    from agentictrading.integrations.vllm import VLLMTradingAgent,VLLMChatClient,VLLMConfig,MockChatTransport
    transport=MockChatTransport(reply if isinstance(reply,list) else [reply])
    agent=VLLMTradingAgent(VLLMChatClient(VLLMConfig(),transport=transport))
    parsed=Step.from_dict(step);agent.on_step(parsed)
    return agent,transport,agent.decide(parsed.observation)


def post(client,run,key,step,decision,idempotency='model-decision'):
    body=decision.to_body(run_id=run,step_id=step['step_id'],idempotency_key=idempotency)
    response=client.post(f"/api/v1/runs/{run}/steps/{step['step_id']}/decision",headers={'X-API-Key':key},json=body)
    assert response.status_code==200,response.text
    return response.json()


def reply(orders):return json.dumps({'orders':orders,'confidence':1,'rationale':'synthetic integration probe'})


def test_model_buy_sell_hold_and_http_duplicate_use_real_costed_ledger(client):
    run,key,step=start(client);session=ebs.get_session(runs._runs[run].backtest_id)
    session.manager.transaction_cost_profile=cost_profile()
    agent,transport,decision=model_decision(step,[(500,b'{"error":{}}'),reply([{'symbol':'AAPL','side':'buy','quantity':1}])])
    assert agent.records[-1].attempts==2 and len(decision.orders)==1
    accepted=post(client,run,key,step,decision)
    again=post(client,run,key,step,decision)
    assert accepted==again and len(session.manager.trades)==1
    assert len(session.decision_log[-1]['actions_submitted'])==1  # risk-approved stage
    assert accepted['accepted'] and not accepted['validation']['rejections']
    assert len(accepted['fills'])==1
    assert session.manager.positions['AAPL']==1
    before=copy.deepcopy(session.manager.equity_history)
    step=_wait_for_step(client,run,key)
    sell_agent,_,sell=model_decision(step,reply([{'symbol':'AAPL','side':'sell','quantity':1}]))
    sold=post(client,run,key,step,sell,'sell')
    assert sold['fills'][0]['side']=='sell' and not session.manager.positions
    assert session.manager.equity_history[:len(before)]==before
    buy=calculate_transaction_costs(side='buy',reference_price=100.37,shares=1,transaction_cost_profile=cost_profile())
    exit=calculate_transaction_costs(side='sell',reference_price=100.49,shares=1,transaction_cost_profile=cost_profile())
    assert session.manager.cash==pytest.approx(1000+buy['net_cash_impact']+exit['net_cash_impact'])
    assert session.manager.transaction_cost_totals['total_fees']>0
    assert session.manager.transaction_cost_totals['slippage_amount']>0
    assert [pd.Timestamp(t['timestamp']) for t in session.manager.trades]==[pd.Timestamp('2026-04-15T14:30Z'),pd.Timestamp('2026-04-15T15:30Z')]
    step=_wait_for_step(client,run,key)
    hold_agent,_,hold=model_decision(step,reply([]))
    held=post(client,run,key,step,hold,'hold')
    assert hold_agent.records[-1].status=='valid' and hold_agent.records[-1].action=='HOLD'
    assert held['fills']==[] and len(session.manager.trades)==2


@pytest.mark.parametrize('bad,reason',[
    ('not JSON','malformed_json'),
    ((400,b'{"error":{"message":"bad request","code":400}}'),'http_error'),
    (TimeoutError('synthetic timeout'),'timeout'),
    (reply([{'symbol':'NOT_ALLOWED','side':'buy','quantity':1}]),'unknown_symbol'),
    (reply([{'symbol':'AAPL','side':'sell','quantity':1}]),'exceeds_position'),
])
def test_bad_model_replies_hold_without_unauthorized_trade(client,bad,reason):
    run,key,step=start(client);session=ebs.get_session(runs._runs[run].backtest_id)
    agent,_,decision=model_decision(step,bad)
    assert agent.records[-1].reason==reason
    assert decision.orders==[]
    result=post(client,run,key,step,decision)
    assert result['fills']==[] and session.manager.trades==[]


def test_parsed_order_can_be_rejected_by_position_risk(client):
    run,key,step=start(client);session=ebs.get_session(runs._runs[run].backtest_id)
    agent,_,decision=model_decision(step,reply([{'symbol':'AAPL','side':'buy','quantity':5}]))
    assert agent.records[-1].status=='valid' and len(decision.orders)==1
    result=post(client,run,key,step,decision)
    assert result['fills']==[] and not session.manager.positions
    assert result['validation']['rejections'][0]['reason']=='exceeds_max_position_weight'
    assert session.decision_log[-1]['actions_submitted']==[]


def test_cash_rejection_and_partial_batch_do_not_create_leverage(client):
    run,key,step=start(client);session=ebs.get_session(runs._runs[run].backtest_id)
    # Controlled pre-existing holding supports equity while cash is unavailable.
    # This is an isolated risk fixture, not a historical portfolio result.
    session.manager.positions['KO']=10
    # US same-day settlement uses None to mean unrestricted sellable shares.
    if session.manager.sellable_positions is not None:
        session.manager.sellable_positions['KO']=10
    session.manager.entry_prices['KO']=100
    session.manager.cash=0
    _,_,decision=model_decision(step,reply([{'symbol':'AAPL','side':'buy','quantity':1}]))
    result=post(client,run,key,step,decision)
    assert result['fills']==[] and result['validation']['rejections'][0]['reason']=='insufficient_cash'
    assert session.manager.cash==0 and session.manager.positions=={'KO':10}
    step=_wait_for_step(client,run,key);session.manager.cash=300
    _,_,decision=model_decision(step,reply([{'symbol':s,'side':'buy','quantity':2} for s in ['AAPL','MSFT']]))
    result=post(client,run,key,step,decision,'partial-batch')
    assert len(session.decision_log[-1]['actions_submitted'])==2
    assert len(result['fills'])==1 and result['validation']['rejections']
    assert result['fills'][0]['filled_quantity']==2
    assert session.manager.cash>=0
