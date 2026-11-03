"""Regression tests for the Stage 1 long-context failure diagnostics."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from agentictrading import Step
from agentictrading.integrations.vllm import (
    ChatTransportError, VLLMChatClient, VLLMConfig, VLLMTradingAgent, build_messages,
)
from _vllm_helpers import FakeClock, chat_body, make_step


def smoke_module():
    path = Path(__file__).resolve().parents[3] / 'dashboard/examples/vllm_agent/vllm_inference_smoke.py'
    spec = importlib.util.spec_from_file_location('stage1_smoke_diagnostics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_http_400_keeps_sanitized_json_and_status_on_agent_record():
    key = 'local-key-do-not-persist'
    message = "This model's maximum context length is 4096 tokens. Your request has 5337 input tokens."
    raw = json.dumps({'error': {'message': message + ' Bearer ' + key,
        'type': 'BadRequestError', 'param': 'max_model_len', 'code': 400,
        'input': {'messages': 'private-position-data'}, 'api_key': key}}).encode()
    config = VLLMConfig(api_key=key)
    client = VLLMChatClient(config, transport=lambda *a: (400, raw))
    with pytest.raises(ChatTransportError) as caught:
        client.complete([{'role': 'user', 'content': 'private-position-data'}], timeout=5)
    error = caught.value
    assert error.error_response['error']['code'] == 400
    assert '5337' in error.error_response['error']['message']
    assert key not in json.dumps(error.error_response)
    assert 'private-position-data' not in json.dumps(error.error_response)
    assert set(error.error_response['error']) == {'message', 'type', 'param', 'code'}
    clock = FakeClock()
    agent = VLLMTradingAgent(client, wall_clock=clock.wall)
    step = make_step();agent.on_step(step);decision = agent.decide(step.observation)
    record = agent.records[-1]
    assert decision.orders == [] and record.status == 'failed'
    assert record.http_status == 400 and record.error_response == error.error_response
    assert record.attempts == 1  # don't retry a permanent 400


@pytest.mark.parametrize('raw', [b'<html>private-position-data</html>', b'not json', b'[]'])
def test_unstructured_error_bodies_are_not_persisted(raw):
    client = VLLMChatClient(VLLMConfig(), transport=lambda *a: (400, raw))
    with pytest.raises(ChatTransportError) as caught:
        client.complete([{'role':'user','content':'private-position-data'}],timeout=5)
    assert caught.value.error_response is None
    assert 'private-position-data' not in str(caught.value)


def test_repeated_smoke_requests_are_byte_identical_with_fresh_deadlines():
    smoke = smoke_module()
    scenario = smoke.scenarios()[0]
    client = VLLMChatClient(VLLMConfig())
    a = smoke.repeat_step(scenario, now=datetime(2026,11,3,10,0,tzinfo=timezone.utc))
    b = smoke.repeat_step(scenario, now=datetime(2026,11,3,10,5,tzinfo=timezone.utc))
    assert a.deadline_at != b.deadline_at
    assert a.timestamp == b.timestamp
    assert json.dumps(client.request_body(build_messages(a)),sort_keys=True).encode() == json.dumps(client.request_body(build_messages(b)),sort_keys=True).encode()
    assert scenario['deadline_at'] != a.deadline_at  # don't mutate the frozen scenario


def test_same_model_text_different_http_envelopes_is_same_decision():
    replies = [chat_body('{"orders":[],"confidence":0.5,"rationale":"wait"}') for _ in range(3)]
    for i,r in enumerate(replies):r.update(id=f'chat-{i}',created=i)
    client = VLLMChatClient(VLLMConfig(),transport=lambda *a:(200,json.dumps(replies.pop(0)).encode()))
    agent = VLLMTradingAgent(client,wall_clock=FakeClock().wall)
    for _ in range(3):
        step=make_step();agent.on_step(step);agent.decide(step.observation)
    assert len({r.response_sha256 for r in agent.records}) == 3
    summary = smoke_module().scenario_summary(agent.records, repeats=3)['step_1']
    assert summary['identical_requests_across_repeats'] is True
    assert summary['identical_replies_across_repeats'] is True


def test_failed_requests_never_count_as_identical_replies():
    client=VLLMChatClient(VLLMConfig(),transport=lambda *a:(400,b'{}'))
    agent=VLLMTradingAgent(client,wall_clock=FakeClock().wall)
    for _ in range(3):
        step=make_step();agent.on_step(step);agent.decide(step.observation)
    summary=smoke_module().scenario_summary(agent.records,repeats=3)['step_1']
    assert summary['identical_replies_across_repeats'] is None
    assert summary['determinism_eligible'] is False


def test_context_budget_includes_output_allowance_and_chat_template():
    smoke=smoke_module()
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):
            assert messages[0]['role']=='system' and messages[1]['role']=='user'
            assert kw == {'tokenize':True,'add_generation_prompt':True,'return_dict':False}
            return list(range(5337))
    request=VLLMChatClient(VLLMConfig()).request_body(build_messages(Step.from_dict(smoke.scenarios()[-1])))
    budget=smoke.context_budget(request,Tokenizer(),4096)
    assert budget['input_tokens']==5337 and budget['total_context_tokens']==5593
    assert budget['fits_context'] is False
    assert smoke.context_budget(request,Tokenizer(),5593)['fits_context'] is True
    assert smoke.context_budget(request,Tokenizer(),5592)['fits_context'] is False


def test_changed_prompts_are_not_eligible_for_determinism():
    client=VLLMChatClient(VLLMConfig(),transport=lambda *a:(200,json.dumps(chat_body('{"orders":[],"confidence":0.5,"rationale":"wait"}')).encode()))
    agent=VLLMTradingAgent(client,wall_clock=FakeClock().wall)
    for i in range(3):
        step=make_step()
        step.observation.portfolio['cash'] += i
        agent.on_step(step);agent.decide(step.observation)
    summary=smoke_module().scenario_summary(agent.records,repeats=3)['step_1']
    assert summary['identical_requests_across_repeats'] is False
    assert summary['determinism_eligible'] is False
    assert summary['identical_replies_across_repeats'] is None


def test_actual_pinned_qwen_token_budget_when_fixture_available():
    import os
    path=os.environ.get('ATL_QWEN_TOKENIZER_FIXTURE')
    if not path:
        pytest.skip('set ATL_QWEN_TOKENIZER_FIXTURE to the frozen pinned Qwen tokenizer; no downloads in unit tests')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    smoke=smoke_module();client=VLLMChatClient(VLLMConfig())
    expected=[465,479,510,5337]
    for payload,n in zip(smoke.scenarios(),expected):
        payload['timestamp']='2026-11-03T09:52:31+00:00'
        for bar in payload['observation']['market']['bars'].values():bar['timestamp']=payload['timestamp']
        request=client.request_body(build_messages(Step.from_dict(payload)))
        budget=smoke.context_budget(request,tokenizer,4096)
        assert budget['input_tokens']==n
        assert budget['total_context_tokens']==n+256
        assert budget['fits_context']==(n+256<=4096)
        assert smoke.context_budget(request,tokenizer,6144)['fits_context']


@pytest.mark.parametrize('long_http_error,expected_exit', [(False,0),(True,1)])
def test_smoke_cli_twelve_requests_and_honest_exit(monkeypatch,tmp_path,long_http_error,expected_exit):
    """Offline orchestration regression; this is not a GPU test."""
    import io
    import sys
    import urllib.request
    smoke=smoke_module();requests=[];generated=[]
    real_scenarios=smoke.scenarios
    def scenarios():
        generated.append(1)
        return real_scenarios()
    class NoGPU:
        available=False;info=[];peak_mib={}
        def start(self):pass
        def stop(self):pass
    def transport(method,url,body,headers,timeout):
        if url.endswith('/health'):return 200,b''
        if url.endswith('/models'):
            return 200,json.dumps({'data':[{'id':VLLMConfig().model_id}]}).encode()
        request=json.loads(body);requests.append(body)
        if long_http_error and len(request['messages'][1]['content'])>5000:
            return 400,json.dumps({'error':{'message':'Input 5337 exceeds 4096.','type':'BadRequestError','param':'max_model_len','code':400}}).encode()
        # Change envelope metadata on every response; identical content is what counts.
        response=chat_body('{"orders":[],"confidence":0.5,"rationale":"wait"}')
        response['id']='reply-'+str(len(requests))
        return 200,json.dumps(response).encode()
    monkeypatch.setattr(smoke,'GpuSampler',NoGPU)
    monkeypatch.setattr(smoke,'scenarios',scenarios)
    monkeypatch.setattr(smoke,'VLLMChatClient',lambda config:VLLMChatClient(config,transport=transport))
    monkeypatch.setattr(smoke,'scrape_metrics',lambda *a:{})
    monkeypatch.setattr(smoke,'free_form_probe',lambda *a:{'http_status':200,'verdict':'valid'})
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'{"version":"0.31.0"}'))
    monkeypatch.setattr(sys,'argv',['smoke','--out',str(tmp_path/'run'),'--repeats','3','--scenario-timestamp','2026-11-03T09:52:31+00:00'])
    assert smoke.main()==expected_exit
    assert len(generated)==1 and len(requests)==12
    for i in range(4):assert requests[i]==requests[i+4]==requests[i+8]
    assert all(json.loads(body)['messages'][1]['content'].count('2026-11-03T09:52:31+00:00') >= 2 for body in requests)
    report=json.loads((tmp_path/'run/inference-smoke.json').read_text())
    assert report['summary']['total_requests']==12
    assert report['gpu_available'] is False
    if long_http_error:
        records=[r for r in report['records'] if r['step_id']=='long_djia30']
        assert all(r['http_status']==400 and r['error_response']['error']['code']==400 for r in records)
        assert report['by_scenario']['long_djia30']['identical_replies_across_repeats'] is None
    # A second run must not overwrite the first report or launch a server.
    previous=(tmp_path/'run/inference-smoke.json').read_bytes()
    with pytest.raises(SystemExit) as caught:smoke.main()
    assert caught.value.code==2
    assert (tmp_path/'run/inference-smoke.json').read_bytes()==previous
