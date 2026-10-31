"""VLLMTradingAgent: deadline budget, bounded retries, safe holds, telemetry, runner fit."""

from __future__ import annotations

import json
import socket
import urllib.error

import pytest

from agentictrading import AgentRunner, ATLClient, Decision
from agentictrading.integrations.vllm import (
    DecisionStatus,
    MockChatTransport,
    RejectionReason,
    VLLMChatClient,
    VLLMConfig,
    VLLMTradingAgent,
    summarize_records,
)

from _vllm_helpers import DEADLINE_EPOCH, FakeClock, chat_body, decision_json, make_step, step_payload

BUY = decision_json([{"symbol": "AAPL", "side": "buy", "quantity": 1}])
SELL = decision_json([{"symbol": "MSFT", "side": "sell", "quantity": 1}])
HOLD = decision_json([])


def _agent(replies, clock=None, delays=None, **cfg):
    clock = clock or FakeClock()
    transport = MockChatTransport(replies, delays=delays, advance=clock.advance)
    config = VLLMConfig(**cfg)
    client = VLLMChatClient(config, transport=transport, monotonic=clock.monotonic)
    agent = VLLMTradingAgent(client, wall_clock=clock.wall)
    return agent, transport, clock


def _decide(agent, step=None):
    step = step or make_step()
    agent.on_step(step)
    return agent.decide(step.observation)


def test_valid_reply_becomes_the_submitted_decision():
    agent, transport, _ = _agent([BUY])
    d = _decide(agent)
    assert isinstance(d, Decision)
    assert d.to_dict()["orders"] == [{"symbol": "AAPL", "side": "buy", "quantity_type": "shares",
                                      "quantity": 1, "order_type": "market"}]
    rec = agent.records[-1]
    assert rec.status == DecisionStatus.VALID and rec.action == "BUY" and rec.attempts == 1
    assert d.trace["prompt_sha256"] == rec.prompt_sha256
    assert d.trace["config_hash"] == agent.client.config.config_hash()
    assert len(transport.requests) == 1


@pytest.mark.parametrize("reply,reason", [
    ("garbage", RejectionReason.MALFORMED_JSON),
    ("", RejectionReason.EMPTY_RESPONSE),
    (decision_json([{"symbol": "TSLA", "side": "buy", "quantity": 1}]), RejectionReason.UNKNOWN_SYMBOL),
    (decision_json([{"symbol": "AAPL", "side": "buy", "quantity": -3}]), RejectionReason.INVALID_QUANTITY),
])
def test_unusable_reply_is_an_explicit_hold_with_its_reason(reply, reason):  # E F G H N
    agent, transport, _ = _agent([reply])
    d = _decide(agent)
    assert d.orders == []
    assert d.trace["status"] == DecisionStatus.REJECTED
    assert d.trace["reason"] == reason
    assert agent.records[-1].reason == reason
    # A rejected reply is deterministic under greedy decoding: no retry.
    assert len(transport.requests) == 1


def test_late_reply_is_discarded_and_cannot_reach_the_next_step():  # I
    clock = FakeClock()
    # Step 1's reply is a BUY that lands after the deadline; step 2 replies HOLD.
    agent, transport, _ = _agent([BUY, HOLD], clock=clock, delays=[58.0, 0.5], request_timeout=59.0)
    first = _decide(agent)
    assert first.orders == []
    assert agent.records[0].reason == RejectionReason.DEADLINE_EXCEEDED
    assert agent.records[0].timed_out is True

    nxt = step_payload(step_id="step_2", sequence=1, timestamp="2026-04-15T12:00:00-04:00",
                       deadline_at="2026-04-15T12:01:00-04:00")
    clock.now = DEADLINE_EPOCH + 3600 - 60  # step 2 starts on time
    from agentictrading import Step

    second = _decide(agent, Step.from_dict(nxt))
    assert second.orders == []
    assert agent.records[1].status == DecisionStatus.VALID and agent.records[1].step_id == "step_2"
    assert len(transport.requests) == 2


def test_request_timeout_is_capped_by_the_step_deadline():  # I
    clock = FakeClock()
    clock.advance(40.0)  # 20s of the 60s window left
    agent, transport, _ = _agent([HOLD], clock=clock, request_timeout=30.0)
    _decide(agent)
    assert transport.timeouts[0] <= 20.0 - agent.deadline_margin_seconds + 1e-9


def test_no_request_once_the_budget_is_spent():  # I
    clock = FakeClock()
    clock.advance(58.0)
    agent, transport, _ = _agent([BUY], clock=clock)
    d = _decide(agent)
    assert d.orders == []
    assert transport.requests == []
    assert agent.records[-1].reason == RejectionReason.DEADLINE_EXCEEDED


def test_connection_failure_retries_once_then_holds():  # J
    err = urllib.error.URLError(ConnectionRefusedError(61, "refused"))
    agent, transport, _ = _agent([err, err, err])
    d = _decide(agent)
    assert d.orders == []
    assert len(transport.requests) == 2  # 1 + max_retries(1), never a storm
    rec = agent.records[-1]
    assert rec.status == DecisionStatus.FAILED and rec.reason == RejectionReason.CONNECTION_ERROR
    assert rec.attempts == 2


def test_connection_blip_then_success_uses_the_second_reply():  # J
    err = ConnectionResetError(54, "reset")
    agent, transport, _ = _agent([err, SELL])
    d = _decide(agent)
    assert d.to_dict()["orders"][0]["side"] == "sell"
    assert agent.records[-1].attempts == 2


def test_timeout_is_not_retried():  # J
    agent, transport, _ = _agent([socket.timeout("timed out"), BUY])
    d = _decide(agent)
    assert d.orders == []
    assert len(transport.requests) == 1
    assert agent.records[-1].reason == RejectionReason.TIMEOUT and agent.records[-1].timed_out


def test_http_400_is_not_retried():  # J
    agent, transport, _ = _agent([(400, b'{"error": "bad"}'), BUY])
    _decide(agent)
    assert len(transport.requests) == 1
    assert agent.records[-1].reason == RejectionReason.HTTP_ERROR


def test_unexpected_transport_exception_holds_instead_of_crashing():  # J
    agent, transport, _ = _agent([RuntimeError("boom")])
    d = _decide(agent)
    assert d.orders == [] and agent.records[-1].status == DecisionStatus.FAILED


def test_records_carry_tokens_latency_and_hashes():  # K
    clock = FakeClock()
    agent, _, _ = _agent([(200, json.dumps(chat_body(HOLD, prompt_tokens=500, completion_tokens=30)).encode())],
                         clock=clock, delays=[0.8])
    _decide(agent)
    rec = agent.records[-1].to_dict()
    assert rec["prompt_tokens"] == 500 and rec["completion_tokens"] == 30
    assert rec["inference_seconds"] == pytest.approx(0.8)
    assert rec["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert len(rec["prompt_sha256"]) == 64 and len(rec["response_sha256"]) == 64
    assert rec["parsed"] is True and rec["timed_out"] is False
    assert rec["step_id"] == "step_1" and rec["timestamp"]
    assert "api_key" not in json.dumps(rec)


def test_decide_without_step_context_fails_loudly():  # M
    agent, _, _ = _agent([HOLD])
    with pytest.raises(RuntimeError, match="on_step"):
        agent.decide(make_step().observation)


def test_mismatched_observation_fails_loudly():  # M
    agent, _, _ = _agent([HOLD])
    agent.on_step(make_step())
    with pytest.raises(RuntimeError):
        agent.decide(make_step().observation)


def test_summary_aggregates_the_records():
    clock = FakeClock()
    agent, _, _ = _agent([BUY, "garbage", HOLD, SELL], clock=clock, delays=[1.0, 2.0, 3.0, 4.0])
    for _ in range(4):
        clock.now = DEADLINE_EPOCH - 60
        _decide(agent)
    s = summarize_records(agent.records)
    assert s["total_requests"] == 4
    assert s["valid_decision_rate"] == 0.75
    assert s["invalid_response_rate"] == 0.25
    assert s["action_rates"] == {"BUY": 0.25, "SELL": 0.25, "HOLD": 0.5}
    assert s["timeout_rate"] == 0.0
    assert s["latency_p50_seconds"] == 2.0 and s["latency_p95_seconds"] == 4.0
    assert s["total_tokens"] == 4 * (420 + 24)
    assert s["output_tokens_per_second"] == pytest.approx(4 * 24 / 10.0)


# -- through the real SDK runner (fake ATL backend) ----------------------


class Backend:
    """Two awaiting steps, then completion; records what the runner submits."""

    def __init__(self):
        self.steps = [step_payload(), step_payload(step_id="step_2", sequence=1)]
        self.submitted = []

    def __call__(self, req):
        url, method = req.full_url, req.get_method()
        if method == "POST" and url.endswith("/api/v1/runs"):
            return (200, {"run_id": "run_1", "status": "created"})
        if url.endswith("/steps/next"):
            return (200, self.steps.pop(0) if self.steps else {"status": "completed", "run_id": "run_1"})
        if url.endswith("/decision"):
            self.submitted.append(json.loads(req.data.decode()))
            return (200, {"run_id": "run_1", "accepted": True, "validation": {"passed": True, "rejections": []},
                          "fills": [], "portfolio_after": {}, "run_status": "running"})
        if url.endswith("/result"):
            return (200, {"run_id": "run_1", "status": "completed", "metrics": {}})
        raise AssertionError(url)


def test_agent_runs_under_agent_runner_and_failed_inference_submits_no_orders(fake_http):  # M N
    backend = Backend()
    fake_http(backend)
    clock = FakeClock()
    agent, transport, _ = _agent([BUY, "{broken"], clock=clock)
    AgentRunner(ATLClient("http://atl.test", "ag_key"), agent).run_backtest(
        "agv_1", environment_id="us-equity-hourly-v1", start_date="2026-04-15",
        end_date="2026-04-15", poll_interval=0.001)
    assert [s["step_id"] for s in backend.submitted] == ["step_1", "step_2"]
    assert backend.submitted[0]["orders"][0]["symbol"] == "AAPL"
    assert backend.submitted[1]["orders"] == []
    assert backend.submitted[1]["trace"]["reason"] == RejectionReason.MALFORMED_JSON
    assert len(transport.requests) == 2


def test_unreadable_step_state_holds_instead_of_crashing():
    payload = step_payload()
    payload["observation"]["portfolio"]["positions"][0]["quantity"] = "two"
    from agentictrading import Step

    agent, _, _ = _agent([SELL])
    d = _decide(agent, Step.from_dict(payload))
    assert d.orders == []
    assert agent.records[-1].reason == RejectionReason.INTERNAL_ERROR
