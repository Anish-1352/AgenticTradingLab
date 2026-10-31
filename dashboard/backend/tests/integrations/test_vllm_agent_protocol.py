"""The SDK's vLLM adapter driving a real protocol run, with the model mocked.

Everything below the model is real: ``AgentRunner`` and ``ATLClient`` speak
HTTP (bridged into the FastAPI app) to the v1 Run API and the external
backtest session, on synthetic bars. The scripted replies mix valid trades with
unusable ones, so the run shows both that a valid reply executes exactly as
given and that a failed or invalid inference reaches ATL as an empty HOLD and
never as a trade. No GPU and no network are needed.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_SDK_SRC = Path(__file__).resolve().parents[4] / "packaging" / "agentictrading" / "src"
if str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

import agentictrading.atl_client as atl_client  # noqa: E402
from agentictrading import AgentRunner, ATLClient  # noqa: E402
from agentictrading.integrations.vllm import (  # noqa: E402
    DecisionStatus,
    MockChatTransport,
    RejectionReason,
    VLLMChatClient,
    VLLMConfig,
    VLLMTradingAgent,
    summarize_records,
)

from dashboard.backend.tests.test_protocol_api import _new_agent, _new_version, client  # noqa: E402,F401


def _reply(orders):
    return json.dumps({"orders": orders, "confidence": 0.6, "rationale": "scripted"})


SCRIPT = [
    _reply([{"symbol": "AAPL", "side": "buy", "quantity": 2}]),        # valid BUY -> one fill
    "I think you should buy AAPL",                                       # malformed -> HOLD
    _reply([{"symbol": "TSLA", "side": "buy", "quantity": 1}]),          # not allowed -> HOLD
    _reply([{"symbol": "MSFT", "side": "sell", "quantity": 5}]),         # sells unheld -> HOLD
    "",                                                                  # empty -> HOLD
    _reply([{"symbol": "AAPL", "side": "sell", "quantity": 2}]),         # valid SELL -> one fill
]


class _Response:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def bridged(client, monkeypatch):  # noqa: F811
    """Route the SDK's urllib calls into the FastAPI app."""

    def urlopen(req, timeout=None):
        path = req.full_url.replace("http://atl.test", "", 1)
        resp = client.request(req.get_method(), path, content=req.data, headers=dict(req.header_items()))
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(req.full_url, resp.status_code, "error", {}, io.BytesIO(resp.content))
        return _Response(resp.status_code, resp.content)

    monkeypatch.setattr(atl_client.urllib.request, "urlopen", urlopen)
    return client


def test_vllm_agent_runs_a_protocol_backtest_and_only_valid_replies_trade(bridged):
    agent_id, key, _ = _new_agent(bridged)
    version_id = _new_version(bridged, agent_id, key)
    replies = SCRIPT + [_reply([])] * 40  # the rest of the window holds
    transport = MockChatTransport(replies)
    agent = VLLMTradingAgent(VLLMChatClient(VLLMConfig(base_url="http://gpu.test:8000/v1"), transport=transport))

    sdk = ATLClient("http://atl.test", key)
    result = AgentRunner(sdk, agent).run_backtest(
        version_id, environment_id="us-equity-hourly-v1", start_date="2026-04-15",
        end_date="2026-04-16", symbols=["AAPL", "MSFT"], poll_interval=0.01, max_wait_seconds=30,
    )
    assert result.status == "completed"

    records = agent.records
    assert len(records) == len(transport.requests) >= len(SCRIPT)
    assert len({r.step_id for r in records}) == len(records)  # one decision per step
    assert [r.reason for r in records[:len(SCRIPT)]] == [
        None, RejectionReason.MALFORMED_JSON, RejectionReason.UNKNOWN_SYMBOL,
        RejectionReason.EXCEEDS_POSITION, RejectionReason.EMPTY_RESPONSE, None,
    ]
    assert [r.status for r in records[:len(SCRIPT)]] == [
        DecisionStatus.VALID, DecisionStatus.REJECTED, DecisionStatus.REJECTED,
        DecisionStatus.REJECTED, DecisionStatus.REJECTED, DecisionStatus.VALID,
    ]

    # Exactly the two valid orders traded, at the scripted size.
    trades = sdk.get_run_trades(result.run_id)
    assert [(t["symbol"], t["side"].lower(), t["quantity"]) for t in trades] == [("AAPL", "buy", 2), ("AAPL", "sell", 2)]

    # ATL recorded every step, and the held ones as held.
    decisions = sdk.get_run_decisions(result.run_id)
    assert len(decisions) == len(records)

    summary = summarize_records(records)
    assert summary["total_requests"] == len(records)
    assert summary["timeout_rate"] == 0.0
    # Every prompt went out with ATL's real constraints for this run.
    snapshot = json.loads(transport.requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert snapshot["constraints"]["allowed_symbols"] == ["AAPL", "MSFT"]
    assert snapshot["constraints"]["max_orders"] == 10
    assert set(snapshot["market"]["bars"]) == {"AAPL", "MSFT"}
