#!/usr/bin/env python3
"""ATL historical backtest smoke test: HOLD baseline vs. the vLLM agent (Phase 28-infra Stage 1).

Runs two protocol backtests with the identical, fixed configuration below
through the existing ATL Run API and SDK runner; nothing in the simulator,
data source or execution timing is touched. This is a functionality test: it
checks that steps arrive, the model is called, replies are validated, ATL's
own limits apply and both runs complete with their logs saved. Its returns are
not a result: no prompt or setting may be changed because of them, and
economic outputs are not validated research P&L.

    # mock model, no GPU (exercises everything but the GPU):
    PYTHONPATH=packaging/agentictrading/src python3 dashboard/examples/vllm_agent/vllm_backtest_smoke.py \\
        --atl-base-url http://127.0.0.1:8765 --out /path/outside/git/backtest-smoke --mock
    # real model: drop --mock, with VLLM_BASE_URL etc. pointing at the server

Agent credentials: ATL_AGENT_ID + ATL_API_KEY for an existing agent, or (on a
local server only) a throwaway guest agent is created per invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agentictrading import AgentRunner, ATLClient, Decision
from agentictrading.integrations.vllm import (
    MockChatTransport,
    VLLMChatClient,
    VLLMConfig,
    VLLMTradingAgent,
    summarize_records,
)

# Fixed before any model output was seen; never changed because of results.
SMOKE = {
    "environment_id": "us-equity-hourly-v1",
    "symbols": ["AAPL"],
    "start_date": "2026-03-02",
    "end_date": "2026-03-03",
}
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class HoldAgent:
    """Deterministic baseline: every step is an explicit empty-orders HOLD."""

    def decide(self, observation) -> Decision:
        return Decision(orders=[], rationale="hold baseline")


def mock_replies() -> List[str]:
    """Offline stand-in: one buy, one malformed reply, one sell, then holds."""
    def reply(orders):
        return json.dumps({"orders": orders, "confidence": 0.6, "rationale": "mock"})

    return [reply([{"symbol": "AAPL", "side": "buy", "quantity": 1}]), "not json",
            reply([{"symbol": "AAPL", "side": "sell", "quantity": 1}])] + [reply([])] * 200


def guest_agent(base_url: str) -> Dict[str, str]:
    if urlsplit(base_url).hostname not in _LOCAL_HOSTS:
        raise SystemExit("guest agents are for a local ATL only; set ATL_AGENT_ID and ATL_API_KEY")
    req = urllib.request.Request(base_url + "/api/v1/agents", method="POST",
                                 data=json.dumps({"name": "vllm-stage1-smoke"}).encode(),
                                 headers={"Content-Type": "application/json", "X-Session-Id": str(uuid.uuid4())})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return {"agent_id": body["agent"]["agent_id"], "api_key": body["api_key"]}


def run(sdk: ATLClient, agent_id: str, label: str, backbone: str, agent: Any,
        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    version = sdk.create_agent_version(agent_id, version=f"stage1-{label}-{uuid.uuid4().hex[:8]}",
                                       model_backbones=[backbone], metadata=metadata)
    result = AgentRunner(sdk, agent).run_backtest(
        version.id, environment_id=SMOKE["environment_id"], start_date=SMOKE["start_date"],
        end_date=SMOKE["end_date"], symbols=SMOKE["symbols"], poll_interval=1.0, max_wait_seconds=600)
    return {"label": label, "agent_version_id": version.id, "run_id": result.run_id, "status": result.status,
            "metrics": result.metrics, "trades": sdk.get_run_trades(result.run_id),
            "decisions": sdk.get_run_decisions(result.run_id)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atl-base-url", default=os.environ.get("ATL_BASE_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--out", required=True, type=Path, help="audit directory (keep it outside git)")
    ap.add_argument("--mock", action="store_true", help="scripted model replies instead of a vLLM server")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.atl_base_url.rstrip("/")

    creds = ({"agent_id": os.environ["ATL_AGENT_ID"], "api_key": os.environ["ATL_API_KEY"]}
             if os.environ.get("ATL_API_KEY") else guest_agent(base))
    sdk = ATLClient(base, creds["api_key"], timeout=60)
    config = VLLMConfig.from_env()
    transport = MockChatTransport(mock_replies()) if args.mock else None
    records_path = args.out / "vllm-records.jsonl"
    sink_file = open(records_path, "w")

    def sink(record):
        sink_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        sink_file.flush()

    vllm_agent = VLLMTradingAgent(VLLMChatClient(config, transport=transport), sink=sink)
    report: Dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(), "smoke": SMOKE,
                              "mode": "mock" if args.mock else "vllm", "atl_base_url": base,
                              "config": config.public_dict()}
    try:
        report["hold"] = run(sdk, creds["agent_id"], "hold", "none", HoldAgent())
        report["vllm"] = run(sdk, creds["agent_id"], "vllm", config.model_id, vllm_agent,
                               metadata={"config_hash": config.config_hash()})
    finally:
        sink_file.close()
        report["vllm_summary"] = summarize_records(vllm_agent.records)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.out / "backtest-smoke.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))

    hold, llm = report["hold"], report["vllm"]
    checks = {
        "both_completed": hold["status"] == llm["status"] == "completed",
        "same_step_count": len(hold["decisions"]) == len(llm["decisions"]),
        "model_called_every_step": len(vllm_agent.records) == len(llm["decisions"]),
        "hold_baseline_traded_nothing": hold["trades"] == [],
        "trades_only_from_valid_replies": len(llm["trades"]) <= sum(r.orders for r in vllm_agent.records
                                                                   if r.status == "valid"),
    }
    report["checks"] = checks
    (args.out / "backtest-smoke.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(json.dumps({"checks": checks, "vllm_summary": report["vllm_summary"]}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
