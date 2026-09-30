"""Frozen tape and nonzero costs must use the real HTTP/ATL execution path."""
import pandas as pd
import pytest
from dashboard.backend.tests.test_protocol_api import client, _new_agent, _new_version, _wait_for_step
from dashboard.backend.tests.test_phase27_dataset import tape
from dashboard.backend.domain.research.phase27 import load_spec, build_dataset, round_trip
from dashboard.backend.domain.backtesting.replay import register_replay, clear_replays
from dashboard.backend.domain.research.replay_dataset import make_replay
from dashboard.backend.domain.backtesting import external_run_service as ebs
from dashboard.backend.domain.runs import service as runs


@pytest.fixture(autouse=True)
def clean_replay_registry():
    clear_replays()
    yield
    clear_replays()


def create(client, replay_id, start="2023-01-03", end="2023-01-23"):
    agent, key, _ = _new_agent(client)
    version = _new_version(client, agent, key)
    response = client.post("/api/v1/runs", headers={"X-API-Key": key}, json={
        "agent_version_id": version, "environment": {"type": "backtest", "environment_id": "us-equity-hourly-v1"},
        "config": {"start_date": start, "end_date": end, "symbols": ["AAPL"], "initial_cash": 3000, "replay_id": replay_id}})
    return response, key


def test_replay_is_fail_closed_and_cannot_select_arbitrary_files(client):
    response, _ = create(client, "/etc/passwd")
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_replay"


def test_registered_window_must_match_request(client):
    raw, spec = tape(), load_spec()
    inputs, _, _ = build_dataset(raw, spec)
    register_replay("frozen-test", make_replay(raw, inputs, spec, "2023-01-03", "2023-01-23"))
    response, _ = create(client, "frozen-test", end="2023-01-24")
    assert response.status_code == 400


def test_frozen_http_roundtrip_costs_match_labels_and_persist(client, monkeypatch):
    raw, spec = tape(), load_spec()
    inputs, outcomes, _ = build_dataset(raw, spec)
    # One labeled opportunity plus its exit-control step.
    selected = inputs[:1]
    register_replay("frozen-test", make_replay(raw, selected, spec, "2023-01-03", "2023-01-23"))
    monkeypatch.setattr(ebs.AlpacaDataLoader, "fetch_bars", lambda *a, **k: pytest.fail("replay downloaded live data"))
    monkeypatch.setattr(ebs.baseline_worker, "submit", lambda *a, **k: pytest.fail("unmatched background baseline"))
    response, key = create(client, "frozen-test")
    assert response.status_code == 200, response.text
    run = response.json()["run_id"]
    timestamps, fills = [], []
    for side in ("buy", "sell"):
        step = _wait_for_step(client, run, key)
        timestamps.append(step["timestamp"])
        result = client.post(f"/api/v1/runs/{run}/steps/{step['step_id']}/decision", headers={"X-API-Key": key},
            json={"idempotency_key": side, "confidence": 1, "orders": [{"symbol": "AAPL", "side": side, "quantity": 1}]})
        assert result.status_code == 200, result.text
        assert not result.json()["validation"]["rejections"]
        fills.extend(result.json()["fills"])
    assert pd.Timestamp(timestamps[1]) - pd.Timestamp(timestamps[0]) == pd.Timedelta(hours=1)
    assert len(fills) == 2
    result = client.get(f"/api/v1/runs/{run}/result", headers={"X-API-Key": key}).json()
    entry = raw.loc[pd.Timestamp(timestamps[0]), "open"]
    expected_cash = 3000 + outcomes[0]["cost_adjusted_forward_return"] * entry
    assert result["metrics"]["final_equity"] == pytest.approx(expected_cash)
    assert sum(t["total_fees"] for t in result["trades"]) > 0
    assert sum(t["slippage_amount"] for t in result["trades"]) > 0
    assert all(pd.Timestamp(t["timestamp"]) == pd.Timestamp(ts) for t,ts in zip(result["trades"], timestamps))
    runs._runs.clear()
    ebs._sessions.clear()
    again = client.get(f"/api/v1/runs/{run}/result", headers={"X-API-Key": key})
    assert again.status_code == 200
    assert again.json() == result
