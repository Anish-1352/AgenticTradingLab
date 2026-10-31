"""Snapshot serializer and prompt: deterministic, and built only from what a step may see."""

from __future__ import annotations

import json

import pytest

from agentictrading.integrations.vllm import (
    DECISION_JSON_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    serialize_snapshot,
)

from _vllm_helpers import STEP_TS, make_step, step_payload


def test_same_snapshot_gives_the_same_prompt():  # A
    a = build_messages(make_step())
    b = build_messages(make_step())
    assert a == b
    assert a[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert a[1]["role"] == "user"


def test_serialization_ignores_dict_and_list_order():  # A
    payload = step_payload()
    reordered = step_payload()
    obs = reordered["observation"]
    obs["market"]["bars"] = dict(reversed(list(obs["market"]["bars"].items())))
    obs["portfolio"]["positions"] = list(reversed(obs["portfolio"]["positions"]))
    reordered["constraints"]["allowed_symbols"] = ["MSFT", "AAPL"]
    from agentictrading import Step

    assert serialize_snapshot(Step.from_dict(payload)) == serialize_snapshot(Step.from_dict(reordered))


def test_snapshot_is_canonical_json_with_the_documented_sections():
    snap = json.loads(serialize_snapshot(make_step()))
    assert set(snap) == {"timestamp", "constraints", "portfolio", "market"}
    assert snap["timestamp"] == STEP_TS
    assert snap["constraints"] == {"allowed_symbols": ["AAPL", "MSFT"], "allow_short": False,
                                   "max_orders": 10, "max_position_weight": 0.25}
    assert snap["portfolio"]["positions"][0]["symbol"] == "MSFT"
    assert set(snap["market"]) == {"bars", "features"}


def test_a_bar_dated_after_the_step_never_reaches_the_prompt():  # L
    payload = step_payload()
    payload["observation"]["market"]["bars"]["AAPL"]["timestamp"] = "2026-04-15T12:00:00-04:00"
    snap = json.loads(serialize_snapshot(make_step(**{"observation": payload["observation"]})))
    assert "AAPL" not in snap["market"]["bars"]
    assert "MSFT" in snap["market"]["bars"]


def test_future_only_fields_cannot_alter_the_prompt():  # L
    base = build_messages(make_step())
    payload = step_payload()
    obs = payload["observation"]
    obs["market"]["features"]["AAPL"]["forward_return_5d"] = 0.12
    obs["market"]["features"]["AAPL"]["label"] = 1
    obs["market"]["bars"]["AAPL"]["next_close"] = 140.0
    obs["market"]["events"] = [{"type": "earnings", "timestamp": "2026-04-20T16:00:00-04:00"}]
    obs["future_fills"] = [{"symbol": "AAPL"}]
    payload["future_metrics"] = {"total_return": 0.3}
    from agentictrading import Step

    assert build_messages(Step.from_dict(payload)) == base


def test_symbols_outside_the_allow_list_are_dropped():
    payload = step_payload()
    payload["observation"]["market"]["bars"]["TSLA"] = dict(payload["observation"]["market"]["bars"]["AAPL"])
    payload["observation"]["market"]["features"]["TSLA"] = {"price": 1.0}
    from agentictrading import Step

    snap = json.loads(serialize_snapshot(Step.from_dict(payload)))
    assert "TSLA" not in snap["market"]["bars"]
    assert "TSLA" not in snap["market"]["features"]


def test_nonfinite_numbers_serialize_as_null_not_nan():
    payload = step_payload()
    payload["observation"]["market"]["features"]["AAPL"]["rsi"] = float("nan")
    from agentictrading import Step

    text = serialize_snapshot(Step.from_dict(payload))
    assert "NaN" not in text
    assert json.loads(text)["market"]["features"]["AAPL"]["rsi"] is None


def test_unparseable_step_timestamp_fails_closed_to_no_bars():
    payload = step_payload(timestamp=None)
    from agentictrading import Step

    snap = json.loads(serialize_snapshot(Step.from_dict(payload)))
    assert snap["market"]["bars"] == {}


def test_step_without_observation_is_refused():
    from agentictrading import Step

    with pytest.raises(ValueError):
        serialize_snapshot(Step.from_dict({"status": "awaiting_decision", "step_id": "s"}))


def test_schema_is_a_strict_subset_of_orders_v1():
    order = DECISION_JSON_SCHEMA["properties"]["orders"]["items"]
    assert DECISION_JSON_SCHEMA["additionalProperties"] is False
    assert order["additionalProperties"] is False
    assert order["properties"]["side"]["enum"] == ["buy", "sell"]
    assert order["properties"]["quantity"]["type"] == "integer"
    assert PROMPT_VERSION
