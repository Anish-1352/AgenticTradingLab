"""Strict reply validation: a reply becomes an ATL Decision exactly, or it is refused."""

from __future__ import annotations

import json

import pytest

from agentictrading import Decision, Order
from agentictrading.integrations.vllm import DecisionRejected, RejectionReason, parse_decision

from _vllm_helpers import decision_json, make_step


def _parse(content):
    return parse_decision(content, make_step())


def _reason(content):
    with pytest.raises(DecisionRejected) as ei:
        _parse(content)
    return ei.value.reason


def test_valid_buy_maps_to_an_atl_share_order():  # B
    d = _parse(decision_json([{"symbol": "AAPL", "side": "buy", "quantity": 1}], confidence=0.7))
    assert isinstance(d, Decision)
    assert d.to_dict()["orders"] == [Order("AAPL", "buy", 1).to_dict()]
    assert d.to_dict()["orders"][0]["quantity_type"] == "shares"
    assert d.confidence == 0.7


def test_valid_sell_maps_to_an_atl_share_order():  # C
    d = _parse(decision_json([{"symbol": "MSFT", "side": "sell", "quantity": 2}]))
    assert d.to_dict()["orders"] == [Order("MSFT", "sell", 2).to_dict()]


def test_valid_hold_is_an_empty_orders_list():  # D
    d = _parse(decision_json([], confidence=0.5, rationale="nothing to do"))
    assert d.orders == []
    assert d.to_dict()["orders"] == []
    assert d.rationale == "nothing to do"


def test_confidence_and_rationale_are_optional():
    d = _parse(json.dumps({"orders": []}))
    assert d.orders == [] and d.confidence is None


@pytest.mark.parametrize("content", [
    "not json",
    "{\"orders\": []",
    "Sure! {\"orders\": []}",
    "```json\n{\"orders\": []}\n```",
    "[]",
    "\"HOLD\"",
    "{\"orders\": [], \"orders\": [{\"symbol\": \"AAPL\", \"side\": \"buy\", \"quantity\": 1}]}",
    "{\"orders\": [], \"confidence\": NaN}",
])
def test_malformed_reply_is_rejected(content):  # E
    assert _reason(content) == RejectionReason.MALFORMED_JSON


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_empty_reply_is_an_explicit_rejection(content):  # F
    assert _reason(content) == RejectionReason.EMPTY_RESPONSE


@pytest.mark.parametrize("payload", [
    {},
    {"orders": "buy AAPL"},
    {"orders": [], "action": "BUY"},
    {"orders": [{"symbol": "AAPL", "side": "buy"}]},
    {"orders": [{"symbol": "AAPL", "side": "buy", "quantity": 1, "quantity_type": "weight"}]},
    {"orders": [["AAPL", "buy", 1]]},
    {"orders": [], "rationale": 5},
])
def test_shape_violations_are_rejected(payload):
    assert _reason(json.dumps(payload)) == RejectionReason.INVALID_SCHEMA


@pytest.mark.parametrize("side", ["BUY", "short", "hold", "", 1])
def test_unknown_side_is_rejected(side):
    assert _reason(decision_json([{"symbol": "AAPL", "side": side, "quantity": 1}])) == RejectionReason.INVALID_SIDE


@pytest.mark.parametrize("symbol", ["TSLA", "aapl", " AAPL", "AAPL.US", ""])
def test_symbol_outside_the_allow_list_is_rejected_not_normalized(symbol):  # G
    assert _reason(decision_json([{"symbol": symbol, "side": "buy", "quantity": 1}])) == RejectionReason.UNKNOWN_SYMBOL


@pytest.mark.parametrize("quantity", [0, -1, 1.5, 2.0, "3", True, None, 1e12])
def test_invalid_quantity_is_rejected(quantity):  # H
    assert _reason(decision_json([{"symbol": "AAPL", "side": "buy", "quantity": quantity}])) == RejectionReason.INVALID_QUANTITY


def test_selling_more_than_held_is_rejected_when_shorting_is_off():  # H
    assert _reason(decision_json([{"symbol": "MSFT", "side": "sell", "quantity": 3}])) == RejectionReason.EXCEEDS_POSITION
    assert _reason(decision_json([{"symbol": "AAPL", "side": "sell", "quantity": 1}])) == RejectionReason.EXCEEDS_POSITION


def test_two_orders_for_one_symbol_are_ambiguous():
    orders = [{"symbol": "AAPL", "side": "buy", "quantity": 1}, {"symbol": "AAPL", "side": "sell", "quantity": 1}]
    assert _reason(decision_json(orders)) == RejectionReason.DUPLICATE_SYMBOL


def test_more_orders_than_max_orders_is_rejected():
    step = make_step(constraints={"allowed_symbols": ["AAPL", "MSFT"], "allow_short": False, "max_orders": 1})
    orders = [{"symbol": "AAPL", "side": "buy", "quantity": 1}, {"symbol": "MSFT", "side": "buy", "quantity": 1}]
    with pytest.raises(DecisionRejected) as ei:
        parse_decision(decision_json(orders), step)
    assert ei.value.reason == RejectionReason.TOO_MANY_ORDERS


@pytest.mark.parametrize("confidence", [-0.1, 1.5, "high", True, float("inf")])
def test_out_of_range_confidence_is_rejected(confidence):
    content = json.dumps({"orders": [], "confidence": confidence}) if confidence != float("inf") else '{"orders": [], "confidence": 1e999}'
    assert _reason(content) == RejectionReason.INVALID_CONFIDENCE


def test_one_bad_order_voids_the_whole_reply():  # N
    orders = [{"symbol": "AAPL", "side": "buy", "quantity": 1}, {"symbol": "TSLA", "side": "buy", "quantity": 1}]
    assert _reason(decision_json(orders)) == RejectionReason.UNKNOWN_SYMBOL


def test_long_rationale_is_capped():
    d = _parse(decision_json([], rationale="x" * 2000))
    assert len(d.rationale) == 500
