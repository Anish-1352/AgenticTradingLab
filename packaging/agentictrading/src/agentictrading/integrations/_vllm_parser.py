"""Strict reading of a model reply into an ATL ``Decision``.

JSON-or-nothing, like the backend's own parsers: the whole reply must be one
JSON object (no prose, no code fences, no duplicate keys, no NaN), every order
must be fully and unambiguously specified, and one bad order refuses the whole
reply instead of trading the rest. Nothing is coerced: ``"aapl"`` is not
``AAPL`` and ``2.0`` is not two shares.

The only limits checked here are the ones the step itself advertises
(``allowed_symbols``, ``max_orders``, ``allow_short`` against the shares held),
because those decide whether the reply is even a well-formed answer to this
step. Position weights, cash and per-order size stay ATL's to enforce.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from ..models import Decision, Order, Step

_MAX_RATIONALE = 500
_ORDER_KEYS = {"symbol", "side", "quantity"}
_DECISION_KEYS = {"orders", "confidence", "rationale"}
# Matches the backend's OrderIn bound (quantity < 1e12).
_MAX_QUANTITY = 10 ** 12


class RejectionReason:
    """Why a step ended in a safe HOLD instead of the model's reply."""

    EMPTY_RESPONSE = "empty_response"
    MALFORMED_JSON = "malformed_json"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_SIDE = "invalid_side"
    UNKNOWN_SYMBOL = "unknown_symbol"
    INVALID_QUANTITY = "invalid_quantity"
    EXCEEDS_POSITION = "exceeds_position"
    DUPLICATE_SYMBOL = "duplicate_symbol"
    TOO_MANY_ORDERS = "too_many_orders"
    INVALID_CONFIDENCE = "invalid_confidence"
    # Inference failures (no reply to parse), set by the agent.
    DEADLINE_EXCEEDED = "deadline_exceeded"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    BAD_RESPONSE = "bad_response"
    INTERNAL_ERROR = "internal_error"


class DecisionRejected(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def parse_decision(content: Optional[str], step: Step) -> Decision:
    if content is None or not content.strip():
        raise DecisionRejected(RejectionReason.EMPTY_RESPONSE)
    try:
        payload = json.loads(content, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants)
    except (ValueError, TypeError) as exc:
        raise DecisionRejected(RejectionReason.MALFORMED_JSON, str(exc)[:200]) from None
    if not isinstance(payload, dict):
        raise DecisionRejected(RejectionReason.MALFORMED_JSON, "reply is not a JSON object")
    if "orders" not in payload or set(payload) - _DECISION_KEYS or not isinstance(payload["orders"], list):
        raise DecisionRejected(RejectionReason.INVALID_SCHEMA, f"keys {sorted(payload)}")

    constraints = step.constraints
    allowed = set(constraints.get("allowed_symbols") or [])
    max_orders = constraints.get("max_orders")
    held = _held_shares(step)
    orders: List[Order] = []
    seen = set()
    for raw in payload["orders"]:
        if not isinstance(raw, dict) or set(raw) != _ORDER_KEYS:
            raise DecisionRejected(RejectionReason.INVALID_SCHEMA, "order must have exactly symbol, side, quantity")
        symbol, side, quantity = raw["symbol"], raw["side"], raw["quantity"]
        if side not in ("buy", "sell"):
            raise DecisionRejected(RejectionReason.INVALID_SIDE, repr(side))
        if not isinstance(symbol, str) or symbol not in allowed:
            raise DecisionRejected(RejectionReason.UNKNOWN_SYMBOL, repr(symbol))
        if symbol in seen:
            raise DecisionRejected(RejectionReason.DUPLICATE_SYMBOL, symbol)
        seen.add(symbol)
        if type(quantity) is not int or not 0 < quantity < _MAX_QUANTITY:
            raise DecisionRejected(RejectionReason.INVALID_QUANTITY, repr(quantity))
        if side == "sell" and not constraints.get("allow_short", False) and quantity > held.get(symbol, 0):
            raise DecisionRejected(RejectionReason.EXCEEDS_POSITION,
                                   f"sell {quantity} {symbol}, hold {held.get(symbol, 0)}")
        orders.append(Order(symbol=symbol, side=side, quantity=quantity))
    if isinstance(max_orders, int) and len(orders) > max_orders:
        raise DecisionRejected(RejectionReason.TOO_MANY_ORDERS, f"{len(orders)} > {max_orders}")

    confidence = payload.get("confidence")
    if confidence is not None and (
        type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1
    ):
        raise DecisionRejected(RejectionReason.INVALID_CONFIDENCE, repr(confidence))
    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise DecisionRejected(RejectionReason.INVALID_SCHEMA, "rationale must be a string")
    return Decision(
        orders=orders,
        confidence=float(confidence) if confidence is not None else None,
        rationale=rationale[:_MAX_RATIONALE] if rationale is not None else None,
    )


def _held_shares(step: Step) -> Dict[str, float]:
    held: Dict[str, float] = {}
    for p in step.observation.positions if step.observation else []:
        if isinstance(p, dict) and isinstance(p.get("symbol"), str):
            held[p["symbol"]] = held.get(p["symbol"], 0) + float(p.get("quantity") or 0)
    return held


def _no_duplicate_keys(pairs: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constants(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")
