"""Deterministic prompt for one ATL step: a fixed system prompt plus a canonical snapshot.

The snapshot is built from an allow-list, not by copying the observation: only
the step's timestamp, its advertised constraints, the portfolio, and each
allowed symbol's current bar and indicator fields reach the model. A field ATL
adds later (or a field that should never have been there, like a forward
return) is dropped until someone adds it here on purpose, and a bar stamped
after the step is dropped too, so nothing dated in the future can enter a
prompt through this module. Serialization is canonical JSON (sorted keys, no
whitespace, non-finite numbers as null), so the same visible state always
yields byte-identical messages.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import Step

PROMPT_VERSION = "atl-vllm-prompt.1"

SYSTEM_PROMPT = (
    "You are a trading agent in a simulated US-equity backtest run by the Agentic Trading Lab. "
    "Each message is one decision step: a JSON snapshot of the current hourly bar, indicator "
    "values, your portfolio and the constraints for this step. Decide what to trade now.\n"
    "Reply with exactly one JSON object and nothing else:\n"
    '{"orders": [{"symbol": "<allowed symbol>", "side": "buy" | "sell", "quantity": <whole shares>}], '
    '"confidence": <number from 0 to 1>, "rationale": "<one short sentence>"}\n'
    "Rules:\n"
    "- An empty orders list means HOLD (no trade).\n"
    "- Use only symbols in constraints.allowed_symbols, at most one order per symbol, and at "
    "most constraints.max_orders orders.\n"
    "- quantity is a positive whole number of shares.\n"
    "- If constraints.allow_short is false, never sell more shares than the portfolio holds.\n"
    "- Keep each position within constraints.max_position_weight of equity, and buy only what "
    "the cash allows.\n"
    "Orders fill at the next bar; the environment rejects anything outside its limits."
)

# orders-v1 restricted to market orders sized in whole shares; the adapter adds
# quantity_type/order_type itself. Static on purpose: per-step symbol enums
# would make every request schema different (and uncacheable server-side),
# while the parser enforces the allow-list exactly anyway.
DECISION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["orders", "confidence", "rationale"],
    "properties": {
        "orders": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "side", "quantity"],
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"enum": ["buy", "sell"]},
                    "quantity": {"type": "integer", "minimum": 1},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 500},
    },
}

_BAR_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")
# The per-symbol fields ATL's external snapshot publishes (top_signals).
_FEATURE_FIELDS = (
    "price", "rsi", "macd", "macd_signal", "sma20", "sma50", "bb_upper", "bb_lower",
    "turnover", "market_cap_usd", "sic_code", "market_cap_status", "industry", "sector",
)
_POSITION_FIELDS = (
    "symbol", "quantity", "entry_price", "current_price", "market_value", "unrealized_pnl_pct",
)
_CONSTRAINT_FIELDS = ("allow_short", "max_orders", "max_position_weight")


def build_messages(step: Step) -> List[Dict[str, str]]:
    """The chat messages for ``step``: system prompt, then the snapshot."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Snapshot:\n" + serialize_snapshot(step)},
    ]


def serialize_snapshot(step: Step) -> str:
    if step.observation is None:
        raise ValueError("step has no observation to serialize")
    allowed = sorted({s for s in step.constraints.get("allowed_symbols") or [] if isinstance(s, str)})
    step_time = parse_iso_time(step.timestamp)
    market = step.observation.market
    bars = market.get("bars") or {}
    features = market.get("features") or {}
    portfolio = step.observation.portfolio

    snapshot = {
        "timestamp": step.timestamp,
        "constraints": {
            "allowed_symbols": allowed,
            **{k: step.constraints.get(k) for k in _CONSTRAINT_FIELDS},
        },
        "portfolio": {
            "cash": portfolio.get("cash"),
            "equity": portfolio.get("equity"),
            "positions": sorted(
                (_pick(p, _POSITION_FIELDS) for p in portfolio.get("positions") or [] if isinstance(p, dict)),
                key=lambda p: str(p.get("symbol")),
            ),
        },
        "market": {
            "bars": {
                sym: _pick(bars[sym], _BAR_FIELDS)
                for sym in allowed
                if isinstance(bars.get(sym), dict) and _not_after(bars[sym].get("timestamp"), step_time)
            },
            "features": {
                sym: _pick(features[sym], _FEATURE_FIELDS)
                for sym in allowed
                if isinstance(features.get(sym), dict)
            },
        },
    }
    return json.dumps(_finite(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pick(row: Dict[str, Any], fields) -> Dict[str, Any]:
    return {k: row[k] for k in fields if k in row}


def parse_iso_time(value: Any) -> Optional[datetime]:
    """A timezone-aware datetime, or None for anything unparseable or naive."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _not_after(bar_time: Any, step_time: Optional[datetime]) -> bool:
    """True only when the bar is provably at or before the step.

    An unparseable or naive timestamp on either side fails closed (the bar is
    dropped): a prompt missing a bar is visible, a leaked future bar is not.
    """
    parsed = parse_iso_time(bar_time)
    return step_time is not None and parsed is not None and parsed <= step_time


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value
