"""Shared builders for the vLLM integration tests: steps, chat replies, a fake clock."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from agentictrading import Step

STEP_TS = "2026-04-15T11:00:00-04:00"
DEADLINE_TS = "2026-04-15T11:01:00-04:00"
# Wall-clock seconds for STEP_TS / DEADLINE_TS (the fake clock starts at the step).
STEP_EPOCH = 1776265200.0
DEADLINE_EPOCH = STEP_EPOCH + 60.0

_STEP: Dict[str, Any] = {
    "protocol_version": "1.0",
    "run_id": "run_1",
    "step_id": "step_1",
    "sequence": 0,
    "timestamp": STEP_TS,
    "deadline_at": DEADLINE_TS,
    "status": "awaiting_decision",
    "observation": {
        "market": {
            "bars": {
                "AAPL": {"timestamp": STEP_TS, "open": 100.0, "high": 101.0,
                         "low": 99.0, "close": 100.5, "volume": 1000000.0},
                "MSFT": {"timestamp": STEP_TS, "open": 200.0, "high": 201.0,
                         "low": 199.0, "close": 200.5, "volume": 2000000.0},
            },
            "features": {
                "AAPL": {"price": 100.5, "rsi": 55.0, "macd": 0.1, "macd_signal": 0.05,
                         "sma20": 99.0, "sma50": 98.0, "bb_upper": 103.0, "bb_lower": 96.0},
                "MSFT": {"price": 200.5, "rsi": 45.0, "macd": -0.1, "macd_signal": -0.05,
                         "sma20": 201.0, "sma50": 202.0, "bb_upper": 205.0, "bb_lower": 196.0},
            },
            "events": [],
        },
        "portfolio": {
            "cash": 600.0,
            "equity": 1000.0,
            "positions": [
                {"symbol": "MSFT", "quantity": 2, "entry_price": 199.0, "current_price": 200.5,
                 "market_value": 401.0, "unrealized_pnl_pct": 0.0075},
            ],
        },
    },
    "constraints": {
        "allowed_symbols": ["AAPL", "MSFT"],
        "allow_short": False,
        "max_position_weight": 0.25,
        "max_orders": 10,
    },
}


def step_payload(**overrides: Any) -> Dict[str, Any]:
    payload = copy.deepcopy(_STEP)
    payload.update(overrides)
    return payload


def make_step(**overrides: Any) -> Step:
    return Step.from_dict(step_payload(**overrides))


def chat_body(content: Optional[str], *, prompt_tokens: int = 420, completion_tokens: int = 24,
              finish_reason: str = "stop", model: str = "Qwen/Qwen2.5-3B-Instruct") -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def decision_json(orders: List[Dict[str, Any]], confidence: Any = 0.6,
                  rationale: str = "test rationale") -> str:
    return json.dumps({"orders": orders, "confidence": confidence, "rationale": rationale})


class FakeClock:
    """One clock for both monotonic and wall time, advanced by fake transports."""

    def __init__(self, start: float = STEP_EPOCH) -> None:
        self.now = start
        self._mono = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self._mono += seconds

    def wall(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self._mono
