"""``VLLMTradingAgent``: one model, one decision per ATL step, inside the step's deadline.

Per step: build the deterministic prompt, call the model within the time left
before ``deadline_at`` (less a safety margin for the submission itself), and
read the reply strictly. Anything short of a valid reply (no time left, a
timeout, a dead server, an empty or malformed answer, an order the step does
not allow) becomes an explicit HOLD: an empty ``orders`` list whose ``trace``
names the reason. An inference failure therefore never raises into the run
and never produces a trade.

``decide`` is synchronous and every request is bounded by its timeout, so no
reply can outlive its step and be read by the next one; a reply that does
return after the deadline is discarded rather than submitted. ATL is the
second fence: it pins every decision to its own ``step_id`` and refuses one
for an auto-held step (``decision_deadline_exceeded``).

Retries are bounded by ``config.max_retries`` and spent only on transient
transport failures. A timeout is not retried (the budget is gone), and nor is
an unusable reply (greedy decoding would return it again).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..models import Decision, Observation, Step
from ._tradingagents_core import sanitize_error_message, sha256_text
from ._vllm_client import ChatErrorKind, ChatResult, ChatTransportError, VLLMChatClient
from ._vllm_config import PARSER_VERSION
from ._vllm_parser import DecisionRejected, RejectionReason, parse_decision
from ._vllm_prompt import PROMPT_VERSION, build_messages, parse_iso_time


class DecisionStatus:
    VALID = "valid"          # the model's reply, submitted as given
    REJECTED = "rejected"    # a reply arrived but was unusable -> HOLD
    FAILED = "failed"        # no usable reply (deadline, timeout, transport) -> HOLD


_ERROR_REASONS = {
    ChatErrorKind.TIMEOUT: RejectionReason.TIMEOUT,
    ChatErrorKind.CONNECTION: RejectionReason.CONNECTION_ERROR,
    ChatErrorKind.HTTP_STATUS: RejectionReason.HTTP_ERROR,
    ChatErrorKind.BAD_RESPONSE: RejectionReason.BAD_RESPONSE,
}


@dataclass
class DecisionRecord:
    """Telemetry for one step: what was asked, how inference went, what was sent."""

    timestamp: str
    run_id: Optional[str]
    step_id: Optional[str]
    sequence: Optional[int]
    model_id: str
    model_revision: Optional[str]
    config_hash: str
    prompt_sha256: str
    status: str
    reason: Optional[str]
    action: str
    orders: int
    attempts: int
    timed_out: bool
    parsed: bool
    latency_seconds: float
    inference_seconds: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    finish_reason: Optional[str]
    response_model: Optional[str]
    response_sha256: Optional[str]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLLMTradingAgent:
    """An ``AgentRunner`` agent backed by an OpenAI-compatible vLLM endpoint."""

    def __init__(self, client: VLLMChatClient, *, deadline_margin_seconds: float = 5.0,
                 min_request_seconds: float = 1.0, wall_clock: Callable[[], float] = time.time,
                 sink: Optional[Callable[[DecisionRecord], None]] = None) -> None:
        self.client = client
        self.deadline_margin_seconds = deadline_margin_seconds
        self.min_request_seconds = min_request_seconds
        self._wall = wall_clock
        self._sink = sink
        self._step: Optional[Step] = None
        self.records: List[DecisionRecord] = []

    def on_step(self, step: Step) -> None:
        self._step = step

    def decide(self, observation: Observation) -> Decision:
        step = self._step
        self._step = None
        if step is None:
            raise RuntimeError("VLLMTradingAgent needs the step: run it under AgentRunner (on_step hook)")
        if step.observation is not observation:
            raise RuntimeError("decide() got an observation from a different step than on_step()")

        config = self.client.config
        started = self.client.monotonic()
        messages = build_messages(step)
        prompt_sha = sha256_text(json.dumps(messages, sort_keys=True, separators=(",", ":")))
        deadline = parse_iso_time(step.deadline_at)
        submit_by = deadline.timestamp() - self.deadline_margin_seconds if deadline else None

        result: Optional[ChatResult] = None
        failure: Optional[ChatTransportError] = None
        reason: Optional[str] = None
        attempts = 0
        timed_out = False
        error: Optional[str] = None
        while attempts <= config.max_retries:
            timeout = config.request_timeout
            if submit_by is not None:
                timeout = min(timeout, submit_by - self._wall())
            if timeout < self.min_request_seconds:
                reason, timed_out = RejectionReason.DEADLINE_EXCEEDED, True
                break
            attempts += 1
            try:
                result = self.client.complete(messages, timeout=timeout)
                failure, error = None, None
            except ChatTransportError as exc:
                failure, error = exc, str(exc)
                if exc.retryable and attempts <= config.max_retries:
                    continue
                reason = _ERROR_REASONS[exc.kind]
                timed_out = exc.kind == ChatErrorKind.TIMEOUT
                break
            except Exception as exc:  # a transport bug must not end the run; it holds and is logged
                error = sanitize_error_message(exc)
                reason = RejectionReason.INTERNAL_ERROR
                break
            if submit_by is not None and self._wall() > submit_by:
                reason, timed_out = RejectionReason.DEADLINE_EXCEEDED, True
            break

        decision: Optional[Decision] = None
        parsed = False
        if result is not None and reason is None:
            try:
                decision = parse_decision(result.content, step)
                parsed = True
            except DecisionRejected as exc:
                reason, error = exc.reason, exc.detail or None
            except Exception as exc:  # e.g. an unreadable position in the step: hold and log, never crash the run
                reason, error = RejectionReason.INTERNAL_ERROR, sanitize_error_message(exc)
        if decision is not None:
            status = DecisionStatus.VALID
        elif result is not None and not timed_out:
            status = DecisionStatus.REJECTED
        else:
            status = DecisionStatus.FAILED

        trace = {
            "adapter": "vllm",
            "status": status,
            "reason": reason,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "config_hash": config.config_hash(),
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "prompt_sha256": prompt_sha,
            "response_sha256": result.response_sha256 if result else None,
        }
        if decision is None:
            decision = Decision(orders=[], confidence=None, rationale=f"vllm adapter hold: {reason}")
        decision.trace = trace

        record = DecisionRecord(
            timestamp=datetime.fromtimestamp(self._wall(), tz=timezone.utc).isoformat(),
            run_id=step.run_id,
            step_id=step.step_id,
            sequence=step.sequence,
            model_id=config.model_id,
            model_revision=config.model_revision,
            config_hash=trace["config_hash"],
            prompt_sha256=prompt_sha,
            status=status,
            reason=reason,
            action=_action(decision),
            orders=len(decision.orders),
            attempts=attempts,
            timed_out=timed_out,
            parsed=parsed,
            latency_seconds=self.client.monotonic() - started,
            inference_seconds=result.latency_seconds if result else (failure.latency_seconds if failure else None),
            prompt_tokens=result.prompt_tokens if result else None,
            completion_tokens=result.completion_tokens if result else None,
            finish_reason=result.finish_reason if result else None,
            response_model=result.model if result else None,
            response_sha256=result.response_sha256 if result else None,
            error=error,
        )
        self.records.append(record)
        if self._sink is not None:
            self._sink(record)
        return decision


def _action(decision: Decision) -> str:
    sides = {o.side for o in decision.orders}
    if not sides:
        return "HOLD"
    if sides == {"buy"}:
        return "BUY"
    if sides == {"sell"}:
        return "SELL"
    return "MIXED"
