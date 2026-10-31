"""Minimal OpenAI-compatible chat client for a vLLM server (standard library only).

One ``complete`` call is one HTTP request: no SDK, no hidden retries, no
streaming. The caller passes the timeout for that request (the agent derives
it from the step deadline) and gets either a ``ChatResult`` or a classified
``ChatTransportError`` saying whether a repeat could help.

Requests never carry ``tools``/``tool_choice``: the reply is read as text and
parsed JSON-or-nothing, the same boundary ATL's own LLM paths keep
(``CLAUDE.md`` § LLM safety boundary).
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit, urlunsplit

from ._tradingagents_core import sanitize_error_message, sha256_text
from ._vllm_config import VLLMConfig
from ._vllm_prompt import DECISION_JSON_SCHEMA

# transport(method, url, body_or_None, headers, timeout) -> (status, body_bytes)
Transport = Callable[[str, str, Optional[bytes], Dict[str, str], float], Tuple[int, bytes]]


class ChatErrorKind:
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    HTTP_STATUS = "http_status"
    BAD_RESPONSE = "bad_response"


class ChatTransportError(Exception):
    def __init__(self, kind: str, message: str, *, status: Optional[int] = None,
                 retryable: bool = False, latency_seconds: Optional[float] = None) -> None:
        super().__init__(sanitize_error_message(message))
        self.kind = kind
        self.status = status
        self.retryable = retryable
        self.latency_seconds = latency_seconds


@dataclass(frozen=True)
class ChatResult:
    content: Optional[str]
    finish_reason: Optional[str]
    model: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_seconds: float
    response_sha256: str


def urllib_transport(method: str, url: str, body: Optional[bytes],
                     headers: Dict[str, str], timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp is not None else b""


class VLLMChatClient:
    def __init__(self, config: VLLMConfig, *, transport: Optional[Transport] = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._transport = transport or urllib_transport
        self.monotonic = monotonic

    def request_body(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        c = self.config
        return {
            "model": c.model_id,
            "messages": list(messages),
            "temperature": c.temperature,
            "top_p": c.top_p,
            "seed": c.seed,
            "max_completion_tokens": c.max_output_tokens,
            "n": 1,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "atl_decision", "schema": DECISION_JSON_SCHEMA, "strict": True},
            },
        }

    def complete(self, messages: Sequence[Dict[str, str]], *, timeout: float) -> ChatResult:
        if not timeout > 0:
            raise ValueError("timeout must be positive")
        body = json.dumps(self.request_body(messages), sort_keys=True).encode("utf-8")
        start = self.monotonic()
        status, raw = self._send("POST", self.config.base_url + "/chat/completions", body, timeout)
        latency = self.monotonic() - start
        if status != 200:
            raise ChatTransportError(ChatErrorKind.HTTP_STATUS, f"HTTP {status}", status=status,
                                     retryable=status == 429 or status >= 500, latency_seconds=latency)
        return _read_completion(raw, latency)

    def list_models(self, *, timeout: float) -> List[str]:
        status, raw = self._send("GET", self.config.base_url + "/models", None, timeout)
        if status != 200:
            raise ChatTransportError(ChatErrorKind.HTTP_STATUS, f"HTTP {status}", status=status)
        try:
            return [m["id"] for m in json.loads(raw)["data"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise ChatTransportError(ChatErrorKind.BAD_RESPONSE, "unreadable /models body") from exc

    def health(self, *, timeout: float) -> bool:
        """vLLM's unauthenticated liveness route, on the server root (not /v1)."""
        parts = urlsplit(self.config.base_url)
        root = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        status, _ = self._send("GET", root + "/health", None, timeout)
        return status == 200

    def _send(self, method: str, url: str, body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            return self._transport(method, url, body, headers, timeout)
        except (socket.timeout, TimeoutError) as exc:
            raise ChatTransportError(ChatErrorKind.TIMEOUT, f"timed out after {timeout:.1f}s") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
                raise ChatTransportError(ChatErrorKind.TIMEOUT, f"timed out after {timeout:.1f}s") from exc
            raise ChatTransportError(ChatErrorKind.CONNECTION, f"connection error: {reason}", retryable=True) from exc
        except (ConnectionError, OSError) as exc:
            raise ChatTransportError(ChatErrorKind.CONNECTION, f"connection error: {exc}", retryable=True) from exc


def _read_completion(raw: bytes, latency: float) -> ChatResult:
    try:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ChatTransportError(ChatErrorKind.BAD_RESPONSE, "unreadable chat completion body",
                                 latency_seconds=latency) from exc
    if content is not None and not isinstance(content, str):
        raise ChatTransportError(ChatErrorKind.BAD_RESPONSE, "message content is not text", latency_seconds=latency)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return ChatResult(
        content=content,
        finish_reason=choice.get("finish_reason"),
        model=payload.get("model"),
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        latency_seconds=latency,
        response_sha256=sha256_text(raw.decode("utf-8", errors="replace")),
    )


def _count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


Reply = Union[str, Tuple[int, bytes], BaseException]


class MockChatTransport:
    """Offline stand-in for a vLLM server: replays scripted replies in order.

    A ``str`` reply is wrapped as a chat completion with fixed token counts, a
    ``(status, body)`` tuple is returned as-is, and an exception is raised as
    the network would. ``delays`` plus ``advance`` move a fake clock per call,
    so deadline behaviour is testable without sleeping. Every request body and
    timeout is recorded.
    """

    def __init__(self, replies: Sequence[Reply], *, delays: Optional[Sequence[float]] = None,
                 advance: Optional[Callable[[float], None]] = None,
                 prompt_tokens: int = 420, completion_tokens: int = 24) -> None:
        self._replies = list(replies)
        self._delays = list(delays or [])
        self._advance = advance
        self._usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                       "total_tokens": prompt_tokens + completion_tokens}
        self.requests: List[Dict[str, Any]] = []
        self.timeouts: List[float] = []

    def __call__(self, method: str, url: str, body: Optional[bytes],
                 headers: Dict[str, str], timeout: float) -> Tuple[int, bytes]:
        self.requests.append(json.loads(body) if body else {"method": method, "url": url})
        self.timeouts.append(timeout)
        if self._advance is not None and self._delays:
            self._advance(self._delays.pop(0))
        if not self._replies:
            raise AssertionError("MockChatTransport ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, tuple):
            return reply
        model = self.requests[-1].get("model")
        return 200, json.dumps({
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            "usage": self._usage,
        }).encode("utf-8")
