"""Frozen, hashable configuration for the vLLM decision adapter.

Everything that can change what the model is asked, or how its reply is read,
is a field here and feeds ``config_hash()``: two runs with the same hash sent
the same request shape to the same model identity. The API key is the one
exception: it authenticates, it never changes a decision, and it never appears
in ``repr``, ``public_dict`` or the hash.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

from ._tradingagents_core import sha256_text

ENV_BASE_URL = "VLLM_BASE_URL"
ENV_MODEL_ID = "VLLM_MODEL_ID"
ENV_MODEL_REVISION = "VLLM_MODEL_REVISION"
ENV_API_KEY = "VLLM_API_KEY"
ENV_REQUEST_TIMEOUT = "VLLM_REQUEST_TIMEOUT"
ENV_MAX_OUTPUT_TOKENS = "VLLM_MAX_OUTPUT_TOKENS"

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# Hugging Face commit of Qwen/Qwen2.5-3B-Instruct the Stage 1 experiment pins
# (weights and tokenizer share the repo, so one revision covers both).
DEFAULT_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_MAX_OUTPUT_TOKENS = 256

# ATL's protocol decision window (EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS,
# default 60). One request may never be allowed to outlast it; the per-step
# budget in the agent then caps each attempt by the step's own deadline_at.
ATL_DECISION_TIMEOUT_SECONDS = 60.0

PARSER_VERSION = "vllm-orders-v1-strict.1"


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION
    api_key: Optional[str] = field(default=None, repr=False)
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    # Greedy decoding: temperature 0 picks the argmax token, so top_p is moot
    # and pinned at 1.0. GPU kernels are still not bit-identical across
    # hardware or batch shapes, so equal hashes promise the same request, not
    # the same reply.
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    # Extra attempts after a transient transport failure (connection refused or
    # reset, HTTP 429/5xx). Never after a timeout or an unusable reply.
    max_retries: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", (self.base_url or "").strip().rstrip("/"))
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"{ENV_BASE_URL} must be an http(s) URL, got {self.base_url!r}")
        if not (self.model_id or "").strip():
            raise ValueError(f"{ENV_MODEL_ID} must be non-empty")
        if not 0 < self.request_timeout < ATL_DECISION_TIMEOUT_SECONDS:
            raise ValueError(
                f"{ENV_REQUEST_TIMEOUT} must be in (0, {ATL_DECISION_TIMEOUT_SECONDS:g}) seconds, "
                f"below ATL's decision window; got {self.request_timeout}"
            )
        if self.max_output_tokens <= 0:
            raise ValueError(f"{ENV_MAX_OUTPUT_TOKENS} must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "VLLMConfig":
        env = os.environ if env is None else env
        return cls(
            base_url=env.get(ENV_BASE_URL, DEFAULT_BASE_URL),
            model_id=env.get(ENV_MODEL_ID, DEFAULT_MODEL_ID),
            model_revision=env.get(ENV_MODEL_REVISION, DEFAULT_MODEL_REVISION) or None,
            api_key=env.get(ENV_API_KEY) or None,
            request_timeout=_number(env, ENV_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT, float),
            max_output_tokens=_number(env, ENV_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS, int),
        )

    def public_dict(self) -> Dict[str, Any]:
        """Every frozen field plus versions, with the key reduced to a flag."""
        from ._vllm_prompt import PROMPT_VERSION

        return {
            "base_url": self.base_url,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "api_key_set": self.api_key is not None,
            "request_timeout": self.request_timeout,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_retries": self.max_retries,
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
        }

    def config_hash(self) -> str:
        """SHA-256 of everything that shapes a request or its reading.

        ``base_url`` and the key are left out: where the server listens and
        how it authenticates do not change what the model is asked.
        """
        from ._vllm_prompt import DECISION_JSON_SCHEMA, SYSTEM_PROMPT

        frozen = self.public_dict()
        frozen.pop("base_url")
        frozen.pop("api_key_set")
        frozen["system_prompt"] = SYSTEM_PROMPT
        frozen["decision_json_schema"] = DECISION_JSON_SCHEMA
        return sha256_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")))


def _number(env: Mapping[str, str], name: str, default: Any, kind: Any) -> Any:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
