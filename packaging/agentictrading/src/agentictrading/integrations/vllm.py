"""Public facade for the vLLM (OpenAI-compatible) decision adapter.

Drive an ATL protocol backtest with a model served by vLLM, using nothing but
the standard library::

    from agentictrading import AgentRunner, ATLClient
    from agentictrading.integrations.vllm import VLLMChatClient, VLLMConfig, VLLMTradingAgent

    agent = VLLMTradingAgent(VLLMChatClient(VLLMConfig.from_env()))
    AgentRunner(ATLClient(atl_url, api_key), agent).run_backtest(...)
    # agent.records: per-step telemetry; summarize_records(agent.records)

The model identity is configuration only (``VLLM_MODEL_ID`` names whatever the
server serves, including a LoRA adapter registered under its own name), so a
base model and a fine-tuned variant run through the identical prompt, parser
and ATL interface.
"""

from ._vllm_agent import DecisionRecord, DecisionStatus, VLLMTradingAgent
from ._vllm_client import (
    ChatErrorKind,
    ChatResult,
    ChatTransportError,
    MockChatTransport,
    VLLMChatClient,
    urllib_transport,
)
from ._vllm_config import (
    ATL_DECISION_TIMEOUT_SECONDS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    PARSER_VERSION,
    VLLMConfig,
)
from ._vllm_metrics import summarize_records
from ._vllm_parser import DecisionRejected, RejectionReason, parse_decision
from ._vllm_prompt import (
    DECISION_JSON_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_messages,
    serialize_snapshot,
)

__all__ = [
    "ATL_DECISION_TIMEOUT_SECONDS",
    "ChatErrorKind",
    "ChatResult",
    "ChatTransportError",
    "DECISION_JSON_SCHEMA",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DecisionRecord",
    "DecisionRejected",
    "DecisionStatus",
    "MockChatTransport",
    "PARSER_VERSION",
    "PROMPT_VERSION",
    "RejectionReason",
    "SYSTEM_PROMPT",
    "VLLMChatClient",
    "VLLMConfig",
    "VLLMTradingAgent",
    "build_messages",
    "parse_decision",
    "serialize_snapshot",
    "summarize_records",
    "urllib_transport",
]
