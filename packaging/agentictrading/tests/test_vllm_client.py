"""OpenAI-compatible chat client: request shape, auth, usage/latency, error mapping, config."""

from __future__ import annotations

import json
import socket
import urllib.error

import pytest

from agentictrading.integrations.vllm import (
    ChatErrorKind,
    ChatTransportError,
    DECISION_JSON_SCHEMA,
    VLLMChatClient,
    VLLMConfig,
)

from _vllm_helpers import FakeClock, chat_body


class RecordingTransport:
    def __init__(self, *results, clock=None, delay=0.0):
        self.results = list(results)
        self.calls = []
        self.clock = clock
        self.delay = delay

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append({"method": method, "url": url, "body": body, "headers": headers, "timeout": timeout})
        if self.clock is not None:
            self.clock.advance(self.delay)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _client(transport, clock=None, **cfg):
    clock = clock or FakeClock()
    config = VLLMConfig(**{"base_url": "http://gpu.internal:8000/v1", **cfg})
    return VLLMChatClient(config, transport=transport, monotonic=clock.monotonic)


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_request_is_deterministic_json_schema_chat_completion():
    t = RecordingTransport((200, json.dumps(chat_body('{"orders": []}')).encode()))
    _client(t).complete(MESSAGES, timeout=10)
    call = t.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://gpu.internal:8000/v1/chat/completions"
    body = json.loads(call["body"])
    assert body["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert body["messages"] == MESSAGES
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["seed"] == 0
    assert body["max_completion_tokens"] == 256
    assert body["stream"] is False and body["n"] == 1
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == DECISION_JSON_SCHEMA
    # The ATL LLM boundary: no tools, ever.
    assert "tools" not in body and "tool_choice" not in body
    assert call["timeout"] == 10


def test_api_key_is_sent_as_bearer_and_never_repr_or_public():
    t = RecordingTransport((200, json.dumps(chat_body("{}")).encode()))
    client = _client(t, api_key="sk-local-secret")
    client.complete(MESSAGES, timeout=5)
    assert t.calls[0]["headers"]["Authorization"] == "Bearer sk-local-secret"
    assert "sk-local-secret" not in repr(client.config)
    assert "sk-local-secret" not in json.dumps(client.config.public_dict())
    assert client.config.public_dict()["api_key_set"] is True


def test_no_auth_header_without_a_key():
    t = RecordingTransport((200, json.dumps(chat_body("{}")).encode()))
    _client(t).complete(MESSAGES, timeout=5)
    assert "Authorization" not in t.calls[0]["headers"]


def test_usage_latency_and_hashes_are_collected():  # K
    clock = FakeClock()
    t = RecordingTransport((200, json.dumps(chat_body('{"orders": []}', prompt_tokens=400, completion_tokens=20)).encode()),
                           clock=clock, delay=1.25)
    r = _client(t, clock=clock).complete(MESSAGES, timeout=5)
    assert r.content == '{"orders": []}'
    assert r.prompt_tokens == 400 and r.completion_tokens == 20
    assert r.latency_seconds == pytest.approx(1.25)
    assert r.finish_reason == "stop"
    assert r.model == "Qwen/Qwen2.5-3B-Instruct"
    assert len(r.response_sha256) == 64


def test_missing_usage_is_none_not_zero():  # K
    body = chat_body("{}")
    del body["usage"]
    t = RecordingTransport((200, json.dumps(body).encode()))
    r = _client(t).complete(MESSAGES, timeout=5)
    assert r.prompt_tokens is None and r.completion_tokens is None


@pytest.mark.parametrize("exc,kind,retryable", [
    (socket.timeout("timed out"), ChatErrorKind.TIMEOUT, False),
    (TimeoutError("timed out"), ChatErrorKind.TIMEOUT, False),
    (urllib.error.URLError(socket.timeout("timed out")), ChatErrorKind.TIMEOUT, False),
    (urllib.error.URLError(ConnectionRefusedError(61, "refused")), ChatErrorKind.CONNECTION, True),
    (ConnectionResetError(54, "reset"), ChatErrorKind.CONNECTION, True),
])
def test_transport_failures_are_classified(exc, kind, retryable):  # J
    t = RecordingTransport(exc)
    with pytest.raises(ChatTransportError) as ei:
        _client(t).complete(MESSAGES, timeout=5)
    assert ei.value.kind == kind and ei.value.retryable is retryable


@pytest.mark.parametrize("status,retryable", [(500, True), (503, True), (429, True), (400, False), (401, False), (404, False)])
def test_http_errors_are_classified(status, retryable):  # J
    t = RecordingTransport((status, b'{"error": "x"}'))
    with pytest.raises(ChatTransportError) as ei:
        _client(t).complete(MESSAGES, timeout=5)
    assert ei.value.kind == ChatErrorKind.HTTP_STATUS
    assert ei.value.status == status and ei.value.retryable is retryable


@pytest.mark.parametrize("raw", [b"<html>", b"{}", b'{"choices": []}', b'{"choices": [{"message": 3}]}'])
def test_unusable_response_bodies_are_bad_response(raw):
    t = RecordingTransport((200, raw))
    with pytest.raises(ChatTransportError) as ei:
        _client(t).complete(MESSAGES, timeout=5)
    assert ei.value.kind == ChatErrorKind.BAD_RESPONSE and ei.value.retryable is False


def test_nonpositive_timeout_is_refused_before_any_request():
    t = RecordingTransport()
    with pytest.raises(ValueError):
        _client(t).complete(MESSAGES, timeout=0)
    assert t.calls == []


def test_list_models_and_health_use_the_server_root():
    t = RecordingTransport((200, b""), (200, json.dumps({"data": [{"id": "Qwen/Qwen2.5-3B-Instruct"}]}).encode()))
    client = _client(t)
    assert client.health(timeout=2) is True
    assert client.list_models(timeout=2) == ["Qwen/Qwen2.5-3B-Instruct"]
    assert t.calls[0]["url"] == "http://gpu.internal:8000/health"
    assert t.calls[1]["url"] == "http://gpu.internal:8000/v1/models"


# -- config -------------------------------------------------------------


def test_config_defaults_are_documented_and_local():
    c = VLLMConfig.from_env({})
    assert c.base_url == "http://127.0.0.1:8000/v1"
    assert c.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert c.api_key is None
    assert c.request_timeout == 30.0
    assert c.max_output_tokens == 256
    assert c.temperature == 0.0


def test_config_reads_the_five_env_vars():
    c = VLLMConfig.from_env({
        "VLLM_BASE_URL": "http://10.0.0.5:9000/v1/", "VLLM_MODEL_ID": "my/model",
        "VLLM_API_KEY": "k", "VLLM_REQUEST_TIMEOUT": "12.5", "VLLM_MAX_OUTPUT_TOKENS": "128",
        "VLLM_MODEL_REVISION": "abc123",
    })
    assert (c.base_url, c.model_id, c.api_key, c.request_timeout, c.max_output_tokens, c.model_revision) == (
        "http://10.0.0.5:9000/v1", "my/model", "k", 12.5, 128, "abc123")


@pytest.mark.parametrize("env", [
    {"VLLM_REQUEST_TIMEOUT": "0"},
    {"VLLM_REQUEST_TIMEOUT": "60"},
    {"VLLM_REQUEST_TIMEOUT": "abc"},
    {"VLLM_MAX_OUTPUT_TOKENS": "0"},
    {"VLLM_BASE_URL": "ftp://x"},
    {"VLLM_MODEL_ID": " "},
])
def test_bad_config_fails_fast(env):
    with pytest.raises(ValueError):
        VLLMConfig.from_env(env)


def test_config_hash_tracks_every_frozen_field_but_not_the_key():
    base = VLLMConfig()
    assert base.config_hash() == VLLMConfig().config_hash()
    assert VLLMConfig(api_key="x").config_hash() == base.config_hash()
    for change in ({"model_id": "m"}, {"model_revision": "r"}, {"temperature": 0.1},
                   {"max_output_tokens": 100}, {"seed": 1}, {"request_timeout": 20.0}):
        assert VLLMConfig(**change).config_hash() != base.config_hash()
