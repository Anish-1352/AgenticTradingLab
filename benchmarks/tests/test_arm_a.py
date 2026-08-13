"""Tests for arm A, the pooling comparison, the rate-limit probe and the model.

**Nothing here touches the network or spends money.** The HTTP client is a mock
that returns canned responses and headers; the one test that would hit a real
endpoint asserts instead that the runner *refuses* to start without a key.

The most important tests are the ones that pin the money-safety properties:
the API key never reaches a manifest, and an over-budget sweep is refused
before a single request goes out.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import arm_a_vs_bc_model as model  # noqa: E402
from analysis import context_pooling as pooling  # noqa: E402
from analysis import rate_limit_probe as probe  # noqa: E402
from common.metrics import RequestRecord  # noqa: E402
from runners import bench_api_baseline as arma  # noqa: E402

KEY = "sk-or-v1-THIS-IS-NOT-A-REAL-KEY-0123456789"


# --------------------------------------------------------------------------
# money safety
# --------------------------------------------------------------------------


def test_api_key_is_hashed_never_stored():
    h = arma.api_key_hash(KEY)
    assert h and len(h) == 64
    assert KEY not in h
    # The hash must not be reversible to any fragment of the key.
    assert "sk-or" not in h


def test_api_key_hash_is_stable_and_distinguishes_keys():
    assert arma.api_key_hash(KEY) == arma.api_key_hash(KEY)
    assert arma.api_key_hash(KEY) != arma.api_key_hash(KEY + "x")
    assert arma.api_key_hash(None) is None


def test_dry_run_submits_nothing(monkeypatch, capsys, tmp_path):
    """--dry-run must not construct a client or read a key."""
    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("dry-run must not issue requests")

    monkeypatch.setattr(arma, "run_level", explode)
    monkeypatch.setattr(arma, "load_api_key", explode)
    rc = arma.main([
        "--dry-run", "--stub-tokenizer", "--n-requests", "2",
        "--fixtures-dir", str(tmp_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing submitted, nothing billed" in out
    assert "ESTIMATED COST" in out


def test_refuses_without_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = arma.main([
        "--stub-tokenizer", "--n-requests", "1", "--concurrency", "1",
        "--fixtures-dir", str(tmp_path),
    ])
    assert rc == 2
    assert "is not set" in capsys.readouterr().err


def test_refuses_when_estimate_exceeds_budget(monkeypatch, tmp_path, capsys):
    """A mistyped sweep must be stopped before it bills anything."""
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("must not run when over budget")

    monkeypatch.setattr(arma, "run_level", explode)
    rc = arma.main([
        "--stub-tokenizer", "--n-requests", "100",
        "--concurrency", "1", "8", "32", "64", "128",
        "--max-cost-usd", "0.0001", "--fixtures-dir", str(tmp_path),
    ])
    assert rc == 3
    assert "REFUSING TO START" in capsys.readouterr().err


# --------------------------------------------------------------------------
# cost arithmetic
# --------------------------------------------------------------------------


def test_request_cost_uses_billed_tokens():
    # 2620 in @ $0.05/M + 256 out @ $0.20/M
    cost = arma.request_cost(2620, 256, 0.05, 0.20)
    assert cost == pytest.approx(2620 / 1e6 * 0.05 + 256 / 1e6 * 0.20)
    assert cost == pytest.approx(0.0001822)


def test_cost_scales_linearly():
    assert arma.request_cost(5240, 512) == pytest.approx(
        2 * arma.request_cost(2620, 256)
    )


def test_parse_usage_prefers_provider_numbers():
    assert arma.parse_usage({"usage": {"prompt_tokens": 3011,
                                       "completion_tokens": 256}}) == (3011, 256)
    # Absent usage must read as zero, never be back-filled from the fixture.
    assert arma.parse_usage({}) == (0, 0)


def test_estimate_is_labelled_as_an_estimate():
    est = arma.estimate_cost(5, 32, 2620, 256, 0.05, 0.20)
    assert est["total_requests"] == 160
    assert "Estimate only" in est["basis"]
    assert "own tokenisation" in est["basis"]


# --------------------------------------------------------------------------
# header parsing
# --------------------------------------------------------------------------


class FakeHeaders(dict):
    pass


def test_extract_headers_captures_rate_limit_and_timing():
    got = arma.extract_headers(FakeHeaders({
        "X-RateLimit-Remaining": "17", "Retry-After": "3",
        "X-OpenRouter-Processing-Ms": "812.5", "Content-Type": "application/json",
    }))
    assert got["x-ratelimit-remaining"] == "17"
    assert got["retry-after"] == "3"
    assert "content-type" not in got  # irrelevant headers dropped


def test_server_processing_and_network_split():
    hdrs = {"x-openrouter-processing-ms": "800"}
    assert arma.server_processing_ms(hdrs) == pytest.approx(800.0)
    assert arma.server_processing_ms({"server-timing": "inbound;dur=123.4"}) == \
        pytest.approx(123.4)
    assert arma.server_processing_ms({}) is None


def test_retry_after_parsing():
    assert arma.retry_after_seconds({"retry-after": "5"}) == 5.0
    assert arma.retry_after_seconds({"retry-after": "not-a-number"}) is None
    assert arma.retry_after_seconds({}) is None


# --------------------------------------------------------------------------
# request behaviour, fully mocked
# --------------------------------------------------------------------------


class MockResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {
            "usage": {"prompt_tokens": 3011, "completion_tokens": 256}
        }
        self.headers = FakeHeaders(headers or {})

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = Exception(f"HTTP {self.status_code}")
            err.response = self
            raise err


class MockClient:
    """Records calls; returns a scripted sequence of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        r = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        return r

    async def aclose(self):
        pass


def _run_one(client, **over):
    kwargs = dict(
        client=client, url="https://example.invalid/v1", prompt_text="hello",
        request_id="r0", model="m", max_tokens=256, temperature=0.0,
        ignore_eos=True, api_key=KEY, sem=asyncio.Semaphore(1), stream=False,
        timeout=5.0, max_retries=3,
    )
    kwargs.update(over)
    return asyncio.run(arma.run_one(**kwargs))


def test_successful_request_records_billed_tokens_and_cost():
    client = MockClient([MockResponse(headers={"x-openrouter-processing-ms": "700"})])
    rec, meta = _run_one(client)
    assert rec.ok
    assert rec.prompt_tokens == 3011   # provider's count, not the fixture's 2620
    assert rec.output_tokens == 256
    assert meta["cost_usd"] == pytest.approx(arma.request_cost(3011, 256))
    assert meta["server_processing_ms"] == pytest.approx(700.0)
    assert meta["network_latency_estimated_ms"] is not None


def test_429_is_retried_then_succeeds():
    client = MockClient([
        MockResponse(status=429, headers={"Retry-After": "0"}),
        MockResponse(status=200),
    ])
    rec, meta = _run_one(client)
    assert rec.ok
    assert meta["rate_limited"] == 1
    assert meta["attempts"] == 2


def test_persistent_429_gives_up_and_is_recorded_not_hidden():
    client = MockClient([MockResponse(status=429, headers={"Retry-After": "0"})])
    rec, meta = _run_one(client, max_retries=2)
    assert not rec.ok
    assert rec.error == "RATE_LIMITED"
    assert meta["rate_limited"] == 2


def test_401_is_fatal_and_not_retried():
    """Auth failures will not fix themselves; retrying just wastes time."""
    client = MockClient([MockResponse(status=401)])
    rec, meta = _run_one(client, max_retries=5)
    assert not rec.ok
    assert rec.error.startswith("FATAL_401")
    assert client.calls == 1


def test_500_is_retried():
    client = MockClient([MockResponse(status=500), MockResponse(status=200)])
    rec, _ = _run_one(client)
    assert rec.ok
    assert client.calls == 2


def test_non_streaming_collapses_ttft_into_e2e():
    """Without a token stream there is no first-token signal to measure.

    TTFT and e2e are stamped a few lines apart — the response lands, then usage
    is parsed — so they differ by microseconds of JSON work, not by a decode
    phase. The property that matters is that the gap is negligible and there
    are no inter-token samples at all.
    """
    rec, _ = _run_one(MockClient([MockResponse()]))
    assert rec.t_first_token is not None
    assert rec.e2e - rec.ttft < 0.001      # sub-millisecond: parsing, not decode
    assert rec.itl == []                    # one timestamp -> no gaps to measure


# --------------------------------------------------------------------------
# rate-limit probe
# --------------------------------------------------------------------------


def test_parse_rate_limit_headers_normalises():
    p = probe.parse_rate_limit_headers({
        "x-ratelimit-limit-requests": "200",
        "x-ratelimit-remaining-requests": "17",
        "x-ratelimit-reset": "60", "retry-after": "3",
    })
    assert p["limit_requests"] == 200
    assert p["remaining_requests"] == 17
    assert p["reset"] == 60
    assert p["retry_after_s"] == 3
    assert p["any_rate_limit_header"] is True


def test_parse_rate_limit_headers_preserves_unknown():
    p = probe.parse_rate_limit_headers({"x-weird-limit": "9"})
    assert p["raw"] == {"x-weird-limit": "9"}
    assert p["any_rate_limit_header"] is False


def test_projection_is_labelled_a_floor():
    proj = probe.project_exhaustion({"remaining_requests": 100.0}, 5.0)
    assert proj["seconds_to_exhaustion"] == pytest.approx(20.0)
    assert "floor" in proj["note"]
    assert probe.project_exhaustion({}, 5.0) is None


def test_probe_identifies_binding_concurrency():
    levels = [
        {"concurrency": 8, "rate_limited_fraction": 0.0},
        {"concurrency": 16, "rate_limited_fraction": 0.0},
        {"concurrency": 32, "rate_limited_fraction": 0.25},
    ]
    s = probe.summarise_probe(levels, 0.10)
    assert s["binding_concurrency"] == 32
    assert s["max_clean_concurrency"] == 16
    assert "binding at concurrency 32" in s["verdict"]


def test_probe_reports_when_no_wall_found():
    levels = [{"concurrency": 8, "rate_limited_fraction": 0.0}]
    s = probe.summarise_probe(levels, 0.10)
    assert s["binding_concurrency"] is None
    assert "did not find the wall" in s["verdict"]


# --------------------------------------------------------------------------
# context pooling
# --------------------------------------------------------------------------


def test_in_process_store_roundtrip():
    store = pooling.InProcessRedis(latency_s=0.0)
    store.set("k", "value")
    assert store.get("k") == "value"
    assert store.gets == 1 and store.sets == 1
    assert store.simulated is True


def test_pooling_reduces_cost_when_the_prompt_shrinks():
    result = pooling.compare(
        agents=10, context_tokens=2620, digest_tokens=200, output_tokens=256,
        price_in=0.05, price_out=0.20, cached_input_price_per_m=None,
        store=pooling.InProcessRedis(latency_s=0.0),
        context_text="word " * 3000, api_latency_s=0.0,
    )
    a = result["scenario_a_naive"]["cost_usd"]
    b = result["scenario_b_pooled"]["cost_usd"]
    assert b < a
    assert result["savings"]["cost_ratio_pooled_over_naive"] < 1.0
    # The quality caveat must ride with the number.
    assert "lossy" in result["savings"]["caveat"]
    assert "upper bound" in result["savings"]["caveat"]


def test_provider_cache_scenario_is_lossless_and_cheaper_than_naive():
    result = pooling.compare(
        agents=10, context_tokens=2620, digest_tokens=200, output_tokens=256,
        price_in=0.05, price_out=0.20, cached_input_price_per_m=0.005,
        store=pooling.InProcessRedis(latency_s=0.0),
        context_text="word " * 3000, api_latency_s=0.0,
    )
    c = result["scenario_c_provider_cache"]
    assert c["lossless"] is True
    assert c["cost_usd"] < result["scenario_a_naive"]["cost_usd"]


def test_store_latency_is_flagged_as_simulated():
    result = pooling.compare(
        agents=3, context_tokens=100, digest_tokens=10, output_tokens=8,
        price_in=0.05, price_out=0.20, cached_input_price_per_m=None,
        store=pooling.InProcessRedis(latency_s=0.0),
        context_text="word " * 200, api_latency_s=0.0,
    )
    assert result["scenario_b_pooled"]["store_latency_simulated"] is True
    assert "SIMULATED" in pooling.format_report(result)


def test_digest_truncates_to_requested_tokens():
    tok = pooling.StubTokenizer()
    digest = pooling.make_digest("a b c d e f g h", 3, tok)
    assert len(tok.encode(digest)) == 3


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------


def test_gpu_cost_is_a_step_function():
    """A GPU bills whole, busy or not — that step is what creates a crossover."""
    rows = model.model_costs(
        [1, 1000], api_cost_per_request=0.0002,
        decisions_per_agent_per_day=390, gpu_usd_per_hour=2.0,
        arm_b_rps=0.054, arm_c_rps=9.28,
    )
    assert rows[0]["arm_c_gpus"] == 1
    assert rows[0]["arm_c_cost_per_day"] == pytest.approx(48.0)
    # One agent barely uses the card; utilisation should be tiny.
    assert rows[0]["arm_c_utilisation"] < 0.01


def test_arm_b_needs_far_more_gpus_than_arm_c():
    rows = model.model_costs(
        [1000], api_cost_per_request=0.0002,
        decisions_per_agent_per_day=390, gpu_usd_per_hour=2.0,
        arm_b_rps=0.054, arm_c_rps=9.28,
    )
    assert rows[0]["arm_b_gpus"] > rows[0]["arm_c_gpus"]


def test_crossover_math_is_exact():
    """At the crossover, API cost per day must equal GPU cost per day."""
    api_price, per_day, gpu_rate, rps = 0.0002, 390, 2.0, 9.28
    cross = model.find_crossover(
        api_cost_per_request=api_price, decisions_per_agent_per_day=per_day,
        gpu_usd_per_hour=gpu_rate, rps=rps,
    )
    assert cross["available"]
    n = cross["crossover_agents_exact"]
    api_cost = n * per_day * api_price
    gpu_cost = cross["gpus_at_crossover"] * 24.0 * gpu_rate
    assert api_cost == pytest.approx(gpu_cost)


def test_crossover_moves_the_right_way_with_price():
    """Cheaper API -> you need MORE agents before self-hosting wins."""
    base = model.find_crossover(
        api_cost_per_request=0.0002, decisions_per_agent_per_day=390,
        gpu_usd_per_hour=2.0, rps=9.28)
    cheaper = model.find_crossover(
        api_cost_per_request=0.0001, decisions_per_agent_per_day=390,
        gpu_usd_per_hour=2.0, rps=9.28)
    assert cheaper["crossover_agents"] > base["crossover_agents"]

    pricier_gpu = model.find_crossover(
        api_cost_per_request=0.0002, decisions_per_agent_per_day=390,
        gpu_usd_per_hour=4.0, rps=9.28)
    assert pricier_gpu["crossover_agents"] > base["crossover_agents"]


def test_sensitivity_covers_both_price_axes():
    rows = model.sensitivity(
        {}, api_cost_per_request=0.0002, decisions_per_agent_per_day=390,
        gpu_usd_per_hour=2.0, rps=9.28)
    names = {r["scenario"] for r in rows}
    assert "baseline" in names
    assert "API price x2" in names
    assert "GPU $3/hr" in names
    assert all(r["crossover_agents"] for r in rows)


def test_model_refuses_to_invent_a_price(capsys):
    rc = model.main([])
    assert rc == 2
    assert "will not invent a price" in capsys.readouterr().err


def test_report_names_its_omissions():
    rows = model.model_costs([10], api_cost_per_request=0.0002,
                             decisions_per_agent_per_day=390,
                             gpu_usd_per_hour=2.0, arm_b_rps=0.054, arm_c_rps=9.28)
    result = {
        "inputs": {"api_cost_per_request": 0.0002, "api_cost_source": "test",
                   "decisions_per_agent_per_day": 390, "gpu_usd_per_hour": 2.0,
                   "context_sync_cost_per_day": 0.0},
        "throughput": {"arm_b": {"available": False, "reason": "test"},
                       "arm_c": {"available": True, "requests_per_s": 9.28,
                                 "concurrency": 32, "requests_per_day": 801792}},
        "rows": rows,
        "crossover": {"arm_b": {"available": False, "reason": "x"},
                      "arm_c": model.find_crossover(
                          api_cost_per_request=0.0002,
                          decisions_per_agent_per_day=390,
                          gpu_usd_per_hour=2.0, rps=9.28)},
        "sensitivity": model.sensitivity(
            {}, api_cost_per_request=0.0002, decisions_per_agent_per_day=390,
            gpu_usd_per_hour=2.0, rps=9.28),
        "omissions": ["Engineering and on-call cost", "Rate limits"],
    }
    text = model.format_report(result)
    assert "OMITTED" in text
    assert "CROSSOVER" in text
