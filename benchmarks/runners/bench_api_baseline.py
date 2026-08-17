#!/usr/bin/env python3
"""Arm A — hosted API serving. Same workload, routed through OpenRouter.

    python benchmarks/runners/bench_api_baseline.py --dry-run
    python benchmarks/runners/bench_api_baseline.py --n-requests 4 --concurrency 1

**This spends real money.** Every non-dry-run request is billed. `--dry-run`
prints the plan and a cost estimate and submits nothing; start there, then a
tiny `--n-requests 4` run, and only then a sweep.

WHAT THIS ARM IS FOR
--------------------
Arms B and C answered a compute question: continuous batching recovers the
host-bound collapse (arm C reaches 96.9% GPU utilisation and 2,375 output tok/s
at C=32 where arm B sits at 24% and 13.7). Arm A answers the question that
actually governs a fleet of trading agents — what it costs and how long it takes
to get a decision out of the API the platform uses today, and where the ceiling
is.

THE TOKENISER CAVEAT — READ BEFORE COMPARING COST OR THROUGHPUT
---------------------------------------------------------------
The fixture is **Qwen-tokenised**: 2,620 token ids, 99.4% shared prefix. An HTTP
API takes *text*, not token ids, and Nemotron tokenises that text with a
different vocabulary. So:

* the **prompt text** is byte-identical to what arms B and C ran;
* the **token count is not**, and neither is the billed input size.

Every number here is therefore reported against the **actual** `usage` the API
returns, never against 2,620. ``context_tokens_fixture`` (Qwen) and
``prompt_tokens_actual`` (Nemotron) are both recorded so the gap is visible
rather than assumed away. A cost figure computed from the local arms' token
count would be wrong by whatever the two vocabularies disagree by.

CONCURRENCY MEANS THE SAME THING IT DID IN ARMS B AND C
-------------------------------------------------------
``asyncio.Semaphore(concurrency)`` around submission — at most N requests
outstanding at once — and ``t_submit`` is stamped *after* the semaphore is
acquired, exactly as in arm C. So TTFT excludes client-side queue wait in all
three arms and measures the server's response to an admitted request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from common import fixtures as fx_mod  # noqa: E402
from common.config import apply_overrides, config_sha256, load_config  # noqa: E402
from common.fixtures import resolve_fixture  # noqa: E402
from common.manifest import build_manifest, make_run_id  # noqa: E402
from common.metrics import (  # noqa: E402
    RequestRecord,
    Summary,
    attach_resources,
    console_lines,
    summarize,
    write_results,
)

ARM = "A"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

# USD per million tokens. Provider pricing changes; these are defaults, they are
# overridable, and the values actually used are written into every manifest so a
# future reader can tell which schedule produced a cost figure.
DEFAULT_PRICE_INPUT_PER_M = 0.05
DEFAULT_PRICE_OUTPUT_PER_M = 0.20

# Header names vary by provider and change over time; all of these are captured
# verbatim when present rather than parsed into a fixed shape.
RATE_LIMIT_HEADER_HINTS = (
    "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
    "retry-after", "x-request-id",
)
# Provider-reported server-side timings, when exposed. The difference between
# these and our measured wall time is the network component.
LATENCY_HEADER_HINTS = (
    "x-openrouter-processing-ms", "openrouter-processing-ms",
    "x-processing-ms", "server-timing",
)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {400, 401, 403, 404}


# --------------------------------------------------------------------------
# key handling
# --------------------------------------------------------------------------


def api_key_hash(key: Optional[str]) -> Optional[str]:
    """SHA-256 of the key. **The key itself is never stored or printed.**

    A hash still lets a future reader confirm two runs used the same
    credential — which matters, because tier and rate limit ride on the key —
    without the manifest becoming a secret.
    """
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_api_key(env_var: str = "OPENROUTER_API_KEY") -> Optional[str]:
    return os.environ.get(env_var) or None


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def request_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_m: float = DEFAULT_PRICE_INPUT_PER_M,
    price_out_per_m: float = DEFAULT_PRICE_OUTPUT_PER_M,
) -> float:
    """USD for one request, from ACTUAL billed tokens."""
    return (prompt_tokens / 1e6) * price_in_per_m + (
        completion_tokens / 1e6
    ) * price_out_per_m


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------


def extract_headers(headers: Any) -> Dict[str, str]:
    """Capture rate-limit and timing headers verbatim, lowercased."""
    out: Dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        return out
    for key, value in items:
        low = str(key).lower()
        if any(h in low for h in RATE_LIMIT_HEADER_HINTS) or any(
            h in low for h in LATENCY_HEADER_HINTS
        ):
            out[low] = str(value)
    return out


def server_processing_ms(headers: Dict[str, str]) -> Optional[float]:
    """Provider-reported server time, if any header carries it."""
    for key in ("x-openrouter-processing-ms", "openrouter-processing-ms",
                "x-processing-ms"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                continue
    # Server-Timing: dur=123.4
    st = headers.get("server-timing")
    if st and "dur=" in st:
        try:
            return float(st.split("dur=")[1].split(";")[0].split(",")[0])
        except (IndexError, ValueError):
            return None
    return None


def retry_after_seconds(headers: Dict[str, str]) -> Optional[float]:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_usage(payload: Dict[str, Any]) -> Tuple[int, int]:
    """(prompt_tokens, completion_tokens) as BILLED by the provider.

    Never estimated from the local fixture — the provider tokenises with its own
    vocabulary and bills on that, so an estimate would be wrong in exactly the
    place cost matters.
    """
    usage = payload.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    )


def parse_cache_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt-cache fields the provider reports, if any.

    Whether OpenRouter exposes prompt caching for this model is itself the
    question — arm C measured a LOCAL prefix cache, and the hosted analogue has
    never been measured on any arm. So this records what came back rather than
    assuming a shape: ``cached_tokens`` under ``prompt_tokens_details`` is the
    OpenAI-compatible spelling, ``cache_read_input_tokens`` the Anthropic one,
    and ``cache_discount`` is OpenRouter's own.

    ``available`` false means the provider said nothing about caching — which is
    evidence of absence only in the weak sense that it is not being reported,
    not that no cache exists.
    """
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    found: Dict[str, Any] = {}
    for key, value in (
        ("cached_tokens", details.get("cached_tokens")),
        ("cache_read_input_tokens", usage.get("cache_read_input_tokens")),
        ("cache_creation_input_tokens", usage.get("cache_creation_input_tokens")),
        ("cache_discount", usage.get("cache_discount")),
        # OpenRouter reports the amount it ACTUALLY billed. That beats any
        # price-table multiplication, which can only ever be a reconstruction.
        ("provider_reported_cost_usd", usage.get("cost")),
    ):
        if value is not None:
            found[key] = value
    return {
        "available": bool(found),
        "fields": found,
        "usage_keys_seen": sorted(usage.keys()),
    }


# --------------------------------------------------------------------------
# one request
# --------------------------------------------------------------------------


class RateLimited(Exception):
    def __init__(self, retry_after: Optional[float], headers: Dict[str, str]):
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.headers = headers


async def _post_once(
    client: Any,
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool,
    rec: RequestRecord,
    timeout: float,
) -> Dict[str, Any]:
    """One HTTP attempt. Raises RateLimited on 429 so the caller can back off."""
    info: Dict[str, Any] = {}

    if not stream:
        resp = await client.post(url, json=body, headers=headers, timeout=timeout)
        hdrs = extract_headers(resp.headers)
        info["headers"] = hdrs
        info["status"] = resp.status_code
        if resp.status_code == 429:
            raise RateLimited(retry_after_seconds(hdrs), hdrs)
        resp.raise_for_status()
        payload = resp.json()
        now = time.perf_counter()
        # Without streaming there is no first-token signal; TTFT and e2e
        # collapse to the same instant. Recorded as such rather than
        # fabricating an intermediate timestamp.
        rec.t_first_token = now
        rec.per_token_timestamps.append(now)
        info["payload"] = payload
        return info

    async with client.stream(
        "POST", url, json=body, headers=headers, timeout=timeout
    ) as resp:
        hdrs = extract_headers(resp.headers)
        info["headers"] = hdrs
        info["status"] = resp.status_code
        if resp.status_code == 429:
            await resp.aread()
            raise RateLimited(retry_after_seconds(hdrs), hdrs)
        resp.raise_for_status()

        usage: Dict[str, Any] = {}
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                now = time.perf_counter()
                if rec.t_first_token is None:
                    rec.t_first_token = now
                # One timestamp per streamed chunk. As in arms B and C, chunks
                # are NOT filtered on content — dropping whitespace-only chunks
                # is what corrupted the v1 per-token figure.
                rec.per_token_timestamps.append(now)
        info["payload"] = {"usage": usage}
        return info


async def run_one(
    client: Any,
    url: str,
    prompt_text: str,
    request_id: str,
    model: str,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    api_key: str,
    sem: "asyncio.Semaphore",
    stream: bool = True,
    timeout: float = 300.0,
    max_retries: int = 4,
    price_in: float = DEFAULT_PRICE_INPUT_PER_M,
    price_out: float = DEFAULT_PRICE_OUTPUT_PER_M,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Tuple[RequestRecord, Dict[str, Any]]:
    rec = RequestRecord(request_id=request_id)
    meta: Dict[str, Any] = {
        "request_id": request_id, "attempts": 0, "rate_limited": 0,
        "headers": {}, "status": None, "cost_usd": 0.0,
        "request_bytes": 0, "response_bytes": 0,
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if ignore_eos:
        # Parity with arms B/C, which force exactly max_new_tokens. Not every
        # provider honours this; the actual completion_tokens is recorded and
        # checked, so a provider that ignores it is visible rather than assumed.
        body["min_tokens"] = max_tokens
    if stream:
        body["stream_options"] = {"include_usage": True}
    # Reasoning is a cost and comparability variable, not a detail. OpenRouter
    # enables extended thinking by default for Nemotron, and thinking tokens are
    # billed as output — which would break comparison against the seed DB's
    # measured 860 output tokens/call for this model. Sent explicitly so the
    # setting is recorded rather than inherited from a provider default.
    if reasoning is not None:
        body["reasoning"] = reasoning

    hdrs = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter attribution headers; harmless elsewhere.
        "HTTP-Referer": "https://github.com/Open-Finance-Lab/AgenticTrading",
        "X-Title": "ATL GPU serving benchmark (arm A)",
    }
    meta["request_bytes"] = len(json.dumps(body).encode("utf-8"))

    async with sem:
        # AFTER the semaphore — same convention as arms B and C.
        rec.t_submit = time.perf_counter()
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            meta["attempts"] = attempt
            try:
                info = await _post_once(
                    client, url, body, hdrs, stream, rec, timeout
                )
                meta["headers"] = info.get("headers", {})
                meta["status"] = info.get("status")
                payload = info.get("payload") or {}
                p_tok, c_tok = parse_usage(payload)
                rec.prompt_tokens = p_tok
                rec.output_tokens = c_tok or len(rec.per_token_timestamps)
                rec.t_done = time.perf_counter()
                meta["cost_usd"] = request_cost(p_tok, c_tok, price_in, price_out)
                meta["cache"] = parse_cache_usage(payload)
                meta["response_bytes"] = len(json.dumps(payload).encode("utf-8"))
                meta["server_processing_ms"] = server_processing_ms(meta["headers"])
                if meta["server_processing_ms"] is not None and rec.e2e:
                    # Wall time minus provider-reported server time. An estimate
                    # of everything outside the provider: DNS, TLS, transit,
                    # client parsing.
                    meta["network_latency_estimated_ms"] = (
                        rec.e2e * 1000.0 - meta["server_processing_ms"]
                    )
                return rec, meta

            except RateLimited as exc:
                meta["rate_limited"] += 1
                meta["headers"] = exc.headers
                meta["status"] = 429
                if attempt >= max_retries:
                    rec.t_done = time.perf_counter()
                    rec.error = "RATE_LIMITED"
                    return rec, meta
                # Respect Retry-After when the provider sends one; otherwise
                # exponential backoff with jitter so retries do not resynchronise
                # into a second thundering herd.
                wait = exc.retry_after if exc.retry_after is not None else delay
                await asyncio.sleep(wait + random.uniform(0, 0.25))
                delay = min(delay * 2, 30.0)

            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                meta["status"] = status
                if status in FATAL_STATUS:
                    # Auth/permission/model errors will not fix themselves;
                    # retrying just spends more time and, on some providers,
                    # more money.
                    rec.t_done = time.perf_counter()
                    rec.error = f"FATAL_{status}: {type(exc).__name__}"
                    return rec, meta
                if attempt >= max_retries:
                    rec.t_done = time.perf_counter()
                    rec.error = f"{type(exc).__name__}: {exc}"
                    return rec, meta
                await asyncio.sleep(delay + random.uniform(0, 0.25))
                delay = min(delay * 2, 30.0)

    rec.t_done = time.perf_counter()
    rec.error = rec.error or "UNKNOWN"
    return rec, meta


# --------------------------------------------------------------------------
# one level
# --------------------------------------------------------------------------


async def run_level(
    prompts: Sequence[Tuple[str, str]],
    concurrency: int,
    *,
    url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    stream: bool,
    timeout: float,
    max_retries: int,
    price_in: float,
    price_out: float,
    client: Any = None,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Tuple[List[RequestRecord], Dict[str, Any]]:
    import httpx  # noqa: PLC0415

    sem = asyncio.Semaphore(concurrency)
    own_client = client is None
    if own_client:
        limits = httpx.Limits(
            max_connections=max(concurrency * 2, 16),
            max_keepalive_connections=max(concurrency, 8),
        )
        client = httpx.AsyncClient(limits=limits)

    t0 = time.perf_counter()
    try:
        results = await asyncio.gather(*[
            run_one(
                client, url, text, rid, model, max_tokens, temperature,
                ignore_eos, api_key, sem, stream=stream, timeout=timeout,
                max_retries=max_retries, price_in=price_in, price_out=price_out,
                reasoning=reasoning,
            )
            for rid, text in prompts
        ])
    finally:
        if own_client:
            await client.aclose()

    wall = time.perf_counter() - t0
    records = [r for r, _ in results]
    metas = [m for _, m in results]

    rate_limited = sum(m["rate_limited"] for m in metas)
    cost = sum(m["cost_usd"] for m in metas)
    _cache = [m.get("cache") or {} for m in metas]
    cached_tok = [c["fields"].get("cached_tokens") for c in _cache
                  if c.get("fields", {}).get("cached_tokens") is not None]
    provider_cost = [c["fields"].get("provider_reported_cost_usd") for c in _cache
                     if c.get("fields", {}).get("provider_reported_cost_usd") is not None]
    net = [m["network_latency_estimated_ms"] for m in metas
           if m.get("network_latency_estimated_ms") is not None]
    srv = [m["server_processing_ms"] for m in metas
           if m.get("server_processing_ms") is not None]

    last_headers: Dict[str, str] = {}
    for m in metas:
        if m.get("headers"):
            last_headers = m["headers"]

    info: Dict[str, Any] = {
        "concurrency": concurrency,
        "section_wall_s": wall,
        "api_cost_total": cost,
        "api_cost_per_request": (cost / len(records)) if records else None,
        "rate_limit_encountered": rate_limited > 0,
        "rate_limit_429_count": rate_limited,
        "prompt_cache": {
            "reported_by_provider": bool(cached_tok),
            "cached_tokens_total": sum(cached_tok) if cached_tok else None,
            "cached_tokens_mean": (sum(cached_tok) / len(cached_tok))
            if cached_tok else None,
            "requests_with_any_cache_hit": sum(1 for c in cached_tok if c),
            "note": (
                "cached_tokens is what the PROVIDER says it reused. Zero across "
                "the level means no prompt-cache benefit was granted on this "
                "run — not that the fixture lacks a shared prefix (it is 99.4% "
                "shared by construction)."
            ),
        },
        "provider_reported_cost_usd_total": (
            sum(provider_cost) if provider_cost else None),
        "rate_limit_headers_last": last_headers,
        "server_processing_ms_mean": (sum(srv) / len(srv)) if srv else None,
        "network_latency_estimated_ms_mean": (sum(net) / len(net)) if net else None,
        "request_bytes_mean": (
            sum(m["request_bytes"] for m in metas) / len(metas) if metas else None
        ),
        "response_bytes_mean": (
            sum(m["response_bytes"] for m in metas) / len(metas) if metas else None
        ),
        "status_counts": _count(m.get("status") for m in metas),
        "error_counts": _count(r.error for r in records if r.error),
        "per_request": metas,
    }
    if not srv:
        info["network_latency_note"] = (
            "The provider exposed no server-timing header, so the network "
            "component cannot be separated from processing. e2e latency here is "
            "the sum of both."
        )
    return records, info


def _count(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def resolve_reasoning(cli_value: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """The reasoning payload to send, and a label for the manifest.

    Defaults to OFF. Thinking tokens are billed as output, so a provider default
    of "on" would inflate the output-token figure and break comparison against
    the seed DB's measured 860 output tokens/call for this model. Off-by-default
    makes that an explicit choice rather than an inherited one.
    """
    raw = cli_value or os.environ.get("OPENROUTER_REASONING_EFFORT") or "none"
    effort = str(raw).strip().lower()
    if effort in ("auto", "default"):
        return None, "auto (provider default, nothing sent)"
    if effort in ("none", "off", "false", "0", "disabled"):
        return {"enabled": False, "effort": "none", "exclude": True}, "none"
    return {"enabled": True, "effort": effort}, effort


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Arm A: hosted API benchmark.")
    # Shared with arms B and C.
    ap.add_argument("--config", default=None)
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--n-requests", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fixture", default=None, choices=sorted(fx_mod.FIXTURE_BUILDERS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and a cost estimate; submit nothing")
    ap.add_argument("--stub-tokenizer", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    # Arm A only.
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--price-input-per-m", type=float, default=DEFAULT_PRICE_INPUT_PER_M)
    ap.add_argument("--price-output-per-m", type=float, default=DEFAULT_PRICE_OUTPUT_PER_M)
    ap.add_argument("--reasoning", default=None,
                    choices=["none", "auto", "low", "medium", "high"],
                    help="Extended-thinking setting sent with every request. "
                         "Defaults to $OPENROUTER_REASONING_EFFORT, else "
                         "'none'. Thinking tokens bill as OUTPUT, so leaving "
                         "this to the provider default silently changes both "
                         "cost and the output-token figure. 'auto' sends "
                         "nothing and inherits the provider default.")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable SSE streaming. TTFT and ITL become "
                         "unavailable — e2e collapses to a single instant.")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-cost-usd", type=float, default=1.0,
                    help="refuse to start if the ESTIMATED cost exceeds this. "
                         "A guard against a mistyped sweep billing real money.")
    return ap


def estimate_cost(
    n_levels: int, n_requests: int, prompt_tokens: int, max_tokens: int,
    price_in: float, price_out: float,
) -> Dict[str, Any]:
    total = n_levels * n_requests
    cost = total * request_cost(prompt_tokens, max_tokens, price_in, price_out)
    return {
        "total_requests": total,
        "prompt_tokens_each_estimated": prompt_tokens,
        "output_tokens_each": max_tokens,
        "estimated_cost_usd": cost,
        "basis": (
            "Estimate only. Priced on the FIXTURE's Qwen token count; the "
            "provider bills its own tokenisation, so the true figure will "
            "differ. Actual usage is recorded per request at run time."
        ),
    }


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, {
        "workload.fixture": args.fixture,
        "workload.requests_per_level": args.n_requests,
        "workload.concurrency": args.concurrency,
        "generation.max_new_tokens": args.max_new_tokens,
        "api.model": args.model,
        "api.base_url": args.base_url,
        "api.price_input_per_m": args.price_input_per_m,
        "api.price_output_per_m": args.price_output_per_m,
        "api.stream": not args.no_stream,
    })

    gen = cfg["generation"]
    wl = cfg["workload"]
    max_tokens = int(gen["max_new_tokens"])
    ignore_eos = bool(gen.get("ignore_eos", True))
    temperature = float(gen.get("temperature", 0.0))
    n_requests = int(wl["requests_per_level"])
    levels = list(args.concurrency
                  or (cfg.get("workload_api") or {}).get("concurrency")
                  or wl["concurrency"])
    context_tokens = int(wl["context_tokens"])
    seed = int(cfg.get("seed", 0))
    fixture_name = wl["fixture"]
    out_dir = args.out_dir or os.path.join(_BENCH_ROOT, "results")

    fixture, how = resolve_fixture(
        fixture_name, n_requests, context_tokens, seed,
        cfg["model"]["id"], args.fixtures_dir, use_stub=args.stub_tokenizer,
    )
    meta = fixture.to_meta()

    print("=" * 72)
    print(f"ARM {ARM} — hosted API ({args.model})")
    print("=" * 72)
    print(f"endpoint           {args.base_url}")
    print(f"fixture            {fixture_name} ({how})  sha256 {meta['sha256'][:16]}…")
    print(f"  tokens/request   {meta['token_count_min']} (Qwen tokenisation)")
    print(f"  common prefix    {meta['common_prefix_tokens']} "
          f"({meta['common_prefix_fraction'] * 100:.1f}%)")
    print(f"max_tokens         {max_tokens}  (ignore_eos: {ignore_eos})")
    print(f"concurrency        {levels}   (offered load, asyncio.Semaphore)")
    print(f"pricing            ${args.price_input_per_m}/M in, "
          f"${args.price_output_per_m}/M out")
    print(f"streaming          {not args.no_stream}")
    reasoning_payload, reasoning_label = resolve_reasoning(args.reasoning)
    print(f"reasoning          {reasoning_label}  "
          f"(thinking tokens bill as OUTPUT)")

    est = estimate_cost(len(levels), n_requests, meta["token_count_min"],
                        max_tokens, args.price_input_per_m, args.price_output_per_m)
    print(f"\nESTIMATED COST     ${est['estimated_cost_usd']:.4f} "
          f"for {est['total_requests']} requests")
    print(f"  {est['basis']}")

    if args.dry_run:
        print("\n--dry-run: nothing submitted, nothing billed.")
        print("\nPlanned requests:")
        for c in levels:
            print(f"  concurrency {c:>4}: {n_requests} requests")
        print("\nThe prompt TEXT is identical to arms B/C. The token COUNT is "
              "not —\nNemotron tokenises differently, and bills on its own count.")
        return 0

    api_key = load_api_key(args.api_key_env)
    if not api_key:
        print(f"\nERROR: {args.api_key_env} is not set. Export it and re-run:",
              file=sys.stderr)
        print(f"  export {args.api_key_env}='sk-or-...'", file=sys.stderr)
        print("  (never put the key in a file that gets committed)", file=sys.stderr)
        return 2

    if est["estimated_cost_usd"] > args.max_cost_usd:
        print(f"\nREFUSING TO START: estimated ${est['estimated_cost_usd']:.4f} "
              f"exceeds --max-cost-usd ${args.max_cost_usd:.4f}.", file=sys.stderr)
        print("Lower --n-requests / --concurrency, or raise the cap "
              "deliberately.", file=sys.stderr)
        return 3

    run_id = args.run_id or make_run_id(ARM, levels[0] if levels else 0)
    prompts_all = [(r["request_id"], r["prompt_text"]) for r in fixture.requests]

    all_records: List[RequestRecord] = []
    summaries: List[Summary] = []
    level_infos: List[Dict[str, Any]] = []
    total_cost = 0.0

    for concurrency in levels:
        prompts = prompts_all[:n_requests]
        print(f"\n[level] concurrency={concurrency}  requests={len(prompts)}")
        records, info = await run_level(
            prompts, concurrency, reasoning=reasoning_payload,
            url=args.base_url, model=args.model, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
            ignore_eos=ignore_eos, stream=not args.no_stream,
            timeout=args.timeout, max_retries=args.max_retries,
            price_in=args.price_input_per_m, price_out=args.price_output_per_m,
        )
        total_cost += info["api_cost_total"]

        summary = summarize(run_id, concurrency, records, requested=len(prompts))
        attach_resources(summary, {})
        summary.notes.update({
            "api_cost_total": info["api_cost_total"],
            "api_cost_per_request": info["api_cost_per_request"],
            "rate_limit_encountered": info["rate_limit_encountered"],
            "rate_limit_429_count": info["rate_limit_429_count"],
            "prompt_cache": info["prompt_cache"],
            "provider_reported_cost_usd_total":
                info["provider_reported_cost_usd_total"],
            "server_processing_ms_mean": info["server_processing_ms_mean"],
            "network_latency_estimated_ms_mean":
                info["network_latency_estimated_ms_mean"],
            "context_tokens_fixture": meta["token_count_min"],
            "prompt_tokens_actual_mean": (
                summary.input_tokens_total / summary.completed
                if summary.completed else None
            ),
        })
        all_records.extend(records)
        summaries.append(summary)
        level_infos.append(info)

        for line in console_lines(summary):
            print(line)
        print(f"  cost          ${info['api_cost_total']:.5f} "
              f"(${info['api_cost_per_request']:.6f}/request)"
              if info["api_cost_per_request"] is not None else "  cost          —")
        if info["rate_limit_encountered"]:
            print(f"  [rate limit]  {info['rate_limit_429_count']} x 429 — "
                  f"THIS IS A FINDING, not a failure")
        if info["status_counts"]:
            print(f"  statuses      {info['status_counts']}")

        write_results(out_dir, run_id, all_records, summaries, extra={
            "arm": ARM,
            "levels_detail": level_infos,
            "api": {
                "model": args.model, "base_url": args.base_url,
                "price_input_per_m": args.price_input_per_m,
                "price_output_per_m": args.price_output_per_m,
                "stream": not args.no_stream,
                "cost_total_usd": total_cost,
            },
        })
        print(f"  [write] {out_dir}/{run_id}_summary.json")

    print(f"\nTOTAL BILLED COST  ${total_cost:.5f}")

    build_manifest(
        arm=ARM, concurrency=levels[0] if levels else 0,
        config_resolved=cfg, fixture_sha256=meta["sha256"],
        fixture_name=fixture_name, max_new_tokens=max_tokens,
        profiling_layer=1, run_id=run_id, out_dir=out_dir,
        context_tokens=context_tokens, model=args.model,
        extra={
            "config_sha256_recomputed": config_sha256(cfg),
            "concurrency_sweep": levels,
            "requests_per_level": n_requests,
            # Hash only. The key itself never touches the manifest.
            "openrouter_api_key_sha256": api_key_hash(api_key),
            "reasoning": reasoning_label,
            "reasoning_payload": reasoning_payload,
            "openrouter_tier": _detect_tier(level_infos),
            "api_base_url": args.base_url,
            "api_model": args.model,
            "price_input_per_m": args.price_input_per_m,
            "price_output_per_m": args.price_output_per_m,
            "pricing_captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
            "api_cost_total_usd": total_cost,
            "context_tokens_fixture_qwen": meta["token_count_min"],
            "tokeniser_caveat": (
                "The fixture is Qwen-tokenised; the provider bills on its own "
                "tokenisation of the same TEXT. Token counts and therefore cost "
                "are not comparable to arms B/C on a per-token basis."
            ),
        },
    )
    print(f"[manifest] {out_dir}/{run_id}_manifest.json")
    return 0


def _detect_tier(level_infos: Sequence[Dict[str, Any]]) -> str:
    """Best-effort tier from rate-limit headers. 'unknown' when undetectable."""
    for info in level_infos:
        hdrs = info.get("rate_limit_headers_last") or {}
        for key, value in hdrs.items():
            if "limit" in key and value.isdigit():
                # A very low request ceiling is characteristic of a free tier,
                # but this is a heuristic and is labelled as such.
                return "free (inferred from low limit)" if int(value) <= 20 else "paid (inferred)"
    return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    return asyncio.run(_amain(build_parser().parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
