"""Does pooling shared context cut the API bill for a fleet of agents?

    python -m analysis.context_pooling --mock          # offline, no cost
    python -m analysis.context_pooling --live --agents 10

If 100 agents analyse the same SEC filing, does sending that filing through the
API 100 times cost 100x? This measures the two ends of that question.

WHAT POOLING ACTUALLY SAVES — AND WHAT IT DOES NOT
--------------------------------------------------
There is a trap here worth naming before any number is quoted.

Caching context in Redis does **not**, by itself, reduce the API bill. The model
still has to see the tokens it reasons over; if each agent fetches the filing
from Redis and then sends the same 2,620 tokens to the provider, the bill is
unchanged. What Redis saves is the *upstream* fetch — a round trip to the SEC,
a database, a vendor feed — not provider tokens.

The API bill only falls when the **prompt itself gets smaller**: context is
summarised once into a digest, and the agents reason over the digest. That is a
genuine saving and it is what scenario B measures.

But it is not free, and this benchmark does not measure the part that costs:
**a digest is lossy.** Whether a decision made from a 200-token summary matches
one made from the full filing is a decision-quality question, and nothing here
tests it. So the savings figure below is an upper bound on the *cost* axis with
the *quality* axis unmeasured. Reported as such.

A third option exists and is cheaper than either on the provider side —
**provider-side prompt caching**, where the shared prefix is billed at a reduced
cached-input rate with no loss at all. It is the hosted-API analogue of arm C's
prefix caching. It is not simulated here because it is provider- and
model-specific; ``--cached-input-price-per-m`` prices that scenario if the
provider publishes a rate.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from common.fixtures import StubTokenizer, resolve_fixture  # noqa: E402
from runners.bench_api_baseline import (  # noqa: E402
    DEFAULT_PRICE_INPUT_PER_M,
    DEFAULT_PRICE_OUTPUT_PER_M,
    request_cost,
)

__all__ = ["InProcessRedis", "pool_context", "compare", "format_report"]


# --------------------------------------------------------------------------
# context store
# --------------------------------------------------------------------------


class InProcessRedis:
    """Dict-backed stand-in for Redis, with realistic-shaped latency.

    Ships so the comparison runs with no server and no docker. It is NOT a
    performance model of Redis — ``latency_s`` is a configurable constant, and
    the report labels the fetch timings as simulated whenever this backend is
    used. A real Redis is used instead when ``--redis-url`` is given.
    """

    backend = "in-process mock"
    simulated = True

    def __init__(self, latency_s: float = 0.0003) -> None:
        self._data: Dict[str, str] = {}
        self.latency_s = latency_s
        self.gets = 0
        self.sets = 0

    def set(self, key: str, value: str) -> None:
        time.sleep(self.latency_s)
        self._data[key] = value
        self.sets += 1

    def get(self, key: str) -> Optional[str]:
        time.sleep(self.latency_s)
        self.gets += 1
        return self._data.get(key)


class RealRedis:  # pragma: no cover - requires a server
    backend = "redis"
    simulated = False

    def __init__(self, url: str) -> None:
        import redis  # noqa: PLC0415

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.gets = 0
        self.sets = 0

    def set(self, key: str, value: str) -> None:
        self._r.set(key, value)
        self.sets += 1

    def get(self, key: str) -> Optional[str]:
        self.gets += 1
        return self._r.get(key)


def make_store(redis_url: Optional[str], latency_s: float = 0.0003):
    if redis_url:
        return RealRedis(redis_url)
    return InProcessRedis(latency_s=latency_s)


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------


def make_digest(context_text: str, digest_tokens: int, tokenizer=None) -> str:
    """Truncate the shared context to ``digest_tokens``.

    A real system would summarise with a model. Truncation is used here on
    purpose: it makes the *token accounting* exact and honest, and it avoids
    implying a summarisation quality this benchmark never measured. Either way
    the cost arithmetic is identical — what differs is the decision quality,
    which is out of scope and flagged everywhere the savings figure appears.
    """
    tok = tokenizer or StubTokenizer()
    ids = tok.encode(context_text)
    return tok.decode(ids[:digest_tokens])


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------


def _timed_fetch(store, key: str) -> Tuple[Optional[str], float]:
    t0 = time.perf_counter()
    value = store.get(key)
    return value, time.perf_counter() - t0


def pool_context(store, key: str, context_text: str) -> float:
    t0 = time.perf_counter()
    store.set(key, context_text)
    return time.perf_counter() - t0


async def _mock_api_call(prompt_tokens: int, output_tokens: int,
                         latency_s: float) -> Dict[str, Any]:
    await asyncio.sleep(latency_s)
    return {"usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": output_tokens}}


async def run_scenarios(
    *,
    agents: int,
    context_tokens: int,
    digest_tokens: int,
    output_tokens: int,
    price_in: float,
    price_out: float,
    cached_input_price_per_m: Optional[float],
    store,
    context_text: str,
    api_latency_s: float,
    live_call=None,
) -> Dict[str, Any]:
    """Scenario A (full context each time) vs B (pooled digest)."""
    tok = StubTokenizer()

    # ---- A: naive. Every agent sends the whole context. ----
    a_latencies: List[float] = []
    a_prompt_tokens = 0
    t0 = time.perf_counter()
    for _ in range(agents):
        t = time.perf_counter()
        if live_call is not None:
            usage = await live_call(context_text)
        else:
            usage = await _mock_api_call(context_tokens, output_tokens, api_latency_s)
        a_latencies.append(time.perf_counter() - t)
        a_prompt_tokens += int((usage.get("usage") or {}).get("prompt_tokens") or 0)
    a_wall = time.perf_counter() - t0
    a_cost = request_cost(a_prompt_tokens, agents * output_tokens, price_in, price_out)

    # ---- B: pooled. Context stored once; agents reason over a digest. ----
    b_latencies: List[float] = []
    fetch_latencies: List[float] = []
    b_prompt_tokens = 0
    pool_write_s = pool_context(store, "ctx:filing", context_text)
    digest = make_digest(context_text, digest_tokens, tok)

    t0 = time.perf_counter()
    for _ in range(agents):
        t = time.perf_counter()
        _, fetch_s = _timed_fetch(store, "ctx:filing")
        fetch_latencies.append(fetch_s)
        if live_call is not None:
            usage = await live_call(digest)
        else:
            usage = await _mock_api_call(digest_tokens, output_tokens, api_latency_s)
        b_latencies.append(time.perf_counter() - t)
        b_prompt_tokens += int((usage.get("usage") or {}).get("prompt_tokens") or 0)
    b_wall = time.perf_counter() - t0
    b_cost = request_cost(b_prompt_tokens, agents * output_tokens, price_in, price_out)

    # ---- C: provider-side prompt caching, if a cached rate is known. ----
    cached = None
    if cached_input_price_per_m is not None:
        # First request pays full price; the shared prefix is cached for the
        # rest. Lossless, unlike the digest.
        first = request_cost(context_tokens, output_tokens, price_in, price_out)
        rest = (agents - 1) * (
            (context_tokens / 1e6) * cached_input_price_per_m
            + (output_tokens / 1e6) * price_out
        )
        cached = {
            "scenario": "provider prompt cache",
            "cost_usd": first + rest,
            "cached_input_price_per_m": cached_input_price_per_m,
            "lossless": True,
            "note": (
                "Shared prefix billed at the cached-input rate after the first "
                "request. Unlike the digest this loses nothing, so it is the "
                "option to reach for first if the provider offers it."
            ),
        }

    return {
        "agents": agents,
        "output_tokens_each": output_tokens,
        "scenario_a_naive": {
            "prompt_tokens_total": a_prompt_tokens,
            "cost_usd": a_cost,
            "cost_per_agent_usd": a_cost / agents if agents else None,
            "wall_s": a_wall,
            "latency_mean_s": statistics.fmean(a_latencies) if a_latencies else None,
            "latency_p95_s": _p95(a_latencies),
        },
        "scenario_b_pooled": {
            "prompt_tokens_total": b_prompt_tokens,
            "digest_tokens": digest_tokens,
            "cost_usd": b_cost,
            "cost_per_agent_usd": b_cost / agents if agents else None,
            "wall_s": b_wall,
            "latency_mean_s": statistics.fmean(b_latencies) if b_latencies else None,
            "latency_p95_s": _p95(b_latencies),
            "pool_write_s": pool_write_s,
            "store_fetch_mean_s": (
                statistics.fmean(fetch_latencies) if fetch_latencies else None
            ),
            "store_backend": store.backend,
            "store_latency_simulated": getattr(store, "simulated", False),
        },
        "scenario_c_provider_cache": cached,
        "savings": {
            "cost_ratio_pooled_over_naive": (b_cost / a_cost) if a_cost else None,
            "cost_saved_usd": a_cost - b_cost,
            "token_ratio": (
                b_prompt_tokens / a_prompt_tokens if a_prompt_tokens else None
            ),
            "caveat": (
                "COST axis only. The digest is lossy and this benchmark does "
                "not test whether a decision made from it matches one made "
                "from the full context. Treat the ratio as an upper bound on "
                "savings with decision quality unmeasured."
            ),
        },
        "store_vs_api_latency": {
            "store_fetch_mean_s": (
                statistics.fmean(fetch_latencies) if fetch_latencies else None
            ),
            "api_call_mean_s": (
                statistics.fmean(a_latencies) if a_latencies else None
            ),
            "ratio_api_over_store": (
                statistics.fmean(a_latencies) / statistics.fmean(fetch_latencies)
                if fetch_latencies and a_latencies
                and statistics.fmean(fetch_latencies) > 0 else None
            ),
            "note": (
                "A local store fetch and an API round trip differ by orders of "
                "magnitude. That gap is why context distribution, not compute, "
                "governs how many agents can stay in sync with a live market."
            ),
        },
    }


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def compare(**kwargs) -> Dict[str, Any]:
    return asyncio.run(run_scenarios(**kwargs))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_report(result: Dict[str, Any]) -> str:
    a = result["scenario_a_naive"]
    b = result["scenario_b_pooled"]
    s = result["savings"]
    lines: List[str] = []
    A = lines.append

    A("=" * 78)
    A(f"CONTEXT POOLING — {result['agents']} agents over one shared context")
    A("=" * 78)
    A("")
    A(f"  {'':<26} {'A: naive':>16} {'B: pooled':>16}")
    A("  " + "-" * 62)
    A(f"  {'prompt tokens (total)':<26} {a['prompt_tokens_total']:>16,} "
      f"{b['prompt_tokens_total']:>16,}")
    A(f"  {'cost (USD)':<26} {a['cost_usd']:>16.6f} {b['cost_usd']:>16.6f}")
    A(f"  {'cost per agent (USD)':<26} {a['cost_per_agent_usd']:>16.6f} "
      f"{b['cost_per_agent_usd']:>16.6f}")
    A(f"  {'wall time (s)':<26} {a['wall_s']:>16.3f} {b['wall_s']:>16.3f}")
    if a["latency_mean_s"] and b["latency_mean_s"]:
        A(f"  {'latency mean (ms)':<26} {a['latency_mean_s'] * 1000:>16.1f} "
          f"{b['latency_mean_s'] * 1000:>16.1f}")

    A("")
    if s["cost_ratio_pooled_over_naive"] is not None:
        A(f"  Pooled costs {s['cost_ratio_pooled_over_naive'] * 100:.1f}% of naive "
          f"(saving ${s['cost_saved_usd']:.6f} over {result['agents']} agents)")

    c = result.get("scenario_c_provider_cache")
    if c:
        A(f"  Provider prompt cache would cost ${c['cost_usd']:.6f} — "
          f"and is LOSSLESS")

    lat = result["store_vs_api_latency"]
    if lat["ratio_api_over_store"]:
        A("")
        A(f"  Store fetch {lat['store_fetch_mean_s'] * 1000:.3f} ms vs API "
          f"{lat['api_call_mean_s'] * 1000:.1f} ms "
          f"— {lat['ratio_api_over_store']:.0f}x")
    if b.get("store_latency_simulated"):
        A("  (store latency is SIMULATED — in-process mock, not a real Redis)")

    A("")
    A("  CAVEAT: " + s["caveat"])
    return "\n".join(lines)


def to_csv(result: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["scenario", "prompt_tokens_total", "cost_usd",
                "cost_per_agent_usd", "wall_s", "latency_mean_s", "lossless"])
    a, b = result["scenario_a_naive"], result["scenario_b_pooled"]
    w.writerow(["A_naive", a["prompt_tokens_total"], f"{a['cost_usd']:.8f}",
                f"{a['cost_per_agent_usd']:.8f}", f"{a['wall_s']:.4f}",
                f"{a['latency_mean_s']:.6f}" if a["latency_mean_s"] else "",
                "yes"])
    w.writerow(["B_pooled_digest", b["prompt_tokens_total"], f"{b['cost_usd']:.8f}",
                f"{b['cost_per_agent_usd']:.8f}", f"{b['wall_s']:.4f}",
                f"{b['latency_mean_s']:.6f}" if b["latency_mean_s"] else "",
                "NO - digest is lossy"])
    c = result.get("scenario_c_provider_cache")
    if c:
        w.writerow(["C_provider_cache", "", f"{c['cost_usd']:.8f}",
                    f"{c['cost_usd'] / result['agents']:.8f}", "", "", "yes"])
    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Context pooling cost comparison.")
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--context-tokens", type=int, default=2620)
    ap.add_argument("--digest-tokens", type=int, default=200)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--price-input-per-m", type=float, default=DEFAULT_PRICE_INPUT_PER_M)
    ap.add_argument("--price-output-per-m", type=float, default=DEFAULT_PRICE_OUTPUT_PER_M)
    ap.add_argument("--cached-input-price-per-m", type=float, default=None,
                    help="provider's cached-input rate, if published. Prices "
                         "the lossless prompt-cache scenario.")
    ap.add_argument("--redis-url", default=None,
                    help="real Redis. Omitted -> in-process mock, and the "
                         "report labels store latency as simulated.")
    ap.add_argument("--api-latency-ms", type=float, default=800.0,
                    help="simulated API round trip for --mock. Use a value "
                         "measured by arm A, not a guess.")
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--live", dest="mock", action="store_false",
                    help="issue REAL API calls (bills money)")
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args(argv)

    try:
        fixture, _ = resolve_fixture(
            "shared_prefix", 1, args.context_tokens, 20260820,
            "Qwen/Qwen2.5-7B-Instruct", args.fixtures_dir, use_stub=True,
        )
        context_text = fixture.requests[0]["prompt_text"]
    except SystemExit:
        context_text = "SHARED CONTEXT " * (args.context_tokens // 2)

    live_call = None
    if not args.mock:
        print("ERROR: --live requires wiring an authenticated client; run the "
              "mock comparison first and confirm the arithmetic.", file=sys.stderr)
        return 2

    result = compare(
        agents=args.agents, context_tokens=args.context_tokens,
        digest_tokens=args.digest_tokens, output_tokens=args.output_tokens,
        price_in=args.price_input_per_m, price_out=args.price_output_per_m,
        cached_input_price_per_m=args.cached_input_price_per_m,
        store=make_store(args.redis_url), context_text=context_text,
        api_latency_s=args.api_latency_ms / 1000.0, live_call=live_call,
    )

    print(format_report(result))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n[pooling] {args.json_out}")
    if args.csv_out:
        with open(args.csv_out, "w") as fh:
            fh.write(to_csv(result))
        print(f"[pooling] {args.csv_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
