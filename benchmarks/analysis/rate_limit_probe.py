"""Where does the provider's rate limit become the binding constraint?

    python -m analysis.rate_limit_probe --dry-run
    python -m analysis.rate_limit_probe --max-concurrency 64

Escalates offered concurrency until 429s exceed a threshold, then stops.

**Hitting the limit is the result, not a failure.** Arm C's ceiling is VRAM and
kernel scheduling; arm A's is a number in someone else's config. If the free
tier caps at C=32, that is the practical ceiling for a hosted-API agent fleet
regardless of what the cost model says — above it the option is not expensive,
it is unavailable without a contract change.

Small N per level (8 by default) so the probe finds the wall without burning
quota getting there — the escalation is the measurement, not the throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from common.fixtures import resolve_fixture  # noqa: E402
from runners.bench_api_baseline import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PRICE_INPUT_PER_M,
    DEFAULT_PRICE_OUTPUT_PER_M,
    api_key_hash,
    estimate_cost,
    load_api_key,
    run_level,
)

__all__ = ["DEFAULT_LADDER", "parse_rate_limit_headers", "project_exhaustion",
           "summarise_probe", "format_report"]

DEFAULT_LADDER = (8, 16, 32, 64, 128, 256, 512)
DEFAULT_THRESHOLD = 0.10  # stop once >10% of a level is rate-limited


def parse_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """Normalise whatever rate-limit headers the provider happened to send.

    Header naming is not standardised and changes; anything unrecognised is
    preserved under ``raw`` so a future reader can re-interpret it rather than
    finding it silently dropped.
    """
    out: Dict[str, Any] = {"raw": dict(headers)}

    def _num(*names):
        for n in names:
            if n in headers:
                try:
                    return float(headers[n])
                except (TypeError, ValueError):
                    continue
        return None

    out["limit_requests"] = _num("x-ratelimit-limit-requests", "x-ratelimit-limit")
    out["remaining_requests"] = _num(
        "x-ratelimit-remaining-requests", "x-ratelimit-remaining"
    )
    out["limit_tokens"] = _num("x-ratelimit-limit-tokens")
    out["remaining_tokens"] = _num("x-ratelimit-remaining-tokens")
    out["reset"] = _num("x-ratelimit-reset", "x-ratelimit-reset-requests")
    out["retry_after_s"] = _num("retry-after")
    out["any_rate_limit_header"] = any(
        v is not None for k, v in out.items() if k != "raw"
    )
    return out


def project_exhaustion(
    parsed: Dict[str, Any], requests_per_s: float
) -> Optional[Dict[str, Any]]:
    """How long the remaining quota lasts at a given request rate."""
    remaining = parsed.get("remaining_requests")
    if remaining is None or requests_per_s <= 0:
        return None
    seconds = remaining / requests_per_s
    return {
        "remaining_requests": remaining,
        "requests_per_s": requests_per_s,
        "seconds_to_exhaustion": seconds,
        "note": (
            "Linear projection from the quota remaining at this instant. It "
            "ignores the reset window, so it is a floor on time-to-exhaustion, "
            "not a schedule."
        ),
    }


def summarise_probe(levels: Sequence[Dict[str, Any]],
                    threshold: float) -> Dict[str, Any]:
    binding = None
    for lv in levels:
        if lv["rate_limited_fraction"] > threshold:
            binding = lv["concurrency"]
            break

    return {
        "levels": list(levels),
        "threshold": threshold,
        "binding_concurrency": binding,
        "max_clean_concurrency": max(
            (lv["concurrency"] for lv in levels
             if lv["rate_limited_fraction"] == 0.0), default=None
        ),
        "verdict": (
            f"Rate limiting became binding at concurrency {binding} "
            f"(>{threshold * 100:.0f}% of requests returned 429)."
            if binding else
            "No level exceeded the threshold — the ladder did not find the "
            "wall. Either the limit is above the range probed, or the account "
            "tier has no request-rate cap that this workload reached."
        ),
        # The single most misquotable number in this file. A ceiling found here
        # belongs to one API key on one plan at one moment; it is not a property
        # of the provider, the model, or hosted APIs in general. Attached to the
        # result rather than left to the reader's memory.
        "scope_qualifier": (
            "THIS RESULT IS A PROPERTY OF THIS ACCOUNT TIER, NOT OF THE API. "
            "Rate limits are per-key and per-plan, and providers change them "
            "without notice. A ceiling found here says what this credential "
            "could do on this date — nothing about what a funded or "
            "contracted account can do. Any claim that hosted serving is "
            "'blocked by rate limits' must carry this qualifier or it is "
            "simply false."
        ),
        "not_a_ceiling_on": [
            "the provider's capacity",
            "the model's capacity",
            "what a higher tier or a negotiated contract would allow",
        ],
    }


def _wrap_qualifier(text: str, width: int = 74) -> List[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def format_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    A = lines.append
    A("=" * 84)
    A("RATE LIMIT PROBE")
    A("=" * 84)
    A(f"  {'C':>6} {'sent':>6} {'429s':>6} {'429 %':>8} {'ok req/s':>10} "
      f"{'remaining':>10} {'reset':>8}")
    A("  " + "-" * 74)
    for lv in result["levels"]:
        parsed = lv.get("rate_limit_parsed") or {}
        A(f"  {lv['concurrency']:>6} {lv['sent']:>6} {lv['rate_limited']:>6} "
          f"{lv['rate_limited_fraction'] * 100:>7.1f}% "
          f"{lv['completed_per_s']:>10.2f} "
          f"{_fmt(parsed.get('remaining_requests')):>10} "
          f"{_fmt(parsed.get('reset')):>8}")
    A("")
    A(f"  {result['verdict']}")
    A("")
    A("  " + "!" * 76)
    for chunk in _wrap_qualifier(result["scope_qualifier"]):
        A(f"  {chunk}")
    A("  Not a ceiling on: " + "; ".join(result["not_a_ceiling_on"]) + ".")
    A("  " + "!" * 76)
    if result.get("max_clean_concurrency") is not None:
        A(f"  Highest concurrency with zero 429s: {result['max_clean_concurrency']}")
    for lv in result["levels"]:
        proj = lv.get("exhaustion_projection")
        if proj:
            A(f"  At C={lv['concurrency']}: quota {proj['remaining_requests']:.0f} "
              f"exhausts in ~{proj['seconds_to_exhaustion']:.0f}s at "
              f"{proj['requests_per_s']:.2f} req/s")
    if not any(
        (lv.get("rate_limit_parsed") or {}).get("any_rate_limit_header")
        for lv in result["levels"]
    ):
        A("")
        A("  NOTE: the provider sent no recognisable rate-limit headers, so "
          "remaining quota and reset time are unknown. Only the observed 429 "
          "rate is evidence here.")
    return "\n".join(lines)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.0f}"


async def _probe(args) -> int:
    fixture, _ = resolve_fixture(
        "shared_prefix", max(args.n_requests, 1), args.context_tokens,
        20260820, "Qwen/Qwen2.5-7B-Instruct", args.fixtures_dir,
        use_stub=args.stub_tokenizer,
    )
    meta = fixture.to_meta()
    ladder = [c for c in (args.ladder or DEFAULT_LADDER)
              if c <= args.max_concurrency]

    est = estimate_cost(len(ladder), args.n_requests, meta["token_count_min"],
                        args.max_tokens, args.price_input_per_m,
                        args.price_output_per_m)

    print("=" * 84)
    print("RATE LIMIT PROBE — plan")
    print("=" * 84)
    print(f"  model          {args.model}")
    print(f"  ladder         {ladder}")
    print(f"  N per level    {args.n_requests}")
    print(f"  stop when      >{args.threshold * 100:.0f}% of a level is 429")
    print(f"  ESTIMATED COST ${est['estimated_cost_usd']:.4f} "
          f"({est['total_requests']} requests, if none are rejected)")

    if args.dry_run:
        print("\n--dry-run: nothing submitted, nothing billed.")
        return 0

    api_key = load_api_key(args.api_key_env)
    if not api_key:
        print(f"\nERROR: {args.api_key_env} not set.", file=sys.stderr)
        return 2
    if est["estimated_cost_usd"] > args.max_cost_usd:
        print(f"\nREFUSING: estimated ${est['estimated_cost_usd']:.4f} exceeds "
              f"--max-cost-usd ${args.max_cost_usd:.4f}.", file=sys.stderr)
        return 3

    prompts_all = [(r["request_id"], r["prompt_text"]) for r in fixture.requests]
    levels: List[Dict[str, Any]] = []

    for c in ladder:
        prompts = prompts_all[:args.n_requests]
        print(f"\n[probe] concurrency={c} ...")
        records, info = await run_level(
            prompts, c, url=args.base_url, model=args.model, api_key=api_key,
            max_tokens=args.max_tokens, temperature=0.0, ignore_eos=False,
            stream=False, timeout=args.timeout, max_retries=1,
            price_in=args.price_input_per_m, price_out=args.price_output_per_m,
        )
        sent = len(records)
        limited = info["rate_limit_429_count"]
        ok = sum(1 for r in records if r.ok)
        frac = (limited / sent) if sent else 0.0
        parsed = parse_rate_limit_headers(info.get("rate_limit_headers_last") or {})
        rps = ok / info["section_wall_s"] if info["section_wall_s"] else 0.0

        level = {
            "concurrency": c, "sent": sent, "completed": ok,
            "rate_limited": limited, "rate_limited_fraction": frac,
            "completed_per_s": rps, "wall_s": info["section_wall_s"],
            "cost_usd": info["api_cost_total"],
            "status_counts": info["status_counts"],
            "rate_limit_parsed": parsed,
            "exhaustion_projection": project_exhaustion(parsed, rps),
        }
        levels.append(level)
        print(f"  sent {sent}  ok {ok}  429s {limited} ({frac * 100:.1f}%)  "
              f"{rps:.2f} req/s")

        if frac > args.threshold:
            print(f"  [stop] {frac * 100:.1f}% rate-limited exceeds the "
                  f"{args.threshold * 100:.0f}% threshold — the wall is here.")
            break

    result = summarise_probe(levels, args.threshold)
    result["model"] = args.model
    result["api_key_sha256"] = api_key_hash(api_key)
    result["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["total_cost_usd"] = sum(lv["cost_usd"] for lv in levels)

    print("\n" + format_report(result))
    print(f"\n  total billed ${result['total_cost_usd']:.5f}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[probe] {args.json_out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Probe the provider's rate limit.")
    ap.add_argument("--ladder", type=int, nargs="*", default=None)
    ap.add_argument("--max-concurrency", type=int, default=512)
    ap.add_argument("--n-requests", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="short outputs: the probe measures admission, not "
                         "generation, and short completions burn less quota")
    ap.add_argument("--context-tokens", type=int, default=2620)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--price-input-per-m", type=float, default=DEFAULT_PRICE_INPUT_PER_M)
    ap.add_argument("--price-output-per-m", type=float, default=DEFAULT_PRICE_OUTPUT_PER_M)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-cost-usd", type=float, default=0.50)
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--stub-tokenizer", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    return asyncio.run(_probe(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
