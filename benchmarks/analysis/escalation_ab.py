#!/usr/bin/env python3
"""Pool the paired escalation runs and test the claim they were built for.

    python benchmarks/analysis/escalation_ab.py

The claim is narrow: escalating the output ceiling on the FIRST retry removes
the long tail of decisions that burn four or five attempts at a ceiling that
has already failed. It is not a claim that every run gets cheaper -- an
escalated retry returns a longer reply, so calls fall while output tokens do
not.

Arms are paired and interleaved, because a single run of fourteen decisions has
enough variance to show a 33% swing from chance alone (2.93 and 2.21 were the
same configuration on the same window).
"""

from __future__ import annotations

import json
import os
import sys
from math import comb
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS = os.path.join(_BENCH_ROOT, "results")

OFF_ARMS = ["retry_ESCoff.json", "retry_ESCoff2.json", "retry_ESCoff3.json"]
ON_ARMS = ["retry_ESCon.json", "retry_ESCon2.json", "retry_ESCon3.json"]
# portfolio_manager.py: range(no_text_retries + 1) == 5 attempts.
TAIL_THRESHOLD = 4
PRICE_IN, PRICE_OUT = 0.05, 0.2


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pool(names: List[str]) -> Dict[str, Any]:
    runs = [d for d in (_load(n) for n in names) if d]
    if not runs:
        return {}
    dec = att = ret = tin = tout = 0
    secs = 0.0
    tail = 0
    hist: Dict[int, int] = {}
    for d in runs:
        s = d["summary"]
        dec += s["decisions_with_requests"]
        att += s["total_attempts"]
        ret += s["retries"]
        tin += s["total_input_tokens"]
        tout += s["total_output_tokens"]
        secs += s["total_seconds"]
        for k, v in s["attempts_histogram"].items():
            hist[int(k)] = hist.get(int(k), 0) + v
            if int(k) >= TAIL_THRESHOLD:
                tail += v
    returns = [d["quality"]["total_return"] for d in runs
               if d.get("quality") and d["quality"].get("total_return") is not None]
    return {
        "returns": returns,
        "mean_return": sum(returns) / len(returns) if returns else None,
        "n_runs": len(runs), "decisions": dec, "attempts": att,
        "attempts_per_decision": att / dec if dec else None,
        "retries": ret, "tail_decisions": tail,
        "tail_rate": tail / dec if dec else None,
        "histogram": dict(sorted(hist.items())),
        "input_tokens": tin, "output_tokens": tout, "seconds": secs,
        "cost_usd": tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT,
        "max_attempts_seen": max(hist) if hist else None,
    }


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided p for a 2x2 table, hypergeometric tail.

    ``a`` tail decisions with the flag off, ``b`` the rest; ``c``/``d`` the same
    with it on. Answers: if the flag did nothing, how often would the off arm
    show at least this many tail decisions? Written out rather than pulled in,
    because scipy is not a dependency of this suite.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = comb(n, col1)
    if total == 0:
        return 1.0
    p = 0.0
    for k in range(a, min(row1, col1) + 1):
        if col1 - k > n - row1:
            continue
        p += comb(row1, k) * comb(n - row1, col1 - k) / total
    return min(1.0, p)


def compare() -> Dict[str, Any]:
    off, on = pool(OFF_ARMS), pool(ON_ARMS)
    if not off or not on:
        return {"off": off, "on": on, "paired": False}
    a, b = off["tail_decisions"], off["decisions"] - off["tail_decisions"]
    c, d = on["tail_decisions"], on["decisions"] - on["tail_decisions"]
    out = {
        "off": off, "on": on, "paired": True,
        "tail_table": {"off_tail": a, "off_rest": b,
                       "on_tail": c, "on_rest": d},
        "tail_p_one_sided": fisher_exact_greater(a, b, c, d),
        "deltas": {},
    }
    for k in ("attempts_per_decision", "retries", "input_tokens",
              "output_tokens", "seconds", "cost_usd"):
        if off.get(k):
            out["deltas"][k] = on[k] / off[k] - 1.0
    return out


def main(argv: Optional[List[str]] = None) -> int:
    r = compare()
    if not r["paired"]:
        print("not enough arms present", file=sys.stderr)
        return 2
    off, on = r["off"], r["on"]
    print(f"pooled over {off['n_runs']} paired runs "
          f"({off['decisions']} vs {on['decisions']} decisions)\n")
    print(f"{'':26s} {'off':>10s} {'on':>10s} {'delta':>9s}")
    for k in ("attempts_per_decision", "retries", "tail_decisions",
              "input_tokens", "output_tokens", "seconds", "cost_usd"):
        dv = r["deltas"].get(k)
        ds = f"{dv*100:+8.1f}%" if dv is not None else ""
        print(f"  {k:24s} {off[k]:10.2f} {on[k]:10.2f} {ds:>9s}")
    print(f"\n  histogram off: {off['histogram']}")
    print(f"  histogram on : {on['histogram']}")
    t = r["tail_table"]
    print(f"\n  decisions needing >= {TAIL_THRESHOLD} attempts: "
          f"{t['off_tail']}/{off['decisions']} off vs "
          f"{t['on_tail']}/{on['decisions']} on")
    print(f"  one-sided Fisher p = {r['tail_p_one_sided']:.4f}")
    out = os.path.join(RESULTS, "escalation_ab.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
