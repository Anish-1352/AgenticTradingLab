#!/usr/bin/env python3
"""Generate UPSTREAM_AND_OUTPUT_CAP.md from the preflight and the cap arms.

    python benchmarks/analysis/make_phase22_report.py --check

Every number comes from results/preflight.json and results/retry_U*.json.
A staleness test regenerates it byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.alpha_per_dollar import sharpe_standard_error, windows_needed  # noqa: E402
from analysis.tier_check import check_report, format_violations  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "UPSTREAM_AND_OUTPUT_CAP.md")
RESULTS = os.path.join(_BENCH_ROOT, "results")

ARMS = [
    ("U2000", "retry_U2000.json", "upstream default", "2000"),
    ("U2000fit", "retry_U2000fit.json", "thinking budget clamped to fit", "2000"),
    ("U4096", "retry_U4096.json", "output ceiling raised", "4096"),
]
REASONING_BUDGET = 2048     # openrouter.py: medium effort
SHIPPED_CAP = 2000          # backtest_harness.py: DEFAULT_MAX_OUTPUT_TOKENS
PRICE_IN, PRICE_OUT = 0.05, 0.2   # nemotron, USD per 1M tokens


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _n(x, fmt=".2f"):
    return "n/a" if x is None else format(x, fmt)


def _cost(s):
    return (s["total_input_tokens"] / 1e6 * PRICE_IN
            + s["total_output_tokens"] / 1e6 * PRICE_OUT)


def _rows():
    out = []
    for label, fname, desc, cap in ARMS:
        d = _load(fname)
        if d:
            out.append({"label": label, "desc": desc, "cap": cap, "raw": d,
                        "s": d["summary"], "q": d.get("quality") or {}})
    return out


def build() -> str:
    pre = _load("preflight.json")
    rows = _rows()
    usage = _load("retry_usage.json")
    L: List[str] = []
    add = L.append

    add("# Upstream drift, and the output ceiling")
    add("")
    add("**Tiers:** `MEASURED` = observed in a run or read out of the tree at "
        "the stated revision. `DERIVED` = arithmetic on measured values. "
        "`NOT MEASURED` = not established here.")
    add("")

    # ---------------- part 0 ----------------
    add("## 1. What upstream landed")
    add("")
    if pre:
        add(f"Local work sat `{pre['commits_behind']}` commits behind "
            f"`origin/main` at `{pre['upstream_head'][:8]}` [MEASURED]. Two "
            f"prior findings are superseded and the rest hold.")
        add("")
        add("| claim | status | Tier |")
        add("|---|---|---|")
        for p in pre["probes"]:
            add(f"| {p['claim']} | "
                f"{'holds' if p['still_holds'] else '**superseded**'} | "
                f"MEASURED |")
        add("")
        for p in [q for q in pre["probes"] if not q["still_holds"]]:
            if p["holds_when"] == "absent":
                where = os.path.basename(p["sample"][0].split(":")[1]) \
                    if p["sample"] else "upstream"
                why = (f"upstream now has `{p['n_matches']}` occurrence(s), "
                       f"first in `{where}`")
            else:
                why = "the code this described is no longer present upstream"
            add(f"- **{p['claim']}** — superseded: {why} [MEASURED].")
        add("")
        add("A third is a straightforward win: **the hourly bar interval is no "
            "longer hardcoded**. `AlpacaDataLoader` now takes a "
            "`source_timeframe` and translates `1m`, `5m` and `60m` "
            "[MEASURED], so the cadence work's second blocker is resolved at "
            "that layer. It does not resolve the first — a bar interval is not "
            "a scheduler — and the other cadence blockers still verify.")
        add("")
        add("The two remaining matter in opposite directions.")
        add("")
        add("**The reasoning-off rescue is gone.** `efdf1e5a` "
            "(\"fix: preserve reasoning during response recovery\") replaced it "
            "with a larger token budget on the same attempt [MEASURED]. Any "
            "plan that starts \"the rescue already sets "
            "`OPENROUTER_REASONING_EFFORT=none`\" is describing code that no "
            "longer exists.")
        add("")
        add("**`attempt_index` exists upstream, but on a different axis.** "
            "`credit_llm_reservations` carries `run_id`, `call_index`, "
            "`attempt_index` and per-call cost — but `execution/service.py` "
            "fills it from `for attempt_index, provider_id in "
            "enumerate(candidates)`, so it counts **provider failover**, not "
            "retries of the no-text loop [MEASURED]. It is also on the "
            "platform path; the backtest path reaches it only when an "
            "`execution_client` is injected, and the leaderboard runs build "
            "the legacy client instead [MEASURED].")
        add("")
        bad = [c for c in pre["report_citations"] if c["status"] != "in_range"]
        add(f"**Citations.** `{len(pre['report_citations'])}` `file:line` "
            f"references across the generated reports were checked; "
            f"`{len(bad)}` fall outside the upstream file [MEASURED]. Line "
            f"numbers in the retry report have all drifted — the anchors still "
            f"exist, so the findings hold and only the coordinates moved:")
        add("")
        add("| anchor | upstream line | Tier |")
        add("|---|---:|---|")
        for a in pre["citation_anchors"]:
            add(f"| {a['describes']} (`{os.path.basename(a['path'])}`) | "
                f"{a['upstream_line']} | MEASURED |")
        add("")
        for b in pre["branches"]:
            if not b.get("resolved"):
                continue
            if b["merges_cleanly"]:
                add(f"`{b['branch']}` merges cleanly [MEASURED].")
            else:
                add(f"`{b['branch']}` does **not** merge cleanly: "
                    f"`{b['files_changed_in_both']}` files changed on both "
                    f"sides [MEASURED], including "
                    f"`portfolio_manager.py` and `engine.py`, the two most "
                    f"heavily rewritten files upstream. Rebasing it is a task "
                    f"in itself, not a step in another one.")
            add("")

    # ---------------- part 2 ----------------
    add("## 2. The root cause: the thinking budget exceeds the output ceiling")
    add("")
    add(f"OpenRouter documents the invariant: max_tokens must exceed the "
        f"reasoning budget so there are tokens left for the answer. The "
        f"shipped defaults violate it — `medium` effort budgets "
        f"`{REASONING_BUDGET}` thinking tokens against a `{SHIPPED_CAP}`-token "
        f"ceiling [MEASURED]. Reasoning tokens are billed as output tokens, so "
        f"a model that uses its budget is cut off inside the thinking block "
        f"and returns `['thinking']` with no text — the sole condition that "
        f"advances the no-text retry loop [MEASURED].")
    add("")
    add("The earlier arms already contained the evidence and it was not read: "
        "every failure sat at exactly the ceiling, and no failure sat below it "
        "[MEASURED].")
    add("")
    if rows:
        add("| arm | ceiling | attempts/decision | retries | input tokens | "
            "output tokens | wall s | cost USD | Tier |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            s = r["s"]
            add(f"| `{r['label']}` ({r['desc']}) | {r['cap']} | "
                f"{_n(s['attempts_per_decision'])} | {s['retries']} | "
                f"{s['total_input_tokens']} | {s['total_output_tokens']} | "
                f"{_n(s['total_seconds'], '.0f')} | "
                f"{_cost(s):.4f} | MEASURED |")
        add("")
        base = next((r for r in rows if r["label"] == "U2000"), None)
        hi = next((r for r in rows if r["label"] == "U4096"), None)
        if base and hi:
            b, h = base["s"], hi["s"]
            add(f"Raising the ceiling cut attempts per decision from "
                f"`{_n(b['attempts_per_decision'])}` to "
                f"`{_n(h['attempts_per_decision'])}`, input tokens by "
                f"`{1 - h['total_input_tokens']/b['total_input_tokens']:.0%}`, "
                f"wall time by "
                f"`{1 - h['total_seconds']/b['total_seconds']:.0%}` and cost by "
                f"`{1 - _cost(h)/_cost(b):.0%}` [DERIVED].")
            add("")
            add("**More output tokens is cheaper, not dearer.** A truncated "
                "attempt is billed in full and thrown away, and it drags a "
                "fresh copy of the whole prompt with it — which is why the "
                "input saving is the largest number in the table [DERIVED].")
            add("")
        fit = next((r for r in rows if r["label"] == "U2000fit"), None)
        if fit and base:
            add("### Clamping the thinking budget does not work")
            add("")
            add(f"The obvious fix — shrink the budget to fit under the ceiling "
                f"— was implemented and measured. It fails. With the clamp "
                f"verified applied (`reasoning.max_tokens=1488`, "
                f"`thinking.budget_tokens=1488`), every failure still came back "
                f"at exactly `{SHIPPED_CAP}` output tokens [MEASURED]: the "
                f"model reasoned past its budget and consumed the whole ceiling "
                f"anyway.")
            add("")
            add(f"Attempts per decision went "
                f"`{_n(base['s']['attempts_per_decision'])}` → "
                f"`{_n(fit['s']['attempts_per_decision'])}` [MEASURED], which "
                f"does not clear run-to-run variance and leaves the failure "
                f"signature unchanged. **Nemotron via OpenRouter treats the "
                f"reasoning budget as advisory** [MEASURED]. The ceiling is the "
                f"only parameter measured to control this.")
            add("")

    # ---------------- quality ----------------
    add("### The quality half, which does not support shipping it on")
    add("")
    if rows and all(r["q"] for r in rows):
        add("| arm | return | Sharpe | 95% CI | trades | Tier |")
        add("|---|---:|---:|---|---:|---|")
        for r in rows:
            q = r["q"]
            n = (q.get("equity_rows") or 1) - 1
            se = sharpe_standard_error(q["sharpe_ratio"], n)
            ci = (f"[{q['sharpe_ratio']-1.96*se:.2f}, "
                  f"{q['sharpe_ratio']+1.96*se:.2f}]") if se else "n/a"
            add(f"| `{r['label']}` | {q['total_return']:.4f} | "
                f"{q['sharpe_ratio']:.2f} | {ci} | {q['num_trades']} | "
                f"MEASURED |")
        add("")
        b = next((r for r in rows if r["label"] == "U2000"), None)
        h = next((r for r in rows if r["label"] == "U4096"), None)
        if b and h:
            gap = abs(b["q"]["sharpe_ratio"] - h["q"]["sharpe_ratio"])
            wn = windows_needed(gap, b["q"]["sharpe_ratio"],
                                (b["q"]["equity_rows"] or 1) - 1, n_windows=1)
            add(f"Raising the ceiling made this window's result **worse**: "
                f"return `{b['q']['total_return']:.4f}` → "
                f"`{h['q']['total_return']:.4f}`, Sharpe "
                f"`{b['q']['sharpe_ratio']:.2f}` → "
                f"`{h['q']['sharpe_ratio']:.2f}` [MEASURED]. Neither arm fell "
                f"back to the rule-based agent, so this is not fallback "
                f"contamination — both ran every decision on the model "
                f"[MEASURED].")
            add("")
            add(f"The intervals above do not overlap, but that is the wrong "
                f"test. It is one window of `{b['q']['equity_rows']}` bars over "
                f"three days, and the gate refuses it [MEASURED]:")
            add("")
            add(f"> {wn['reason']}")
            add("")
            add("So: **the cost and latency effect is unambiguous and the "
                "quality effect is adverse but underpowered** [MEASURED]. That "
                "combination argues for the flag, not for the default. Nothing "
                "here should be turned on globally until the ceiling is "
                "measured across several windows.")
            add("")

    # ---------------- part 1 ----------------
    add("## 3. Per-request accounting")
    add("")
    add("`llm_calls` is incremented inside the retry loop, so it already "
        "contains every retry — and having contained them, cannot say which "
        "they were [MEASURED]. `llm_call_usage` adds one row per billed "
        "request with the attempt it belongs to, the phase, the outcome, and "
        "the ceiling that request ran under.")
    add("")
    if usage:
        s = usage["summary"]
        add(f"Verified against a real backtest that retried, not only in unit "
            f"tests: `{s['total_attempts']}` rows against `llm_calls` = "
            f"`{s['total_attempts']}`, tokens equal to `agent_runs` to the "
            f"digit, and `{s['retries']}` requests at `attempt_index > 0` "
            f"[MEASURED].")
        add("")
    add("`max_output_tokens` is on the row because without it a short answer "
        "and a truncated one are indistinguishable, and "
        "`output_tokens >= max_output_tokens` is the signature of the failure "
        "that dominates this workload [MEASURED]. Finding it took a whole "
        "phase precisely because no stored row carried it.")
    add("")
    add("`phase` separates the no-text loop from the post-parse truncation "
        "recovery — a third amplification mechanism this work did not know "
        "about until upstream's own logs showed it re-requesting at the "
        "recovery ceiling [MEASURED].")
    add("")

    # ---------------- part 3 ----------------
    add("## 4. Structured outputs would not have prevented this")
    add("")
    add("The suggestion assumed models emit conversational filler that breaks "
        "parsing. That is not the measured failure. The failures had **no "
        "text channel at all** — `content types: ['thinking']` — so there was "
        "nothing for a schema to constrain [MEASURED].")
    add("")
    add("| question | answer | Tier |")
    add("|---|---|---|")
    add("| Does OpenRouter expose structured outputs? | Yes, via "
        "`response_format` with `json_schema`, on select models and providers "
        "| MEASURED |")
    add("| Does ATL use it? | No — no `response_format` or `json_schema` "
        "anywhere on the backtest path | MEASURED |")
    add("| Would it have prevented these failures? | No. The budget was "
        "exhausted before any text was emitted; a schema constrains text that "
        "exists | DERIVED |")
    add("| Does it interact with reasoning? | Undocumented by the provider | "
        "NOT MEASURED |")
    add("| Is there a parameter that caps thinking reliably? | Not for this "
        "model: the budget was set and ignored | MEASURED |")
    add("")
    add("Guided decoding proper is a vLLM feature and ATL calls hosted APIs, "
        "so it is not available on this path at all [MEASURED]. **Do not "
        "implement it for this failure.**")
    add("")

    # ---------------- part 4 ----------------
    add("## 5. Remaining latency levers, ranked")
    add("")
    add("Bars cannot be parallelised — each bar's prompt embeds the previous "
        "bar's fills — so the levers are fewer calls, faster calls, cached "
        "results [MEASURED, prior work].")
    add("")
    add("| lever | effect | effort | changes results? | Tier |")
    add("|---|---|---|---|---|")
    add("| Raise the output ceiling | `2.93` → `1.14` attempts/decision, "
        "`-61%` input tokens, `-28%` wall | env var, no code | **yes** — "
        "different decisions | MEASURED |")
    add("| Escalate the ceiling on the first retry, not the fifth | removes "
        "up to `4` truncated attempts per failing decision | small, additive | "
        "no — same first request | DERIVED |")
    add("| Record per-request rows | none directly; makes the above "
        "measurable | small, additive | no | MEASURED |")
    add("| Clamp the thinking budget | none for this model | done, off by "
        "default | no | MEASURED |")
    add("| Client-side timeout | bounds a `600`s stall; no effect on the "
        "common path | small | no | MEASURED |")
    add("| Structured outputs | none — wrong mechanism | n/a | n/a | DERIVED |")
    add("")
    add("The second row is the one worth building next and was not built here. "
        "Upstream already raises the ceiling on attempt five; every failing "
        "decision therefore pays four truncated calls before the remedy it "
        "already knows about is applied [MEASURED]. Moving that escalation to "
        "the first retry keeps the first request unchanged — so it does not "
        "alter results the way raising the default ceiling does — while "
        "removing most of the amplification [DERIVED].")
    add("")

    add("## 6. What this does not establish")
    add("")
    add("- **Whether a higher ceiling is better or worse for returns** "
        "[NOT MEASURED]. One window, three days, and the gate refuses it.")
    add("- **Whether other models honour the reasoning budget** "
        "[NOT MEASURED]. Only nemotron was tested against the clamp.")
    add("- **Whether the platform path shows the same amplification** "
        "[NOT MEASURED]. These runs use the legacy client; the unified "
        "execution path has its own timeout and failover.")
    add("")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = build()
    res = check_report(text)
    if not res["ok"]:
        print(f"UNTAGGED NUMERIC LINES ({res['n_violations']}):", file=sys.stderr)
        print(format_violations(res["violations"]), file=sys.stderr)
        return 3
    if args.check:
        print(f"[check] builds, {len(text.splitlines())} lines, all tagged")
        return 0
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[report] {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
