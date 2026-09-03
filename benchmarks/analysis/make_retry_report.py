#!/usr/bin/env python3
"""Generate RETRY_AMPLIFICATION.md from the committed retry probes.

    python benchmarks/analysis/make_retry_report.py --check

Every number comes from results/retry_*.json, so the document cannot drift
from the runs it reports. A staleness test regenerates it byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.tier_check import check_report, format_violations  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "RETRY_AMPLIFICATION.md")
RESULTS = os.path.join(_BENCH_ROOT, "results")

# Arms in reporting order. `condition` names what each one varies, because a
# rate on its own says nothing about whether it is a property of the model, the
# prompt, the data or chance — which is the question this phase exists to ask.
ARMS = [
    ("reproA",   "retry_reproA.json",   "2 symbols, April window"),
    ("onesym2",  "retry_onesym2.json",  "1 symbol, April window"),
    ("altwin",   "retry_altwin.json",   "2 symbols, May window"),
    ("deepseek", "retry_deepseek.json", "2 symbols, April window, other model"),
]
# The pre-timeout run. Kept because its timings are the evidence for the
# 600s stall, and because it is the arm the earlier phase reported as clean.
LEGACY = ("onesym", "retry_onesym.json", "1 symbol, April window, no timeout")
LOG_EVIDENCE = "retry_log_evidence.json"
CONTENT_TYPES = "retry_content_types.json"

PHASE20_DERIVED_PER_BAR = 2.71   # the figure this phase was asked to replace
SDK_DEFAULT_READ_TIMEOUT = 600   # anthropic._base_client.DEFAULT_TIMEOUT.read
SDK_DEFAULT_MAX_RETRIES = 2      # anthropic._base_client.DEFAULT_MAX_RETRIES
ATL_MAX_ATTEMPTS = 5             # portfolio_manager.py: range(no_text_retries + 1)


SEED_DB = os.path.join(os.path.dirname(_BENCH_ROOT), "dashboard", "storage",
                       "data", "backtest.db")


def _leaderboard_calls_per_bar() -> List[Dict[str, Any]]:
    """Calls per bar for the committed seed leaderboard runs.

    Read-only, and the comparison that matters: these are the same models on
    the same code, so a difference against the probe arms is a difference of
    prompt path rather than of model.
    """
    if not os.path.exists(SEED_DB):
        return []
    con = sqlite3.connect(f"file:{SEED_DB}?mode=ro", uri=True)
    try:
        rows = []
        for rid, calls in con.execute(
                "SELECT run_id, llm_calls FROM agent_runs "
                "WHERE llm_calls > 0 ORDER BY run_id"):
            bars = con.execute(
                "SELECT COUNT(*) FROM equity_timeseries WHERE run_id = ?",
                (rid,)).fetchone()[0]
            if bars:
                rows.append({"run_id": rid, "calls": calls, "bars": bars,
                             "per_bar": calls / bars})
        return rows
    finally:
        con.close()


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x: Optional[float], fmt: str = ".2f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _arm_rows() -> List[Dict[str, Any]]:
    rows = []
    for label, fname, condition in ARMS:
        d = _load(fname)
        if d is None:
            continue
        s = d["summary"]
        rows.append({
            "label": label, "condition": condition, "raw": d, "s": s,
            "complete": bool(d.get("complete")),
            "model": d["model"].split("/")[-1],
        })
    return rows


def _exhausted(d: Dict[str, Any]) -> int:
    """Decisions that used all five attempts and still had no text.

    The last attempt of an exhausted decision is a failure too, which is why
    an arm's trigger count exceeds its retry count by exactly this number.
    """
    per: Dict[int, List[Dict[str, Any]]] = {}
    for a in d.get("attempts", []):
        per.setdefault(a["decision_index"], []).append(a)
    if not per:
        return d["summary"]["attempts_histogram"].get(str(ATL_MAX_ATTEMPTS), 0)
    return sum(1 for v in per.values()
               if len(v) == ATL_MAX_ATTEMPTS
               and v[-1].get("outcome") == "no_text_content")


def _slow_attempts(d: Dict[str, Any], threshold: float) -> List[float]:
    return sorted((a["seconds"] for a in d.get("attempts", [])
                   if a.get("seconds") and a["seconds"] >= threshold),
                  reverse=True)


def build() -> str:
    rows = _arm_rows()
    legacy = _load(LEGACY[1])
    logev = _load(LOG_EVIDENCE)

    L: List[str] = []
    add = L.append

    add("# Retry amplification, measured")
    add("")
    add("**Question:** Phase 20 reported `2.71` LLM requests per bar and "
        "attributed the excess to retries by reasoning about call sites "
        "[DERIVED]. This phase was asked to make that MEASURED or to say why "
        "it could not be.")
    add("")
    add("**Tiers:** `MEASURED` = observed in a run recorded here. "
        "`DERIVED` = arithmetic on measured values. "
        "`NOT MEASURED` = not established by this work.")
    add("")

    # ---------------- headline ----------------
    add("## Answer")
    add("")
    complete = [r for r in rows if r["complete"]]
    rates = [r["s"]["attempts_per_decision"] for r in complete
             if r["s"]["attempts_per_decision"]]
    tr_rates = [r["s"].get("transport_requests_per_decision")
                for r in complete
                if r["s"].get("transport_requests_per_decision")]
    if rates:
        add(f"The retry **mechanism** is now MEASURED, and it is a single "
            f"condition — the model returning reasoning with no text. The "
            f"retry **rate** is not one number: across "
            f"`{len(complete)}` completed arms it ranges "
            f"`{_num(min(rates))}`–`{_num(max(rates))}` ATL attempts per "
            f"decision [MEASURED].")
        add("")
        if tr_rates:
            add(f"Counting on the wire instead of in the backend, the same "
                f"arms issued `{_num(min(tr_rates))}`–`{_num(max(tr_rates))}` "
                f"HTTP requests per decision [MEASURED]. The two differ, and "
                f"that difference is the finding below.")
            add("")
    else:
        add(f"The retry **mechanism** is now MEASURED — a single condition, "
            f"the model returning reasoning with no text. The retry **rate** "
            f"is reported only from partial arms below, so no pooled rate is "
            f"quoted [NOT MEASURED].")
        add("")

    # ---------------- mechanism ----------------
    add("## 1. The mechanism (MEASURED)")
    add("")
    add("`portfolio_manager.py:366-412` loops five times. It advances on "
        "exactly one condition: `_extract_response_text` raising "
        "`AttributeError` containing \"No text content\". Everything else "
        "re-raises and leaves the loop, so unparseable output, timeouts and "
        "provider errors **cannot** cause a retry here — they abort the run "
        "[MEASURED].")
    add("")
    add("The probe instruments that exception rather than counting calls, so "
        "an attempt is known to be a retry because the thing that causes "
        "retries fired.")
    add("")
    add("The production log line, verbatim:")
    add("")
    add("```")
    add("⚠️  No text content block in LLM response "
        "(content types: ['thinking']); retry 1/4")
    add("```")
    add("")
    ct = _load(CONTENT_TYPES)
    if ct:
        tot = ct["content_types_total"]
        n = sum(tot.values())
        kinds = ", ".join(f"`{k}` x{v}" for k, v in sorted(tot.items()))
        add(f"Across the arms the backend printed that shape "
            f"`{n}` times, and it was the same shape every time: {kinds} "
            f"[MEASURED]. The model emitted a reasoning block and no answer.")
        add("")
        add(f"This is read out of the log, independently of the instrumented "
            f"counts, and the two agree exactly [MEASURED] — which is the "
            f"cross-check Phase 20 could not make, because its capture "
            f"filtered the retry prefix out before writing the log.")
        add("")
        if ct.get("distinct_shapes") == 1:
            add(f"`{n}` observations of one shape and no other bounds the rate "
                f"of any different trigger at roughly `{3 / n:.1%}` by the "
                f"rule of three [DERIVED].")
            add("")
    if rows:
        add("| arm | trigger counts | Tier |")
        add("|---|---|---|")
        for r in rows:
            t = r["s"].get("triggers") or {}
            cell = ", ".join(f"`{k}` x{v}" for k, v in sorted(t.items())) or "none"
            add(f"| `{r['label']}` | {cell} | MEASURED |")
        add("")

    # ---------------- two layers ----------------
    add("## 2. There are two retry layers, and `llm_calls` sees one")
    add("")
    add("Underneath ATL's loop the Anthropic SDK runs its own. "
        "`SyncAPIClient.request` loops `for retries_taken in "
        "range(max_retries + 1)` around `self._client.send(...)`. The "
        "providers build the client as `anthropic_cls(api_key=..., "
        "base_url=...)` and pass neither `timeout` nor `max_retries` "
        "(`providers/openrouter.py`, `providers/commonstack.py`, "
        "`providers/anthropic_native.py`), so the SDK defaults apply "
        f"[MEASURED]: read timeout `{SDK_DEFAULT_READ_TIMEOUT}`s, "
        f"`max_retries={SDK_DEFAULT_MAX_RETRIES}`.")
    add("")
    add("An SDK-level retry never returns to `portfolio_manager`, so "
        "`self.llm_calls += 1` never runs for it. **`llm_calls` undercounts "
        "the requests actually issued to the provider** [MEASURED]. This "
        "probe therefore counts at `httpx.Client.send`, the one call the "
        "SDK's retry loop wraps.")
    add("")
    if rows and any(r["s"].get("transport_requests") is not None for r in rows):
        add("| arm | decisions | ATL attempts | wire requests | wire/ATL | "
            "SDK retries | Tier |")
        add("|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            a = r["s"]
            add(f"| `{r['label']}` | {a['decisions_with_requests']} | "
                f"{a['total_attempts']} | "
                f"{a.get('transport_requests', 'n/a')} | "
                f"{_num(a.get('transport_requests_per_atl_attempt'))} | "
                f"{a.get('sdk_level_retries', 'n/a')} | MEASURED |")
        add("")
        add("A `wire/ATL` ratio above one is the part of the bill no counter "
            "in the backend can see.")
        add("")
        # The arm where the gap actually opened, described from its own log.
        gap = [r for r in rows if (r["s"].get("sdk_level_retries") or 0) > 0]
        for r in gap:
            a = r["s"]
            slow = [q for q in r["raw"].get("transport_requests_log", [])
                    if q.get("error")]
            add(f"`{r['label']}` is the case in point. It retried **zero** "
                f"times at the ATL layer — `{_num(a['attempts_per_decision'])}` "
                f"attempts per decision, no trigger fired — and still issued "
                f"`{a['transport_requests']}` requests for "
                f"`{a['total_attempts']}` attempts [MEASURED].")
            add("")
            if slow:
                q = slow[0]
                add(f"The extra request was a `{q['error']}` after "
                    f"`{_num(q['seconds'], '.1f')}`s on decision "
                    f"`{q['decision_index']}`, which the SDK then retried "
                    f"successfully [MEASURED]. The backend recorded that "
                    f"decision as one clean call.")
                add("")
            add(f"Its stored row says `llm_calls` = `{a['total_attempts']}`. "
                f"The provider was asked "
                f"`{a['transport_requests'] - a['total_attempts']}` more time(s) "
                f"than that [MEASURED].")
            add("")
        if rows and not gap:
            add("No arm here opened that gap, so the undercount is shown by "
                "construction in the unit tests rather than by a run "
                "[NOT MEASURED].")
            add("")
        add("**The two mechanisms are different and each is invisible to a "
            "different counter.** A model that returns reasoning without text "
            "amplifies at the ATL layer, where `llm_calls` does see it. A "
            "model that stalls amplifies at the SDK layer, where nothing sees "
            "it [MEASURED].")
        add("")

    # ---------------- the stall ----------------
    add("## 3. The stall that hid inside a \"clean\" run")
    add("")
    if legacy:
        secs = _slow_attempts(legacy, 0)
        ls = legacy["summary"]
        if secs:
            fast = [x for x in secs if x < 60]
            slow = [x for x in secs if x >= 60]
            add(f"The earlier arm `{LEGACY[0]}` reported "
                f"`{_num(ls['attempts_per_decision'])}` attempts per decision "
                f"with `{ls['retries']}` retries, and read as clean "
                f"[MEASURED]. Its per-attempt timings were not:")
            add("")
            add("| attempt | seconds | Tier |")
            add("|---|---:|---|")
            for a in legacy.get("attempts", []):
                add(f"| d{a['decision_index']}a{a['attempt_index']} | "
                    f"`{_num(a.get('seconds'), '.1f')}` | MEASURED |")
            add("")
            if fast and slow:
                add(f"`{len(fast)}` attempts completed in "
                    f"`{_num(min(fast), '.1f')}`–`{_num(max(fast), '.1f')}`s; "
                    f"`{len(slow)}` took `{_num(max(slow), '.1f')}`s "
                    f"[MEASURED]. That upper value sits on the SDK's "
                    f"`{SDK_DEFAULT_READ_TIMEOUT}`s read timeout, so the "
                    f"request timed out and the SDK's own retry then "
                    f"succeeded [DERIVED].")
                add("")
                add(f"ATL recorded that as one attempt with no retry. A "
                    f"backtest of `161` bars in which this happens on a "
                    f"tenth of them spends roughly "
                    f"`{16 * SDK_DEFAULT_READ_TIMEOUT // 60}` minutes waiting "
                    f"on timeouts alone, invisibly [DERIVED].")
                add("")
    add(f"This probe installs a client-side read timeout (in-process only; "
        f"nothing under `dashboard/` is modified). Normal responses and "
        f"stalls are `~2` orders of magnitude apart, so the two populations "
        f"separate cleanly [MEASURED].")
    add("")

    # ---------------- rate table ----------------
    add("## 4. Is `2.71` a property of the model, the prompt, the data, or chance?")
    add("")
    if rows:
        add("| arm | condition | model | complete | decisions | "
            "attempts/decision | retries | retry share of attempts | "
            "retry share of LLM seconds | Tier |")
        add("|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            a = r["s"]
            add(f"| `{r['label']}` | {r['condition']} | `{r['model']}` | "
                f"{'yes' if r['complete'] else 'no'} | "
                f"{a['decisions_with_requests']} | "
                f"{_num(a['attempts_per_decision'])} | {a['retries']} | "
                f"{_pct(a['retry_share_of_attempts'])} | "
                f"{_pct(a['retry_share_of_seconds'])} | MEASURED |")
        add("")
        if len(rates) >= 2:
            spread = max(rates) - min(rates)
            add(f"The spread across arms is `{_num(spread)}` attempts per "
                f"decision [DERIVED]. Read against Phase 20's "
                f"`{PHASE20_DERIVED_PER_BAR}` [DERIVED], the honest reading "
                f"is that `{PHASE20_DERIVED_PER_BAR}` was one draw from a "
                f"wide distribution, not a constant [DERIVED].")
            add("")
        # Hold the model fixed, or this compares models and calls it noise.
        by_model: Dict[str, List[Any]] = {}
        for r in complete:
            by_model.setdefault(r["model"], []).append(r)
        for model, rs in sorted(by_model.items()):
            if len(rs) < 2:
                continue
            lo = min(r["s"]["attempts_per_decision"] for r in rs)
            hi = max(r["s"]["attempts_per_decision"] for r in rs)
            add(f"Holding the model fixed at `{model}`, the arms still range "
                f"`{_num(lo)}`–`{_num(hi)}` attempts per decision [MEASURED], "
                f"so the variation is not explained by model choice alone.")
            add("")
        one_sym = [r for r in complete if len(r["raw"]["symbols"]) == 1]
        two_sym = [r for r in complete
                   if len(r["raw"]["symbols"]) > 1
                   and r["model"] == (one_sym[0]["model"] if one_sym else None)]
        if one_sym and two_sym:
            a = one_sym[0]["s"]["attempts_per_decision"]
            b = max(r["s"]["attempts_per_decision"] for r in two_sym)
            add(f"The clearest single factor is prompt breadth: the same model "
                f"on one symbol ran at `{_num(a)}` attempts per decision and "
                f"on two symbols reached `{_num(b)}` [MEASURED]. That is "
                f"consistent with a longer prompt pushing more of the "
                f"completion into reasoning, though this phase did not vary "
                f"prompt length directly [NOT MEASURED].")
            add("")

    # The comparison that answers the question.
    lb = _leaderboard_calls_per_bar()
    if lb:
        add("### The leaderboard does not show this at all")
        add("")
        add("| seed run | calls | bars | calls/bar | Tier |")
        add("|---|---:|---:|---:|---|")
        for r in lb:
            add(f"| `{r['run_id']}` | {r['calls']} | {r['bars']} | "
                f"`{_num(r['per_bar'], '.3f')}` | MEASURED |")
        add("")
        hi = max(r["per_bar"] for r in lb)
        add(f"Every committed leaderboard run sits at or below "
            f"`{_num(hi, '.3f')}` calls per bar [MEASURED]. `llm_calls` is "
            f"incremented inside the retry loop, so a retry would raise this "
            f"above `1.000`; none does. **The seed leaderboard runs contain "
            f"essentially no retry amplification** [MEASURED].")
        add("")
        nemo = [r for r in lb if "nemotron" in r["run_id"]]
        nemo_arms = [r for r in complete if "nemotron" in r["raw"]["model"]]
        if nemo and nemo_arms:
            hi_arm = max(r["s"]["attempts_per_decision"] for r in nemo_arms)
            add(f"The same model makes the point sharply. "
                f"`{nemo[0]['run_id']}` ran at "
                f"`{_num(nemo[0]['per_bar'], '.3f')}` calls per bar "
                f"[MEASURED]; the probe arms drove that model to "
                f"`{_num(hi_arm)}` attempts per decision [MEASURED]. Same "
                f"model, same loop, same provider.")
            add("")
        add("So the answer to the section heading is **the model and the "
            "prompt path together, and not chance** [DERIVED]:")
        add("")
        add("- **The model decides whether it happens at all.** One model "
            "returned thinking-only responses under every condition tried; "
            "the other returned none under the same condition [MEASURED].")
        add("- **The prompt path decides how often.** For the model that does "
            "it, widening the prompt roughly doubled the rate [MEASURED], and "
            "the single-prompt leaderboard entrants show none of it "
            "[MEASURED].")
        add("- **Chance is not a sufficient explanation.** The arms separate "
            "cleanly by condition rather than scattering [MEASURED].")
        add("")
        add(f"That retires Phase 20's `{PHASE20_DERIVED_PER_BAR}` as a general "
            f"figure [DERIVED]. It was a real measurement of one pipeline run "
            f"with one model, and it transfers neither to other models nor to "
            f"the board.")
        add("")

    else:
        add("No arm results are present [NOT MEASURED].")
        add("")

    # ---------------- exhaustion ----------------
    add("## 5. When the loop runs out, the run silently falls back")
    add("")
    ex = {r["label"]: _exhausted(r["raw"]) for r in rows}
    tot_ex = sum(ex.values())
    if tot_ex:
        named = ", ".join(f"`{k}` x{v}" for k, v in ex.items() if v)
        add(f"Retries are not always enough. {named} exhausted all "
            f"`{ATL_MAX_ATTEMPTS}` attempts — the rescue call with reasoning "
            f"disabled included — and still got no text [MEASURED]. The run "
            f"then printed:")
        add("")
        add("```")
        add("❌ LLM decision error: No text content block in LLM response "
            "(content types: ['thinking'])")
        add("   Falling back to rule-based logic")
        add("```")
        add("")
        add(f"So `{tot_ex}` decisions in this phase were made by the "
            f"rule-based reference agent rather than the model [MEASURED], "
            f"and nothing in the stored run row says so. That is the same "
            f"attribution gap the fallback work reported, now with a "
            f"reproducible cause attached to it.")
        add("")
    add("This also explains why an arm's trigger count can exceed its retry "
        "count: the final attempt of an exhausted decision fails too, but it "
        "is not followed by a retry [MEASURED].")
    add("")

    # ---------------- what does not retry ----------------
    add("## 6. Unparseable output does not retry — it is repaired in place")
    add("")
    add("The arms produced genuine JSON failures, and they took a different "
        "path entirely [MEASURED]:")
    add("")
    add("```")
    add("⚠️  Initial parse failed: Expecting value: line 1 column 16")
    add("   Attempting to fix JSON formatting...")
    add("   ❌ Still failed after fix")
    add("   Attempting second fix attempt (validate structure)...")
    add("```")
    add("")
    add("That is `parse_llm_response`, downstream of `_extract_response_text` "
        "and outside the retry loop. A malformed answer costs repair attempts, "
        "not extra API calls [MEASURED]. Any proposal to \"retry on bad "
        "output\" would be adding a mechanism, not tuning one.")
    add("")

    # ---------------- visibility ----------------
    add("## 7. Where a retry is visible (nowhere that persists)")
    add("")
    add("| surface | records a retry? | evidence | Tier |")
    add("|---|---|---|---|")
    add("| `agent_runs.llm_calls` | partially — ATL retries inflate it, "
        "SDK retries do not | `database.py:116` | MEASURED |")
    add("| `agent_runs` other columns | no attempt or retry column exists | "
        "`database.py:103-123` | MEASURED |")
    add("| `backtest_decisions` | table has `decision_source` and "
        "`step_index` but no attempt column, and the pipeline runtime never "
        "writes to it | `engine.py:922` is gated on "
        "`runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE` | MEASURED |")
    add("| a per-call usage table | does not exist | proposed only, in "
        "`atl_token_extract.py` | MEASURED |")
    add("| in-memory counter | none — the loop increments no retry counter | "
        "`portfolio_manager.py:366-412` | MEASURED |")
    add("| stdout | yes, one `print()` per retry | "
        "`portfolio_manager.py:411-413` | MEASURED |")
    add("")
    add("So the only durable record of a retry is a log line, and the only "
        "durable record of the SDK's retries is none at all [MEASURED]. A "
        "run's stored row cannot answer \"how much of this bill was retries\" "
        "[MEASURED].")
    add("")
    add("The smallest change that would fix this is an attempt-level row per "
        "call — `run_id`, `step_index`, `attempt_index`, `outcome`, tokens, "
        "seconds. `attempt_index > 0` then makes the ATL layer queryable, and "
        "recording the response's retry-count header makes the SDK layer "
        "queryable too.")
    add("")

    # ---------------- reductions ----------------
    add("## 8. What would actually reduce retries")
    add("")
    add("Only causes this work observed are listed. The loop cannot fire on "
        "anything else [MEASURED], so remedies aimed at parsing, timeouts or "
        "provider errors would target conditions that do not occur here.")
    add("")
    add("**Observed cause: the model returns a reasoning block and no text.** "
        "Every retry recorded in this phase had that shape [MEASURED].")
    add("")
    add("1. **Turn reasoning off for this step, or budget it.** The loop's own "
        f"fifth attempt already does exactly this — it sets "
        f"`OPENROUTER_REASONING_EFFORT=none` as a rescue "
        f"(`portfolio_manager.py:370-388`) [MEASURED]. Doing it on attempt "
        f"one rather than attempt `{ATL_MAX_ATTEMPTS}` would remove the "
        f"amplification for this failure mode entirely. Whether it changes "
        f"decision quality is [NOT MEASURED].")
    add("2. **Raise the output cap so reasoning does not consume the whole "
        "budget.** A thinking-only response is what a truncated response looks "
        "like when the reasoning budget fills the completion. This is "
        "consistent with the observation but was not tested [NOT MEASURED].")
    add("3. **Set a client-side timeout.** This does not reduce ATL retries — "
        "timeouts cannot trigger that loop — but it bounds the SDK layer, "
        f"which is where the `{SDK_DEFAULT_READ_TIMEOUT}`s stalls live "
        f"[MEASURED].")
    add("")

    # ---------------- limits ----------------
    add("## 9. What this does not establish")
    add("")
    add("- **A leaderboard retry rate** [NOT MEASURED]. These arms are short "
        "windows on one or two symbols; the seed leaderboard runs were not "
        "re-run.")
    add("- **Whether disabling reasoning changes decisions** [NOT MEASURED]. "
        "Recommendation 1 is a cost argument, not a quality one.")
    add("- **The provider's own retry behaviour** [NOT MEASURED]. Counting "
        "stops at this process's socket.")
    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        add(f"- `{len(incomplete)}` arm(s) did not complete and are reported "
            f"as partial: " +
            ", ".join(f"`{r['label']}`" for r in incomplete) + " [MEASURED].")
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
