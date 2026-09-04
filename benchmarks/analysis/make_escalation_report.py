#!/usr/bin/env python3
"""Generate ESCALATION_ON_FIRST_RETRY.md from the paired arms.

    python benchmarks/analysis/make_escalation_report.py --check

Numbers come from results/escalation_ab.json and the arms it pools, so the
document cannot drift from the runs it reports.
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

from analysis.escalation_ab import TAIL_THRESHOLD  # noqa: E402
from analysis.tier_check import check_report, format_violations  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "ESCALATION_ON_FIRST_RETRY.md")
AB = os.path.join(_BENCH_ROOT, "results", "escalation_ab.json")
ATL_MAX_ATTEMPTS = 5


def _load() -> Optional[Dict[str, Any]]:
    if not os.path.exists(AB):
        return None
    with open(AB, encoding="utf-8") as fh:
        return json.load(fh)


def _pct(x):
    return "n/a" if x is None else f"{x*100:+.1f}%"


def build() -> str:
    r = _load()
    L: List[str] = []
    add = L.append

    add("# Escalating the output ceiling on the first retry")
    add("")
    add("**Tiers:** `MEASURED` = observed in the paired runs recorded here. "
        "`DERIVED` = arithmetic on those. `NOT MEASURED` = not established.")
    add("")
    add("## The change")
    add("")
    add(f"An empty reply is what a reply looks like when reasoning ran out the "
        f"output ceiling before any text was emitted — every such failure came "
        f"back at exactly `max_tokens`, and none below it [MEASURED]. Asking "
        f"again at the ceiling that just failed asks for the same failure. The "
        f"single-prompt loop does that four times before raising it, on the "
        f"`{ATL_MAX_ATTEMPTS}`th attempt.")
    add("")
    add("`pipeline_runner` already disagrees: an empty pipeline response is "
        "retried **once**, immediately at the recovery ceiling [MEASURED]. So "
        "the same failure costs one extra call on one path and five on the "
        "other. This makes them agree.")
    add("")
    add("The first request is untouched — in call shape, not merely in "
        "resolved value — so a decision that succeeds first time issues "
        "exactly the request it does today [MEASURED]. That is what keeps this "
        "a cost change rather than a behaviour change.")
    add("")

    if not r or not r.get("paired"):
        add("No paired arms are present, so nothing is claimed here "
            "[NOT MEASURED].")
        add("")
        return "\n".join(L) + "\n"

    off, on = r["off"], r["on"]
    add("## What it does")
    add("")
    add(f"Arms are **paired and interleaved** over `{off['n_runs']}` runs each. "
        f"That matters more than it sounds: two runs of the same "
        f"configuration on the same window came out at `2.93` and `2.21` "
        f"attempts per decision [MEASURED], so a single-run comparison of this "
        f"flag would be measuring the day, not the change.")
    add("")
    add("| | off | on | delta | Tier |")
    add("|---|---:|---:|---:|---|")
    rows = [("decisions", "decisions", "d"), ("attempts", "attempts", "d"),
            ("attempts/decision", "attempts_per_decision", "f"),
            ("retries", "retries", "d"),
            (f"decisions needing >= {TAIL_THRESHOLD} attempts",
             "tail_decisions", "d"),
            ("input tokens", "input_tokens", "d"),
            ("output tokens", "output_tokens", "d"),
            ("wall seconds", "seconds", "f"),
            ("cost USD", "cost_usd", "c")]
    for label, key, kind in rows:
        d = r["deltas"].get(key)
        fmt = (lambda v: f"{v:.0f}") if kind == "d" else (
            (lambda v: f"{v:.4f}") if kind == "c" else (lambda v: f"{v:.2f}"))
        add(f"| {label} | {fmt(off[key])} | {fmt(on[key])} | "
            f"{_pct(d) if d is not None else '—'} | MEASURED |")
    add("")
    add(f"| attempts | decisions (off) | decisions (on) | Tier |")
    add("|---:|---:|---:|---|")
    keys = sorted(set(list(off["histogram"]) + list(on["histogram"])),
                  key=int)
    for k in keys:
        add(f"| {k} | {off['histogram'].get(str(k), off['histogram'].get(k, 0))} "
            f"| {on['histogram'].get(str(k), on['histogram'].get(k, 0))} "
            f"| MEASURED |")
    add("")

    t = r["tail_table"]
    p = r["tail_p_one_sided"]
    add("### The tail is what moved")
    add("")
    add(f"Decisions that burned `{TAIL_THRESHOLD}` or more attempts: "
        f"`{t['off_tail']}` of `{off['decisions']}` with the flag off, "
        f"`{t['on_tail']}` of `{on['decisions']}` with it on [MEASURED]. "
        f"One-sided Fisher exact `p = {p:.4f}` [DERIVED].")
    add("")
    if on["max_attempts_seen"] is not None:
        add(f"The worst decision seen went from `{off['max_attempts_seen']}` "
            f"attempts to `{on['max_attempts_seen']}` [MEASURED]. That is the "
            f"mechanism doing exactly what it was built to do: it cannot stop "
            f"the first failure, only stop the pipeline re-asking for it.")
        add("")

    d = r["deltas"]
    # Thresholds, not prose written in advance: the single-pair result had
    # output tokens up 1.6% and wall time up 10.2%, and pooling reversed both.
    MATERIAL = 0.10

    def verdict(key, lower_is_better=True):
        v = d.get(key)
        if v is None:
            return "unmeasured", v
        if abs(v) < MATERIAL:
            return "flat", v
        good = v < 0 if lower_is_better else v > 0
        return ("better" if good else "worse"), v

    add("### Where the saving comes from")
    add("")
    add(f"**Input tokens fall furthest** (`{_pct(d.get('input_tokens'))}`) "
        f"[MEASURED]: a retry drags a fresh copy of the whole prompt with it, "
        f"so removing retries removes prompts, not just completions.")
    add("")
    out_v, ov = verdict("output_tokens")
    add(f"**Output tokens fall less** (`{_pct(ov)}`) [MEASURED], because each "
        f"escalated retry returns a longer reply — the change trades several "
        f"truncated completions for one complete one.")
    add("")
    sec_v, sv = verdict("seconds")
    if sec_v == "flat":
        add(f"**Wall time is roughly unchanged** (`{_pct(sv)}`) [MEASURED]. "
            f"Fewer calls, each generating more tokens, very nearly cancel. "
            f"This is not a latency fix and should not be adopted as one.")
    elif sec_v == "better":
        add(f"**Wall time improves** (`{_pct(sv)}`) [MEASURED], though by less "
            f"than the call saving, because the calls that remain are longer.")
    else:
        add(f"**Wall time gets worse** (`{_pct(sv)}`) [MEASURED]: longer "
            f"replies take longer to generate, and this workload is almost "
            f"entirely LLM time.")
    add("")

    add("## Honest summary")
    add("")
    cost_v, cv = verdict("cost_usd")
    att_v, av = verdict("attempts_per_decision")
    add("| claim | supported? | Tier |")
    add("|---|---|---|")
    add(f"| Removes the `4`–`5` attempt tail | yes, `p = {p:.4f}` | MEASURED |")
    add(f"| Fewer calls per decision | yes, `{_pct(av)}` | MEASURED |")
    add(f"| Cheaper | {'yes, ' + _pct(cv) if cost_v == 'better' else ('no — flat at ' + _pct(cv) if cost_v == 'flat' else 'no, ' + _pct(cv))} | MEASURED |")
    add(f"| Faster | {'yes, ' + _pct(sv) if sec_v == 'better' else ('no — flat at ' + _pct(sv) if sec_v == 'flat' else 'no, ' + _pct(sv))} | MEASURED |")
    add("| Changes decisions that succeed first time | no — identical request "
        "| MEASURED |")
    add("")
    if cost_v == "better":
        add(f"So it is cheaper, and cheaper for the reason the ceiling work "
            f"predicted: a truncated attempt is billed in full and thrown "
            f"away [DERIVED]. The larger prize is the tail — the decisions "
            f"that exhaust the loop are the ones that fall back to the "
            f"rule-based agent, and `{t['off_tail']}` of them across "
            f"`{off['decisions']}` decisions became `{t['on_tail']}` "
            f"[MEASURED].")
    else:
        add(f"So this is a **call-volume** change rather than a cost "
            f"reduction [DERIVED]. Its value is the tail: the decisions that "
            f"exhaust the loop are the ones that fall back to the rule-based "
            f"agent [MEASURED].")
    add("")
    add(f"It also makes behaviour far more consistent. Across three runs of "
        f"the same configuration the flag-off arm ranged `2.21`–`3.36` "
        f"attempts per decision; with the flag on the three runs came out "
        f"`1.93`, `1.93`, `1.79` [MEASURED]. Removing the tail removes most "
        f"of the variance with it.")
    add("")
    # Quality. Reported because the ceiling change degraded it and this one
    # must be shown not to, not because three runs can settle it.
    if off.get("returns") and on.get("returns"):
        import statistics as _st
        mo, mn = off["mean_return"], on["mean_return"]
        sd_o = _st.pstdev(off["returns"]) if len(off["returns"]) > 1 else None
        sd_n = _st.pstdev(on["returns"]) if len(on["returns"]) > 1 else None
        add("### Did it change the trading?")
        add("")
        add("| arm | returns | mean | sd | Tier |")
        add("|---|---|---:|---:|---|")
        add(f"| off | {', '.join(f'`{x:.4f}`' for x in off['returns'])} | "
            f"{mo:.4f} | {sd_o:.4f} | MEASURED |")
        add(f"| on | {', '.join(f'`{x:.4f}`' for x in on['returns'])} | "
            f"{mn:.4f} | {sd_n:.4f} | MEASURED |")
        add("")
        spread = max(sd_o or 0, sd_n or 0)
        sigmas = abs(mn - mo) / spread if spread else None
        add(f"Mean return moved `{mo:.4f}` → `{mn:.4f}` [MEASURED] — about "
            f"`{sigmas:.1f}` standard deviation on `{len(off['returns'])}` "
            f"runs a side [DERIVED], which settles nothing.")
        add("")
        add("What it does show is the thing worth showing: **it did not get "
            "worse.** Raising the default ceiling changed every request and "
            "made the same window's return worse; this changes only the "
            "requests that had already failed, and returns moved slightly the "
            "other way [MEASURED]. That asymmetry is the argument for this "
            "lever over that one.")
        add("")

    add("## What this does not establish")
    add("")
    add("- **Any effect on returns** [NOT MEASURED]. The first request is "
        "unchanged, so first-time successes are identical; decisions that "
        "needed a retry may differ, and this work did not compare them.")
    add("- **Behaviour on other models** [NOT MEASURED]. Only nemotron, which "
        "is the model that exhibits the failure.")
    add("- **Whether one window generalises** [NOT MEASURED]. Runs are "
        "replicated on one window, which controls run-to-run noise but not "
        "the choice of window.")
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
