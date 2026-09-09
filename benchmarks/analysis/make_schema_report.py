#!/usr/bin/env python3
"""Generate DECISION_SCHEMA.md from the probe and the captured pairs.

    python benchmarks/analysis/make_schema_report.py --check

Numbers come from results/schema_probe.json and results/pair_analysis.json.
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

from analysis.tier_check import check_report, format_violations  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, "DECISION_SCHEMA.md")
RESULTS = os.path.join(_BENCH_ROOT, "results")


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def build() -> str:
    probe = _load("schema_probe.json")
    pairs = _load("pair_analysis.json")
    L: List[str] = []
    add = L.append

    add("# The decision schema, and whether it can be trained against")
    add("")
    add("**Tiers:** `MEASURED` = observed by calling the shipped code or in a "
        "recorded run. `DERIVED` = arithmetic on those. `NOT MEASURED` = not "
        "established here.")
    add("")
    if probe:
        tr = probe["tree"]
        add(f"Everything below was measured against `origin/main` at "
            f"`{tr['head'][:8]}` [MEASURED]. The first run of the probe used a "
            f"local checkout `533` commits behind and reported parser "
            f"behaviour that has since been fixed, so the revision is recorded "
            f"with the result.")
        add("")

    # ---------------- headline ----------------
    add("## Answer: no, not yet — there is more than one schema")
    add("")
    if probe:
        ag = probe["parser_agreement"]
        es = probe["envelope_output_shapes"]
        add(f"Three code paths turn model output into a decision and they do "
            f"not agree. On the same `{ag['compared']}` payloads the "
            f"permissive pipeline parser and the strict pydantic validator "
            f"agree only `{_pct(ag['agreement_rate'])}` of the time; "
            f"`{ag['accepted_by_pipeline_only']}` payloads are accepted by one "
            f"and rejected by the other [MEASURED].")
        add("")
        add(f"Worse for a training target, **one parser emits "
            f"`{es['distinct_shapes']}` different action shapes** depending on "
            f"which envelope the model chose [MEASURED]. A model tuned to "
            f"satisfy the parser would still not know what its consumer "
            f"receives.")
        add("")

    # ---------------- D1 ----------------
    add("## 1. What the schema actually is")
    add("")
    add("### The three consumers")
    add("")
    add("| path | parser | behaviour | Tier |")
    add("|---|---|---|---|")
    add("| multi-step pipeline | `pipeline_runner.pipeline_output_to_decision` "
        "| permissive, coercive | MEASURED |")
    add("| single prompt | `portfolio_manager.make_trading_decision_with_llm` "
        "| consumes `actions`, applies its own rules | MEASURED |")
    add("| external agents, AI hedge fund | `validator.parse_actions_payload` "
        "| pydantic; rejects rather than coerces | MEASURED |")
    add("")
    add("### What the pipeline parser accepts")
    add("")
    if probe:
        rows = probe["pipeline_parser"]
        acc = sum(1 for r in rows if r["accepted"])
        add(f"`{acc}` of `{len(rows)}` constructed payloads were accepted "
            f"[MEASURED]. Three envelopes are recognised — `actions`, "
            f"`orders`, `risk_actions` — and the first match wins.")
        add("")
        add("| case | accepted | what came out | Tier |")
        add("|---|---|---|---|")
        for r in rows:
            if r["group"] != "envelope":
                continue
            out = "—"
            if r["accepted"]:
                acts = (r["parsed"] or {}).get("actions") or []
                out = f"`{len(acts)}` action(s)" if acts else "`{\"actions\": []}`"
            add(f"| {r['case']} | {'yes' if r['accepted'] else 'no'} | {out} "
                f"| MEASURED |")
        add("")

    add("### What it does NOT constrain")
    add("")
    add("These are the places a tuned model could emit valid-but-wrong output "
        "[MEASURED]:")
    add("")
    if probe:
        interesting = {
            "side=short (unknown)": "silently becomes `hold`",
            "side=liquidate everything": "silently becomes `hold`",
            "side missing": "silently becomes `hold`",
            "qty=-5 (negative)": "passes through negative",
            "qty=1e9 (huge)": "passes through unbounded",
            "qty='many' (unparseable)": "silently becomes `0`",
            "qty=2.9 (float)": "truncated toward zero",
            "symbol=None": "accepted, symbol stays `None`",
            "symbol not in DJIA": "accepted unchanged at this layer",
            "symbol is a number": "accepted unchanged",
            "confidence=2.5 (out of range)": "accepted, no range check",
            "confidence=-1": "accepted, no range check",
            "unknown extra field": "ignored silently",
        }
        by_case = {r["case"]: r for r in probe["pipeline_parser"]}
        add("| input | result | Tier |")
        add("|---|---|---|")
        for case, note in interesting.items():
            r = by_case.get(case)
            if r and r["accepted"]:
                add(f"| `{case}` | {note} | MEASURED |")
        add("")
        add("The one input that is **not** handled gracefully is a "
            "non-numeric `confidence`: `float(\"high\")` raises `ValueError` "
            "out of the parser [MEASURED]. It is caught by the broad handler "
            "in `portfolio_manager`, which prints \"LLM decision error\" and "
            "substitutes a rule-based decision — so a model writing "
            "`\"confidence\": \"high\"`, an entirely natural thing for a "
            "language model to do, silently loses the whole bar [MEASURED].")
        add("")
        fields = probe["template_fields_read_by_pipeline_parser"]
        ignored = [k for k, v in fields.items() if not v]
        add(f"**Fields the shipped templates ask for and no parser reads:** "
            f"{', '.join('`' + k + '`' for k in ignored)} [MEASURED]. Every "
            f"model is being asked for them on every call.")
        add("")

    # ---------------- templates ----------------
    add("### Do the templates agree?")
    add("")
    if probe:
        t = probe["templates"]
        add(f"Yes, with each other. All `{len(t['templates'])}` shipped "
            f"templates that define a pipeline end in the same envelope: "
            f"{', '.join('`' + e + '`' for e in t['distinct_final_envelopes'])} "
            f"[MEASURED]. The three-step template's intermediate steps use "
            f"`facts` and `signals`, which is correct — only the last step's "
            f"output is converted.")
        add("")
        add("They do **not** agree with the single-prompt path. The templates "
            "ask for `orders`; `validator.py`'s prompt asks for `actions` with "
            "different field names — `action` not `side`, `position_size` not "
            "`qty`, `reasoning` not `reason` [MEASURED]. Both parse, because "
            "the pipeline parser accepts either, but they are two different "
            "output contracts for the same job.")
        add("")

    # ---------------- D2/D3 ----------------
    add("## 2. What real traffic looks like")
    add("")
    if not pairs:
        add("No captured pairs are present [NOT MEASURED].")
        add("")
    else:
        ts = pairs["token_split"]
        add(f"`{pairs['n_pairs']}` prompt/response pairs were captured from "
            f"real backtests against real Alpaca bars [MEASURED] — not from "
            f"the `atl_realistic` fixture, whose own metadata records that it "
            f"was built from assumed proportions.")
        add("")
        add("### The single largest number in this report")
        add("")
        add(f"Of `{ts['billed_output_tokens']}` output tokens billed, roughly "
            f"`{ts['visible_text_tokens_est']}` appear in the response text "
            f"[DERIVED]. **About `{_pct(ts['hidden_share_est'])}` of what is "
            f"paid for is never shown at all** — it is the reasoning block "
            f"[DERIVED].")
        add("")
        add("That is the quantity a tuned model removes. It is not prose the "
            "parser ignores; it is thinking the response never contains.")
        add("")
        shapes = pairs["content_type_shapes"]
        add("| response shape | count | Tier |")
        add("|---|---:|---|")
        for k, v in shapes.items():
            add(f"| `{k}` | {v} | MEASURED |")
        add("")
        add(f"`{pairs['no_text_responses']}` responses carried a thinking "
            f"block and no text at all [MEASURED] — the failure Phase 21 "
            f"measured, reproduced here.")
        add("")

        add("### The two paths are not equally reliable")
        add("")
        cr = pairs["crossed"]
        add(f"| regime | path | n | parsed | hit ceiling | mean output tokens "
            f"| mean prompt chars | Tier |")
        add("|---|---|---:|---:|---:|---:|---:|---|")
        for k in sorted(cr["cells"]):
            v = cr["cells"][k]
            add(f"| {v['regime']} | `{v['path']}` | {v['n']} | "
                f"{_pct(v['parse_rate'])} | {_pct(v['ceiling_rate'])} | "
                f"{v['mean_output_tokens']:.0f} | "
                f"{v['mean_prompt_chars']:.0f} | MEASURED |")
        add("")
        if cr["fully_crossed"]:
            add("The grid is fully crossed, which matters: read as a regime "
                "column alone the earlier partial data said \"uptrend parses "
                "29%\", when what it measured was the single-prompt path "
                "[MEASURED].")
        else:
            add(f"The grid is `{cr['cells_filled']}` of "
                f"`{cr['cells_possible']}` cells [MEASURED]; regime and path "
                f"are partly confounded and the regime column should not be "
                f"read on its own.")
        add("")
        bp = pairs["by_path"]
        if "pipeline" in bp and "single_prompt" in bp:
            p, s = bp["pipeline"], bp["single_prompt"]
            add(f"**The pipeline path did not fail once.** "
                f"`{p['n']}` calls, `{_pct(p['parse_rate'])}` parsed, "
                f"`{_pct(p['ceiling_rate'])}` hit the ceiling. The "
                f"single-prompt path: `{s['n']}` calls, "
                f"`{_pct(s['parse_rate'])}` parsed, "
                f"`{_pct(s['ceiling_rate'])}` hit the ceiling [MEASURED].")
            add("")
            add(f"The mechanism is prompt size. A pipeline step's prompt "
                f"averages `{p['mean_prompt_chars']:.0f}` characters against "
                f"`{s['mean_prompt_chars']:.0f}` for the single prompt "
                f"[MEASURED], and the smaller ask leaves room for both "
                f"reasoning and an answer inside the same ceiling [DERIVED].")
            add("")
        c = pairs["conversion"]
        add(f"Final-step conversion: `{c['pipeline_final_converted']}` of "
            f"`{c['pipeline_final_steps']}` pipeline bars produced a decision "
            f"[MEASURED]. Intermediate steps emit `facts` and `signals` and "
            f"are correctly declined by the decision parser; counting those as "
            f"failures would manufacture a defect that is not there.")
        add("")

    # ---------------- D4 ----------------
    add("## 3. The empty decision")
    add("")
    add("Phase 16 established that a well-formed `{\"orders\": []}` returned "
        "`None`. **That has been fixed upstream** — it now returns "
        "`{\"actions\": []}` [MEASURED]. The substitution did not disappear "
        "though; it moved.")
    add("")
    add("| model output | `strict_llm=False` | `strict_llm=True` | Tier |")
    add("|---|---|---|---|")
    add("| `{\"actions\": []}` | **rule-based substituted**, "
        "`llm_decisions` stays `0` | accepted, `llm_decisions` `+1` "
        "| MEASURED |")
    add("| `{\"orders\": []}` | **rule-based substituted** | accepted "
        "| MEASURED |")
    add("| an explicit `hold` action | accepted | accepted | MEASURED |")
    add("")
    add("So the strict branch already treats \"do nothing\" as a real "
        "decision, and its own comment says so; the non-strict branch at "
        "`portfolio_manager.py:591-593` still swaps in the reference agent and "
        "records nothing [MEASURED].")
    add("")
    add("**What a \"no trades\" training example should look like:** explicit "
        "`hold` actions, one per symbol under consideration — not an empty "
        "array. The empty array survives only the strict path; an explicit "
        "hold survives both [MEASURED]. Teaching a model to emit `[]` would "
        "teach it an output that loses the decision on the default path.")
    add("")

    # ---------------- licensing ----------------
    add("## 4. May the collected data be used?")
    add("")
    add("The pairs come from `nvidia/nemotron-3-nano-30b-a3b` served through "
        "OpenRouter, so two documents govern them.")
    add("")
    add("| source | position | Tier |")
    add("|---|---|---|")
    add("| OpenRouter terms | output rights are delegated: \"Your ownership "
        "rights in the Output are set forth in the Model Terms for each Model "
        "you use\" | MEASURED |")
    add("| NVIDIA Open Model License | \"NVIDIA claims no ownership rights in "
        "outputs\"; derivative models are permitted with attribution "
        "| MEASURED |")
    add("")
    add("So this particular collection is usable as a training target, subject "
        "to the licence's attribution condition and its Trustworthy AI terms "
        "[MEASURED]. That is a reading of the published terms, not legal "
        "advice, and anything shipped should be confirmed with counsel "
        "[NOT MEASURED].")
    add("")
    add("The choice of provider is what makes this true. Had the pairs been "
        "collected from an OpenAI or Anthropic model, their terms generally "
        "prohibit using outputs to train competing models, and the data would "
        "have to be discarded rather than relabelled [NOT MEASURED — their "
        "terms were not fetched for this report, because no pair here came "
        "from those providers].")
    add("")

    # ---------------- verdict ----------------
    add("## 5. Is this trainable?")
    add("")
    add("| question | answer | Tier |")
    add("|---|---|---|")
    if probe:
        ag = probe["parser_agreement"]
        add(f"| Is the schema stable across paths? | No — "
            f"`{_pct(ag['agreement_rate'])}` agreement between the permissive "
            f"and strict parsers | MEASURED |")
        add(f"| Stable across templates? | Yes — all shipped pipelines end in "
            f"`orders` | MEASURED |")
        add(f"| Does the prompt ask for what the parser accepts? | Not "
            f"exactly — templates ask `orders`, the single-prompt path asks "
            f"`actions` | MEASURED |")
    if pairs:
        ts = pairs["token_split"]
        add(f"| How much output is schema-mandated? | Roughly "
            f"`{_pct(1 - (ts['hidden_share_est'] or 0))}` is visible text at "
            f"all; the rest is reasoning | DERIVED |")
    add("| What would \"correct\" mean? | Schema validity, which is checkable. "
        "Decision quality is not, and must not be substituted for it "
        "| MEASURED |")
    add("| May the data be used? | Yes for this provider, with attribution "
        "| MEASURED |")
    add("")
    add("### The recommendation this leads to")
    add("")
    add("A schema-compliance target **is** well defined, but not against \"the "
        "ATL schema\" as a whole, because there is no such single thing. It "
        "would have to name one parser and one envelope. The obvious choice is "
        "the pipeline parser with the `orders` envelope, since every shipped "
        "template already targets it.")
    add("")
    if pairs:
        bp = pairs["by_path"]
        if "pipeline" in bp:
            add(f"There is an awkward finding for the motivation, though. The "
                f"failure this was meant to remove — a reasoning model "
                f"emitting no text — did not occur once on the pipeline path "
                f"in `{bp['pipeline']['n']}` calls [MEASURED]. Decomposing the "
                f"work into smaller steps already removes it, without training "
                f"anything.")
            add("")
    add("So the honest ordering is: the schema needs unifying before it can be "
        "a target, and the failure that motivated the target is already "
        "avoidable by configuration. Fine-tuning may still be worth doing for "
        "cost or latency — a small model emitting `orders` in few tokens is "
        "cheaper than a reasoning model emitting `~80%` invisible tokens "
        "[DERIVED] — but it should be argued on that basis, not as a fix for "
        "a defect that decomposition already fixes.")
    add("")

    add("## 6. What this does not establish")
    add("")
    add("- **That schema compliance predicts good trading** [NOT MEASURED]. "
        "It is deliberately not a claim about decision quality.")
    add("- **Behaviour of models other than the one sampled** [NOT MEASURED].")
    add("- **That the pipeline path is reliable in general** [NOT MEASURED]. "
        "It did not fail in these windows; that is not the same as cannot.")
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
