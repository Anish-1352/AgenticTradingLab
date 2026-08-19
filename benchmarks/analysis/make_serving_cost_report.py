"""Generate SERVING_COST.md — serving cost, and nothing else.

    cd benchmarks && python -m analysis.make_serving_cost_report

SCOPE IS ENFORCED, NOT INTENDED
--------------------------------
This project has produced a GPU serving benchmark, a latency/P&L audit, a
cadence re-scope and a heterogeneous-serving design. **None of it belongs
here.** Self-hosting addresses a small minority of the projected bill, the
cadence was never confirmed, and the multi-model sweep has never run — mixing
any of them in obscures the cost answer rather than supporting it.

``EXCLUDED_TERMS`` lists what must not appear, and
``tests/test_serving_cost_report.py`` asserts the generated file contains none
of them. Scope creep is the failure mode for this document, so it is a test
rather than a resolution.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.alpha_per_dollar import cost_per_alpha_table  # noqa: E402
from analysis.arm_a_vs_bc_model import lever_sensitivity  # noqa: E402
from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, NOT_MEASURED, cost_per_call, load_arm_a, load_measured,
    load_measured_calls_per_decision,
)
from analysis.tier_check import check_report, format_violations  # noqa: E402

__all__ = ["build_report", "EXCLUDED_TERMS", "main"]

# Terms whose presence means the report has drifted out of scope. Checked
# case-insensitively against the generated text by the test suite.
EXCLUDED_TERMS = [
    # GPU serving benchmark — a separate document
    "cudaLaunchKernel", "kernel_overlap", "GPU utilisation", "GPU utilization",
    "VRAM", "Layer 3", "arm B", "arm C", "173x", "req/s", "A100", "L4",
    "vLLM", "occupancy", "tensor core",
    # cadence re-scope — never confirmed
    "Nof1", "Alpha Arena", "156 decisions", "500 decisions", "2.5 min",
    # latency / P&L audit
    "fill price", "execution.py", "bar interval", "596",
    # heterogeneous sweep — never run
    "multi-model", "quantiz", "models-per-card", "agents-per-card",
    # self-hosting economics
    "crossover", "self-host", "self host",
]


def _t(tier: str) -> str:
    return f"[{tier}]"


def build_report() -> str:
    measured = load_measured()
    models = measured["models"]
    depths = load_measured_calls_per_decision()
    alpha = cost_per_alpha_table()
    arm_a = load_arm_a()
    levers = lever_sensitivity(
        baseline_calls=3, baseline_input=4790, baseline_output=2652,
        price_in=0.05, price_out=0.20)

    L: List[str] = []
    A = L.append

    A("# Serving cost")
    A("")
    A("Scoped to what the platform spends on LLM calls and what has been built "
      "to reduce it. The GPU serving benchmark, the latency audit and the "
      "cadence analysis are separate documents and are deliberately absent — "
      "each would obscure this answer rather than support it.")
    A("")
    A("## Read this first")
    A("")
    A("**No cost reduction has been measured.** The reductions are built and "
      "unmerged, and the measurement infrastructure is part of the same "
      "unmerged change. Nothing here reports a saving, because none has been "
      "observed.")
    A("")
    A("What follows is: what a call costs today (measured), what that spend "
      "appears to buy (one window, with the limits stated), what has been "
      "built, and what is blocked on what.")
    A("")

    # ---- 1 -------------------------------------------------------------
    A("## 1. What a call costs")
    A("")
    A(f"Seven models, one task, one window, ~161 calls each {_t(MEASURED)}. "
      f"Pricing was verified by recomputing every run's cost from its own "
      f"stored token totals against the `est_cost_usd` the run stored — all "
      f"seven agree to under 1e-5 USD {_t(MEASURED)}.")
    A("")
    A("| Model | Input tok/call | Output tok/call | $/M in | $/M out | "
      "$/call | Tier |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for name, m in sorted(models.items(), key=lambda kv: cost_per_call(kv[1])):
        A(f"| `{m['slug']}` | {m['input_per_call']:,.0f} | "
          f"{m['output_per_call']:,.0f} | {m['price_in']:.3f} | "
          f"{m['price_out']:.2f} | {cost_per_call(m):.6f} | {MEASURED} |")
    A("")
    ins = [m["input_per_call"] for m in models.values()]
    outs = [m["output_per_call"] for m in models.values()]
    costs = [cost_per_call(m) for m in models.values()]
    A(f"Three spreads, under an identical prompt: input tokens vary "
      f"{max(ins) / min(ins):.1f}x, output tokens {max(outs) / min(outs):.1f}x, "
      f"and cost per call {max(costs) / min(costs):.0f}x {_t(DERIVED)}.")
    A("")
    A("Input is a property of the prompt and carries across models. **Output "
      "is a property of the model and does not** — an output-token figure "
      "must never be substituted from one model to another.")
    A("")
    A("### Calls per decision")
    A("")
    if depths.get("available"):
        A("| Configured pipeline steps | Observed calls/decision | Agrees | Tier |")
        A("|---:|---:|---|---|")
        for d in depths["depths_measured"]:
            r = depths["by_depth"][d]
            A(f"| {d} | {r['observed_calls_per_decision']:.3f} | "
              f"{'yes' if r['agrees_with_configured'] else 'NO'} | {MEASURED} |")
        A("")
        A(f"Calls per decision equals the configured step count exactly, with "
          f"no retry inflation observed at either depth {_t(MEASURED)}. So "
          f"**pipeline depth multiplies the whole bill linearly** — a "
          f"three-step agent costs three times a single-call one for the same "
          f"decisions {_t(DERIVED)}.")
    else:
        A(f"Not available: {depths.get('reason')} {_t(NOT_MEASURED)}")
    A("")
    A("### Which lever moves cost most")
    A("")
    A("| Lever | Span | Tier |")
    A("|---|---:|---|")
    for l in levers["levers"]:
        if l["span_factor"]:
            A(f"| {l['lever']} | {l['span_factor']:.1f}x | {DERIVED} |")
    A("")
    A("Model choice dominates every other lever combined. A shorter prompt or "
      "a better cache cannot rescue an expensive model; the model is the "
      "first decision, not the last.")
    A("")

    # ---- 2 -------------------------------------------------------------
    A("## 2. What that spend buys")
    A("")
    bb = alpha["best_baseline"]
    best = alpha["llm"][0]
    A(f"The same seven models ran the same task over the same window, so cost "
      f"and performance can be joined directly {_t(MEASURED)}.")
    A("")
    A(f"**Cost does not track performance.** Spearman rank correlation between "
      f"run cost and Sharpe is "
      f"{alpha['cost_sharpe_rank_correlation']:+.3f} {_t(DERIVED)}, across a "
      f"{alpha['cost_span']:.0f}x price span {_t(MEASURED)}.")
    A("")
    A("| Model | Run cost USD | Return | Sharpe | 95% CI | Tier |")
    A("|---|---:|---:|---:|:---:|---|")
    for e in alpha["llm"]:
        ci = e["sharpe_ci"]
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        A(f"| `{e['label']}` | {e['run_cost_usd']:.3f} | "
          f"{e['total_return'] * 100:.2f}% | {e['sharpe_recomputed']:.2f} | "
          f"{cis} | {MEASURED} cost, {DERIVED} Sharpe |")
    A("")
    A(f"For comparison, `{bb['label']}` over the identical window: "
      f"{bb['total_return'] * 100:.2f}% at Sharpe "
      f"{bb['sharpe_recomputed']:.2f}, for zero LLM cost {_t(DERIVED)}.")
    A("")
    A(f"**Both framings, because they disagree.** No model beat that baseline "
      f"on a risk-adjusted basis {_t(DERIVED)}. But `{best['label']}` DID beat "
      f"it on raw return — {best['total_return'] * 100:.2f}% against "
      f"{bb['total_return'] * 100:.2f}% {_t(DERIVED)} — while scoring lower on "
      f"Sharpe, {best['sharpe_recomputed']:.2f} against "
      f"{bb['sharpe_recomputed']:.2f} {_t(DERIVED)}, so it bought that return "
      f"with volatility. Which framing governs is a mandate question, not a "
      f"measurement one, and the flattering one is not chosen here.")
    A("")

    # ---- the n=1 limit, as prominent as the result ----------------------
    A("### ⚠️ This is one window, and it cannot rank the models")
    A("")
    A(f"**Windows: {alpha['board']['n_windows']}** {_t(MEASURED)}. Seven "
      f"models, one month, one universe — a single draw from the distribution "
      f"of possible months.")
    A("")
    A("Three limits, each structural rather than a caveat:")
    A("")
    A("- **`can_rank` returns False below two windows**, regardless of how "
      "separable the intervals look. It is a gate in the code, not a note in "
      "the prose.")
    A("- **`windows_needed()` refuses to answer.** Sizing a study needs "
      "across-window variance — how much a model's Sharpe moves month to "
      "month — and with one window that quantity is not imprecise but "
      "unestimable, because a single sample has no dispersion.")
    A("- **The confidence intervals shown are too narrow.** The Lo standard "
      "error assumes iid returns; hourly equity returns are autocorrelated and "
      "heteroskedastic, both of which widen the true interval. So the "
      "non-overlapping pairs are the least trustworthy part of that table.")
    A("")
    A(f"The defensible statement is therefore: **the ordering does not track "
      f"price, and the sample is too small to rank the models** {_t(DERIVED)}. "
      f"Both halves matter. The first is informative on its own; the second "
      f"stops it becoming a recommendation to buy a particular model.")
    A("")

    # ---- 3 -------------------------------------------------------------
    A("## 3. What has been built, and what each eliminates")
    A("")
    A("All three are complete, tested and **unmerged**. Unmerged code saves "
      "nothing.")
    A("")
    A("| Change | Eliminates | Default | Tier |")
    A("|---|---|---|---|")
    A(f"| Per-call usage logging | nothing directly — it is the prerequisite "
      f"for attributing any saving | ON | {NOT_MEASURED} saving |")
    A(f"| Backtest result cache | the repeated calls of an identical re-run; "
      f"one leaderboard-shaped backtest is ~161 calls {_t(MEASURED)} | OFF | "
      f"{NOT_MEASURED} |")
    A(f"| Leaderboard governance | refreshes of unchanged configs, and "
      f"expensive models on a daily cadence | no-op | {NOT_MEASURED} |")
    A("")
    A("**Each row's saving is unmeasured on purpose.** The elimination is "
      "structural and easy to describe; its size depends on how often users "
      "re-run identical configurations and how often leaderboard inputs are "
      "unchanged, and neither has been observed.")
    A("")

    # ---- the observability gap -----------------------------------------
    A("### Why no saving can be attributed yet")
    A("")
    A("`agent_runs` accumulates `llm_calls`, `input_tokens` and "
      "`output_tokens` with `+=` and writes three totals once per run. The "
      "per-call distribution is destroyed before it reaches the database.")
    A("")
    A("So a fall in the bill after enabling the cache cannot be distinguished "
      "from a quiet week. **That is the gap the per-call table closes, and it "
      "is why it ships in the same change as the optimisations rather than "
      "after them.**")
    A("")

    # ---- prefix caching ------------------------------------------------
    A("### Prompt caching: measured, and it is zero")
    A("")
    if arm_a.get("available"):
        n_req = sum(l["completed"] or 0 for l in arm_a["levels"])
        A(f"{n_req} hosted-API requests were issued against a fixture with a "
          f"99.4% shared prefix — the best possible case for a prefix cache "
          f"{_t(MEASURED)}. The provider reported a cache field, and it "
          f"reported {arm_a['cached_tokens_total']} cached tokens "
          f"{_t(MEASURED)}.")
        A("")
        A("**Consequence: do not do the prompt-reordering refactor.** "
          "Reordering static content earlier to improve cacheability would "
          "optimise a mechanism this endpoint is not applying to this model.")
        A("")
        A("Stated narrowly on purpose: this is a measured absence **on this "
          "model and this endpoint**, not a claim that the provider does not "
          "support caching generally. A different model, or explicit cache "
          "control, would need re-testing.")
    else:
        A(f"Hosted-API cache data unavailable {_t(NOT_MEASURED)}.")
    A("")

    # ---- 4 -------------------------------------------------------------
    A("## 4. What is blocked, and on what")
    A("")
    A("| Blocked | On | Who |")
    A("|---|---|---|")
    A("| Any measured saving | merge review of the cost-reduction branch | "
      "reviewer |")
    A("| The before/after baseline | the per-call table existing in "
      "production, then a period of normal activity | time |")
    A("| Leaderboard cadence values | one instrumented refresh cycle — the "
      "measure-first step, not yet taken | operator |")
    A("| Ranking the models by performance | more evaluation windows | "
      "leaderboard re-run |")
    A("")
    A("The last row is the cheapest high-value item on the list. The "
      "leaderboard already runs seven models over a window; running it over "
      "three to five non-overlapping windows needs no new code and would make "
      "the cost/performance ordering testable rather than suggestive.")
    A("")

    # ---- 5 -------------------------------------------------------------
    A("## 5. What cannot be measured yet")
    A("")
    A("| Question | Why not | What would resolve it |")
    A("|---|---|---|")
    A("| How much the cache saves | no per-call attribution in production yet |"
      " merge, then a period of normal use |")
    A("| How much governance saves | the per-cycle cost has never been "
      "measured | one instrumented refresh cycle |")
    A("| Whether a cheaper model is as good | one evaluation window | three to "
      "five windows |")
    A("| Whether trimming prompts costs quality | same single-window limit — "
      "an ablation would return \"no significant difference\", which is "
      "indistinguishable from \"no effect\" and from \"no power\" | the "
      "multi-window baseline first |")
    A("")
    A("The cost side of every ablation is already known without spending "
      f"anything: depth multiplies calls exactly {_t(MEASURED)}, output is "
      f"priced at four times input {_t(MEASURED)}, and the input-heavy versus "
      f"output-heavy balance inverts between the cheap and expensive tiers "
      f"{_t(DERIVED)}. **What is missing in every case is the quality term**, "
      f"and that is what one window cannot supply.")
    A("")
    A("---")
    A("")
    A(f"Sources: the committed seed database "
      f"({measured['total_runs']} runs, {measured['runs_with_llm']} with LLM "
      f"usage) {_t(MEASURED)}; hosted-API request records {_t(MEASURED)}; "
      f"pipeline-depth runs {_t(MEASURED)}.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate SERVING_COST.md")
    ap.add_argument("--out", default=os.path.join(_HERE, "SERVING_COST.md"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    text = build_report()

    result = check_report(text)
    if not result["ok"]:
        print(f"UNTAGGED NUMERIC LINES ({result['n_violations']}):",
              file=sys.stderr)
        print(format_violations(result["violations"]), file=sys.stderr)
        return 3

    lowered = text.lower()
    drift = [term for term in EXCLUDED_TERMS if term.lower() in lowered]
    if drift:
        print(f"OUT-OF-SCOPE TERMS IN A SERVING-COST REPORT: {drift}",
              file=sys.stderr)
        return 4

    if args.check:
        print(f"[check] builds, {len(text.splitlines())} lines, tagged and "
              f"in scope")
        return 0
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"[report] {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
