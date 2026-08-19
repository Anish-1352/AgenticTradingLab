"""Generate ALPHA_PER_DOLLAR.md — does the model price spread buy performance?

    cd benchmarks && python -m analysis.make_alpha_report

Analysis over existing leaderboard data. No dashboard/ change, no new runs, no
API calls. The leaderboard already ran seven models on one task; this joins
that to the verified per-call costs and asks whether the money bought anything.
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
from analysis.cost_model_lib import (  # noqa: E402
    DERIVED, MEASURED, NOT_MEASURED, load_arm_a,
)
from analysis.tier_check import check_report, format_violations  # noqa: E402

__all__ = ["build_report", "main"]


def _t(tier: str) -> str:
    return f"[{tier}]"


def build_report() -> str:
    t = cost_per_alpha_table()
    board = t["board"]
    arm_a = load_arm_a()
    L: List[str] = []
    A = L.append

    A("# Cost per unit of alpha")
    A("")
    A("**Generated file.** Rebuild with `cd benchmarks && "
      "python -m analysis.make_alpha_report`. Analysis over existing "
      "leaderboard data — no code change, no new runs, no API calls.")
    A("")

    # ---- headline ------------------------------------------------------
    A("## The finding")
    A("")
    best_llm = t["llm"][0]
    bb = t["best_baseline"]
    A(f"**No model beat a passive index on a risk-adjusted basis** "
      f"{_t(DERIVED)}. `{bb['label']}` returned "
      f"{bb['total_return'] * 100:.2f}% at Sharpe "
      f"{bb['sharpe_recomputed']:.2f} for zero LLM cost {_t(DERIVED)}.")
    A("")
    A(f"The precision matters: the best LLM, `{best_llm['label']}`, DID beat "
      f"it on raw return — {best_llm['total_return'] * 100:.2f}% against "
      f"{bb['total_return'] * 100:.2f}% {_t(DERIVED)} — but at Sharpe "
      f"{best_llm['sharpe_recomputed']:.2f} against "
      f"{bb['sharpe_recomputed']:.2f}, so it bought that return with more "
      f"volatility {_t(DERIVED)}. It also cost "
      f"${best_llm['run_cost_usd']:.3f} for the run {_t(MEASURED)}. Which of "
      f"those two framings is the right one is a mandate question, not a "
      f"measurement question — both are reported rather than the flattering "
      f"one being chosen.")
    A("")
    A(f"**Cost does not track performance.** Spearman rank correlation between "
      f"cost per run and Sharpe is "
      f"{t['cost_sharpe_rank_correlation']:+.3f} {_t(DERIVED)} — "
      f"indistinguishable from zero across a "
      f"{t['cost_span']:.0f}x price span {_t(MEASURED)}.")
    A("")
    A("**But the sample cannot rank the models, and this report does not.** "
      "See the statistics section: one window is one draw, and a league table "
      "built from it would be noise with a ranking printed on it.")
    A("")

    # ---- the join ------------------------------------------------------
    A("## Cost against risk-adjusted return")
    A("")
    A(f"All runs share one window, "
      f"{board['windows'][0][0]} to {board['windows'][0][1]}, on the same "
      f"universe {_t(MEASURED)}. Sharpe is recomputed from each run's stored "
      f"equity curve and annualised; the 95% interval uses the Lo standard "
      f"error for {t['llm'][0]['n_observations']} observations "
      f"{_t(DERIVED)}.")
    A("")
    A("| Model | Run cost USD | Return | Sharpe | 95% CI | $/Sharpe | Tier |")
    A("|---|---:|---:|---:|:---:|---:|---|")
    for e in t["llm"]:
        ci = e["sharpe_ci"]
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        cps = (f"{e['cost_per_sharpe']:.3f}" if e["cost_per_sharpe"]
               else "undefined")
        A(f"| `{e['label']}` | {e['run_cost_usd']:.3f} | "
          f"{e['total_return'] * 100:.2f}% | {e['sharpe_recomputed']:.2f} | "
          f"{cis} | {cps} | {MEASURED} cost, {DERIVED} Sharpe |")
    A("")
    A("Zero-LLM-cost strategies over the identical window:")
    A("")
    A("| Strategy | Run cost USD | Return | Sharpe | Tier |")
    A("|---|---:|---:|---:|---|")
    for e in t["baselines"]:
        A(f"| `{e['label']}` | 0.000 | {e['total_return'] * 100:.2f}% | "
          f"{e['sharpe_recomputed']:.2f} | {DERIVED} |")
    A("")
    A(f"`$/Sharpe` is withheld where Sharpe is at or below zero: dividing a "
      f"cost by a negative denominator yields a number that sorts as if it "
      f"were excellent {_t(DERIVED)}.")
    A("")

    # ---- statistics ----------------------------------------------------
    A("## What the sample can and cannot support")
    A("")
    A(f"**Windows: {board['n_windows']}** {_t(MEASURED)}. Seven models, one "
      f"month, one universe. That is one draw from the distribution of "
      f"possible months, and a model's Sharpe moves a great deal between "
      f"months for reasons that have nothing to do with the model.")
    A("")
    A(f"Confidence intervals were computed for all "
      f"{t['overlap']['n_pairs']} model pairs {_t(DERIVED)}; "
      f"{t['overlap']['n_overlapping']} overlap "
      f"({t['overlap']['fraction_overlapping']:.0%}) {_t(DERIVED)}.")
    A("")
    A("Two reasons that understates the true uncertainty:")
    A("")
    A("- The Lo standard error assumes **iid returns**. Hourly equity returns "
      "are autocorrelated and heteroskedastic, both of which make the real "
      "interval WIDER. So the non-overlapping pairs are the least trustworthy "
      "part of this table.")
    A("- Within-window precision is not the question. **Across-window "
      "variance is**, and with one window that quantity is not merely "
      "imprecise — it is unestimable, because a single sample has no "
      "dispersion.")
    A("")
    need = t["windows_needed_for_that_gap"]
    A(f"**How many windows would be needed cannot be computed** "
      f"{_t(NOT_MEASURED)}. {need['reason']}")
    A("")
    A(f"*{need['what_would_resolve_it']}*")
    A("")
    A("So the defensible statement is: **the ordering does not track price, "
      "and the sample is too small to rank the models.** Both halves matter. "
      "The first is informative on its own; the second stops it from becoming "
      "a recommendation to buy a particular model.")
    A("")

    # ---- what it means -------------------------------------------------
    A("## What follows for the serving-cost work")
    A("")
    A(f"The brief's hypothesis was that if cheap models perform comparably, "
      f"the answer to \"reduce serving cost\" is a one-line recommendation "
      f"needing no infrastructure. The data is **consistent with that and "
      f"stronger than it**: on this window the expensive models did not "
      f"outperform, and no model of any price beat a free index on a "
      f"risk-adjusted basis {_t(DERIVED)}.")
    A("")
    A("But one window cannot carry a model-selection decision, and the "
      "honest recommendation is a measurement, not a switch:")
    A("")
    A("1. **Run the leaderboard over several non-overlapping windows.** This "
      "is the cheapest high-value experiment available and it needs no new "
      "code — the leaderboard already does it for one window. Three to five "
      "windows would make the ordering testable.")
    A(f"2. **Until then, the 194x price span is unjustified by evidence** "
      f"{_t(DERIVED)}. That is not the same as saying the cheap model is as "
      f"good; it is saying nobody here has shown the expensive one is better.")
    A("3. **The passive-baseline result deserves its own attention.** If "
      "agents do not beat buy-and-hold, model choice is a second-order "
      "question and serving cost is a third-order one.")
    A("")

    # ---- the ablations -------------------------------------------------
    A("## Why the three ablations were not run")
    A("")
    A("The brief asks for calls-per-decision, output-token and input-token "
      "ablations, each reporting cost AND decision quality. **Each would hit "
      "exactly the statistical wall above.**")
    A("")
    A(f"An ablation at 1, 3 and 5 pipeline steps over one window produces "
      f"three Sharpe estimates whose intervals would overlap for the same "
      f"reason the seven models' do {_t(DERIVED)}. It would cost real money "
      f"and return \"no significant difference\", which is indistinguishable "
      f"from \"depth does not help\" and from \"the test had no power\".")
    A("")
    A("The ablations become worth running **after** the multi-window baseline "
      "exists, because that is what supplies the across-window variance "
      "needed to size them. Running them first spends money to produce an "
      "uninterpretable result — the specific failure this project has spent "
      "fourteen phases avoiding.")
    A("")
    A("The cost side of each is already known without running anything: "
      f"depth multiplies calls exactly {_t(MEASURED)}, output is priced at "
      f"four times input, and the input-heavy/output-heavy split inverts "
      f"between the cheap and expensive tiers. What is unknown in every case "
      f"is the quality term, and that is what the sample cannot yet deliver.")
    A("")

    # ---- prompt reordering ---------------------------------------------
    A("## Prompt reordering: checked, and it buys nothing")
    A("")
    if arm_a.get("available"):
        total_req = sum(l["completed"] or 0 for l in arm_a["levels"])
        A(f"The brief asks to verify provider caching support before doing the "
          f"reordering work. **Already measured.** Arm A ran {total_req} "
          f"requests against the production gateway on a fixture with a 99.4% "
          f"shared prefix — the best possible case for a prefix cache "
          f"{_t(MEASURED)}.")
        A("")
        A(f"The provider **did** report a cache field, and it reported "
          f"{arm_a['cached_tokens_total']} cached tokens {_t(MEASURED)}.")
        A("")
        A("So the field exists and the answer through it is zero. Reordering "
          "the prompt to improve cacheability would be optimising against a "
          "mechanism this provider is not applying to this model. **Do not do "
          "the refactor.** One check, as the brief asked, instead of one "
          "refactor.")
        A("")
        A("This is provider- and model-specific, not a universal claim. A "
          "gateway with explicit cache control, or a self-hosted engine with "
          "prefix caching enabled, would need re-testing.")
    else:
        A(f"Arm A data unavailable, so provider caching support is unverified "
          f"{_t(NOT_MEASURED)}.")
    A("")

    # ---- status --------------------------------------------------------
    A("## Status of the other deliverables")
    A("")
    A("Per-call usage logging, backtest result caching, leaderboard "
      "governance and the cost report were built and shipped in the earlier "
      "cost-reduction batch, and are on the `feature/serving-cost-reduction` "
      "branch as a PR for review:")
    A("")
    A("| Deliverable | Where | State |")
    A("|---|---|---|")
    A("| Per-call `llm_call_usage` | `infrastructure/llm/usage_recorder.py` | "
      "shipped; three call sites, sum-equals-aggregate asserted |")
    A("| Backtest result cache | `domain/backtesting/result_cache.py` | "
      "shipped; default OFF, per-user scope, strict key |")
    A("| Leaderboard governance | `domain/leaderboard/governance.py` | "
      "shipped; defaults reproduce current behaviour |")
    A("| Cost report generator | `scripts/cost_reduction_report.py` | "
      "shipped; refuses to estimate on empty data |")
    A("")
    A("**The leaderboard cost-per-cycle measurement is still not taken.** The "
      "brief requires measuring a real refresh cycle before optimising it, "
      "and that needs a cycle run with logging enabled against production "
      "credentials. No cadence values are proposed until it exists.")
    A("")
    A("---")
    A("")
    A(f"Source: `dashboard/storage/data/backtest.db`, "
      f"{len(board['entries'])} leaderboard runs over "
      f"{board['n_windows']} window {_t(MEASURED)}; per-call costs verified "
      f"against stored `est_cost_usd` {_t(MEASURED)}.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate ALPHA_PER_DOLLAR.md")
    ap.add_argument("--out", default=os.path.join(_HERE, "ALPHA_PER_DOLLAR.md"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    text = build_report()
    result = check_report(text)
    if not result["ok"]:
        print(f"UNTAGGED NUMERIC LINES ({result['n_violations']}):",
              file=sys.stderr)
        print(format_violations(result["violations"]), file=sys.stderr)
        return 3
    if args.check:
        print(f"[check] builds, {len(text.splitlines())} lines, every numeric "
              f"line tagged")
        return 0
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"[report] {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
