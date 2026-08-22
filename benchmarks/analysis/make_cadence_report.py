"""Generate CADENCE_RESCOPE.md — the Nof1-cadence re-computation.

    cd benchmarks && python -m analysis.make_cadence_report

Separate from ADVISOR_REPORT.md on purpose. That report's hourly figures are
correct for ATL as it runs today and are NOT superseded — the comparison
between the two cadences is itself the finding. Replacing them would destroy
it.

Nothing here is a new measurement. Unit costs, calls-per-decision, and all
three arms' throughput and latency are measured elsewhere and verified; this
recombines them at a different decision cadence. The cadence is the single new
input and it is ASSUMED, so every figure downstream carries that tag.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from analysis.cadence_model import (  # noqa: E402
    ASSUMED, A100_USD_PER_MONTH, CADENCES, DEFAULT_AGENTS, FIXTURE_GAP,
    cadence_table, crossover_table, latency_vs_bar_table,
)
from analysis.cost_model_lib import DERIVED, MEASURED, NOT_MEASURED  # noqa: E402
from analysis.tier_check import check_report, format_violations  # noqa: E402

__all__ = ["build_report", "BLOCKERS", "main"]

BUDGET = 50_000.0

# Audited against origin/main — what actually ships — not against this branch,
# whose copies of two of these files have since diverged. Each is re-read and
# verified at generation time.
BLOCKERS: List[Dict[str, str]] = [
    {
        "id": "no_scheduler",
        "ref": "origin/main",
        "file": "dashboard/backend/execution/paper_backend.py",
        "line": 5,
        "expect": "a realtime decision-cadence scheduler",
        "blocker": (
            "There is no scheduler. Paper trading is an explicit stub with no "
            "order submission and no step loop, and a 'realtime "
            "decision-cadence scheduler' is named as one of the things that "
            "would have to be built."),
        "consequence": (
            "Nothing in the codebase can fire a decision every 150 seconds. "
            "This is the primary blocker: every other item is downstream of "
            "having a clock."),
    },
    {
        "id": "hourly_hardcoded",
        "ref": "origin/main",
        "file": "dashboard/backend/infrastructure/market_data/alpaca_bars.py",
        # Moved 114 -> 455 by upstream refactoring. The claim is unchanged and
        # still verified against origin/main; only the line drifted.
        "line": 455,
        "expect": "timeframe=self.TimeFrame.Hour",
        "blocker": (
            "The bar interval is hardcoded to one hour at the fetch site, and "
            "every market profile declares timeframe=\"60m\"."),
        "consequence": (
            "Minute or 2.5-minute bars are unreachable without changing both "
            "the fetch and the profile table."),
    },
    {
        "id": "timeframe_flag_is_an_assertion",
        "ref": "origin/main",
        "file": "dashboard/scripts/backtest_hourly_agent.py",
        "line": 251,
        "expect": "args.timeframe != market_profile.timeframe",
        "blocker": (
            "--timeframe looks like a knob but is a guard: it errors unless the "
            "value equals the profile's own timeframe."),
        "consequence": (
            "Passing --timeframe 1m fails rather than switching cadence. There "
            "is no configuration path to a sub-hourly run."),
    },
    {
        "id": "fill_equals_decision_price",
        "ref": "origin/main",
        "file": "dashboard/backend/domain/trading/execution.py",
        # Moved 169 -> 526 by upstream refactoring; claim unchanged.
        "line": 526,
        "expect": 'price = market_data[symbol]["close"]',
        "blocker": (
            "The fill price is the close of the same bar the decision was "
            "computed from, and the agent is shown that same close as the "
            "current price (portfolio.py:95). Decision price and fill price "
            "are the same number."),
        "consequence": (
            "At an hourly bar this was a harmless simplification. At a 150s "
            "bar with a 596s decision the fill would be roughly four bars "
            "stale, and the engine has no quantity in which to express that."),
    },
]


def _t(tier: str) -> str:
    return f"[{tier}]"


def verify_blockers(repo_root: str = _REPO_ROOT) -> List[Dict[str, Any]]:
    """Re-read every cited line from the named ref before printing it."""
    out: List[Dict[str, Any]] = []
    for b in BLOCKERS:
        rec = {**b, "verified": False, "actual": None}
        try:
            blob = subprocess.run(
                ["git", "show", f"{b['ref']}:{b['file']}"],
                cwd=repo_root, capture_output=True, text=True, check=True).stdout
            actual = blob.split("\n")[int(b["line"]) - 1].strip()
            rec["actual"] = actual
            rec["verified"] = b["expect"] in actual
        except (subprocess.CalledProcessError, IndexError, ValueError) as exc:
            rec["actual"] = f"unreadable: {exc}"
        out.append(rec)
    return out


def build_report() -> str:
    L: List[str] = []
    A = L.append

    cad = cadence_table()
    cross = crossover_table()
    lat = latency_vs_bar_table()
    blockers = verify_blockers()

    A("# Re-scope: ATL's cost and serving models at Nof1 cadence")
    A("")
    A("**Generated file — do not edit by hand.** Regenerate with:")
    A("")
    A("```bash")
    A("cd benchmarks && python -m analysis.make_cadence_report")
    A("```")
    A("")
    A("## What changed, and what did not")
    A("")
    A(f"Every earlier figure assumed ATL's hourly bar — about "
      f"{CADENCES['hourly'].decisions_per_day} decisions per agent per trading "
      f"day {_t(MEASURED)}. Nof1's Alpha Arena runs inference every two to "
      f"three minutes: about {CADENCES['nof1_equity'].decisions_per_day} "
      f"decisions/day on a US equity session and "
      f"{CADENCES['nof1_crypto'].decisions_per_day} on 24/7 crypto "
      f"{_t(ASSUMED)}.")
    A("")
    A(f"**No unit cost is re-derived here.** Cost per call, calls per decision, "
      f"and all three arms' throughput and latency are measured and verified "
      f"{_t(MEASURED)}; this recombines them at a different volume. The cadence "
      f"is the one new input, and it is {_t(ASSUMED)} — the whole re-scope "
      f"rests on it, so it is tagged everywhere it appears.")
    A("")
    A("The hourly figures in `ADVISOR_REPORT.md` are **not superseded**. They "
      "are correct for ATL as it runs today. That the same measured unit costs "
      "produce opposite conclusions at the two cadences is the finding.")
    A("")

    # ---- 1 -------------------------------------------------------------
    A("## 1. Cost at variable cadence")
    A("")
    A(f"{DEFAULT_AGENTS} agents {_t(NOT_MEASURED)}, at each cadence and "
      f"pipeline depth. Depth multiplies calls exactly — measured as 3.000 and "
      f"5.000 calls per decision against the real API {_t(MEASURED)}.")
    A("")
    A("| Cadence | Decisions/agent/day | Calls/decision | Calls/day | "
      "Models over $50K/mo | Tier |")
    A("|---|---:|---:|---:|---|---|")
    for cad_key in ("hourly", "nof1_equity", "nof1_crypto"):
        c = CADENCES[cad_key]
        for depth in (1, 3, 5):
            rows = [r for r in cad["rows"]
                    if r["cadence"] == cad_key and r["calls_per_decision"] == depth]
            over = [r for r in rows if r["over_budget"]]
            names = ", ".join(sorted(
                r["slug"].split("/")[-1] for r in over)) or "none"
            tier = MEASURED if c.tier == MEASURED else "ASSUMED"
            A(f"| {c.name} | {c.decisions_per_day} | {depth} | "
              f"{rows[0]['calls_per_day']:,} | {len(over)}/7 — {names} | "
              f"{tier} cadence, {DERIVED} cost |")
    A("")
    A(f"**At hourly cadence nothing crosses $50K/month** — 0 of 21 cells "
      f"{_t(DERIVED)}. That was the earlier conclusion and it stands for ATL "
      f"as it runs today.")
    A("")
    A(f"**At Nof1 cadence most of the table crosses.** "
      f"{cad['cells_over_budget']} of {cad['cells_total']} cells exceed the "
      f"budget {_t(DERIVED)}, and at crypto cadence with a three-step pipeline "
      f"six of seven models do.")
    A("")
    cheapest_over = cad.get("cheapest_over_budget")
    if cheapest_over:
        A(f"The cheapest model that ever crosses is "
          f"`{cheapest_over['slug']}`, and only at "
          f"{cheapest_over['cadence_name']} with "
          f"{cheapest_over['calls_per_decision']} calls/decision "
          f"{_t(DERIVED)}.")
        A("")
    A("### Monthly cost per model, one call per decision")
    A("")
    A("| Model | $/call | hourly | Nof1 equity | Nof1 crypto | Tier |")
    A("|---|---:|---:|---:|---:|---|")
    per_model: Dict[str, Dict[str, float]] = {}
    for r in cad["rows"]:
        if r["calls_per_decision"] != 1:
            continue
        per_model.setdefault(r["slug"], {"cpc": r["cost_per_call"]})
        per_model[r["slug"]][r["cadence"]] = r["monthly_usd"]
    for slug, v in sorted(per_model.items(), key=lambda kv: kv[1]["cpc"]):
        A(f"| `{slug}` | {v['cpc']:.6f} | {v['hourly']:,.0f} | "
          f"{v['nof1_equity']:,.0f} | {v['nof1_crypto']:,.0f} | "
          f"{DERIVED} from {MEASURED} unit cost x ASSUMED cadence |")
    A("")
    A(f"Only `nvidia/nemotron-3-nano-30b-a3b` stays under budget at every "
      f"cadence and depth {_t(DERIVED)}. The 193x model-cost span "
      f"{_t(MEASURED)} that mattered little at hourly volume decides the "
      f"answer at Nof1 volume.")
    A("")

    # ---- 2 -------------------------------------------------------------
    A("## 2. The self-hosting crossover, recomputed")
    A("")
    A(f"The earlier conclusion — self-hosting never pays — was arithmetic at "
      f"about 6,300 calls/day {_t(DERIVED)}. It is a property of the volume, "
      f"not of the stack, and the volume is what changed.")
    A("")
    A("A GPU bills the same whether saturated or idle, so self-hosting is flat "
      "and the API is a line through the origin. They cross once:")
    A("")
    A(f"> crossover calls/day = ${A100_USD_PER_MONTH:,.0f} per month / 30 days "
      f"/ cost-per-call {_t(DERIVED)}")
    A("")
    A("| Model | $/call | Crossover calls/day | hourly | Nof1 equity | "
      "Nof1 crypto | Tier |")
    A("|---|---:|---:|:---:|:---:|:---:|---|")
    for r in cross["rows"]:
        c = r["by_cadence"]
        marks = [("**yes**" if c[k]["past_crossover"] else "no")
                 for k in ("hourly", "nof1_equity", "nof1_crypto")]
        A(f"| `{r['slug']}` | {r['cost_per_call']:.6f} | "
          f"{r['crossover_calls_per_day']:,.0f} | {marks[0]} | {marks[1]} | "
          f"{marks[2]} | {DERIVED} |")
    A("")
    A(f"At Nof1 equity cadence ({DEFAULT_AGENTS} agents, one call per "
      f"decision) six of seven models are past the crossover {_t(DERIVED)}. "
      f"Only Nemotron stays cheaper on the API, and even it crosses at crypto "
      f"cadence.")
    A("")
    A("### Does one card suffice?")
    A("")
    A(f"Arm C measured {cross['arm_c_rps']:.3f} completed requests/second at "
      f"concurrency 32 {_t(MEASURED)}. That is a hard per-card ceiling for "
      f"this model and prompt shape.")
    A("")
    A("| Cadence | Card capacity req/day | Agents per GPU | A100s for "
      f"{DEFAULT_AGENTS} agents | Tier |")
    A("|---|---:|---:|---:|---|")
    for k in ("hourly", "nof1_equity", "nof1_crypto"):
        g = cross["ceilings"][k]
        A(f"| {CADENCES[k].name} | {g['capacity_requests_per_day']:,.0f} | "
          f"{g['agents_per_gpu']:,.0f} | {g['gpus_for_300_agents']} | "
          f"{DERIVED} from {MEASURED} throughput |")
    A("")
    A(f"**One A100 suffices at every cadence considered** — the tightest case "
      f"still leaves headroom of roughly "
      f"{cross['ceilings']['nof1_equity']['agents_per_gpu'] / DEFAULT_AGENTS:.1f}x "
      f"{_t(DERIVED)}. The constraint is not card count; it is whether the "
      f"stack is arm C rather than arm B.")
    A("")
    A("Assumptions, stated rather than buried:")
    for note in cross["assumptions"]:
        A(f"- {note}")
    A("")

    # ---- 3 -------------------------------------------------------------
    A("## 3. Latency now binds")
    A("")
    A(f"At a 3,600s bar every measured configuration fitted, which is exactly "
      f"why the latency audit concluded the engine's blindness to timing was "
      f"harmless {_t(MEASURED)}. At 120-180s it is not. A decision slower than "
      f"its bar is not late — it is a decision the engine cannot place, "
      f"because the next bar has already arrived.")
    A("")
    A("| Arm | C | e2e p50 s | p95 s | p99 s | " +
      " | ".join(f"{b}s" for b in lat["bar_intervals_s"]) + " | Tier |")
    A("|---|---:|---:|---:|---:|" + ":---:|" * len(lat["bar_intervals_s"]) + "---|")
    for r in lat["rows"]:
        cells = []
        for b in lat["bar_intervals_s"]:
            f = r["fits"][b]
            cells.append("fits" if f["p95"] else ("p50 only" if f["p50"] else "**NO**"))
        A(f"| {r['arm']} | {r['concurrency']} | {r['e2e_p50']:.2f} | "
          f"{r['e2e_p95']:.2f} | {r['e2e_p99']:.2f} | " + " | ".join(cells) +
          f" | {MEASURED} |")
    A("")
    A(f"`fits` = p95 within the bar. `p50 only` = median fits, p95 does not. "
      f"`NO` = the median already overruns. All three columns are "
      f"{_t(MEASURED)}.")
    A("")
    A(f"**Arm B at concurrency 32 overruns every candidate bar.** Its 596.5s "
      f"p50 {_t(MEASURED)} is about 4.0x a 150s bar {_t(DERIVED)}. Arm B at "
      f"concurrency 8 is the marginal case: 148.3s p50 against a 150s bar "
      f"{_t(DERIVED)} — inside it by under two seconds, which is not margin "
      f"anyone should plan around.")
    A("")
    A(f"**Arm C fits every candidate bar at every measured concurrency**, with "
      f"its worst case 3.44s against 60s {_t(MEASURED)}. That is the whole "
      f"argument for the batching runtime restated as a trading constraint "
      f"rather than a throughput number.")
    A("")
    A(f"**Arm A caveat.** Its e2e ran 0.71-1.11s p50 {_t(MEASURED)}, but the "
      f"provider ignored `min_tokens` and returned 53-86 output tokens against "
      f"the 256 arms B and C forced {_t(MEASURED)}. Its latency is real and is "
      f"what a hosted decision would actually cost in wall time, but it "
      f"measures less work, so it is **not directly comparable**.")
    A("")
    A(f"One further multiplier: {lat['note']}")
    A("")
    A(f"**This is the first result in the project where serving performance "
      f"has a consequence rather than being a benchmark number.** At the "
      f"hourly bar the arm B / arm C choice moved a throughput figure; at "
      f"Nof1 cadence {_t(ASSUMED)} it decides whether decisions happen at "
      f"all.")
    A("")

    # ---- 4 -------------------------------------------------------------
    A("## 4. What ATL cannot do at this cadence")
    A("")
    A(f"Audited against `origin/main` — what ships — not against the benchmark "
      f"branch, whose copies of two of these files have since diverged. Every "
      f"citation is re-read at generation time "
      f"({sum(1 for b in blockers if b['verified'])} of {len(blockers)} "
      f"verified {_t(MEASURED)}).")
    A("")
    A("**These are blockers, not work items.** Whether to build any of them is "
      "the advisor's call.")
    A("")
    for b in blockers:
        A(f"### `{b['file']}:{b['line']}` {_t(MEASURED)}")
        A("")
        A("```python")
        A(f"{b['actual']}")
        A("```")
        A("")
        A(f"{b['blocker']} {_t(MEASURED)}")
        A("")
        A(f"*Consequence:* {b['consequence']} {_t(DERIVED)}")
        A("")
    A("### Market data is not the blocker")
    A("")
    A(f"Alpaca returns minute bars on the free tier: a probe for one symbol "
      f"over one day returned 744 minute bars {_t(MEASURED)}, against 32 "
      f"hourly bars for the same window {_t(MEASURED)}. Two-and-a-half-minute "
      f"bars would be aggregated from those.")
    A("")
    A("So the data exists and the credential works. **Every blocker above is "
      "code-side.**")
    A("")
    A("### What would have to change to model a stale fill")
    A("")
    A("The engine cannot express \"this decision arrived four bars late\" "
      "because it has no decision timestamp and takes the fill from the same "
      "bar object it built the prompt from. Modelling it needs, at minimum: a "
      "decision-completion time carried alongside the decision; a fill that "
      "selects the bar at that time rather than the bar the prompt came from; "
      "and a policy for decisions whose bar has already passed — drop, place "
      "late, or queue. That is a change to `dashboard/`, which this project "
      "has held read-only throughout.")
    A("")

    # ---- 5 -------------------------------------------------------------
    A("## 5. The fixture gap — top GPU-session priority")
    A("")
    A(f"{FIXTURE_GAP['what']} {_t(MEASURED)}")
    A("")
    A(f"**{FIXTURE_GAP['never_run']}** {_t(NOT_MEASURED)}")
    A("")
    A(f"{FIXTURE_GAP['why_it_matters']}")
    A("")
    A("### Claims that would change")
    A("")
    for c in FIXTURE_GAP["claims_that_would_change"]:
        A(f"- {c}")
    A("")
    A("### Claims that hold regardless")
    A("")
    for c in FIXTURE_GAP["claims_that_hold"]:
        A(f"- {c}")
    A("")
    A(f"{FIXTURE_GAP['priority']}")
    A("")
    A("Nothing was re-run for this report; the gap is flagged, not closed.")
    A("")

    A("---")
    A("")
    A(f"Sources: `results/armB_shared_summary.json`, "
      f"`results/armC_shared_summary.json`, `results/armA_nemotron_summary.json`, "
      f"the committed seed database, and `origin/main` for every code citation "
      f"{_t(MEASURED)}. Cadence figures {_t(ASSUMED)}.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate CADENCE_RESCOPE.md")
    ap.add_argument("--out", default=os.path.join(_HERE, "CADENCE_RESCOPE.md"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    bad = [b for b in verify_blockers() if not b["verified"]]
    if bad:
        for b in bad:
            print(f"DRIFTED CITATION {b['ref']}:{b['file']}:{b['line']} — "
                  f"expected {b['expect']!r}, found {b['actual']!r}",
                  file=sys.stderr)
        return 2

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
