# Re-scope: ATL's cost and serving models at Nof1 cadence

**Generated file — do not edit by hand.** Regenerate with:

```bash
cd benchmarks && python -m analysis.make_cadence_report
```

## What changed, and what did not

Every earlier figure assumed ATL's hourly bar — about 7 decisions per agent per trading day [MEASURED]. Nof1's Alpha Arena runs inference every two to three minutes: about 156 decisions/day on a US equity session and 500 on 24/7 crypto [ASSUMED — not confirmed with advisor].

**No unit cost is re-derived here.** Cost per call, calls per decision, and all three arms' throughput and latency are measured and verified [MEASURED]; this recombines them at a different volume. The cadence is the one new input, and it is [ASSUMED — not confirmed with advisor] — the whole re-scope rests on it, so it is tagged everywhere it appears.

The hourly figures in `ADVISOR_REPORT.md` are **not superseded**. They are correct for ATL as it runs today. That the same measured unit costs produce opposite conclusions at the two cadences is the finding.

## 1. Cost at variable cadence

300 agents [NOT MEASURED], at each cadence and pipeline depth. Depth multiplies calls exactly — measured as 3.000 and 5.000 calls per decision against the real API [MEASURED].

| Cadence | Decisions/agent/day | Calls/decision | Calls/day | Models over $50K/mo | Tier |
|---|---:|---:|---:|---|---|
| hourly (ATL today) | 7 | 1 | 2,100 | 0/7 — none | MEASURED cadence, DERIVED cost |
| hourly (ATL today) | 7 | 3 | 6,300 | 0/7 — none | MEASURED cadence, DERIVED cost |
| hourly (ATL today) | 7 | 5 | 10,500 | 0/7 — none | MEASURED cadence, DERIVED cost |
| Nof1 equity (2.5 min) | 156 | 1 | 46,800 | 2/7 — gemini-3.1-pro, gpt-5.5 | ASSUMED cadence, DERIVED cost |
| Nof1 equity (2.5 min) | 156 | 3 | 140,400 | 3/7 — claude-sonnet-4-6, gemini-3.1-pro, gpt-5.5 | ASSUMED cadence, DERIVED cost |
| Nof1 equity (2.5 min) | 156 | 5 | 234,000 | 4/7 — claude-haiku-4-5, claude-sonnet-4-6, gemini-3.1-pro, gpt-5.5 | ASSUMED cadence, DERIVED cost |
| Nof1 crypto (2.5 min, 24/7) | 500 | 1 | 150,000 | 3/7 — claude-sonnet-4-6, gemini-3.1-pro, gpt-5.5 | ASSUMED cadence, DERIVED cost |
| Nof1 crypto (2.5 min, 24/7) | 500 | 3 | 450,000 | 6/7 — claude-haiku-4-5, claude-sonnet-4-6, deepseek-v4-pro, gemini-3.1-pro, gpt-5.5, qwen3.7-plus | ASSUMED cadence, DERIVED cost |
| Nof1 crypto (2.5 min, 24/7) | 500 | 5 | 750,000 | 6/7 — claude-haiku-4-5, claude-sonnet-4-6, deepseek-v4-pro, gemini-3.1-pro, gpt-5.5, qwen3.7-plus | ASSUMED cadence, DERIVED cost |

**At hourly cadence nothing crosses $50K/month** — 0 of 21 cells [DERIVED]. That was the earlier conclusion and it stands for ATL as it runs today.

**At Nof1 cadence most of the table crosses.** 24 of 63 cells exceed the budget [DERIVED], and at crypto cadence with a three-step pipeline six of seven models do.

The cheapest model that ever crosses is `deepseek/deepseek-v4-pro`, and only at Nof1 crypto (2.5 min, 24/7) with 3 calls/decision [DERIVED].

### Monthly cost per model, one call per decision

| Model | $/call | hourly | Nof1 equity | Nof1 crypto | Tier |
|---|---:|---:|---:|---:|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 0.000447 | 20 | 439 | 2,012 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `deepseek/deepseek-v4-pro` | 0.004696 | 207 | 4,616 | 21,134 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `qwen/qwen3.7-plus` | 0.009896 | 436 | 9,726 | 44,534 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `anthropic/claude-haiku-4-5` | 0.010719 | 473 | 10,534 | 48,233 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `anthropic/claude-sonnet-4-6` | 0.030690 | 1,353 | 30,162 | 138,106 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `google/gemini-3.1-pro` | 0.069954 | 3,085 | 68,751 | 314,793 | DERIVED from MEASURED unit cost x ASSUMED cadence |
| `openai/gpt-5.5` | 0.086262 | 3,804 | 84,778 | 388,179 | DERIVED from MEASURED unit cost x ASSUMED cadence |

Only `nvidia/nemotron-3-nano-30b-a3b` stays under budget at every cadence and depth [DERIVED]. The 193x model-cost span [MEASURED] that mattered little at hourly volume decides the answer at Nof1 volume.

## 2. The self-hosting crossover, recomputed

The earlier conclusion — self-hosting never pays — was arithmetic at about 6,300 calls/day [DERIVED]. It is a property of the volume, not of the stack, and the volume is what changed.

A GPU bills the same whether saturated or idle, so self-hosting is flat and the API is a line through the origin. They cross once:

> crossover calls/day = $1,440 per month / 30 days / cost-per-call [DERIVED]

| Model | $/call | Crossover calls/day | hourly | Nof1 equity | Nof1 crypto | Tier |
|---|---:|---:|:---:|:---:|:---:|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 0.000447 | 107,352 | no | no | **yes** | DERIVED |
| `deepseek/deepseek-v4-pro` | 0.004696 | 10,221 | no | **yes** | **yes** | DERIVED |
| `qwen/qwen3.7-plus` | 0.009896 | 4,850 | no | **yes** | **yes** | DERIVED |
| `anthropic/claude-haiku-4-5` | 0.010719 | 4,478 | no | **yes** | **yes** | DERIVED |
| `anthropic/claude-sonnet-4-6` | 0.030690 | 1,564 | **yes** | **yes** | **yes** | DERIVED |
| `google/gemini-3.1-pro` | 0.069954 | 686 | **yes** | **yes** | **yes** | DERIVED |
| `openai/gpt-5.5` | 0.086262 | 556 | **yes** | **yes** | **yes** | DERIVED |

At Nof1 equity cadence (300 agents, one call per decision) six of seven models are past the crossover [DERIVED]. Only Nemotron stays cheaper on the API, and even it crosses at crypto cadence.

### Does one card suffice?

Arm C measured 9.280 completed requests/second at concurrency 32 [MEASURED]. That is a hard per-card ceiling for this model and prompt shape.

| Cadence | Card capacity req/day | Agents per GPU | A100s for 300 agents | Tier |
|---|---:|---:|---:|---|
| hourly (ATL today) | 217,154 | 31,022 | 1 | DERIVED from MEASURED throughput |
| Nof1 equity (2.5 min) | 217,154 | 1,392 | 1 | DERIVED from MEASURED throughput |
| Nof1 crypto (2.5 min, 24/7) | 801,800 | 1,604 | 1 | DERIVED from MEASURED throughput |

**One A100 suffices at every cadence considered** — the tightest case still leaves headroom of roughly 4.6x [DERIVED]. The constraint is not card count; it is whether the stack is arm C rather than arm B.

Assumptions, stated rather than buried:
- A100 80GB at $1,440/month on-demand, one card [NOT MEASURED] — a stated rate, not a quote. Provider, region and commitment move every crossover here linearly.
- Self-hosted cost is the card alone: engineering, on-call and failover are not priced, so every crossover here is a LOWER bound on the volume that justifies self-hosting.
- Throughput is Arm C's 9.280 req/s at C=32 [MEASURED] on the shared_prefix fixture — see the fixture gap.

## 3. Latency now binds

At a 3,600s bar every measured configuration fitted, which is exactly why the latency audit concluded the engine's blindness to timing was harmless [MEASURED]. At 120-180s it is not. A decision slower than its bar is not late — it is a decision the engine cannot place, because the next bar has already arrived.

| Arm | C | e2e p50 s | p95 s | p99 s | 60s | 120s | 150s | 180s | 300s | 3600s | Tier |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| B | 1 | 9.14 | 9.23 | 9.28 | fits | fits | fits | fits | fits | fits | MEASURED |
| B | 8 | 148.29 | 149.79 | 150.21 | **NO** | **NO** | fits | fits | fits | fits | MEASURED |
| B | 32 | 596.48 | 597.07 | 597.11 | **NO** | **NO** | **NO** | **NO** | **NO** | fits | MEASURED |
| C | 1 | 2.65 | 2.65 | 2.76 | fits | fits | fits | fits | fits | fits | MEASURED |
| C | 8 | 2.71 | 2.87 | 2.87 | fits | fits | fits | fits | fits | fits | MEASURED |
| C | 32 | 3.44 | 3.45 | 3.45 | fits | fits | fits | fits | fits | fits | MEASURED |
| A | 1 | 0.71 | 0.95 | 0.98 | fits | fits | fits | fits | fits | fits | MEASURED |
| A | 8 | 0.82 | 1.65 | 1.72 | fits | fits | fits | fits | fits | fits | MEASURED |
| A | 32 | 1.11 | 2.45 | 2.45 | fits | fits | fits | fits | fits | fits | MEASURED |
| A | 64 | 0.86 | 1.33 | 1.48 | fits | fits | fits | fits | fits | fits | MEASURED |
| A | 128 | 0.95 | 1.20 | 1.29 | fits | fits | fits | fits | fits | fits | MEASURED |

`fits` = p95 within the bar. `p50 only` = median fits, p95 does not. `NO` = the median already overruns. All three columns are [MEASURED].

**Arm B at concurrency 32 overruns every candidate bar.** Its 596.5s p50 [MEASURED] is about 4.0x a 150s bar [DERIVED]. Arm B at concurrency 8 is the marginal case: 148.3s p50 against a 150s bar [DERIVED] — inside it by under two seconds, which is not margin anyone should plan around.

**Arm C fits every candidate bar at every measured concurrency**, with its worst case 3.44s against 60s [MEASURED]. That is the whole argument for the batching runtime restated as a trading constraint rather than a throughput number.

**Arm A caveat.** Its e2e ran 0.71-1.11s p50 [MEASURED], but the provider ignored `min_tokens` and returned 53-86 output tokens against the 256 arms B and C forced [MEASURED]. Its latency is real and is what a hosted decision would actually cost in wall time, but it measures less work, so it is **not directly comparable**.

One further multiplier: these are ONE call. A multi-step pipeline is sequential, so a 3-step decision costs roughly 3x these figures end to end — from the 3.000 / 5.000 calls per decision [MEASURED].

**This is the first result in the project where serving performance has a consequence rather than being a benchmark number.** At the hourly bar the arm B / arm C choice moved a throughput figure; at Nof1 cadence [ASSUMED — not confirmed with advisor] it decides whether decisions happen at all.

## 4. What ATL cannot do at this cadence

Audited against `origin/main` — what ships — not against the benchmark branch, whose copies of two of these files have since diverged. Every citation is re-read at generation time (4 of 4 verified [MEASURED]).

**These are blockers, not work items.** Whether to build any of them is the advisor's call.

### `dashboard/backend/execution/paper_backend.py:5` [MEASURED]

```python
new code (live order submission, a realtime decision-cadence scheduler, live bar
```

There is no scheduler. Paper trading is an explicit stub with no order submission and no step loop, and a 'realtime decision-cadence scheduler' is named as one of the things that would have to be built. [MEASURED]

*Consequence:* Nothing in the codebase can fire a decision every 150 seconds. This is the primary blocker: every other item is downstream of having a clock. [DERIVED]

### `dashboard/backend/infrastructure/market_data/alpaca_bars.py:455` [MEASURED]

```python
timeframe=self.TimeFrame.Hour,
```

The bar interval is hardcoded to one hour at the fetch site, and every market profile declares timeframe="60m". [MEASURED]

*Consequence:* Minute or 2.5-minute bars are unreachable without changing both the fetch and the profile table. [DERIVED]

### `dashboard/scripts/backtest_hourly_agent.py:251` [MEASURED]

```python
if args.timeframe is not None and args.timeframe != market_profile.timeframe:
```

--timeframe looks like a knob but is a guard: it errors unless the value equals the profile's own timeframe. [MEASURED]

*Consequence:* Passing --timeframe 1m fails rather than switching cadence. There is no configuration path to a sub-hourly run. [DERIVED]

### `dashboard/backend/domain/trading/execution.py:526` [MEASURED]

```python
price = market_data[symbol]["close"]
```

The fill price is the close of the same bar the decision was computed from, and the agent is shown that same close as the current price (portfolio.py:95). Decision price and fill price are the same number. [MEASURED]

*Consequence:* At an hourly bar this was a harmless simplification. At a 150s bar with a 596s decision the fill would be roughly four bars stale, and the engine has no quantity in which to express that. [DERIVED]

### Market data is not the blocker

Alpaca returns minute bars on the free tier: a probe for one symbol over one day returned 744 minute bars [MEASURED], against 32 hourly bars for the same window [MEASURED]. Two-and-a-half-minute bars would be aggregated from those.

So the data exists and the credential works. **Every blocker above is code-side.**

### What would have to change to model a stale fill

The engine cannot express "this decision arrived four bars late" because it has no decision timestamp and takes the fill from the same bar object it built the prompt from. Modelling it needs, at minimum: a decision-completion time carried alongside the decision; a fill that selects the bar at that time rather than the bar the prompt came from; and a policy for decisions whose bar has already passed — drop, place late, or queue. That is a change to `dashboard/`, which this project has held read-only throughout.

## 5. The fixture gap — top GPU-session priority

Arms B and C have only ever run on the shared_prefix fixture: 2,620 tokens per request with a 99.4% common prefix, by construction a deliberate UPPER BOUND on cross-agent overlap. [MEASURED]

**atl_realistic (~10% overlap, built from measured ATL prompt structure) has never been executed on arm B or arm C.** [NOT MEASURED]

Every serving number in this project therefore describes a workload we have separately established is not ATL's. A 99.4% [MEASURED] shared prefix is the best possible case for a prefix cache and for batching efficiency.

### Claims that would change

- The 173x arm C / arm B throughput ratio at C=32 [MEASURED] — taken under maximal prefix sharing, and expected to shrink at ~10% overlap. Anything derived from it, including the GPU ceiling and every crossover in this module, moves with it.
- Arm C's 9.28 req/s per card [MEASURED], and therefore agents-per-GPU and the number of A100s a fleet needs.
- Any statement about prefix-cache benefit, which was never separately ablated on either arm.

### Claims that hold regardless

- Arm B is launch-bound: 2.57M cudaLaunchKernel calls costing 2.6x the GPU's busy time, on a single stream with 79% idle [MEASURED]. A dispatch-mechanism finding; it does not depend on prompt overlap.
- Arm B's throughput INVERTS with concurrency, 0.109 -> 0.054 req/s [MEASURED]. The mechanism is per-request Python dispatch, not the prompts.
- Pipeline steps are sequential with a real data dependency, so intra-decision latency adds regardless of fixture.

TOP priority for the next GPU session: re-run arms B and C on atl_realistic at the same concurrency levels. Until then every throughput-derived figure carries the upper-bound caveat.

Nothing was re-run for this report; the gap is flagged, not closed.

---

Sources: `results/armB_shared_summary.json`, `results/armC_shared_summary.json`, `results/armA_nemotron_summary.json`, the committed seed database, and `origin/main` for every code citation [MEASURED]. Cadence figures [ASSUMED — not confirmed with advisor].
