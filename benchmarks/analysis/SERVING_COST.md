# Serving cost

Scoped to what the platform spends on LLM calls and what has been built to reduce it. The GPU serving benchmark, the latency audit and the cadence analysis are separate documents and are deliberately absent — each would obscure this answer rather than support it.

## Read this first

**No cost reduction has been measured.** The reductions are built and unmerged, and the measurement infrastructure is part of the same unmerged change. Nothing here reports a saving, because none has been observed.

What follows is: what a call costs today (measured), what that spend appears to buy (one window, with the limits stated), what has been built, and what is blocked on what.

## 1. What a call costs

Seven models, one task, one window, ~161 calls each [MEASURED]. Pricing was verified by recomputing every run's cost from its own stored token totals against the `est_cost_usd` the run stored — all seven agree to under 1e-5 USD [MEASURED].

| Model | Input tok/call | Output tok/call | $/M in | $/M out | $/call | Tier |
|---|---:|---:|---:|---:|---:|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 5,502 | 860 | 0.050 | 0.20 | 0.000447 | MEASURED |
| `deepseek/deepseek-v4-pro` | 4,005 | 3,396 | 0.435 | 0.87 | 0.004696 | MEASURED |
| `qwen/qwen3.7-plus` | 5,224 | 4,879 | 0.400 | 1.60 | 0.009896 | MEASURED |
| `anthropic/claude-haiku-4-5` | 4,934 | 1,157 | 1.000 | 5.00 | 0.010719 | MEASURED |
| `anthropic/claude-sonnet-4-6` | 5,017 | 1,043 | 3.000 | 15.00 | 0.030690 | MEASURED |
| `google/gemini-3.1-pro` | 4,950 | 5,005 | 2.000 | 12.00 | 0.069954 | MEASURED |
| `openai/gpt-5.5` | 3,895 | 2,226 | 5.000 | 30.00 | 0.086262 | MEASURED |

Three spreads, under an identical prompt: input tokens vary 1.4x, output tokens 5.8x, and cost per call 193x [DERIVED].

Input is a property of the prompt and carries across models. **Output is a property of the model and does not** — an output-token figure must never be substituted from one model to another.

### Calls per decision

| Configured pipeline steps | Observed calls/decision | Agrees | Tier |
|---:|---:|---|---|
| 3 | 3.000 | yes | MEASURED |
| 5 | 5.000 | yes | MEASURED |

Calls per decision equals the configured step count exactly, with no retry inflation observed at either depth [MEASURED]. So **pipeline depth multiplies the whole bill linearly** — a three-step agent costs three times a single-call one for the same decisions [DERIVED].

### Which lever moves cost most

| Lever | Span | Tier |
|---|---:|---|
| model choice | 134.4x | DERIVED |
| calls per decision | 6.0x | DERIVED |
| input tokens per call | 2.3x | DERIVED |
| prefix cache hit rate | 1.3x | DERIVED |

Model choice dominates every other lever combined. A shorter prompt or a better cache cannot rescue an expensive model; the model is the first decision, not the last.

## 2. What that spend buys

The same seven models ran the same task over the same window, so cost and performance can be joined directly [MEASURED].

**Cost does not track performance.** Spearman rank correlation between run cost and Sharpe is +0.071 [DERIVED], across a 194x price span [MEASURED].

| Model | Run cost USD | Return | Sharpe | 95% CI | Tier |
|---|---:|---:|---:|:---:|---|
| `deepseek_v4_pro` | 0.756 | 7.49% | 5.01 | [4.44, 5.57] | MEASURED cost, DERIVED Sharpe |
| `qwen3_7_plus` | 1.593 | 2.49% | 2.19 | [1.91, 2.48] | MEASURED cost, DERIVED Sharpe |
| `gemini_3_1_pro_preview` | 11.263 | 2.32% | 2.03 | [1.76, 2.31] | MEASURED cost, DERIVED Sharpe |
| `gpt_5_5` | 13.888 | 1.45% | 1.56 | [1.33, 1.79] | MEASURED cost, DERIVED Sharpe |
| `claude_sonnet_4_6` | 4.941 | 0.11% | 0.17 | [0.01, 0.32] | MEASURED cost, DERIVED Sharpe |
| `claude_haiku_4_5` | 1.726 | 0.03% | 0.08 | [-0.07, 0.24] | MEASURED cost, DERIVED Sharpe |
| `nemotron_3_nano_30b` | 0.072 | -0.22% | -0.50 | [-0.66, -0.33] | MEASURED cost, DERIVED Sharpe |

For comparison, `spy_index` over the identical window: 5.95% at Sharpe 5.78, for zero LLM cost [DERIVED].

**Both framings, because they disagree.** No model beat that baseline on a risk-adjusted basis [DERIVED]. But `deepseek_v4_pro` DID beat it on raw return — 7.49% against 5.95% [DERIVED] — while scoring lower on Sharpe, 5.01 against 5.78 [DERIVED], so it bought that return with volatility. Which framing governs is a mandate question, not a measurement one, and the flattering one is not chosen here.

### ⚠️ This is one window, and it cannot rank the models

**Windows: 1** [MEASURED]. Seven models, one month, one universe — a single draw from the distribution of possible months.

Three limits, each structural rather than a caveat:

- **`can_rank` returns False below two windows**, regardless of how separable the intervals look. It is a gate in the code, not a note in the prose.
- **`windows_needed()` refuses to answer.** Sizing a study needs across-window variance — how much a model's Sharpe moves month to month — and with one window that quantity is not imprecise but unestimable, because a single sample has no dispersion.
- **The confidence intervals shown are too narrow.** The Lo standard error assumes iid returns; hourly equity returns are autocorrelated and heteroskedastic, both of which widen the true interval. So the non-overlapping pairs are the least trustworthy part of that table.

The defensible statement is therefore: **the ordering does not track price, and the sample is too small to rank the models** [DERIVED]. Both halves matter. The first is informative on its own; the second stops it becoming a recommendation to buy a particular model.

## 3. What has been built, and what each eliminates

All three are complete, tested and **unmerged**. Unmerged code saves nothing.

| Change | Eliminates | Default | Tier |
|---|---|---|---|
| Per-call usage logging | nothing directly — it is the prerequisite for attributing any saving | ON | NOT MEASURED saving |
| Backtest result cache | the repeated calls of an identical re-run; one leaderboard-shaped backtest is ~161 calls [MEASURED] | OFF | NOT MEASURED |
| Leaderboard governance | refreshes of unchanged configs, and expensive models on a daily cadence | no-op | NOT MEASURED |

**Each row's saving is unmeasured on purpose.** The elimination is structural and easy to describe; its size depends on how often users re-run identical configurations and how often leaderboard inputs are unchanged, and neither has been observed.

### Why no saving can be attributed yet

`agent_runs` accumulates `llm_calls`, `input_tokens` and `output_tokens` with `+=` and writes three totals once per run. The per-call distribution is destroyed before it reaches the database.

So a fall in the bill after enabling the cache cannot be distinguished from a quiet week. **That is the gap the per-call table closes, and it is why it ships in the same change as the optimisations rather than after them.**

### Prompt caching: measured, and it is zero

160 hosted-API requests were issued against a fixture with a 99.4% shared prefix — the best possible case for a prefix cache [MEASURED]. The provider reported a cache field, and it reported 0 cached tokens [MEASURED].

**Consequence: do not do the prompt-reordering refactor.** Reordering static content earlier to improve cacheability would optimise a mechanism this endpoint is not applying to this model.

Stated narrowly on purpose: this is a measured absence **on this model and this endpoint**, not a claim that the provider does not support caching generally. A different model, or explicit cache control, would need re-testing.

## 4. What is blocked, and on what

| Blocked | On | Who |
|---|---|---|
| Any measured saving | merge review of the cost-reduction branch | reviewer |
| The before/after baseline | the per-call table existing in production, then a period of normal activity | time |
| Leaderboard cadence values | one instrumented refresh cycle — the measure-first step, not yet taken | operator |
| Ranking the models by performance | more evaluation windows | leaderboard re-run |

The last row is the cheapest high-value item on the list. The leaderboard already runs seven models over a window; running it over three to five non-overlapping windows needs no new code and would make the cost/performance ordering testable rather than suggestive.

## 5. What cannot be measured yet

| Question | Why not | What would resolve it |
|---|---|---|
| How much the cache saves | no per-call attribution in production yet | merge, then a period of normal use |
| How much governance saves | the per-cycle cost has never been measured | one instrumented refresh cycle |
| Whether a cheaper model is as good | one evaluation window | three to five windows |
| Whether trimming prompts costs quality | same single-window limit — an ablation would return "no significant difference", which is indistinguishable from "no effect" and from "no power" | the multi-window baseline first |

The cost side of every ablation is already known without spending anything: depth multiplies calls exactly [MEASURED], output is priced at four times input [MEASURED], and the input-heavy versus output-heavy balance inverts between the cheap and expensive tiers [DERIVED]. **What is missing in every case is the quality term**, and that is what one window cannot supply.

---

Sources: the committed seed database (17 runs, 7 with LLM usage) [MEASURED]; hosted-API request records [MEASURED]; pipeline-depth runs [MEASURED].
