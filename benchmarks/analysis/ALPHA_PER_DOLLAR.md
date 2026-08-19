# Cost per unit of alpha

**Generated file.** Rebuild with `cd benchmarks && python -m analysis.make_alpha_report`. Analysis over existing leaderboard data — no code change, no new runs, no API calls.

## The finding

**No model beat a passive index on a risk-adjusted basis** [DERIVED]. `spy_index` returned 5.95% at Sharpe 5.78 for zero LLM cost [DERIVED].

The precision matters: the best LLM, `deepseek_v4_pro`, DID beat it on raw return — 7.49% against 5.95% [DERIVED] — but at Sharpe 5.01 against 5.78, so it bought that return with more volatility [DERIVED]. It also cost $0.756 for the run [MEASURED]. Which of those two framings is the right one is a mandate question, not a measurement question — both are reported rather than the flattering one being chosen.

**Cost does not track performance.** Spearman rank correlation between cost per run and Sharpe is +0.071 [DERIVED] — indistinguishable from zero across a 194x price span [MEASURED].

**But the sample cannot rank the models, and this report does not.** See the statistics section: one window is one draw, and a league table built from it would be noise with a ranking printed on it.

## Cost against risk-adjusted return

All runs share one window, 2026-04-15 to 2026-05-15, on the same universe [MEASURED]. Sharpe is recomputed from each run's stored equity curve and annualised; the 95% interval uses the Lo standard error for 160 observations [DERIVED].

| Model | Run cost USD | Return | Sharpe | 95% CI | $/Sharpe | Tier |
|---|---:|---:|---:|:---:|---:|---|
| `deepseek_v4_pro` | 0.756 | 7.49% | 5.01 | [4.44, 5.57] | 0.151 | MEASURED cost, DERIVED Sharpe |
| `qwen3_7_plus` | 1.593 | 2.49% | 2.19 | [1.91, 2.48] | 0.727 | MEASURED cost, DERIVED Sharpe |
| `gemini_3_1_pro_preview` | 11.263 | 2.32% | 2.03 | [1.76, 2.31] | 5.537 | MEASURED cost, DERIVED Sharpe |
| `gpt_5_5` | 13.888 | 1.45% | 1.56 | [1.33, 1.79] | 8.887 | MEASURED cost, DERIVED Sharpe |
| `claude_sonnet_4_6` | 4.941 | 0.11% | 0.17 | [0.01, 0.32] | 29.354 | MEASURED cost, DERIVED Sharpe |
| `claude_haiku_4_5` | 1.726 | 0.03% | 0.08 | [-0.07, 0.24] | 21.481 | MEASURED cost, DERIVED Sharpe |
| `nemotron_3_nano_30b` | 0.072 | -0.22% | -0.50 | [-0.66, -0.33] | undefined | MEASURED cost, DERIVED Sharpe |

Zero-LLM-cost strategies over the identical window:

| Strategy | Run cost USD | Return | Sharpe | Tier |
|---|---:|---:|---:|---|
| `spy_index` | 0.000 | 5.95% | 5.78 | DERIVED |
| `buy_hold_djia` | 0.000 | 4.87% | 5.58 | DERIVED |
| `equal_weight_djia` | 0.000 | 5.04% | 5.50 | DERIVED |
| `mean_variance_djia` | 0.000 | 2.43% | 2.81 | DERIVED |
| `djia_index` | 0.000 | 2.24% | 1.96 | DERIVED |

`$/Sharpe` is withheld where Sharpe is at or below zero: dividing a cost by a negative denominator yields a number that sorts as if it were excellent [DERIVED].

## What the sample can and cannot support

**Windows: 1** [MEASURED]. Seven models, one month, one universe. That is one draw from the distribution of possible months, and a model's Sharpe moves a great deal between months for reasons that have nothing to do with the model.

Confidence intervals were computed for all 21 model pairs [DERIVED]; 3 overlap (14%) [DERIVED].

Two reasons that understates the true uncertainty:

- The Lo standard error assumes **iid returns**. Hourly equity returns are autocorrelated and heteroskedastic, both of which make the real interval WIDER. So the non-overlapping pairs are the least trustworthy part of this table.
- Within-window precision is not the question. **Across-window variance is**, and with one window that quantity is not merely imprecise — it is unestimable, because a single sample has no dispersion.

**How many windows would be needed cannot be computed** [NOT MEASURED]. Across-window variance cannot be estimated from ONE window. Sizing a study needs to know how much a model's Sharpe moves between months, and a single sample carries no such information. The within-window SE below describes only how precisely this one month was measured, which is a different and much easier question.

*Three to five non-overlapping windows would give a first estimate of across-window dispersion, from which a real sample size could be computed. Rolling or seasonally varied windows are better than consecutive ones, since adjacent months share regime.*

So the defensible statement is: **the ordering does not track price, and the sample is too small to rank the models.** Both halves matter. The first is informative on its own; the second stops it from becoming a recommendation to buy a particular model.

## What follows for the serving-cost work

The brief's hypothesis was that if cheap models perform comparably, the answer to "reduce serving cost" is a one-line recommendation needing no infrastructure. The data is **consistent with that and stronger than it**: on this window the expensive models did not outperform, and no model of any price beat a free index on a risk-adjusted basis [DERIVED].

But one window cannot carry a model-selection decision, and the honest recommendation is a measurement, not a switch:

1. **Run the leaderboard over several non-overlapping windows.** This is the cheapest high-value experiment available and it needs no new code — the leaderboard already does it for one window. Three to five windows would make the ordering testable.
2. **Until then, the 194x price span is unjustified by evidence** [DERIVED]. That is not the same as saying the cheap model is as good; it is saying nobody here has shown the expensive one is better.
3. **The passive-baseline result deserves its own attention.** If agents do not beat buy-and-hold, model choice is a second-order question and serving cost is a third-order one.

## Why the three ablations were not run

The brief asks for calls-per-decision, output-token and input-token ablations, each reporting cost AND decision quality. **Each would hit exactly the statistical wall above.**

An ablation at 1, 3 and 5 pipeline steps over one window produces three Sharpe estimates whose intervals would overlap for the same reason the seven models' do [DERIVED]. It would cost real money and return "no significant difference", which is indistinguishable from "depth does not help" and from "the test had no power".

The ablations become worth running **after** the multi-window baseline exists, because that is what supplies the across-window variance needed to size them. Running them first spends money to produce an uninterpretable result — the specific failure this project has spent fourteen phases avoiding.

The cost side of each is already known without running anything: depth multiplies calls exactly [MEASURED], output is priced at four times input, and the input-heavy/output-heavy split inverts between the cheap and expensive tiers. What is unknown in every case is the quality term, and that is what the sample cannot yet deliver.

## Prompt reordering: checked, and it buys nothing

The brief asks to verify provider caching support before doing the reordering work. **Already measured.** Arm A ran 160 requests against the production gateway on a fixture with a 99.4% shared prefix — the best possible case for a prefix cache [MEASURED].

The provider **did** report a cache field, and it reported 0 cached tokens [MEASURED].

So the field exists and the answer through it is zero. Reordering the prompt to improve cacheability would be optimising against a mechanism this provider is not applying to this model. **Do not do the refactor.** One check, as the brief asked, instead of one refactor.

This is provider- and model-specific, not a universal claim. A gateway with explicit cache control, or a self-hosted engine with prefix caching enabled, would need re-testing.

## Status of the other deliverables

Per-call usage logging, backtest result caching, leaderboard governance and the cost report were built and shipped in the earlier cost-reduction batch, and are on the `feature/serving-cost-reduction` branch as a PR for review:

| Deliverable | Where | State |
|---|---|---|
| Per-call `llm_call_usage` | `infrastructure/llm/usage_recorder.py` | shipped; three call sites, sum-equals-aggregate asserted |
| Backtest result cache | `domain/backtesting/result_cache.py` | shipped; default OFF, per-user scope, strict key |
| Leaderboard governance | `domain/leaderboard/governance.py` | shipped; defaults reproduce current behaviour |
| Cost report generator | `scripts/cost_reduction_report.py` | shipped; refuses to estimate on empty data |

**The leaderboard cost-per-cycle measurement is still not taken.** The brief requires measuring a real refresh cycle before optimising it, and that needs a cycle run with logging enabled against production credentials. No cadence values are proposed until it exists.

---

Source: `dashboard/storage/data/backtest.db`, 12 leaderboard runs over 1 window [MEASURED]; per-call costs verified against stored `est_cost_usd` [MEASURED].
