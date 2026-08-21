# How often did the leaderboard fall back to rule-based trading?

**It cannot be determined from the stored data** [MEASURED]. Not "cannot be
determined precisely" — there is no per-decision provenance anywhere in the
seed database, and the one counter that distinguishes a model decision from a
fallback is computed at runtime, used once, and discarded before the run is
saved.

Reproduce with `python benchmarks/analysis/fallback_recoverability.py`. The
seed DB is asserted byte-identical (`414bf53c…`) before and after
[MEASURED], and 21 tests pin the result.

---

## First, a correction to the premise

That defect lives in `pipeline_output_to_decision` [MEASURED], reached only
from `run_pipeline_decision`. **No leaderboard entrant carries a pipeline** —
all `12` of `12` have no `pipeline` key [MEASURED] — and
`llm_agent.py` calls `make_trading_decision_with_llm` without a `pipeline`
argument, so that function was never executed during these runs
[MEASURED]. The specific defect Phase 16 found did not touch the leaderboard.

**The concern survives, via a different code path.** The single-prompt branch
carries the same defect class at `portfolio_manager.py:432-446`: an empty
`actions` list under `strict_llm=False` prints "Falling back to rule-based
logic" and returns `self.make_trading_decision(...)`. `llm_agent.py` passes no
`strict_llm` either, so the default `False` applies and the silent branch is
the one that ran [MEASURED].

The code comment at that site already names the problem — the non-strict branch
"silently swaps in a rule-based decision, which is exactly what must NOT
count." So the risk is real and was known; what is missing is any record of
whether it fired.

## The counter exists. It is never saved.

`PortfolioManager.llm_decisions` counts steps the model actually drove. It is
correct, it is exactly the needed numerator, and its fate is:

```
PortfolioManager.llm_decisions        computed per step
  -> llm_agent.py:188                 copied onto the strategy object
  -> _reject_if_llm_fallback()        read by the H6 publish guard
  -> discarded
```

`insert_run` accepts `llm_calls`, `input_tokens`, `output_tokens`,
`est_cost_usd`, `metadata` — and no decision counter [MEASURED]. The number
existed in memory during every one of these runs and was never written.

## Every signal checked, and what each supports

| Signal | Verdict | Why | Tier |
|---|---|---|---|
| `agent_runs.llm_decisions` | not recoverable | no such column | MEASURED |
| `backtest_decisions.decision_source` | not recoverable | right column, `0` rows | MEASURED |
| `agent_runs.metadata` | not recoverable | empty in `0/17` runs; the builder carries no counter either | MEASURED |
| `trades` | not recoverable | `0` rows against `2148` reported trades | MEASURED |
| `llm_calls` vs bars | not recoverable | the call is billed before parsing | MEASURED |
| `equity_timeseries` | not recoverable | records outcomes, not origins | MEASURED |
| H6 publish guard | **bound only** | conditional `≥95%` floor, `2/7` runs | DERIVED |
| pipeline exposure | n/a | the defect's path was never taken | MEASURED |

### Why `llm_calls` cannot substitute

All seven LLM runs show `1.000` calls per bar (Nemotron `0.994`, `160/161`)
[MEASURED]. That ratio is **consistent with all 161 steps being model-driven
and with none of them being**, because the call is billed before the response
is parsed — a step that falls back still consumes its call.

The codebase already knows this [MEASURED]. The commit that fixed the publish
guard is titled "key coverage on model-driven steps, not billed calls"
[MEASURED] — but it changed the in-memory guard, not what is stored.

### The one real constraint, and its limits

The H6 guard refuses to publish an entry below `95%` coverage [MEASURED], so a
stored row implies it passed the guard *as it existed then*. Converting commit
times from `+0800` and run times from SQLite's UTC `CURRENT_TIMESTAMP`:

| Run | created (UTC) | guard in force then | floor | Tier |
|---|---|---|---|---|
| claude_haiku_4_5 | 2026-07-05 05:39 | none — predates `e11c541` | — | MEASURED |
| qwen3_7_plus | 2026-07-05 15:35 | keyed on `llm_calls` | — | MEASURED |
| gemini_3_1_pro_preview | 2026-07-05 16:39 | keyed on `llm_calls` | — | MEASURED |
| gpt_5_5 | 2026-07-05 16:58 | keyed on `llm_calls` | — | MEASURED |
| deepseek_v4_pro | 2026-07-05 19:32 | keyed on `llm_calls` | — | MEASURED |
| nemotron_3_nano_30b | 2026-07-12 07:49 | keyed on `llm_decisions` | `≥95%` | DERIVED |
| claude_sonnet_4_6 | 2026-07-12 08:44 | keyed on `llm_decisions` | `≥95%` | DERIVED |

Five of seven get **no constraint at all**: the guard either did not exist or
compared billed calls, which a fallback also consumes [DERIVED].

The two that do carry a floor carry it conditionally. It assumes the run was
published through `deploy_model_run` with `allow_fallback=False` — the database
does not record which path published a run, or which revision produced it
[NOT MEASURED]. And a floor is not a rate: `≥95%` of `161` spans `0` to `8`
fallback decisions, and stored data cannot narrow it [DERIVED].

**No fallback fraction is reported here, for any run.** A test enforces that
the audit module cannot emit one — an invented number would look like a
measurement and would corrupt exactly the result it was meant to check.

## What this means for the cost-per-alpha result

The existing result stands as reported and **its exposure to this cannot be
assessed** [MEASURED]:

- Spearman rho `+0.071` across the `194x` price span, one window
- no model beat `spy_index` on Sharpe (`5.95%` at `5.78`, zero LLM cost)
- `deepseek_v4_pro` beat SPY on raw return (`7.49%`) and lost on Sharpe

Fallback contamination would explain the convergence neatly — but "would
explain it" is not evidence. The honest position is that the hypothesis is
**untestable against this data**, in either direction. It cannot be confirmed
and it cannot be dismissed. Nothing was recomputed and no gate was re-run,
because there is no fallback-excluded subset to recompute over; the statistical
gates in `alpha_per_dollar` and `multi_window_alpha` are untouched by this
work.

That the question is unanswerable is itself the strongest argument for the
instrumentation below.

## The minimal instrumentation that would fix it

One column, written at three call sites. This is the **same gap** as the
`llm_call_usage` proposal in `INSTRUMENTATION_PATCH.md`, at the **same three
sites**, and both are needed before the platform can attribute either cost or
behaviour to a model.

**Minimum:** persist the counter that already exists.

```
ALTER TABLE agent_runs ADD COLUMN llm_decisions INTEGER;
```

and pass `llm_decisions=strategy_impl.llm_decisions` through `insert_run`.
That is a one-line schema change plus one argument, and it converts this
question from unanswerable to a `SELECT`. It gives a per-run rate, not a
per-decision record.

**Better, and what `INSTRUMENTATION_PATCH.md` already argues for:** write
`backtest_decisions` on the native path. The table and its `decision_source`
column already exist and are already the right shape — `insert_decisions`
simply never fires outside the `ai_hedge_fund` runtime. Populating it with
`decision_source ∈ {model, fallback_empty_actions, fallback_unparseable,
fallback_no_client}` would answer this question per decision, per model, per
step, and would carry the token columns the cost work needs.

The three sites are the ones that proposal already names:

- `portfolio_manager.py:340-342` — the pipeline path
- `portfolio_manager.py:399-401` — the single-call path
- `engine.py:892` — where the run total is assembled for `insert_run`

**Both proposals are one change.** A decision row that records provenance and
tokens together answers "what did this cost" and "was this the model" from the
same write. Doing them separately means touching the same three sites twice.

## The narrower fix worth considering alongside

Persisting the counter records *how often* the fallback fired. It does not stop
it firing on a valid decision. An empty `actions` list is a legitimate "do
nothing this step", and treating it as a failure is the underlying defect —
tracked separately, and already visible in the strict-LLM branch, which counts
exactly that case as a model decision while the non-strict branch replaces it.
Instrumenting without fixing that would measure a bug faithfully rather than
remove it.
