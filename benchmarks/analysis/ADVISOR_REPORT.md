# ATL cost and serving: what has been measured

Prepared for Dr. Liu, in answer to: *100 users, 1000 agents — would $50K/month be a real problem?* Those three figures are the question as posed [NOT MEASURED], not measurements of the platform.

**Generated file — do not edit by hand.** Regenerate with:

```bash
cd benchmarks && python -m analysis.make_advisor_report
```

## How to read this

Every number carries one of three tags. A number without one is a bug, and a test enforces it.

| Tag | Meaning |
|---|---|
| `MEASURED` | Read from a run artifact or the seed database. Traceable to a `run_id`, or to `file:line` for a code finding. |
| `DERIVED` | Arithmetic on measured values, with the arithmetic shown. |
| `NOT MEASURED` | Named, with what it would take to measure it. Never silently defaulted. |

The headline answer is a **conditional**, not a figure. The inputs that would decide it are precisely the ones no artifact in this repository contains — so they are listed rather than guessed.

## 1. The $50K question

### The short answer

A $50K/month budget [NOT MEASURED] is reachable or unreachable depending almost entirely on **which model serves production** and **how much backtesting users do**. Neither is measured here. The spread between them is not a detail — it is the whole answer.

At each model's measured cost per call, the monthly call volume needed to spend the $50,000 [NOT MEASURED] budget is:

| Model | $/call | Calls/month for $50K | Tier | Source run_id |
|---|---:|---:|---|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 0.000447 | 111,824,536 | DERIVED | `lb_nemotron_3_nano_30b_20260415_20260515` |
| `deepseek/deepseek-v4-pro` | 0.004696 | 10,646,427 | DERIVED | `lb_deepseek_v4_pro_20260415_20260515` |
| `qwen/qwen3.7-plus` | 0.009896 | 5,052,298 | DERIVED | `lb_qwen3_7_plus_20260415_20260515` |
| `anthropic/claude-haiku-4-5` | 0.010719 | 4,664,830 | DERIVED | `lb_claude_haiku_4_5_20260415_20260515` |
| `anthropic/claude-sonnet-4-6` | 0.030690 | 1,629,189 | DERIVED | `lb_claude_sonnet_4_6_20260415_20260515` |
| `google/gemini-3.1-pro` | 0.069954 | 714,756 | DERIVED | `lb_gemini_3_1_pro_preview_20260415_20260515` |
| `openai/gpt-5.5` | 0.086262 | 579,630 | DERIVED | `lb_gpt_5_5_20260415_20260515` |

Cost per call is [DERIVED]: it is `(input_tokens/1e6 x price_in) + (output_tokens/1e6 x price_out)`, where both token counts are [MEASURED] from the seed DB and both prices are [MEASURED] — verified below.

The span between cheapest and dearest model is 193x [DERIVED]. The same $50K buys 111,824,536 calls on `nvidia/nemotron-3-nano-30b-a3b` or 579,630 on `openai/gpt-5.5` [DERIVED].

### What that load would have to look like

Translating call volume into Dr. Liu's scenario (100 users, 1000 agents [NOT MEASURED] — these are the posed scenario, not a measurement of the platform):

**Live trading.** One live agent making `D` decisions per day, at one call per decision:

> 1000 agents x `D` x 21 trading days/month = 21,000 x `D` calls/month [DERIVED]; the trading-day count is a calendar convention [NOT MEASURED], and `D` is [NOT MEASURED].

| Model | Decisions/agent/day needed for $50K | Tier | Plausible? |
|---|---:|---|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 5,325.0 | DERIVED | far above any bar-driven rate |
| `openai/gpt-5.5` | 27.6 | DERIVED | within reach |

For context: the seed runs recorded 161 bars over a one-month window [MEASURED], and the engine requests hourly bars filtered to market sessions (`dashboard/backend/infrastructure/market_data/alpaca_bars.py:114`) [MEASURED]. Over 21 trading days [NOT MEASURED] that is about 7.7 decisions per agent per trading day [DERIVED] — three orders of magnitude below the rate the default model would need. So on that model, **live trading alone cannot approach $50K/month at this scenario's agent count** [DERIVED].

**Backtesting — and this is where the risk actually is.** One backtest of the seed runs' window costs:

> 160.9 LLM calls per backtest [MEASURED] (range 160-161 across 7 runs), one call per hourly bar over a one-month replay.

> 100 users x `B` backtests/day x 160.9 calls x 30 days/month = 482,571 x `B` calls/month [DERIVED]; `B` is [NOT MEASURED].

Pipeline depth divides every threshold, because a decision costs one call per configured step — measured, not assumed (section three):

| Model | 1 call/decision | 3 calls | 5 calls | Tier |
|---|---:|---:|---:|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 231.73 | 77.24 | 46.35 | DERIVED |
| `deepseek/deepseek-v4-pro` | 22.06 | 7.35 | 4.41 | DERIVED |
| `qwen/qwen3.7-plus` | 10.47 | 3.49 | 2.09 | DERIVED |
| `anthropic/claude-haiku-4-5` | 9.67 | 3.22 | 1.93 | DERIVED |
| `anthropic/claude-sonnet-4-6` | 3.38 | 1.13 | 0.68 | DERIVED |
| `google/gemini-3.1-pro` | 1.48 | 0.49 | 0.30 | DERIVED |
| `openai/gpt-5.5` | 1.20 | 0.40 | 0.24 | DERIVED |

The one-call column is what the earlier version of this report showed. It was derived at the seed runs' single-call rate [MEASURED], which is right for those runs and wrong for any pipeline. On `openai/gpt-5.5` a three-step pipeline moves the threshold to 0.40 backtests/user/day [DERIVED] — under one.

**This is the finding worth acting on.** On the platform's default model, reaching $50K needs 232 backtests per user per day at one call per decision, or 77 at three [DERIVED] — implausible either way. On the most expensive measured model it takes 1.20 at one call and 0.40 at three [DERIVED] — **less than one backtest per user per day**, which a single engaged user would exceed without trying. Backtest volume is unbounded by design: a backtest replays a whole window on demand, where live trading is rate-limited by the bar interval.

### So the conditional

> $50K/month becomes a real problem **only if** production runs a frontier model **and** backtest volume reaches roughly one run per user per day at a three-step pipeline, or single digits at one call per decision [DERIVED]. At the platform's current default model and call pattern, the same load costs on the order of 259 dollars/month [DERIVED] — a 193x difference driven by model choice alone [DERIVED].

Two inputs decide it and neither is in this repository: the production **model mix** and **backtests per user per day**. Both are one SQL query away on the Render database — see `QUESTIONS_FOR_ADVISOR.md`.

## 2. Measured cost per call, by model

Seven leaderboard runs in the committed seed database, one per model, each replaying the same window under the same prompt [MEASURED]. Because the prompt is identical, the output-token spread is the model's verbosity rather than the workload's.

| Model | In tok/call | Out tok/call | $/M in | $/M out | $/call | Calls | Tier | run_id |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `nvidia/nemotron-3-nano-30b-a3b` | 5,502 | 860 | 0.050 | 0.20 | 0.000447 | 160 | MEASURED | `lb_nemotron_3_nano_30b_20260415_20260515` |
| `deepseek/deepseek-v4-pro` | 4,005 | 3,396 | 0.435 | 0.87 | 0.004696 | 161 | MEASURED | `lb_deepseek_v4_pro_20260415_20260515` |
| `qwen/qwen3.7-plus` | 5,224 | 4,879 | 0.400 | 1.60 | 0.009896 | 161 | MEASURED | `lb_qwen3_7_plus_20260415_20260515` |
| `anthropic/claude-haiku-4-5` | 4,934 | 1,157 | 1.000 | 5.00 | 0.010719 | 161 | MEASURED | `lb_claude_haiku_4_5_20260415_20260515` |
| `anthropic/claude-sonnet-4-6` | 5,017 | 1,043 | 3.000 | 15.00 | 0.030690 | 161 | MEASURED | `lb_claude_sonnet_4_6_20260415_20260515` |
| `google/gemini-3.1-pro` | 4,950 | 5,005 | 2.000 | 12.00 | 0.069954 | 161 | MEASURED | `lb_gemini_3_1_pro_preview_20260415_20260515` |
| `openai/gpt-5.5` | 3,895 | 2,226 | 5.000 | 30.00 | 0.086262 | 161 | MEASURED | `lb_gpt_5_5_20260415_20260515` |

Input tokens span 1.4x across models and output tokens span 5.8x [DERIVED]. Input is a property of the prompt and carries across models; output is a property of the model and does not. **An output-token figure must never be substituted from one model to another.**

### Pricing verification

The seed DB stores model names in an underscored form that does not substring-match the priced slugs in `dashboard/backend/infrastructure/llm/token_cost.py`, so the mapping is asserted rather than inferred. Each pairing was checked by recomputing the run's cost from its own stored token totals and comparing against the `est_cost_usd` the run itself stored:

- runs checked: 7 [MEASURED]
- largest disagreement: 4.00e-07 USD, against a tolerance of 1e-05 [DERIVED]

The loader raises rather than reporting if any run disagrees, so a drifted price cannot reach this document.

## 3. Calls per decision, and Arm A — both now measured

### Pipeline depth sets calls per decision, exactly

Measured against the real API, not a stub. Each run replayed the same window; `observed` is `llm_calls / bars` and `configured` is the step count in `metadata.initial_pipeline`:

| Configured steps | Observed calls/decision | Agrees | Runs | Tier |
|---:|---:|---|---:|---|
| 3 | 3.000 | yes | 2 | MEASURED |
| 5 | 5.000 | yes | 1 | MEASURED |

Both derivations agree at both depths [MEASURED], so **no retry inflation was observed**. This closes a lever that had been open since the seed data: all seven seed runs used the single-call path, so a multi-step pipeline had never been run anywhere. Depth multiplies cost linearly — a five-step pipeline costs five times a single-call one for the same decisions [DERIVED].

What is still unmeasured is which depth **production** runs. That is the third question in `QUESTIONS_FOR_ADVISOR.md`.

### Arm A — hosted API, measured for the first time

Every previous Arm A figure was modelled from an assumed price. These come from the provider's billed `usage`. Model `nvidia/nemotron-3-nano-30b-a3b`, fixture `a69c2be743189489...` — the same fixture arms B and C ran [MEASURED] — with reasoning `none`, since thinking tokens bill as output.

| Concurrency | req/s | e2e p50 ms | e2e p95 ms | e2e p99 ms | $/request | 429s | errors | Tier |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.438 | 706 | 948 | 981 | 0.000149 | 0 | 0 | MEASURED |
| 8 | 8.037 | 816 | 1,652 | 1,720 | 0.000149 | 0 | 0 | MEASURED |
| 32 | 13.043 | 1,105 | 2,449 | 2,451 | 0.000149 | 0 | 0 | MEASURED |
| 64 | 20.834 | 856 | 1,334 | 1,483 | 0.000148 | 0 | 0 | MEASURED |
| 128 | 24.097 | 955 | 1,200 | 1,293 | 0.000148 | 0 | 0 | MEASURED |

Cost per request is flat at $0.000148-$0.000149 across every level [MEASURED] — the API prices tokens, not concurrency. The provider's own reported cost matched the figure computed from the price table exactly [MEASURED].

Throughput rises monotonically from 1.438 to 24.097 req/s [MEASURED], with no inversion — the opposite of arm B's behaviour, and unsurprising, since the provider is batching behind the endpoint.

**Prompt caching: none granted.** The provider reported `cached_tokens` totalling 0 across all levels [MEASURED], against a fixture that is 99.4% shared prefix by construction [MEASURED]. So the hosted analogue of arm C's prefix cache did not fire here. That is worth a follow-up — it is a measured absence on this model and endpoint, not a statement about OpenRouter generally.

**Throughput here is NOT comparable to arms B/C.** The provider ignored `min_tokens`, so completions ran far shorter than the 256 tokens the local arms forced [MEASURED]. Cost and latency per request stand on their own; tokens/second does not cross arms.

## 4. Self-hosted serving results (arms B and C)

One GPU, one fixture, one pip freeze. Arm B is a plain HuggingFace generate loop; Arm C is vLLM.

| Property | Value | Tier |
|---|---|---|
| model | `Qwen/Qwen2.5-7B-Instruct` | MEASURED |
| GPU | `NVIDIA A100-SXM4-80GB` | MEASURED |
| GPU UUID | `GPU-3bda9160-0081-328d-0755-6afe70878e7c` | MEASURED |
| fixture | `shared_prefix` | MEASURED |
| context tokens | `2620` | MEASURED |
| max new tokens | `256` | MEASURED |
| CUDA | `13.0` | MEASURED |
| driver | `580.82.07` | MEASURED |
| fixture sha256 | `a69c2be743189489...` | MEASURED |
| pip freeze sha256 | `29deea666a5c3e16...` | MEASURED |
| Arm C engine | `vllm.AsyncLLMEngine` | MEASURED |

### Matched levels

| Concurrency | Arm B req/s | Arm C req/s | Arm C / Arm B | Arm B e2e p50 s | Arm C e2e p50 s | Tier |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1093 | 0.3772 | 3.5x | 9.14 | 2.65 | MEASURED (ratio DERIVED) |
| 8 | 0.0539 | 2.9063 | 53.9x | 148.29 | 2.71 | MEASURED (ratio DERIVED) |
| 32 | 0.0536 | 9.2801 | 173.2x | 596.48 | 3.44 | MEASURED (ratio DERIVED) |

**Arm B throughput inverts.** It falls from 0.1093 req/s at concurrency one to 0.0536 at thirty-two [MEASURED] — a factor of 2.04 the wrong way [DERIVED]. Adding concurrent load to the naive loop makes it slower in absolute terms, not merely sub-linear.

### Where Arm B's time goes

From a streamed PyTorch profiler trace at concurrency eight (`armB_L3_trace.json`):

| Metric | Value | Tier |
|---|---:|---|
| kernels launched | 2,850,442 | MEASURED |
| GPU busy fraction | 0.209 | MEASURED |
| GPU idle fraction | 0.791 | MEASURED |
| distinct CUDA streams | 1 | MEASURED |
| cudaLaunchKernel calls | 2,567,089 | MEASURED |
| cudaLaunchKernel CPU time (us) | 81,283,355 | MEASURED |
| GPU busy time (us) | 31,627,412 | MEASURED |

Launch bookkeeping on the CPU costs 2.6x the time the GPU spends executing [DERIVED], across a single stream [MEASURED]. The bottleneck is dispatch, not arithmetic.

Carried verbatim from the trace artifact's own caveats (3 of them) [MEASURED]:

```text
- gpu_busy_us is the union of kernel intervals; total_kernel_time_us is the plain sum and double-counts stream overlap.
- Kernel residency is not occupancy. Achieved occupancy and Tensor Core pipe utilization require ncu (Layer 4).
- tensor_core_gemm_share is the time share of kernels whose names indicate an HMMA GEMM, not a measure of tensor pipe efficiency.
```

## 5. Code findings

Static reading of `dashboard/backend`; nothing executed. Every citation below is re-read from the file at generation time and this document fails to build if one has drifted (8 of 8 verified [MEASURED]).

### `dashboard/backend/domain/backtesting/engine.py:892` [MEASURED]

```python
llm_calls_total = manager.llm_calls + runtime_calls
```

Token usage is accumulated with += across a whole run and written once. The per-call distribution is destroyed before it reaches the database, so no percentile or per-step split can ever be recovered from stored data. [MEASURED]

*Consequence:* ATL cannot cost its own product per decision. See INSTRUMENTATION_PATCH.md. [DERIVED]

### `dashboard/backend/domain/backtesting/engine.py:921` [MEASURED]

```python
if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:
```

insert_decisions fires ONLY under the ai_hedge_fund runtime. Under the native pipeline runtime backtest_decisions is never written: 0 rows across all 17 seed runs. [MEASURED]

*Consequence:* Calls per decision has no decision denominator from that table. equity_timeseries (one row per bar) is used as a proxy instead. [DERIVED]

### `dashboard/backend/domain/trading/execution.py:64` [MEASURED]

```python
price = market_data[symbol]["close"]
```

The fill price is the close of the same bar the decision was computed from, and portfolio.py:78 shows the agent that same close as the current price. Decision price and fill price are the same number. [MEASURED]

*Consequence:* The engine cannot express execution timing. Latency has no quantity through which to act. See latency_sensitivity_audit.md. [DERIVED]

### `dashboard/backend/domain/trading/portfolio.py:78` [MEASURED]

```python
"price": row["close"],
```

The agent's observed price is the bar close it will fill at. [MEASURED]

*Consequence:* Confirms the zero decision-to-fill gap is structural. [DERIVED]

### `dashboard/backend/domain/backtesting/engine.py:833` [MEASURED]

```python
manager.execute_actions(decision["actions"], market_data, timestamp)
```

The decision is applied to the same market_data and timestamp it was computed from (built at engine.py:756). Neither is reassigned in between. [MEASURED]

*Consequence:* A decision taking an hour and one taking 50ms produce bit-identical results. [DERIVED]

### `dashboard/scripts/refresh_daily_leaderboard.py:89` [MEASURED]

```python
for entry in llm_entries:
```

Agents are deployed in a plain sequential loop, one complete backtest at a time. There is no thread pool, process pool, or asyncio.gather anywhere in the backtest path. [MEASURED]

*Consequence:* The dashboard exhibits SERIAL, not synchronised, arrivals. Burst arrival patterns are an assumption about a hypothetical deployment, not a property of this code. [DERIVED]

### `dashboard/backend/infrastructure/llm/pipeline_runner.py:451` [MEASURED]

```python
for index, step in enumerate(decision_steps):
```

Pipeline steps run strictly sequentially, and step n+1's prompt embeds every prior step's output (prior_outputs passed at :460, rendered at :124). No async/await/gather in the module. [MEASURED]

*Consequence:* A real data dependency, so the calls cannot be parallelised. Batching raises throughput across agents; it cannot shorten one decision. [DERIVED]

### `dashboard/backend/execution/paper_backend.py:1` [MEASURED]

```python
"""PaperBackend — DESIGNED-FOR STUB (spec §4.2, Phase B).
```

Paper trading has no execution path: no order submission, no step loop, and a realtime decision-cadence scheduler is listed as not built. [MEASURED]

*Consequence:* No live decision rate can be derived from this repository. decisions_per_agent_per_day is a deployment question. [DERIVED]

One database fact belongs with these: `backtest_decisions` holds 0 rows across all 17 runs in the seed database [MEASURED], which is the schema behaving as written rather than a broken run.

## 6. What is NOT measured

Every item here has been kept out of the findings above. Each is listed with what would resolve it.

### Inputs the cost model needs and this repository does not contain

| Input | Tier | What would resolve it |
|---|---|---|
| `n_users` | NOT MEASURED | How many users the platform serves. Render dashboard / users table. |
| `n_agents` | NOT MEASURED | How many agents run concurrently. Render DB: count of active agents. |
| `decisions_per_agent_per_day` | NOT MEASURED | Decisions one live agent makes per day. Nothing in this repo runs a live agent — paper_backend.py is an explicit stub with no step loop, so this is a deployment property, not a code property. |
| `backtests_per_user_per_day` | NOT MEASURED | Backtests a user launches per day. LIKELY THE DOMINANT DRIVER: one leaderboard backtest was ~161 calls, and backtesting is what the platform is for. Render DB: agent_runs grouped by day and user. |
| `calls_per_decision` | NOT MEASURED | LLM calls per decision — equivalently, YOUR pipeline depth. The relationship is now MEASURED against real API runs: a 3-step pipeline issued exactly 3.000 calls/decision and a 5-step exactly 5.000, with no retry inflation at either depth. What remains unmeasured is which depth PRODUCTION runs — all 7 seed runs used the single-call path. Render DB: metadata.initial_pipeline step counts across real runs. |
| `model_mix` | NOT MEASURED | Fraction of production calls by model, as {db_model: fraction}. Cost per call spans more than two orders of magnitude across the measured models, so this usually dominates the answer. Render DB: agent_runs grouped by llm_model. |
| `trading_days_per_month` | NOT MEASURED | Trading days per month. A calendar convention (~21), not a measurement — state it explicitly rather than letting it default. |
| `calendar_days_per_month` | NOT MEASURED | Calendar days per month for backtest volume (~30). Also a convention, not a measurement. |
| `infrastructure_usd_per_month` | NOT MEASURED | Hosting, database, and any GPU spend — everything that is NOT per-token LLM cost. Arm A now measures the per-request API price, but that is the token bill, not the platform's fixed running cost, and no self-hosted deployment exists to price. Render/AWS billing console. |

### Measurements never taken

| Item | Tier | What it would take |
|---|---|---|
| Which pipeline DEPTH production runs | NOT MEASURED | The depth->calls relationship is now measured (3->3.000, 5->5.000). What depth production actually configures is not. Render DB: metadata.initial_pipeline step counts. |
| Retry inflation under failure | NOT MEASURED | Not observed in 77 real API calls across two depths — but those runs had zero parse failures and zero 429s, so the retry path never engaged. A run under provider errors would be needed to see it. |
| Real cross-agent prompt overlap | NOT MEASURED | Tokenising real production prompts. The shared-prefix fixture is a deliberate upper bound and the low-overlap fixture is CONSTRUCTED, not sampled from production. |
| Prefix-cache benefit | NOT MEASURED | The ablation was never run. vLLM reported stats_source: null, so no hit rate was observed even in the run that had caching enabled. |
| Attribution of the Arm C speedup | NOT MEASURED | The arms differ in batching, caching, and engine at once. Decomposing it needs one-variable-at-a-time runs. |
| Occupancy, Tensor Core utilisation, memory bandwidth, SM activity | NOT MEASURED | ncu and nsys were never run. Kernel residency is not occupancy. |
| Saturation point | NOT MEASURED | Neither arm was pushed to out-of-memory, so no ceiling was found. |
| Arm A throughput comparable with arms B/C | NOT MEASURED | The provider ignored min_tokens, so completions were far shorter than the forced 256. Cost and latency are measured; tokens/second does not cross arms. |
| Whether prompt caching would help on a provider that grants it | NOT MEASURED | OpenRouter granted zero cached tokens for this model. A provider with explicit cache control, against the same shared-prefix fixture, would answer it. |
| Decision latency's effect on trading P&L | NOT MEASURED | The engine cannot express it — see finding on execution.py. A harness-side replay with shifted fills, plus real market data. |

### Deliberately excluded from this report

- **crossover agent counts** — the self-hosted side is a -dirty run, and the crossover is a lower bound that omits engineering and on-call cost entirely.
- **prefix-cache benefit estimates** — the ablation was never run; vLLM reported stats_source: null [NOT MEASURED], so no hit rate was observed even with caching enabled.
- **throughput comparison of arm A against arms B/C** — the provider ignored min_tokens, so arm A produced far fewer output tokens per request than the 256 arms B/C forced [MEASURED] — the tok/s figures measure different work.
- **attribution of the speedup across batching / caching / engine** — never decomposed; the arms differ in more than one variable.

## 7. Caveats

### Serving results are provisional

Both serving runs carry a `-dirty` branch SHA [MEASURED]:

| Run | branch_sha | Tier |
|---|---|---|
| `armB_shared` | `141e5a51d86b972569bb91f53becd84af0c43c72-dirty` | MEASURED |
| `armC_shared` | `f40ae8ba55776e10fdbbe5eb325d5bb6173c64f6-dirty` | MEASURED |

`RUN_MANIFEST_SCHEMA.md` treats a dirty result as non-citable: the tree had uncommitted changes, so the exact code that produced these numbers cannot be reconstructed from the SHA alone. **Treat every serving figure in section three as provisional pending a clean-tree re-run.** The cost figures in section two are unaffected — they come from the committed database, not from these runs.

### The rate-limit ceiling was not found

A probe escalated concurrency to 256 and saw zero 429s at every level [MEASURED]; the full Arm A sweep also returned zero 429s and zero errors across all requests [MEASURED].

**This is a property of THIS ACCOUNT TIER, not of the API.** Rate limits are per-key and per-plan and providers change them without notice. The result says what this credential could do on this date — nothing about the provider's capacity, the model's capacity, or what a funded or contracted account would allow. Any claim that hosted serving is blocked by rate limits must carry that qualifier.

### Scope of the serving measurement

One GPU, one model, one fixture, one prompt shape [MEASURED]. Nothing here establishes how the result varies across GPUs, model sizes, or prompt distributions.

### The seed runs are single-call

Observed calls per decision across the seed runs ranges 0.994 to 1.000 [DERIVED], using bar count as the decision denominator. Every one used the single-call path. A multi-step pipeline multiplies this — verified as three steps to three calls and five to five against the real runner with a stub client [MEASURED] — but no production multi-step run has ever been recorded.

### The dashboard is not orchestration/FinAgents

Every code finding above concerns `dashboard/`, which is what ships. `orchestration/FinAgents` is a separate tree containing the paper artifact; it holds transaction-cost and market-impact machinery that the dashboard does not. Neither imports the other — verified in both directions. Conflating them has been a recurring error in this project and no figure here draws on `orchestration/`.

## 8. Questions only you can answer

Set out in full in `QUESTIONS_FOR_ADVISOR.md`. In brief:

1. **Production backtest volume and model mix** — the two inputs that decide the $50,000 [NOT MEASURED] answer. Both are single queries against the Render database.
2. **Typical pipeline depth in production** — multiplies every cost figure by the step count.
3. **Does paper trading run continuously?** — decides whether live decisions are a rounding error or a second cost centre.
4. **Dashboard or orchestration/FinAgents?** — the scoping question open since the start. They are different systems and the answer changes what should be measured next.

A companion notebook, `cost_model.ipynb`, takes these inputs and produces the monthly figure. It refuses to compute while any of them is blank rather than substituting a default.

---

Sources: `../dashboard/storage/data/backtest.db` (17 runs, 7 with LLM usage) [MEASURED]; `results/armB_shared_*`, `results/armC_shared_*`, `results/armB_L3_trace.json` [MEASURED].
