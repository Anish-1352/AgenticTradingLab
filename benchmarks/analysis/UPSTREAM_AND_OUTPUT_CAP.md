# Upstream drift, and the output ceiling

**Tiers:** `MEASURED` = observed in a run or read out of the tree at the stated revision. `DERIVED` = arithmetic on measured values. `NOT MEASURED` = not established here.

## 1. What upstream landed

Local work sat `533` commits behind `origin/main` at `4388762f` [MEASURED]. Two prior findings are superseded and the rest hold.

| claim | status | Tier |
|---|---|---|
| no per-call usage table exists | holds | MEASURED |
| nothing records which attempt a call was | **superseded** | MEASURED |
| llm_decisions is never written to agent_runs | holds | MEASURED |
| the no-text retry loop is still five attempts on one trigger | holds | MEASURED |
| the only trigger is an AttributeError carrying 'No text content' | holds | MEASURED |
| the fifth attempt rescues by forcing reasoning off | **superseded** | MEASURED |
| the backtest LLM client is built with no timeout or max_retries | holds | MEASURED |
| the pipeline runtime writes no per-decision rows | holds | MEASURED |
| the bar interval is hardcoded to one hour at the fetch site | **superseded** | MEASURED |
| the backtest path requests no JSON schema / structured output | holds | MEASURED |

- **nothing records which attempt a call was** — superseded: upstream now has `55` occurrence(s), first in `models.py` [MEASURED].
- **the fifth attempt rescues by forcing reasoning off** — superseded: the code this described is no longer present upstream [MEASURED].
- **the bar interval is hardcoded to one hour at the fetch site** — superseded: the code this described is no longer present upstream [MEASURED].

A third is a straightforward win: **the hourly bar interval is no longer hardcoded**. `AlpacaDataLoader` now takes a `source_timeframe` and translates `1m`, `5m` and `60m` [MEASURED], so the cadence work's second blocker is resolved at that layer. It does not resolve the first — a bar interval is not a scheduler — and the other cadence blockers still verify.

The two remaining matter in opposite directions.

**The reasoning-off rescue is gone.** `efdf1e5a` ("fix: preserve reasoning during response recovery") replaced it with a larger token budget on the same attempt [MEASURED]. Any plan that starts "the rescue already sets `OPENROUTER_REASONING_EFFORT=none`" is describing code that no longer exists.

**`attempt_index` exists upstream, but on a different axis.** `credit_llm_reservations` carries `run_id`, `call_index`, `attempt_index` and per-call cost — but `execution/service.py` fills it from `for attempt_index, provider_id in enumerate(candidates)`, so it counts **provider failover**, not retries of the no-text loop [MEASURED]. It is also on the platform path; the backtest path reaches it only when an `execution_client` is injected, and the leaderboard runs build the legacy client instead [MEASURED].

**Citations.** `66` `file:line` references across the generated reports were checked; `2` fall outside the upstream file [MEASURED]. Line numbers in the retry report have all drifted — the anchors still exist, so the findings hold and only the coordinates moved:

| anchor | upstream line | Tier |
|---|---:|---|
| the retry loop (`portfolio_manager.py`) | 483 | MEASURED |
| the retry print (`portfolio_manager.py`) | 526 | MEASURED |
| the rescue call (`portfolio_manager.py`) | 493 | MEASURED |
| the billing counter column (`database.py`) | 143 | MEASURED |
| the run table (`database.py`) | 130 | MEASURED |
| the decision write site (`engine.py`) | 1649 | MEASURED |

`feature/serving-cost-reduction` does **not** merge cleanly: `5` files changed on both sides [MEASURED], including `portfolio_manager.py` and `engine.py`, the two most heavily rewritten files upstream. Rebasing it is a task in itself, not a step in another one.

## 2. The root cause: the thinking budget exceeds the output ceiling

OpenRouter documents the invariant: max_tokens must exceed the reasoning budget so there are tokens left for the answer. The shipped defaults violate it — `medium` effort budgets `2048` thinking tokens against a `2000`-token ceiling [MEASURED]. Reasoning tokens are billed as output tokens, so a model that uses its budget is cut off inside the thinking block and returns `['thinking']` with no text — the sole condition that advances the no-text retry loop [MEASURED].

The earlier arms already contained the evidence and it was not read: every failure sat at exactly the ceiling, and no failure sat below it [MEASURED].

| arm | ceiling | attempts/decision | retries | input tokens | output tokens | wall s | cost USD | Tier |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `U2000` (upstream default) | 2000 | 2.93 | 27 | 81236 | 88752 | 434 | 0.0218 | MEASURED |
| `U2000fit` (thinking budget clamped to fit) | 2000 | 2.29 | 18 | 64449 | 67283 | 397 | 0.0167 | MEASURED |
| `U4096` (output ceiling raised) | 4096 | 1.14 | 2 | 31700 | 40048 | 310 | 0.0096 | MEASURED |

Raising the ceiling cut attempts per decision from `2.93` to `1.14`, input tokens by `61%`, wall time by `28%` and cost by `56%` [DERIVED].

**More output tokens is cheaper, not dearer.** A truncated attempt is billed in full and thrown away, and it drags a fresh copy of the whole prompt with it — which is why the input saving is the largest number in the table [DERIVED].

### Clamping the thinking budget does not work

The obvious fix — shrink the budget to fit under the ceiling — was implemented and measured. It fails. With the clamp verified applied (`reasoning.max_tokens=1488`, `thinking.budget_tokens=1488`), every failure still came back at exactly `2000` output tokens [MEASURED]: the model reasoned past its budget and consumed the whole ceiling anyway.

Attempts per decision went `2.93` → `2.29` [MEASURED], which does not clear run-to-run variance and leaves the failure signature unchanged. **Nemotron via OpenRouter treats the reasoning budget as advisory** [MEASURED]. The ceiling is the only parameter measured to control this.

### The quality half, which does not support shipping it on

| arm | return | Sharpe | 95% CI | trades | Tier |
|---|---:|---:|---|---:|---|
| `U2000` | -0.0019 | -1.78 | [-2.03, -1.53] | 6 | MEASURED |
| `U2000fit` | -0.0110 | -14.15 | [-15.72, -12.57] | 11 | MEASURED |
| `U4096` | -0.0071 | -9.99 | [-11.11, -8.88] | 10 | MEASURED |

Raising the ceiling made this window's result **worse**: return `-0.0019` → `-0.0071`, Sharpe `-1.78` → `-9.99` [MEASURED]. Neither arm fell back to the rule-based agent, so this is not fallback contamination — both ran every decision on the model [MEASURED].

The intervals above do not overlap, but that is the wrong test. It is one window of `158` bars over three days, and the gate refuses it [MEASURED]:

> Across-window variance cannot be estimated from ONE window. Sizing a study needs to know how much a model's Sharpe moves between months, and a single sample carries no such information. The within-window SE below describes only how precisely this one month was measured, which is a different and much easier question.

So: **the cost and latency effect is unambiguous and the quality effect is adverse but underpowered** [MEASURED]. That combination argues for the flag, not for the default. Nothing here should be turned on globally until the ceiling is measured across several windows.

## 3. Per-request accounting

`llm_calls` is incremented inside the retry loop, so it already contains every retry — and having contained them, cannot say which they were [MEASURED]. `llm_call_usage` adds one row per billed request with the attempt it belongs to, the phase, the outcome, and the ceiling that request ran under.

Verified against a real backtest that retried, not only in unit tests: `17` rows against `llm_calls` = `17`, tokens equal to `agent_runs` to the digit, and `10` requests at `attempt_index > 0` [MEASURED].

`max_output_tokens` is on the row because without it a short answer and a truncated one are indistinguishable, and `output_tokens >= max_output_tokens` is the signature of the failure that dominates this workload [MEASURED]. Finding it took a whole phase precisely because no stored row carried it.

`phase` separates the no-text loop from the post-parse truncation recovery — a third amplification mechanism this work did not know about until upstream's own logs showed it re-requesting at the recovery ceiling [MEASURED].

## 4. Structured outputs would not have prevented this

The suggestion assumed models emit conversational filler that breaks parsing. That is not the measured failure. The failures had **no text channel at all** — `content types: ['thinking']` — so there was nothing for a schema to constrain [MEASURED].

| question | answer | Tier |
|---|---|---|
| Does OpenRouter expose structured outputs? | Yes, via `response_format` with `json_schema`, on select models and providers | MEASURED |
| Does ATL use it? | No — no `response_format` or `json_schema` anywhere on the backtest path | MEASURED |
| Would it have prevented these failures? | No. The budget was exhausted before any text was emitted; a schema constrains text that exists | DERIVED |
| Does it interact with reasoning? | Undocumented by the provider | NOT MEASURED |
| Is there a parameter that caps thinking reliably? | Not for this model: the budget was set and ignored | MEASURED |

Guided decoding proper is a vLLM feature and ATL calls hosted APIs, so it is not available on this path at all [MEASURED]. **Do not implement it for this failure.**

## 5. Remaining latency levers, ranked

Bars cannot be parallelised — each bar's prompt embeds the previous bar's fills — so the levers are fewer calls, faster calls, cached results [MEASURED, prior work].

| lever | effect | effort | changes results? | Tier |
|---|---|---|---|---|
| Raise the output ceiling | `2.93` → `1.14` attempts/decision, `-61%` input tokens, `-28%` wall | env var, no code | **yes** — different decisions | MEASURED |
| Escalate the ceiling on the first retry, not the fifth | `2.67` → `1.88` attempts/decision, `-33%` input tokens, `-24%` cost, and the `4`–`5` attempt tail goes to zero | small, additive | no — same first request | MEASURED |
| Record per-request rows | none directly; makes the above measurable | small, additive | no | MEASURED |
| Clamp the thinking budget | none for this model | done, off by default | no | MEASURED |
| Client-side timeout | bounds a `600`s stall; no effect on the common path | small | no | MEASURED |
| Structured outputs | none — wrong mechanism | n/a | n/a | DERIVED |

The second row has since been built and measured; see `ESCALATION_ON_FIRST_RETRY.md`. Pooled over three interleaved pairs it removes the `4`–`5` attempt tail completely — `14` of `42` decisions to `0`, one-sided Fisher `p < 0.001` — for `-24%` cost, and it did **not** degrade returns the way raising the default ceiling did [MEASURED]. That asymmetry is the case for preferring it: it changes only the requests that had already failed.

## 6. What this does not establish

- **Whether a higher ceiling is better or worse for returns** [NOT MEASURED]. One window, three days, and the gate refuses it.
- **Whether other models honour the reasoning budget** [NOT MEASURED]. Only nemotron was tested against the clamp.
- **Whether the platform path shows the same amplification** [NOT MEASURED]. These runs use the legacy client; the unified execution path has its own timeout and failover.

