# Retry amplification, measured

**Question:** Phase 20 reported `2.71` LLM requests per bar and attributed the excess to retries by reasoning about call sites [DERIVED]. This phase was asked to make that MEASURED or to say why it could not be.

**Tiers:** `MEASURED` = observed in a run recorded here. `DERIVED` = arithmetic on measured values. `NOT MEASURED` = not established by this work.

## Answer

The retry **mechanism** is now MEASURED, and it is a single condition — the model returning reasoning with no text. The retry **rate** is not one number: across `4` completed arms it ranges `1.00`–`2.57` ATL attempts per decision [MEASURED].

Counting on the wire instead of in the backend, the same arms issued `1.07`–`2.57` HTTP requests per decision [MEASURED]. The two differ, and that difference is the finding below.

## 1. The mechanism (MEASURED)

`portfolio_manager.py:366-412` loops five times. It advances on exactly one condition: `_extract_response_text` raising `AttributeError` containing "No text content". Everything else re-raises and leaves the loop, so unparseable output, timeouts and provider errors **cannot** cause a retry here — they abort the run [MEASURED].

The probe instruments that exception rather than counting calls, so an attempt is known to be a retry because the thing that causes retries fired.

The production log line, verbatim:

```
⚠️  No text content block in LLM response (content types: ['thinking']); retry 1/4
```

Across the arms the backend printed that shape `32` times, and it was the same shape every time: `['thinking']` x32 [MEASURED]. The model emitted a reasoning block and no answer.

This is read out of the log, independently of the instrumented counts, and the two agree exactly [MEASURED] — which is the cross-check Phase 20 could not make, because its capture filtered the retry prefix out before writing the log.

`32` observations of one shape and no other bounds the rate of any different trigger at roughly `9.4%` by the rule of three [DERIVED].

| arm | trigger counts | Tier |
|---|---|---|
| `reproA` | `no_text_content` x16 | MEASURED |
| `onesym2` | `no_text_content` x3 | MEASURED |
| `altwin` | `no_text_content` x13 | MEASURED |
| `deepseek` | none | MEASURED |

## 2. There are two retry layers, and `llm_calls` sees one

Underneath ATL's loop the Anthropic SDK runs its own. `SyncAPIClient.request` loops `for retries_taken in range(max_retries + 1)` around `self._client.send(...)`. The providers build the client as `anthropic_cls(api_key=..., base_url=...)` and pass neither `timeout` nor `max_retries` (`providers/openrouter.py`, `providers/commonstack.py`, `providers/anthropic_native.py`), so the SDK defaults apply [MEASURED]: read timeout `600`s, `max_retries=2`.

An SDK-level retry never returns to `portfolio_manager`, so `self.llm_calls += 1` never runs for it. **`llm_calls` undercounts the requests actually issued to the provider** [MEASURED]. This probe therefore counts at `httpx.Client.send`, the one call the SDK's retry loop wraps.

| arm | decisions | ATL attempts | wire requests | wire/ATL | SDK retries | Tier |
|---|---:|---:|---:|---:|---:|---|
| `reproA` | 14 | 30 | 30 | 1.00 | 0 | MEASURED |
| `onesym2` | 14 | 17 | 17 | 1.00 | 0 | MEASURED |
| `altwin` | 7 | 18 | 18 | 1.00 | 0 | MEASURED |
| `deepseek` | 14 | 14 | 15 | 1.07 | 1 | MEASURED |

A `wire/ATL` ratio above one is the part of the bill no counter in the backend can see.

`deepseek` is the case in point. It retried **zero** times at the ATL layer — `1.00` attempts per decision, no trigger fired — and still issued `15` requests for `14` attempts [MEASURED].

The extra request was a `ReadTimeout` after `121.0`s on decision `12`, which the SDK then retried successfully [MEASURED]. The backend recorded that decision as one clean call.

Its stored row says `llm_calls` = `14`. The provider was asked `1` more time(s) than that [MEASURED].

**The two mechanisms are different and each is invisible to a different counter.** A model that returns reasoning without text amplifies at the ATL layer, where `llm_calls` does see it. A model that stalls amplifies at the SDK layer, where nothing sees it [MEASURED].

## 3. The stall that hid inside a "clean" run

The earlier arm `onesym` reported `1.00` attempts per decision with `0` retries, and read as clean [MEASURED]. Its per-attempt timings were not:

| attempt | seconds | Tier |
|---|---:|---|
| d0a0 | `5.0` | MEASURED |
| d1a0 | `8.8` | MEASURED |
| d2a0 | `5.1` | MEASURED |
| d3a0 | `616.3` | MEASURED |

`3` attempts completed in `5.0`–`8.8`s; `1` took `616.3`s [MEASURED]. That upper value sits on the SDK's `600`s read timeout, so the request timed out and the SDK's own retry then succeeded [DERIVED].

ATL recorded that as one attempt with no retry. A backtest of `161` bars in which this happens on a tenth of them spends roughly `160` minutes waiting on timeouts alone, invisibly [DERIVED].

This probe installs a client-side read timeout (in-process only; nothing under `dashboard/` is modified). Normal responses and stalls are `~2` orders of magnitude apart, so the two populations separate cleanly [MEASURED].

## 4. Is `2.71` a property of the model, the prompt, the data, or chance?

| arm | condition | model | complete | decisions | attempts/decision | retries | retry share of attempts | retry share of LLM seconds | Tier |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `reproA` | 2 symbols, April window | `nemotron-3-nano-30b-a3b` | yes | 14 | 2.14 | 16 | 53.3% | 54.6% | MEASURED |
| `onesym2` | 1 symbol, April window | `nemotron-3-nano-30b-a3b` | yes | 14 | 1.21 | 3 | 17.6% | 13.2% | MEASURED |
| `altwin` | 2 symbols, May window | `nemotron-3-nano-30b-a3b` | yes | 7 | 2.57 | 11 | 61.1% | 59.3% | MEASURED |
| `deepseek` | 2 symbols, April window, other model | `deepseek-v4-pro` | yes | 14 | 1.00 | 0 | 0.0% | 0.0% | MEASURED |

The spread across arms is `1.57` attempts per decision [DERIVED]. Read against Phase 20's `2.71` [DERIVED], the honest reading is that `2.71` was one draw from a wide distribution, not a constant [DERIVED].

Holding the model fixed at `nemotron-3-nano-30b-a3b`, the arms still range `1.21`–`2.57` attempts per decision [MEASURED], so the variation is not explained by model choice alone.

The clearest single factor is prompt breadth: the same model on one symbol ran at `1.21` attempts per decision and on two symbols reached `2.57` [MEASURED]. That is consistent with a longer prompt pushing more of the completion into reasoning, though this phase did not vary prompt length directly [NOT MEASURED].

### The leaderboard does not show this at all

| seed run | calls | bars | calls/bar | Tier |
|---|---:|---:|---:|---|
| `lb_claude_haiku_4_5_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |
| `lb_claude_sonnet_4_6_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |
| `lb_deepseek_v4_pro_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |
| `lb_gemini_3_1_pro_preview_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |
| `lb_gpt_5_5_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |
| `lb_nemotron_3_nano_30b_20260415_20260515` | 160 | 161 | `0.994` | MEASURED |
| `lb_qwen3_7_plus_20260415_20260515` | 161 | 161 | `1.000` | MEASURED |

Every committed leaderboard run sits at or below `1.000` calls per bar [MEASURED]. `llm_calls` is incremented inside the retry loop, so a retry would raise this above `1.000`; none does. **The seed leaderboard runs contain essentially no retry amplification** [MEASURED].

The same model makes the point sharply. `lb_nemotron_3_nano_30b_20260415_20260515` ran at `0.994` calls per bar [MEASURED]; the probe arms drove that model to `2.57` attempts per decision [MEASURED]. Same model, same loop, same provider.

So the answer to the section heading is **the model and the prompt path together, and not chance** [DERIVED]:

- **The model decides whether it happens at all.** One model returned thinking-only responses under every condition tried; the other returned none under the same condition [MEASURED].
- **The prompt path decides how often.** For the model that does it, widening the prompt roughly doubled the rate [MEASURED], and the single-prompt leaderboard entrants show none of it [MEASURED].
- **Chance is not a sufficient explanation.** The arms separate cleanly by condition rather than scattering [MEASURED].

That retires Phase 20's `2.71` as a general figure [DERIVED]. It was a real measurement of one pipeline run with one model, and it transfers neither to other models nor to the board.

## 5. When the loop runs out, the run silently falls back

Retries are not always enough. `altwin` x2 exhausted all `5` attempts — the rescue call with reasoning disabled included — and still got no text [MEASURED]. The run then printed:

```
❌ LLM decision error: No text content block in LLM response (content types: ['thinking'])
   Falling back to rule-based logic
```

So `2` decisions in this phase were made by the rule-based reference agent rather than the model [MEASURED], and nothing in the stored run row says so. That is the same attribution gap the fallback work reported, now with a reproducible cause attached to it.

This also explains why an arm's trigger count can exceed its retry count: the final attempt of an exhausted decision fails too, but it is not followed by a retry [MEASURED].

## 6. Unparseable output does not retry — it is repaired in place

The arms produced genuine JSON failures, and they took a different path entirely [MEASURED]:

```
⚠️  Initial parse failed: Expecting value: line 1 column 16
   Attempting to fix JSON formatting...
   ❌ Still failed after fix
   Attempting second fix attempt (validate structure)...
```

That is `parse_llm_response`, downstream of `_extract_response_text` and outside the retry loop. A malformed answer costs repair attempts, not extra API calls [MEASURED]. Any proposal to "retry on bad output" would be adding a mechanism, not tuning one.

## 7. Where a retry is visible (nowhere that persists)

| surface | records a retry? | evidence | Tier |
|---|---|---|---|
| `agent_runs.llm_calls` | partially — ATL retries inflate it, SDK retries do not | `database.py:116` | MEASURED |
| `agent_runs` other columns | no attempt or retry column exists | `database.py:103-123` | MEASURED |
| `backtest_decisions` | table has `decision_source` and `step_index` but no attempt column, and the pipeline runtime never writes to it | `engine.py:922` is gated on `runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE` | MEASURED |
| a per-call usage table | does not exist | proposed only, in `atl_token_extract.py` | MEASURED |
| in-memory counter | none — the loop increments no retry counter | `portfolio_manager.py:366-412` | MEASURED |
| stdout | yes, one `print()` per retry | `portfolio_manager.py:411-413` | MEASURED |

So the only durable record of a retry is a log line, and the only durable record of the SDK's retries is none at all [MEASURED]. A run's stored row cannot answer "how much of this bill was retries" [MEASURED].

The smallest change that would fix this is an attempt-level row per call — `run_id`, `step_index`, `attempt_index`, `outcome`, tokens, seconds. `attempt_index > 0` then makes the ATL layer queryable, and recording the response's retry-count header makes the SDK layer queryable too.

## 8. What would actually reduce retries

Only causes this work observed are listed. The loop cannot fire on anything else [MEASURED], so remedies aimed at parsing, timeouts or provider errors would target conditions that do not occur here.

**Observed cause: the model returns a reasoning block and no text.** Every retry recorded in this phase had that shape [MEASURED].

1. **Turn reasoning off for this step, or budget it.** The loop's own fifth attempt already does exactly this — it sets `OPENROUTER_REASONING_EFFORT=none` as a rescue (`portfolio_manager.py:370-388`) [MEASURED]. Doing it on attempt one rather than attempt `5` would remove the amplification for this failure mode entirely. Whether it changes decision quality is [NOT MEASURED].
2. **Raise the output cap so reasoning does not consume the whole budget.** A thinking-only response is what a truncated response looks like when the reasoning budget fills the completion. This is consistent with the observation but was not tested [NOT MEASURED].
3. **Set a client-side timeout.** This does not reduce ATL retries — timeouts cannot trigger that loop — but it bounds the SDK layer, which is where the `600`s stalls live [MEASURED].

## 9. What this does not establish

- **A leaderboard retry rate** [NOT MEASURED]. These arms are short windows on one or two symbols; the seed leaderboard runs were not re-run.
- **Whether disabling reasoning changes decisions** [NOT MEASURED]. Recommendation 1 is a cost argument, not a quality one.
- **The provider's own retry behaviour** [NOT MEASURED]. Counting stops at this process's socket.

