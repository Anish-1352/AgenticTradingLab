# Record one row per LLM request, retries included

**Branch:** `feat/llm-call-usage` (from `origin/main` @ `4388762f`)
**Size:** +1 table, +1 writer, ~40 lines in the domain layer, 10 tests. Additive only.

## The problem

`agent_runs.llm_calls` is incremented *inside* the no-text retry loop
(`portfolio_manager.py:483-526`). So it already counts every retry — and having
counted them, it can no longer tell you which they were. A run that billed 17
calls for 7 decisions and a run that billed 17 for 17 store the identical row.

We measured a real run at **2.93 attempts per decision**. Nothing persisted says so.

## What this adds

```sql
CREATE TABLE llm_call_usage (
  run_id, step_index, attempt_index, phase, outcome,
  model, input_tokens, output_tokens, max_output_tokens, created_at
)
```

- **`attempt_index > 0`** is the retry filter. That is the query `llm_calls` cannot express.
- **`phase`** separates the two mechanisms that share the billing counter: the
  no-text retry loop, and the post-parse truncation recovery added in
  `efdf1e5a`. They are different failures and would otherwise be
  indistinguishable in the totals.
- **`max_output_tokens`** is on the row because without it a short answer and a
  truncated one look identical. `output_tokens >= max_output_tokens` turned out
  to be the signature of the failure that dominates this workload — it took a
  full investigation to find precisely because no stored row carried it.

## Why not reuse `credit_llm_reservations`

That table already has `run_id`, `call_index`, `attempt_index` and per-call
cost, and it was the obvious candidate. It does not fit, for two reasons:

1. Its `attempt_index` is a **different axis**. `execution/service.py` fills it
   from `for attempt_index, provider_id in enumerate(candidates)` — it counts
   provider failover, not retries of the no-text loop. A run could have
   `attempt_index = 0` on every row and still have retried four times.
2. It is on the **platform path**. The backtest path reaches it only when an
   `execution_client` is injected; leaderboard and CLI runs build the legacy
   client and never touch it.

The two axes are orthogonal and both are worth having. This patch adds the one
that is missing rather than overloading the one that exists.

## Verification

Not just unit tests — a real backtest that actually retried:

| check | result |
|---|---|
| rows vs `agent_runs.llm_calls` | 17 = 17 |
| tokens vs `agent_runs` | equal to the digit |
| requests at `attempt_index > 0` | 10 (58.8%) |
| requests at the output ceiling | 11 of 17 |
| both phases present | yes |

A column written only on the happy path would be decorative; the test that
matters drives the retry loop and asserts the extra rows appear.

## Design notes for review

- `portfolio_manager` must not import the database singleton (its own module
  docstring forbids it). The manager buffers rows; the engine drains them —
  the same injection shape `execution_client` already uses.
- `(run_id, step_index, attempt_index)` is **indexed, not unique**: a decision
  attempt and a truncation recovery can share an attempt index and are told
  apart by `phase`.
- A response with no `usage` object is still billed at zero tokens. That is
  pre-existing behaviour and differs from what `_record_llm_usage`'s docstring
  claims; a test documents it rather than changing the billing counter here.
- A run that never retries produces exactly one row per decision. No existing
  column, counter, or query changes.

## Not in this PR

Decision provenance (`llm_decisions` on `agent_runs`, or a per-decision source
column). It touches the same call sites and is worth doing next, but it is a
different claim about a different thing, and bundling them makes the review
slower than doing them in sequence.
