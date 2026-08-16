# Proposal: a `llm_call_usage` table

**ATL cannot currently cost its own product.** Not "cannot cost it precisely" —
cannot cost it at all below the level of a whole run. The numbers needed exist
at runtime and are added together before they are saved.

This is a proposal for the lab, not for the benchmark. The benchmark has worked
around the gap; the platform still has it, and it blocks questions the platform
itself needs answered — pricing, per-user attribution, and whether prompt
caching is worth enabling.

---

## The problem

`agent_runs` stores token usage as three run-level totals:

```
llm_calls INTEGER, input_tokens INTEGER, output_tokens INTEGER
```

They are accumulated with `+=` across the entire run and written once:

- `dashboard/backend/domain/backtesting/portfolio_manager.py:340-342` — the
  pipeline path adds a whole decision's totals at once
- `dashboard/backend/domain/backtesting/portfolio_manager.py:399-401` — the
  single-call path adds one call's totals
- `dashboard/backend/domain/backtesting/engine.py:892` — the run total is
  assembled and handed to `insert_run`

`backtest_decisions` records what a decision *did* — actions, source,
context_ref — and carries no token columns at all. It is also, under the native
pipeline runtime, never written: `insert_decisions` fires only under the
`ai_hedge_fund` runtime (`engine.py:921`), which is why the table holds zero
rows across all seventeen runs in the committed seed database.

So for a run of 161 calls, the database retains three numbers. The distribution
across those calls is gone.

### What that costs, concretely

Everything below is impossible from stored data today, and each is a question
someone will eventually ask:

| Question | Why it cannot be answered |
|---|---|
| What does one decision cost? | Only the run total survives; a mean is not a cost, and the spread is unknown |
| Which pipeline step dominates? | Per-step split was summed away |
| Is the first step the expensive one? | Same — and this is exactly what decides whether prefix caching helps |
| p50 / p95 cost per decision | No per-call values exist to take percentiles over |
| Did retries fire? | Extra calls are indistinguishable from configured ones in a total |
| Which user or agent drove spend? | Attribution stops at the run |

The prefix-caching row is worth dwelling on. Whether caching helps ATL depends
on whether input tokens are concentrated in a large shared first step or spread
evenly across steps. Those two shapes have the *same run total* and behave
completely differently under a cache. No amount of querying the current schema
tells them apart.

---

## The proposal

One table, written where the numbers already exist.

```sql
CREATE TABLE IF NOT EXISTS llm_call_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    decision_index  INTEGER NOT NULL,   -- which bar/decision within the run
    step_index      INTEGER NOT NULL,   -- which pipeline step; 0 for single-call
    model           TEXT,               -- the model for THIS call
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_call_usage_run
    ON llm_call_usage(run_id, decision_index, step_index);
```

`model` is per-row rather than per-run deliberately. Output length is
model-specific — the seed runs span 860 to 5,005 output tokens per call under an
identical prompt, a factor of 5.8 — so a row whose model is implied by its run
cannot be reused if a run ever mixes models.

### Where it gets written

The three sites already hold everything a row needs. Nothing new has to be
computed; `extract_token_usage()` returns the pair at each one.

| # | Site | Path | What it covers |
|---|---|---|---|
| 1 | `infrastructure/llm/pipeline_runner.py:473` | multi-step decision | one row per pipeline step — `index` is in scope as the loop variable |
| 2 | `infrastructure/llm/pipeline_runner.py:388` | post-trade analysis | one row per post-trade step, once per trading day |
| 3 | `domain/backtesting/portfolio_manager.py:398` | single-call path | one row per decision, `step_index = 0` |

Site 1 is inside `for index, step in enumerate(decision_steps)`
(`pipeline_runner.py:451`), so the step index needs no plumbing. The decision
index does: `run_pipeline_decision` does not currently know which bar it is
serving. The cheapest fix is to pass it in — the engine's loop counter `i` at
`engine.py:735` is the value, and it already flows as far as
`make_trading_decision_with_llm`.

### Migration

Follow the precedent already in the file. `_ensure_decisions_table`
(`database.py:531`) is called before use and creates the table if absent, so the
schema self-migrates on first write with no migration step and no downtime. An
`_ensure_llm_call_usage_table` alongside it would behave identically.

Existing rows need no backfill — the per-call data never existed, so there is
nothing to recover. `agent_runs`' three totals stay exactly as they are; the new
table is additive, and the totals remain derivable from it as a cross-check:

```sql
SELECT run_id, COUNT(*), SUM(input_tokens), SUM(output_tokens)
FROM llm_call_usage GROUP BY run_id;
```

A disagreement between that and `agent_runs` would itself be a useful alarm.

### Cost of the change

Three insert sites, one table-creation helper, one parameter threaded through
`run_pipeline_decision`. Write volume is one row per LLM call — for a 161-call
backtest, 161 rows, against a call that already took seconds. The write is
irrelevant beside the network round trip it accompanies.

The one design decision worth making deliberately: batch the inserts per run
rather than writing per call, matching how `insert_trades` and
`insert_decisions` already work. That keeps the hot path free of a database
round trip per LLM call.

---

## What it enables that is impossible today

1. **Cost per decision, as a distribution rather than a mean.** The current mean
   is a poor input to any pricing decision when the spread is unknown.
2. **Per-step attribution.** Which step of a pipeline is expensive, and
   therefore which prompt is worth shortening.
3. **A real answer on prefix caching.** Whether input tokens concentrate in a
   shared first step is the whole question, and it becomes a `GROUP BY
   step_index`.
4. **Retry detection.** Rows beyond the configured step count are retries,
   visible directly instead of inferred.
5. **Per-user and per-agent cost attribution**, by joining to `agent_runs`.
6. **Honest customer pricing.** The platform could tell a user what their
   backtest cost, which today it cannot.

The benchmark work has produced per-model cost figures from the seed database,
but those are *run-level means over seven runs* — enough to rank models, not
enough to price a product. This table is what turns every downstream cost figure
from an estimate into a measurement.

---

## Scope note

This concerns `dashboard/` only — the platform that ships.
`orchestration/FinAgents` maintains its own separate cost machinery and does not
import, and is not imported by, the dashboard. Nothing in this proposal touches
it.
