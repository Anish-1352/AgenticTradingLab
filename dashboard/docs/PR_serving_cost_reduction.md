# Serving cost reduction, batch 1

Three changes plus the measurement that makes them provable. **Every default
reproduces current behaviour**, with one deliberate exception called out below.

Branched from `origin/main` (`030b177`). No divergence at time of writing — the
branch fast-forwards cleanly and needed no rebase.

## Tests

| | Collected | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| `origin/main`, before any change | 2518 | 2451 | 1 | 70 |
| this branch | 2624 | 2557 | 1 | 70 |

**+106 tests, zero regressions.**

The single failure is **pre-existing on clean `origin/main`**:
`test_ifind_ashare_engine.py::test_ifind_engine_resolves_csi300_sample20_and_records_provenance`.
It concerns iFinD A-share market data and touches nothing here. Left alone
deliberately — fixing an unrelated red test inside a cost PR hides whether this
PR broke anything.

---

## 1. Per-call usage logging (`llm_call_usage`)

**What it does.** Writes one row per LLM call: `run_id`, `call_index`,
`step_label`, `model`, `input_tokens`, `output_tokens`, `cached_input_tokens`,
`latency_ms`, `timestamp`, `error`.

**What it does not do.** It does not change any prompt, any model, any decision,
or any result. It does not remove the existing `agent_runs` aggregation — both
are kept.

**Why it is first.** `agent_runs` accumulates totals with `+=` and writes three
numbers per run, so the per-call distribution is destroyed before it reaches the
database. Without this, "the cache cut cost 30%" cannot be distinguished from
"users ran fewer backtests that week". Every other change in this PR is
unmeasurable until this lands.

**Three call sites**, located on this branch rather than assumed:

| Site | File |
|---|---|
| pipeline decision step | `infrastructure/llm/pipeline_runner.py` |
| post-trade analysis | `infrastructure/llm/pipeline_runner.py` |
| single-call path (inside the retry loop) | `domain/backtesting/portfolio_manager.py` |

**Design choices a reviewer may want to push back on:**

- **A `ContextVar`, not a threaded parameter.** The call sites sit several
  frames below the engine and have no `run_id`. Threading one down would change
  signatures across a production path purely for observability. The engine
  installs a recorder for the decision loop; outside it, `record_call` is a
  no-op — chat, strategy synthesis and the API are untouched.
- **Buffered, flushed once per run**, matching `insert_trades` /
  `insert_decisions`. A database round trip per LLM call would add write
  amplification to a hot path for no benefit.
- **Recording failures never reach the run.** Both the recorder and the
  persistence step swallow their own exceptions. Losing a usage row is
  acceptable; losing a completed backtest to a logging bug is not.
- **Retries get their own rows.** Each retry in the single-call path is a
  separately billed call, and collapsing them would hide exactly the retry cost
  this table exists to expose.
- **`cached_input_tokens` is `None`, not `0`, when the provider says nothing.**
  "Does not report caching" and "cache returned nothing" are different facts.

**The one flag that defaults ON.** `ATL_LLM_CALL_USAGE=0` disables it. It is on
by default because it changes no result and because nothing else in this PR can
be verified from a system that is not recording. This is the single deviation
from strictly-current defaults and it is flagged here rather than buried.

---

## 2. Backtest result cache — **default OFF**

`ATL_BACKTEST_CACHE` unset ⇒ byte-identical to today.

**What it does.** Maps a fully-specified configuration to the `run_id` of a
previous run with that exact configuration, so an identical re-run does not
re-pay its LLM calls.

**An index, not a copy.** A hit returns a prior `run_id`; the equity curve,
trades and metadata are read from where they already live. A cached answer
cannot drift from the run that produced it, and invalidation is deleting one
row.

### ⚠️ This changes semantics, and the change is visible

`temperature=0` does **not** make a hosted API deterministic — batching and
hardware vary. So a cache hit returns **the first run's result, not a fresh
sample**. That is a real change in meaning, not an implementation detail.

`CacheLookup.cached_from` carries `"cached result from <date>"` for the UI, and
callers are expected to render it and to offer an explicit re-run that bypasses
the cache. **A reviewer should check that the UI actually surfaces this** — a
silent cache hit on a nondeterministic backend is the failure mode that costs
trust.

### 🔍 Scrutinise the cache key

This is the highest-risk part of the PR. **A missed hit costs money; a wrong
hit shows a user results that do not match the configuration in front of them,
and is very hard to notice.** The key is deliberately over-strict.

Keyed on: session, model, start and end date, sorted universe, bar interval,
initial capital, data source, mode, **full prompt text**, **full pipeline
JSON**, runtime type and config, `max_output_tokens`, plus `CACHE_KEY_VERSION`
so a semantic code change can invalidate everything without a migration.

Tests assert a **miss** for every one of those, including a one-character prompt
edit and a reordered pipeline (steps feed each other, so order changes results).
If you can think of an input that changes a result and is not in that list, that
is the review comment worth making.

**Scope is per-user.** Global sharing — two users with byte-identical configs
sharing one result — is implemented and tested behind
`ATL_BACKTEST_CACHE_SCOPE=global`, but it is a product decision about whether
one user's compute may serve another, so it is off. **Its saving is not
estimated**, because that needs a production duplicate rate nobody has measured.

TTL defaults to 7 days so a cached result cannot outlive a market-data
correction. Unknown age reads as expired. An entry pointing at a deleted run
reads as a miss, not a dangling pointer.

---

## 3. Leaderboard governance — defaults reproduce current behaviour exactly

With nothing configured, **every entry refreshes exactly as today**. That
default matters: a leaderboard that silently stops updating looks identical to
one updating with unchanged numbers.

Three independent controls, config file first with env override:

- **change detection** (`skip_unchanged`) — skip when the window and every agent
  config are byte-identical to the last refresh. The only one safe to enable
  without further thought.
- **per-model cadence** — expensive models weekly, cheap ones daily.
- **window length** — how many days the evaluation covers.

Cadence is checked **before** change detection, so a trivial config edit cannot
silently restore daily spend.

### ⚠️ Measure-first is not yet satisfied

The plan called for instrumenting one real refresh cycle and reporting per-model
cost **before** choosing cadence values. The mechanism is in and defaults to a
no-op, but **that measurement has not been taken** — it needs a cycle run
against production credentials with logging on. **No cadence values are proposed
in this PR.** Merging it changes nothing about leaderboard spend until someone
sets config, deliberately.

---

## 4. `llm_spend_baseline.py` — the before number

Reads `llm_call_usage` over a date range and reports spend by **workload**
(backtest / leaderboard / live-paper), by **model**, and by **pipeline step**.
This is what turns "we shipped caching" into "cost fell X%".

On an empty table it says so and **estimates nothing** — including refusing to
fall back to the `agent_runs` aggregate, which would produce a baseline that
looks equivalent and is not. Unpriced models are counted in tokens and excluded
from the dollar total rather than priced at a guess. `--dry-run` works against
an empty or missing table and writes nothing.

It also warns that two windows are only comparable at similar user activity: a
quieter week is not a saving, and the script cannot tell the two apart.

---

## Migration and rollback

Both new tables are `CREATE TABLE IF NOT EXISTS` on **brand-new** tables,
created on first write, mirroring the existing `_ensure_decisions_table`
pattern. **No existing table is altered**, so nothing locks or rewrites under a
live reader. Added to both the SQLite and Postgres layers — the store-parity
tests enforce that, and caught it when the first version added the cache methods
to SQLite only.

**Rollback** is `DROP TABLE llm_call_usage`, `DROP TABLE
backtest_result_cache`, and unsetting the flags. `agent_runs` keeps its own
totals and no read path depends on either table.

---

## Two things to scrutinise, restated

1. **The cache key** (§2). Over-strict is the intended failure direction. A
   missing input is a correctness bug that surfaces as a user seeing the wrong
   results.
2. **`test_per_call_sum_equals_the_agent_runs_aggregate`.** The per-call rows
   and the `agent_runs` totals come from two independent accumulators — `+=` in
   the portfolio manager, and the recorder. If they ever disagree, one of them
   is wrong and the per-call table cannot be trusted for attribution, which is
   its only purpose. That test is the invariant holding the whole thing up.

## Out of scope, deliberately

No change to prompt content, model selection, or output schema — those affect
decision quality and need A/B evaluation on more than one evaluation window.
`benchmarks/` and `orchestration/` untouched.
