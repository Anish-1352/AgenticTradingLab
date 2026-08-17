# Serving cost reduction — batch 1 (PR notes)

Branched from `origin/main`, not from the benchmark branch. Every behaviour
change is behind a flag that defaults to today's behaviour.

## Test baseline

| | Collected | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| `origin/main`, before any change | 2518 | 2451 | 1 | 70 |
| after this batch | 2607 | 2540 | 1 | 70 |

+89 tests, zero regressions. Four regressions were introduced during the work
and fixed before commit — the suite caught all four: a `sys.path` mutation in
the new script (the repo has a bootstrap helper for that), a `sys.modules` pop
in two new fixtures that broke the shared-DB-singleton identity for later
tests, and Postgres twin-parity failures from adding cache methods to the
SQLite layer only. Recorded here because they are the reason the Postgres side
of the cache exists at all.

**The one failure is pre-existing.**
`test_ifind_ashare_engine.py::test_ifind_engine_resolves_csi300_sample20_and_records_provenance`
fails on clean `origin/main` before any change here; it concerns iFinD A-share
market data and touches nothing in this batch. It is recorded rather than
fixed, because fixing an unrelated red test inside a cost PR hides whether this
PR broke anything.

The task brief quoted 2119 as the expected collection count. Main has moved 40
commits since; 2518 is the count on the commit this branched from.

## 1. Per-call usage logging — `llm_call_usage`

`agent_runs` accumulates totals with `+=`, so the per-call distribution is
destroyed before persistence. Any "we cut cost by X%" claim is currently
unfalsifiable. This makes it checkable.

**Three call sites**, located on this branch rather than assumed from the
earlier audit — one had moved:

| Site | Path |
|---|---|
| pipeline step | `infrastructure/llm/pipeline_runner.py` (decision loop) |
| post-trade analysis | `infrastructure/llm/pipeline_runner.py` (post-trade loop) |
| single-call path | `domain/backtesting/portfolio_manager.py` (retry loop) |

Design points worth review:

- **A `ContextVar`, not a threaded parameter.** The call sites are several
  frames below the engine and have no `run_id`. Threading one through would
  change signatures across a production path for observability. The engine
  installs a recorder for the decision loop; outside it every `record_call` is
  a no-op, so chat, strategy synthesis and the API are untouched.
- **Buffered, flushed once per run**, matching `insert_trades` /
  `insert_decisions`. A DB round trip per LLM call would add write
  amplification to a hot path.
- **Failures never reach the run.** Recording and persistence both swallow
  their exceptions. Losing a usage row is acceptable; losing a completed
  backtest to a logging bug is not.
- **Retries get their own rows.** Each retry in the single-call path is a
  separate billed call. Collapsing them would hide retry cost — precisely what
  this table exists to expose.
- **`cached_input_tokens` is `None`, not `0`, when the provider is silent.**
  "Does not report caching" and "cache returned nothing" are different facts;
  deliverable 3 needs to tell them apart.

`agent_runs` aggregation is unchanged. Both are kept, and
`test_per_call_sum_equals_the_agent_runs_aggregate` asserts they agree — they
come from independent accumulators, so disagreement means one is wrong.

**This is the one flag that defaults ON** (`ATL_LLM_CALL_USAGE=0` disables).
It changes no result, prompt or model, and the other deliverables cannot be
verified from a system that is not recording. Flagged explicitly because it is
the one place I did not default to strictly-current behaviour.

## 2. Backtest result cache — default OFF

`ATL_BACKTEST_CACHE` unset ⇒ byte-identical to today.

**An index, not a copy.** A hit returns a prior `run_id`; results are read from
where they already live. A cached answer cannot drift from the run that
produced it, and invalidation is deleting one row.

**The key is deliberately too strict.** A missed hit costs one backtest; a
wrong hit shows results that do not match the configuration on screen and is
very hard to notice. Keyed on: session, model, dates, sorted universe, bar
interval, initial capital, data source, mode, full prompt text, full pipeline
JSON, runtime type and config, `max_output_tokens`, plus `CACHE_KEY_VERSION` so
a semantic code change can invalidate everything without a migration.

Tests assert a miss for every one of those, including a **one-character prompt
edit** and a **reordered pipeline** (steps feed each other, so order changes
results).

**Two things resolved rather than assumed:**

- *Scope.* Per-user by default. Global sharing would let two users with
  identical configs share one result — a larger saving, but a product decision
  about whether one user's compute may serve another. `ATL_BACKTEST_CACHE_SCOPE=global`
  exists and is tested; **estimating its saving needs the production duplicate
  rate, which is not measured.** Deliberately not estimated here.
- *Determinism.* `temperature=0` does **not** make a hosted API deterministic.
  A hit returns the first run's sample, not a fresh one — a real change in
  meaning. `CacheLookup.cached_from` carries "cached result from `<date>`" for
  the UI, and callers are expected to offer a bypass.

TTL defaults to 7 days so a cached result cannot outlive a market-data
correction. Unknown age reads as expired: re-running costs money, serving a
result of unknown vintage costs trust. Entries pointing at a deleted run read
as a miss, not a dangling pointer.

## 3. Leaderboard governance — defaults reproduce current behaviour

With nothing configured, every entry refreshes exactly as today. A leaderboard
that silently stops updating looks identical to one updating with unchanged
numbers, so the default had to be the expensive, correct one.

Three independent controls: **change detection** (`skip_unchanged`, the only one
safe to enable blindly), **per-model cadence** (expensive weekly, cheap daily),
and **window length**. Config file first, env override second.

Cadence is checked *before* change detection, so a trivial config edit cannot
silently restore daily spend — tested.

### Measure-first is not yet satisfied

The brief says to instrument one refresh cycle and report actual cost per model
per cycle **before** optimising. The mechanism is in place and defaults to a
no-op, but **that measurement has not been taken** — it needs a real refresh
cycle against production credentials, which this PR does not run. No cadence
values are proposed here for that reason. The number goes in
`cost_reduction_report.md` once a cycle has run with logging on.

## 4. `cost_reduction_report.md`

Generated by `dashboard/scripts/cost_reduction_report.py` from
`llm_call_usage`. On empty data it says so and estimates nothing; it explicitly
refuses to compare against the pre-instrumentation seed data, which was produced
under a different configuration. Unrecognised models are counted in tokens and
excluded from the cost total rather than priced at a guess. `:free` slugs price
at zero, because substring-matching them to the paid rate fabricates a charge
that never happened.

## Migration and rollback

Both new tables are `CREATE TABLE IF NOT EXISTS` on brand-new tables, created on
first write, mirroring `_ensure_decisions_table`. No existing table is altered,
so nothing locks or rewrites under a live reader. Added to both the SQLite and
Postgres layers.

Rollback is `DROP TABLE llm_call_usage` / `DROP TABLE backtest_result_cache`
plus unsetting the flags. `agent_runs` keeps its own totals and no read path
depends on either table.

## Out of scope, as specified

No change to prompt content, model selection, or output schema — those affect
decision quality and need A/B evaluation (batch 2). `benchmarks/` and
`orchestration/` untouched.
