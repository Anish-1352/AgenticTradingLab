# Does ATL's backtest engine model decision latency?

**Answer: no. Latency is invisible to the engine — completely, not partially.**

Static analysis of `dashboard/backend`. Nothing was executed; no database was
read or written. Every claim below carries a `file:line`.

The question matters because it decides whether one candidate thesis is
measurable. Everything measured in Phases 4–6 — throughput inversion,
launch-bound dispatch, cost dominated by model choice — could have been
measured on any agentic workload. Only one thesis needs a *trading* platform:
that inference latency has a dollar value, because a decision arriving after
the bar has moved gets a worse fill. That thesis requires the engine to model
*when* a decision arrives. It does not.

---

## The path from decision to fill

Four steps, all inside one synchronous loop over bars:

| # | What happens | Where |
|---|---|---|
| 1 | Loop over bar timestamps | [engine.py:735](../../dashboard/backend/domain/backtesting/engine.py#L735) |
| 2 | Build agent's view from the bar at `timestamp` | [engine.py:753,756](../../dashboard/backend/domain/backtesting/engine.py#L753) |
| 3 | Agent decides (LLM call blocks here) | [engine.py:765](../../dashboard/backend/domain/backtesting/engine.py#L765) |
| 4 | Apply decision to **the same** `market_data`, **the same** `timestamp` | [engine.py:833](../../dashboard/backend/domain/backtesting/engine.py#L833) |

```python
# engine.py:756 — the agent's view is built from bar i
state = manager.get_portfolio_state(market_data, price_cache, timestamp)
...
# engine.py:833 — the fill uses the identical objects, however long step 3 took
manager.execute_actions(decision["actions"], market_data, timestamp)
```

`market_data` and `timestamp` are never reassigned between line 756 and line
833. Whatever elapsed in step 3 leaves no trace in the simulation.

### The decision price and the fill price are the same number

This is the sharpest form of the finding, and it is stronger than "latency is
ignored".

The agent is shown the bar's close as the current price:

```python
# portfolio.py:78
"price": row["close"],
```

and the fill is taken at that same close:

```python
# execution.py:64
price = market_data[symbol]["close"]
```

So the agent observes a price and transacts at exactly that price. The
decision-to-fill price gap is **identically zero by construction** — not small,
not noisy, but structurally incapable of being anything else. There is no
quantity through which latency could express itself, because the only price in
the system is the one the agent already saw.

`execution.py`'s own module docstring states the rule normatively, which makes
this a design invariant rather than an incidental detail:

> execution price is always `market_data[symbol]["close"]`
> — [execution.py:15](../../dashboard/backend/domain/trading/execution.py#L15)

---

## The five questions

### 1. Does the engine use a decision timestamp?

**No.** The decision is applied to the bar it was computed from, regardless of
elapsed time.

No wall clock is consulted anywhere in the decision path. The engine's only
uses of `datetime.now()` are run-ID generation
([engine.py:869, 982, 1045](../../dashboard/backend/domain/backtesting/engine.py#L869)) —
never simulation logic. There is no `time.time()`, `perf_counter`, or `sleep`
in the loop.

One place records something called a decision timestamp:

```python
# adapter.py:789
"timestamp": context.timestamp.isoformat(),
```

That is the **simulated bar timestamp**, not a wall-clock arrival time, and it
is written only under the `ai_hedge_fund` runtime — not the default pipeline
runtime. It is an audit label, not an input to execution.

### 2. Where does the fill price come from?

**The close of the same bar the decision was computed from** —
[execution.py:64](../../dashboard/backend/domain/trading/execution.py#L64).

Not the next bar's open, not VWAP, not a spread-adjusted touch. Both buys
(`execution.py:67`) and sells (`execution.py:85`) use the identical `price`
variable. `price_cache` is deliberately *not* consulted here
([execution.py:13](../../dashboard/backend/domain/trading/execution.py#L13)),
so forward-filled prices cannot leak into fills either.

### 3. Is there any slippage, latency, or execution-delay model?

**None, anywhere in `dashboard/backend`.** A search for `slippage`, `latency`,
`spread`, `partial_fill`, `execution_delay`, `commission`, and `fees` across the
non-test backend returns zero execution-related hits. The single `spread` match
is in `vnpy_simulation.py:75`, where it synthesises high/low prices when
*fabricating* bars — it never touches a fill.

Fills are unconditional on liquidity: `execute_actions` checks only
`cost <= cash and shares > 0` for a buy (`execution.py:68`) and position
sufficiency for a sell (`execution.py:83`). There is no volume constraint, so
any size fills instantly at the close. Partial fills exist only as
`min(shares, positions[symbol])` (`execution.py:84`) — a position-capping rule,
not a liquidity model.

#### The orchestration paper describes something the dashboard does not do

The Li et al. description of execution agents that account for "latency, fees,
spreads, and partial fills" refers to `orchestration/FinAgents`. That code is
**not reachable from the dashboard**. Verified in both directions:

```
grep -rE "^\s*(from|import)\s+(orchestration|FinAgents)" dashboard --include=*.py   → 0
grep -rE "^\s*(from|import)\s+dashboard" orchestration --include=*.py              → 0
```

Neither tree imports the other. (The word "orchestration" does appear in four
dashboard docstrings, e.g. `domain/runs/service.py:1`, describing the Run API —
unrelated to `orchestration/`.)

For fairness to the paper: `orchestration` genuinely contains cost machinery —
market-impact and execution pydantic schemas
(`transaction_cost_agent_pool/schema/market_impact_schema.py`,
`execution_schema.py`), participation-rate parameters
(`agents/optimization/cost_optimizer.py:225`), and timing-risk computation over
an execution horizon (`agents/optimization/timing_optimizer.py:636`). The claim
is substantiated *there*. It simply has no bearing on what the dashboard
executes.

Even in `orchestration`, venue latency is declared but inert: `latency_ms` is a
`VenueConfig` field (`config_loader.py:53`) with no reader anywhere in the tree.

**Conflating the two trees has been a recurring error in this project. They
share a repository and nothing else.**

### 4. What if a decision takes longer than one bar interval?

**Nothing. The bar is not skipped, queued, or dropped — the decision is applied
anyway.**

The loop is ordinary synchronous Python. The LLM call at `engine.py:765` blocks;
when it returns, execution continues at `engine.py:833` with the same bar. A
decision that took an hour of wall-clock time and one that took 50 ms produce
**bit-identical** results. Real time changes only how long the backtest takes to
run, never what it computes.

There is no timeout on the pipeline path — no `timeout`, `max_retries`, or
deadline in `pipeline_runner.py` or `backtest_harness.py`.

#### One exception, and it is a cliff rather than a model

The non-default `ai_hedge_fund` runtime does impose a real-time timeout:

- `DEFAULT_TIMEOUT_SECONDS = 300` — [adapter.py:39](../../dashboard/backend/infrastructure/ai_hedge_fund/adapter.py#L39)
- passed to the runner at [adapter.py:782](../../dashboard/backend/infrastructure/ai_hedge_fund/adapter.py#L782)
- on expiry, `AgentRuntimeError` → the step is held: `decision = {"actions": []}` — [engine.py:810-828](../../dashboard/backend/domain/backtesting/engine.py#L810)

This is the only place in the codebase where wall-clock time changes a trading
outcome. It is worth stating precisely what it is and is not. It is **binary**:
under 300 s the decision is applied with a perfect fill; over 300 s it is
discarded entirely. It does not degrade fill quality, so it cannot express "a
decision arrived 40 seconds late and got 3 bps worse." And it applies only to a
runtime the seed data never used.

It is, however, a real collision with the measurements: **Arm B at C=32 measured
596 s per call, which already exceeds this 300 s timeout.** On that runtime, at
that serving configuration, every decision would be dropped.

### 5. Bar interval vs. measured decision latency

The interval is **hourly** — `timeframe=self.TimeFrame.Hour`
([alpaca_bars.py:114](../../dashboard/backend/infrastructure/market_data/alpaca_bars.py#L114)),
confirmed by the log line "Timeframe: Hourly (1h)" at `alpaca_bars.py:110`.
Bars are filtered to market-hours sessions (`engine.py:571`), giving ~7 bars per
trading day — consistent with the seed runs' 161 bars over 2026-04-15→05-15.

So one bar = **3,600 s**. Against measured end-to-end latency:

| Config | e2e p50 | Fraction of one hourly bar | ×5 for a 5-step pipeline |
|---|---|---|---|
| Arm C, C=1 | 2.65 s | 0.07 % | 0.4 % |
| Arm C, C=32 | 3.44 s | 0.10 % | 0.5 % |
| Arm B, C=1 | 9.14 s | 0.25 % | 1.3 % |
| Arm B, C=8 | 148.29 s | 4.1 % | 21 % |
| Arm B, C=32 | 596.48 s | 16.6 % | **83 %** |

Two things follow, and they point in opposite directions.

**At hourly bars the thesis is weak.** Even the worst measured configuration
fits inside one bar with room to spare. A single-call decision never misses its
bar, so even if the engine *did* model arrival time, there would be almost
nothing to detect.

**The margin is thinner than it looks**, for two reasons the table understates:

1. *These are 256-output-token calls.* The fixture pins output length with
   `ignore_eos`. ATL's real decisions emit 860–5,005 tokens (Phase 6b). These
   workloads are decode-bound, so a first-order scaling puts a real Arm-B C=32
   Gemini-length call near 10⁴ s — well beyond an hourly bar. *This is an
   extrapolation from the token ratio, not a measurement, and should be
   measured before it is relied on.*
2. *A 5-step pipeline multiplies* — the calls are strictly sequential
   (§Deliverable 3), so latency adds. At Arm B C=32 that is already 83 % of a
   bar at fixture output lengths.

**The interval is the real lever.** Nothing pins the engine to hourly bars
except the Alpaca request at `alpaca_bars.py:114`. At minute bars (60 s), Arm B
C=32 overruns by ~10× and even Arm C C=32 consumes 5.7 % of the bar. If this
research direction is pursued, *bar interval is the independent variable that
makes latency bite*, not model or serving stack.

---

## Deliverable 2: feasibility verdict

### (b) LATENCY IS INVISIBLE, BUT INJECTABLE

Invisible for the reasons above. Injectable because the seam is unusually
clean — and, decisively, **the change can live entirely in the benchmark
harness without modifying `dashboard/`.**

#### Why no dashboard change is needed

`execute_actions` is a pure, keyword-only function over explicit state:

```python
# execution.py:38
def execute_actions(*, actions, market_data, timestamp, cash, positions,
                    entry_prices, trades) -> float:
```

Its own docstring commits to this: *"pure, domain-level execution logic over
explicit state"* and *"must not import FastAPI, Anthropic, Alpaca clients, the
database singleton, API routers, or scripts"*
([execution.py:4-5, 30-31](../../dashboard/backend/domain/trading/execution.py#L4)).

Because `market_data` is an argument rather than something the function fetches,
a caller may pass the bar from time *t+k* while the decision was computed from
the bar at *t*. That single substitution **is** the latency model, and it
requires no edit to `dashboard/`.

#### Minimal change

A harness-side replay driver that reimplements the bar loop:

1. For bar *i*, build state from bar *i* and obtain a decision (calling the real
   `run_pipeline_decision`, or replaying a recorded decision trace).
2. Draw a latency `L` — a constant, or sampled from a measured e2e
   distribution in `results/`.
3. Compute `k = ceil(L / bar_interval)`.
4. Call the real `execute_actions` with `market_data` from bar *i+k*.
5. Sweep `L` and compare terminal equity, per-trade slippage, and decisions
   dropped past the horizon.

The engine loop's essential logic is ~30 lines (`engine.py:735-846`); the rest
is currency conversion, progress publishing, and post-trade bookkeeping that a
latency experiment can omit.

#### How invasive

**Low, with one honest caveat.** Both dependencies (`execute_actions`,
`run_pipeline_decision`) are importable and pure. No dashboard modification, no
new provider code, no schema change. The work is a new benchmark module plus
tests — the same shape as `atl_pipeline_probe.py`.

The caveat is that the *interesting* version of the experiment needs real bars,
and market data is currently blocked on Alpaca credentials (see
`local_atl_setup.md`). A synthetic price series would demonstrate the mechanism
but could not support a claim about realised trading cost, because the effect
size depends entirely on real intra-bar price dynamics.

#### What this would and would not show

It would show how terminal P&L degrades as decision latency rises, at a chosen
bar interval — turning serving latency into a dollar figure.

It would **not** be a claim about ATL as it ships. The latency sensitivity would
be a property of the harness's execution model, not of the dashboard engine,
which remains a zero-latency simulator. Any writeup must say so, or it
misrepresents the platform. The honest framing is *"here is what latency would
cost if the engine modelled it"*, not *"here is what latency costs ATL users."*

---

## Deliverable 3: two secondary workload properties

### Synchronised arrivals — **NOT SUPPORTED BY THE CODE**

This one does not survive checking, and the claim should be dropped or rewritten.

The dashboard **never runs multiple agents concurrently**. The leaderboard
deploys models in a plain sequential loop, one complete backtest at a time:

```python
# refresh_daily_leaderboard.py:89
for entry in llm_entries:
    ...
    result = deploy_model_run(entry_id, ...)
```

There is no thread pool, process pool, or `asyncio.gather` anywhere in the
backtest path — the only `ThreadPoolExecutor` in the backend is in
`market_data/quotes.py:413`, fetching quotes, unrelated to agent scheduling.

Nor is there a live scheduler that could produce bar-boundary arrivals. The
paper-trading backend is an explicit stub:

> "Paper trading has no execution path in the codebase today... there is no
> order-submission or step loop. Building this means real new code (live order
> submission, a realtime decision-cadence scheduler, live bar assembly)"
> — [paper_backend.py:1-7](../../dashboard/backend/execution/paper_backend.py#L1)

So the arrival pattern in the codebase is **serial, not bursty**. Agents do not
contend, because only one exists at a time.

The harness's burst arrivals therefore model a *hypothetical* deployment — N
agents sharing one endpoint, all triggered at bar close — which is plausible on
its face but has **no support in this codebase**. That is a defensible modelling
choice, but it must be labelled an assumption. Citing ATL as evidence for
synchronised arrivals would be wrong: what ATL actually demonstrates is the
opposite.

*What would resolve it:* a real multi-tenant deployment, or the realtime
scheduler in `paper_backend.py` once built — its trigger policy would settle
whether arrivals are synchronised or staggered.

### Sequential intra-decision dependency — **CONFIRMED**

Pipeline steps cannot overlap, and the constraint is structural rather than an
un-optimised implementation.

The loop is synchronous and blocking:

```python
# pipeline_runner.py:451
for index, step in enumerate(decision_steps):
    ...
    # pipeline_runner.py:466 — blocking; nothing dispatches ahead
    response = client.messages.create(...)
    ...
    # pipeline_runner.py:482 — output appended, feeding the NEXT iteration
    prior_outputs.append({...})
```

`pipeline_runner.py` contains no `async`, `await`, `gather`, `ThreadPool`, or
`concurrent` — zero matches.

More importantly, step *n+1*'s prompt embeds every prior step's output:
`prior_outputs` is passed into `_build_step_prompt` (`pipeline_runner.py:460`)
and rendered under an `=== UPSTREAM PIPELINE OUTPUTS ===` header
(`pipeline_runner.py:120-127`). The dependency is a genuine data dependency, so
these calls **cannot** be parallelised without changing what the pipeline
computes.

This was verified by execution as well as by reading: `atl_pipeline_probe.py`
drives the real `run_pipeline_decision` with a stub client and observes exactly
3 calls for the 3-step pipeline and 5 for the 5-step, with `prior_outputs`
present in every step after the first.

**Consequence for the serving benchmark.** Intra-decision latency *adds* while
inter-agent concurrency is high. One agent's decision costs `steps × per-call
latency` end to end no matter how well the server batches, so per-decision
latency is bounded below by the sequential chain. Batching improves aggregate
throughput across agents; it cannot shorten a single decision. This is a real
distinguishing property of the workload, and unlike the arrival-pattern claim,
the code supports it.

---

## Summary

| Question | Answer |
|---|---|
| Decision timestamp used? | No — same bar in, same bar out (`engine.py:756→833`) |
| Fill price | Same bar's close (`execution.py:64`); equals the price the agent saw (`portfolio.py:78`) |
| Slippage / fees / spread / partial fills | None in dashboard; present in `orchestration`, which the dashboard does not import |
| Decision slower than a bar? | Applied anyway. Exception: 300 s timeout → hold, `ai_hedge_fund` runtime only |
| Bar interval vs latency | 3,600 s vs 2.7–596 s measured; margin large at hourly, gone at minute bars |
| **Verdict** | **(b) invisible, but injectable — harness-side, no `dashboard/` changes** |
| Synchronised arrivals | **Not supported** — agents run serially; no scheduler exists |
| Sequential intra-decision | **Confirmed** — real data dependency, not just un-parallelised |

### What could not be determined from the code

- **The size of the latency effect.** Whether a one-bar delay costs basis points
  or nothing depends on intra-bar price dynamics, which no amount of reading
  resolves. Only the experiment in Deliverable 2 answers it, and it needs real
  market data.
- **Whether a deployed fleet would exhibit synchronised arrivals.** The
  codebase has no deployment mode in which the question arises.
- **Real decision latency at ATL's actual output lengths.** All e2e figures come
  from 256-token fixture calls; ATL emits 860–5,005. The extrapolation in §5 is
  arithmetic on the token ratio, not a measurement.
