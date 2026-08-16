# Questions only you can answer

Five questions. Each is unanswerable from anything in this repository, each
unblocks something specific, and each has a cheap way to get it.

The first two decide the $50K answer outright. The rest change what should be
measured next.

---

## 1. How much backtesting happens in production?

**Why it can't be answered here.** The committed seed database has seventeen
runs, seven with LLM usage — a snapshot, not a production log. Nothing records
how often real users launch backtests.

**Why it matters more than anything else.** One backtest of the seed window cost
~161 LLM calls, one per hourly bar. Backtest volume is unbounded by design: a
backtest replays a whole window on demand, where live trading is rate-limited by
the bar interval. At 100 users, the number of backtests per user per day needed
to reach $50K/month ranges from ~232 on the platform's default model to **1.2 on
GPT-5.5** — which one engaged user could exceed in a morning.

**Cheapest way to get it.** One query against the Render database:

```sql
SELECT DATE(created_at) AS day, COUNT(*) AS runs,
       COUNT(DISTINCT session_id) AS users
FROM agent_runs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY day ORDER BY day;
```

---

## 2. What is the production model mix?

**Why it can't be answered here.** The seed runs are one-per-model leaderboard
entries — a deliberate sweep, not a reflection of what production uses.

**Why it matters.** Cost per call spans **193x** across the seven measured
models, from $0.000447 on Nemotron to $0.086 on GPT-5.5. That single factor
moves the monthly bill more than every other lever combined. Two effects
multiply: price per token (~150x on output) and verbosity under an identical
prompt (5.8x). Nemotron happens to be cheapest on *both*, so any move off the
default pays twice.

**Cheapest way to get it.**

```sql
SELECT llm_model, COUNT(*) AS runs, SUM(llm_calls) AS calls,
       SUM(est_cost_usd) AS cost
FROM agent_runs
WHERE llm_calls > 0 AND created_at > NOW() - INTERVAL '30 days'
GROUP BY llm_model ORDER BY cost DESC;
```

That also gives real per-model call volume, which the cost notebook takes
directly as `model_mix`.

---

## 3. What is the typical pipeline depth in production?

**Why it can't be answered here.** All seven seed runs used the **single-call**
path — six at exactly 1.000 calls per decision, Nemotron at 0.994. None had
`metadata.initial_pipeline` populated, so the multi-step path has no recorded
run anywhere in the artifacts.

**Why it matters.** A pipeline issues exactly one LLM call per configured step,
sequentially — verified against the real runner with a stub client: three steps
produce three calls, five produce five. So depth multiplies every cost figure
linearly. If production typically runs five steps, every number in the report is
5x low.

It also compounds: steps are strictly sequential and each step's prompt embeds
all prior outputs, so a verbose model inflates its own later prompts.

**Cheapest way to get it.**

```sql
SELECT jsonb_array_length(metadata->'initial_pipeline') AS steps, COUNT(*)
FROM agent_runs
WHERE metadata ? 'initial_pipeline'
GROUP BY steps ORDER BY steps;
```

If that returns nothing, production is single-call too — which is itself the
answer, and a reassuring one.

---

## 4. Does paper trading run continuously?

**Why it can't be answered here.** It cannot run at all in this codebase.
`dashboard/backend/execution/paper_backend.py` is an explicit stub: no order
submission, no step loop, and a realtime decision-cadence scheduler listed as
not built. There is also no scheduler, cron entry, or concurrent execution path
anywhere in the dashboard — the leaderboard runs agents in a plain sequential
loop, one complete backtest at a time.

**Why it matters.** Two things hang on it:

- **Cost.** If live agents run continuously, they are a second cost centre that
  scales with agent count. If not, live trading is a rounding error and
  backtesting is the whole bill.
- **The serving benchmark's arrival model.** The harness uses burst arrivals,
  which assumes many agents firing together at a bar boundary. Nothing in the
  code supports that — the dashboard's actual pattern is serial. If there is a
  deployment where agents genuinely run concurrently, that assumption becomes
  defensible; if not, it should be relabelled as hypothetical.

**Cheapest way to get it.** You know the deployment. If there is a scheduler
running outside this repository, its trigger policy answers both parts.

---

## 5. Dashboard, or `orchestration/FinAgents`?

**The scoping question still open since the start**, and worth settling
explicitly because getting it wrong has already cost time.

They are different systems that share a repository and nothing else — neither
imports the other, verified in both directions. They differ in ways that matter
for what should be measured:

| | `dashboard/` | `orchestration/FinAgents/` |
|---|---|---|
| What it is | the platform that ships | the paper artifact |
| Execution model | fills at the same bar's close; no slippage, fees, spread, or partial fills | transaction-cost and market-impact machinery, participation rates, timing risk |
| Latency | invisible to the engine | venue latency is a config field (unread) |
| What has been measured | all of Phases 4-8 | nothing |

Every finding in the advisor report concerns `dashboard/`.

**Why it matters.** If the thesis is about serving performance for the *product*,
the dashboard is right and the next step is the instrumentation patch. If it is
about the *multi-agent architecture* in the paper, essentially none of the
measurement so far applies to it, and that should be known now rather than after
another phase.

**What would resolve it.** Your call on which system the research is about.

---

## What each unlocks

| Question | Unblocks |
|---|---|
| 1 + 2 | The $50K answer becomes a figure instead of a conditional |
| 3 | Whether every cost figure needs multiplying by pipeline depth |
| 4 | Whether live trading is a cost centre; whether burst arrivals are real |
| 5 | Which system the next phase should measure |

Questions 1-3 are single SQL queries against the Render database and would take
minutes. Question 4 needs deployment knowledge. Question 5 needs a decision.

Once 1-3 are answered, `cost_model.ipynb` produces the monthly figure directly —
it is already parameterised on exactly these inputs and refuses to compute
without them.
