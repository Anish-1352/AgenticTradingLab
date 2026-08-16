# Running ATL's pipeline path locally to measure calls-per-decision

**Read the blocker first — it changes which key you need.**

## BLOCKER: a raw Gemini key cannot drive this backend

`dashboard/backend/infrastructure/llm/providers/` contains exactly three
integrations, and all three are **Anthropic-SDK-compatible wrappers**:

| Provider | Env var | Talks to |
|---|---|---|
| `anthropic_native` | `ANTHROPIC_API_KEY` | Anthropic directly |
| `commonstack` | `COMMONSTACK_API_KEY` | CommonStack gateway |
| `openrouter` | `OPENROUTER_API_KEY` | OpenRouter gateway |

`make_llm_client()` constructs `Anthropic(...)` for every path and only swaps
`base_url` and auth. There is **no Google/Gemini client**, no
`GOOGLE_API_KEY`/`GEMINI_API_KEY` anywhere in the backend, and the task
correctly forbids adding one.

So a key from Google AI Studio will not authenticate against any existing path.
Gemini did reach ATL in the seed data — the `lb_gemini_3_1_pro_preview` run used
the slug `google/gemini-3.1-pro`, which `token_cost.py` lists as
**CommonStack-verified**. Gemini arrived *through a gateway*, not through a
Google key.

### A second blocker: the backtest script cannot select OpenRouter

`OPENROUTER_API_KEY` alone will **not** work, despite the engine's own error
message advertising it:

```
⚠️  No LLM key (COMMONSTACK_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY) set.
```
— `dashboard/backend/domain/backtesting/engine.py:242`

`engine.py:235` calls `make_llm_client()` with **no arguments**, so
`resolve_integration(None)` runs — and that returns CommonStack when
`COMMONSTACK_API_KEY` is set, otherwise native Anthropic. It never returns
OpenRouter. The providers package says so outright:

> OpenRouter is never auto-selected — set `integration: "openrouter"` on the
> leaderboard entry (or pass the kwarg).
> — `dashboard/backend/infrastructure/llm/providers/__init__.py:11-12`

`backtest_hourly_agent.py` exposes no `--integration` flag and no env var
selects one. So with only `OPENROUTER_API_KEY` set, the client resolves to
`None` and the run silently falls back to rule-based with `llm_calls = 0` —
the documented "measurement did not happen" state.

**Env-only workaround, no `dashboard/` change.** `commonstack.make_client`
builds `Anthropic(api_key=COMMONSTACK_API_KEY, base_url=COMMONSTACK_BASE_URL)`,
and OpenRouter's Anthropic skin is wire-compatible — which is precisely what
`openrouter.make_client` constructs. So pointing the CommonStack provider at
OpenRouter works:

```bash
export COMMONSTACK_API_KEY="$OPENROUTER_API_KEY"
export COMMONSTACK_BASE_URL="https://openrouter.ai/api"
```

Two things this costs, both worth knowing before trusting a number from it:

1. **The OpenRouter reasoning wrapper is bypassed.** `_OpenRouterMessages`
   normally injects a `reasoning.max_tokens` budget; via the CommonStack path
   it never runs, so the provider's raw default applies. The openrouter module
   warns what that does to reasoning models: they "often return only
   thinking/redacted_thinking and no JSON text", which aborts the pipeline
   step. **Prefer a non-reasoning model on this path.**
2. **Provenance lies.** The run is OpenRouter but every code path calls it
   CommonStack. Record the real gateway alongside any result from this route.

The clean fix is a one-line `integration` passthrough in the engine, but that
is a `dashboard/` change and out of scope here.

### What you actually need

Any **one** of these, in preference order for this measurement:

1. **`OPENROUTER_API_KEY`** — you may already have one from Phase 5's arm A.
   Routes to Gemini via the slug `google/gemini-3.1-pro`. Cheapest to obtain.
2. **`COMMONSTACK_API_KEY`** — the path the seed run actually used.
3. **`ANTHROPIC_API_KEY`** — works out of the box with zero configuration.

**For this specific measurement, the model barely matters.** What is being
measured — calls per decision, retry inflation, prompt growth across steps — is
a property of `run_pipeline_decision`, not of the model. Use whichever key you
have. Claude Haiku is the cheapest of the three paths and is entirely
sufficient.

---

## What this run measures, and what it must not be used for

**Transfers to production (model-independent):**

- calls per decision for a multi-step pipeline
- whether retries inflate that count in practice
- prompt composition and growth as `prior_outputs` accumulates across steps
- sequential latency accumulation

**Does NOT transfer (model-specific):**

- **output token counts.** The seed data measured 860 tokens/call for Nemotron
  and 5,005 for Gemini under the same prompt — a **5.8x spread**, with Gemini
  the most verbose of the seven. An output figure from this run describes the
  model you ran, not Nemotron.
- cost per decision, which is output length times price.

`atl_token_extract.py` tags every output figure with the model that produced it
and refuses to let it be substituted across models. That enforcement is in the
tool, not just in this document.

---

## Setup

### 1. Python and dependencies

```bash
cd /path/to/AgenticTradingLab
python3.13 -m venv .venv-atl && source .venv-atl/bin/activate
pip install -r requirements.txt        # the backend's own deps, NOT benchmarks/
pip install pytest                     # not in requirements.txt
```

### 2. Database — SQLite, no Postgres needed

`DATABASE_PATH` selects the SQLite file and the schema self-migrates on first
import. There is nothing to run.

```bash
export DATABASE_PATH="$PWD/local_atl.db"
```

Leave `USERS_DATABASE_URL`, `CONTENT_DATABASE_URL` and
`AGENT_RUNS_DATABASE_URL` **unset** — unset means local SQLite, which is what
you want. Setting them points at Postgres, and pointing them at production is
the one thing this task forbids.

> Do not use `dashboard/storage/data/backtest.db`. That file is committed, seeds
> `dashboard/config/defaults.json`, and must not gain rows from a local
> experiment.

### 3. Credentials

```bash
export OPENROUTER_API_KEY='sk-or-...'      # or COMMONSTACK_API_KEY / ANTHROPIC_API_KEY
```

Never put the key in a file. It reaches manifests only as a SHA-256.

`app.py` loads `dashboard/.env` if present — that path is gitignored, but
exporting in the shell keeps the key out of the filesystem entirely.

### 4. Configure a multi-step pipeline

The step count **is** the measurement, so run more than one variant. A pipeline
is a list of step dicts on the agent; each step becomes exactly one LLM call in
`run_pipeline_decision`. A step needs `label`, `prompt` and `outputFormat`; the
final step must emit `actions`, `orders` or `risk_actions`.

Run a **3-step** and a **5-step** variant. If calls-per-decision does not come
out as 3 and 5 respectively, the difference is retries — which is the more
interesting result.

> A step with `presetKey: "post_trade_analysis"` is stripped from the hourly
> path and runs once per trading **day**. Keep at most one, and expect its calls
> to amortise rather than multiply.

### 5. Run a small backtest

Keep it structural, not a performance test: **1–2 tickers, a few days**. The
pipeline shape is visible in a handful of decisions; a long run only spends
money.

```bash
python dashboard/scripts/backtest_hourly_agent.py    # from the repo root
```

Run from the repo root — the backend is the `dashboard.backend` package and
imports fail otherwise.

### 6. Extract the measurement

```bash
cd benchmarks
python -m analysis.atl_token_extract --local-db "$DATABASE_PATH" \
    --json-out results/atl_local_pipeline.json
```

It reports calls-per-decision two independent ways — `llm_calls / decisions`
and the step count from `metadata.initial_pipeline` — and flags disagreement,
because disagreement means retries fired.

Omit `--local-db` and it reports the seed DB alone. Supplied, it prints both
extracts and then a three-way comparison against the original assumption; the
earlier figures are kept as their own column rather than overwritten.

### The decision denominator is not `backtest_decisions`

`backtest_decisions` is the obvious source and the wrong one. `engine.py` calls
`insert_decisions` **only** under the `ai_hedge_fund` runtime — under the native
pipeline runtime, the path this measurement is about, the table is never written
at all. It is empty in the seed DB (0 rows, all 17 runs) and it will be empty in
your local DB too. That is the schema behaving as written, not your run failing.

The denominator used instead is `equity_timeseries`, one row per simulated bar,
with one decision per bar. It is a proxy and is labelled as one in every figure
(`decision_count.is_proxy`). On the seed data it gives 1.000 calls/decision for
six runs and 0.994 for Nemotron — 160 calls across 161 bars, i.e. one bar that
never reached the model.

---

## If it does not run

Report what blocked it rather than working around it. Likely stops:

| Symptom | Cause |
|---|---|
| Client is `None`, run falls back to rule-based | No recognised API key env var set |
| `ModuleNotFoundError: dashboard` | Not run from the repo root |
| `llm_calls = 0` in the row | Fell back to rule-based; the key never resolved |
| `steps = 0` in metadata | The agent had no pipeline — this is the single-call path again, and the gap stays open |

That last row is the failure that matters: it is exactly the state all seven
seed runs are in, and it means the measurement did not happen.

## Current status: BLOCKED on TWO credentials, not one

### 1. LLM credential — needed for the measurement

```
$ python -c "from dashboard.backend.infrastructure.llm.providers import make_llm_client; print(make_llm_client())"
None
```

`resolve_integration(None)` falls through to `anthropic`, and
`ANTHROPIC_API_KEY` is unset — as are `OPENROUTER_API_KEY` and
`COMMONSTACK_API_KEY`. `dashboard/.env` does not exist. A `None` client means
the backtest runs the rule-based fallback and records `llm_calls = 0`.

> `dashboard/.env` is loaded by `app.py` only. `backtest_hourly_agent.py` does
> **not** load it, so for this run the key must be in the real environment.

### 2. Alpaca credentials — needed for the backtest to run at all

This is an *hourly* backtest over real Alpaca bars.
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are unset and `credentials/alpaca.json`
does not exist (only `alpaca.json.example`), so `AlpacaCredentialsError` fires
before any pipeline step executes. The committed cache cannot substitute: it
holds **daily** bars only (`*_1d.csv`, AAPL/MSFT, 2024) and zero hourly files.

### What is already done

`configs/atl_pipelines/pipeline_3step.json` and `pipeline_5step.json` are
written and verified. `analysis/atl_pipeline_probe.py` drives the **real**
`run_pipeline_decision` with a stub client — no key, no network — and confirms
on the actual code path:

| | 3-step | 5-step |
|---|---|---|
| configured decision steps | 3 | 5 |
| LLM calls issued | **3** | **5** |
| decision produced | yes | yes |

So derivation (b) holds against execution, not just against a metadata field,
and both pipeline files are known-good before any money is spent. The probe
also shows the market snapshot enters **step 1 only** — later steps carry
`prior_outputs` instead, so the static prefix does not repeat and prompt growth
is driven entirely by upstream model output.

What the probe cannot give: token counts, retries, cost, latency. Its prompt
sizes are lower bounds, because a stub's output is a few dozen characters where
a real model emits 860–5,005 tokens.

### To finish it

Set both credentials in the environment, then:

```bash
export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...
export ANTHROPIC_API_KEY=...
export DATABASE_PATH="$PWD/local_atl.db"

python dashboard/scripts/backtest_hourly_agent.py \
    --start 2026-04-15 --end 2026-04-17 --use-llm \
    --pipeline-file benchmarks/configs/atl_pipelines/pipeline_3step.json \
    --run-id local_pipeline_3step --session-id bench-local

python dashboard/scripts/backtest_hourly_agent.py \
    --start 2026-04-15 --end 2026-04-17 --use-llm \
    --pipeline-file benchmarks/configs/atl_pipelines/pipeline_5step.json \
    --run-id local_pipeline_5step --session-id bench-local

cd benchmarks && python -m analysis.atl_token_extract \
    --local-db "$DATABASE_PATH" --json-out results/atl_local_pipeline.json
```

If calls-per-decision does not come out at 3 and 5, the difference is retries —
the probe has already ruled out a malformed pipeline as the cause.
