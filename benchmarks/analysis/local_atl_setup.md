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
