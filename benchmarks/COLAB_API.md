# Running arm A (hosted API) on Colab

Different from [`COLAB.md`](COLAB.md) in every way that matters:

| | Arms B / C (local inference) | Arm A (hosted API) |
|---|---|---|
| Runtime | A100 required | **CPU is fine** — free tier works |
| Setup | ~50 lines of dependency reconciliation | `pip install httpx` |
| Cost | GPU hours | **real money, per request** |
| Ceiling | VRAM and kernel scheduling | **someone else's rate limit** |
| Main risk | a wasted session | **an unbounded bill** |

**Read the cost section before running anything.** The failure mode here is not
a crash, it is a charge.

---

## Cell 1 — Get the code

```python
%cd /content
!git clone https://github.com/Anish-1352/AgenticTradingLab.git 2>/dev/null || true
%cd /content/AgenticTradingLab
!git fetch origin && git checkout feature/gpu-serving-benchmark
!git reset --hard origin/feature/gpu-serving-benchmark
```

## Cell 2 — Install

No CUDA reconciliation, no torch, no restart:

```python
!pip install -q httpx pyyaml
```

## Cell 3 — The API key

**Never put the key in a notebook cell, a file, or a commit.** A notebook is
saved to Drive with its output; a key pasted into one is a key you have to
rotate.

Use Colab's secret manager:

```python
from google.colab import userdata
import os
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
print('key set:', bool(os.environ.get('OPENROUTER_API_KEY')))   # prints a bool, never the key
```

Add the secret once via the key icon in Colab's left sidebar. If `userdata` is
unavailable, `getpass` keeps it off screen and out of the saved output:

```python
import getpass, os
os.environ['OPENROUTER_API_KEY'] = getpass.getpass('OpenRouter API key: ')
```

The benchmark stores **only a SHA-256 of the key** in its manifest
(`openrouter_api_key_sha256`), so two runs can be confirmed to have used the
same credential — tier and rate limit ride on the key — without the manifest
becoming a secret.

## Cell 4 — Dry run first. Always.

```python
!python benchmarks/runners/bench_api_baseline.py --dry-run
```

Prints the plan and a cost estimate, submits nothing, bills nothing. Check the
estimate before going further.

### What a run costs

At the default sweep — 5 concurrency levels x 32 requests = 160 requests, each
~2,620 input + 256 output tokens, priced at $0.05/M input and $0.20/M output:

| Requests | Approx. input tok | Approx. output tok | Estimated cost |
|---|---|---|---|
| 4 (smoke) | 10,480 | 1,024 | **~$0.0007** |
| 32 (one level) | 83,840 | 8,192 | **~$0.006** |
| 160 (full sweep) | 419,200 | 40,960 | **~$0.029** |

Cheap — but the estimate is priced on the *fixture's Qwen* token count, and
Nemotron tokenises the same text differently and bills on its own count. Treat
these as the right order of magnitude, not a quote. Actual `usage` is recorded
per request.

A budget guard is on by default: `--max-cost-usd` (default `$1.00`) refuses to
start if the estimate exceeds it. It is a guard against a mistyped sweep, not a
substitute for reading the estimate.

## Cell 5 — Smoke test: 4 requests

```python
!python benchmarks/runners/bench_api_baseline.py \
    --n-requests 4 --concurrency 1 \
    --out-dir /content/drive/MyDrive/atl_bench/results
```

Confirm before scaling up:

- **Requests succeed** — a `FATAL_401` means the key is wrong or unauthorised
  for this model.
- **`prompt_tokens` from the API** — compare it to the fixture's 2,620. The gap
  is the tokeniser difference, and everything downstream is priced on the
  provider's number.
- **Cost printed at the end** matches roughly what you expected.

## Cell 6 — The sweep

```python
!python benchmarks/runners/bench_api_baseline.py \
    --out-dir /content/drive/MyDrive/atl_bench/results
```

## Cell 7 — Rate limit probe

Where the API stops being an option at all:

```python
!python benchmarks/analysis/rate_limit_probe.py --dry-run
!python benchmarks/analysis/rate_limit_probe.py --max-concurrency 128
```

Escalates concurrency until >10% of a level returns 429, then stops. Uses short
outputs (32 tokens) because it measures *admission*, not generation.

**Hitting the wall is the result.** If the free tier caps at C=32, that is the
practical ceiling for a hosted-API agent fleet regardless of what the cost model
says — above it the option is not expensive, it is unavailable.

## Cell 8 — Context pooling

Runs offline against an in-process store and a mocked API, so it costs nothing:

```python
!python benchmarks/analysis/context_pooling.py \
    --agents 100 --api-latency-ms 800 \
    --csv-out /content/drive/MyDrive/atl_bench/results/pooling.csv
```

Set `--api-latency-ms` from **your measured arm A e2e p50**, not the default.

## Cell 9 — The economic model

```python
!python benchmarks/analysis/arm_a_vs_bc_model.py \
    --arm-b benchmarks/results/armB_shared_summary.json \
    --arm-c benchmarks/results/armC_shared_summary.json \
    --arm-a /content/drive/MyDrive/atl_bench/results/<armA_run>_summary.json \
    --csv-out /content/drive/MyDrive/atl_bench/results/cost_model.csv
```

With `--arm-a` it uses your **measured** cost per request rather than an
assumed price.

---

## Caveats that change how the numbers read

### Colab egress is not a data centre

Latency measured from a Colab VM includes Google's egress path to OpenRouter and
onward to the model host. A production agent fleet would run somewhere with a
deliberate network position — possibly the same region as the provider.

So **arm A's latency here is an upper bound**, and the gap is not small: a
cross-region round trip can be 100ms+ before the model does any work. The
*cost* numbers transfer (billing does not care where you called from); the
*latency* numbers do not. Report them as Colab-measured and say so.

Run it several times at different hours if you can — network variance is real,
and one sample of e2e p95 is not a distribution.

### Streaming is on by default, and it has to be

Without SSE there is no first-token signal, so TTFT and ITL are unavailable and
e2e collapses to a single instant. `--no-stream` is supported but the resulting
run is not comparable to arms B and C on latency.

### The tokeniser differs, so per-token comparisons do not hold

The prompt **text** is byte-identical to arms B and C. The **token count is
not** — Nemotron has its own vocabulary and bills on it. Cost and throughput are
reported against the provider's `usage`, never against 2,620. Any per-token
comparison across arms is invalid; per-*request* comparisons are fine.

### Pricing and rate limits change

Every manifest records the prices used, a capture date, the key hash, and the
inferred tier. A cost figure without those is not reproducible — providers
change pricing without notice, and a rate limit is a property of the account,
not the model.

---

## Cost safety checklist

- [ ] Key came from `userdata`/`getpass`, never a literal in a cell
- [ ] `--dry-run` inspected first
- [ ] 4-request smoke test before any sweep
- [ ] `--max-cost-usd` set to something you would not mind spending
- [ ] `--out-dir` points at Drive, so a preemption does not lose paid-for results
- [ ] Spend checked on the OpenRouter dashboard afterwards — the benchmark
      reports what it *believes* it spent, from the `usage` it was told
