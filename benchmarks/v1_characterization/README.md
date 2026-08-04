# v1 Hardware Characterization (superseded — preserved as evidence)

These three scripts were the first-pass hardware characterization run on Colab
(A100). They are **superseded** by the Phase 2+ benchmark harness in the parent
`benchmarks/` directory and should not be used to produce new numbers.

They are kept, unmodified and with their git history intact (`git mv` from
`orchestration/benchmarks/`), so that any figure quoted from this phase can be
traced back to the exact code that produced it.

**Read this file before citing any v1 number.** The three scripts measure
different things under different conditions, and the differences are large
enough that a finding from one cannot be used to explain a number from another.

---

## The three scripts

### `concurrent_stress_test.py` — no LLM, no GPU

This script performs **no inference of any kind**. It does not import `torch`
or `transformers`. There is no model.

The "agent work" is:

```python
time.sleep(random.uniform(0.5, 2.0))
```

followed by `generate_market_data(...)` / `run_trading_strategy(...)` — synthetic
price series and a NumPy-level strategy loop, imported from
`orchestration/run_simple_backtest.py`.

Its actual contribution is the **load structure** reused by the GPU scripts:

```python
agents  = ['DeepSeek', 'Claude', 'GPT-4']            # 3
symbols = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA']  # 7
tasks   = [... for i, agent in enumerate(agents * 5) for symbol in symbols]
#          3 agents x 5 multiplier x 7 symbols = 105 tasks
ThreadPoolExecutor(max_workers=15)
```

That is the origin of the **105 tasks / 15 threads** shape. The agent names are
labels on a sleep; no DeepSeek, Claude, or GPT-4 call is made.

Two further notes:

- `THICK_CONTEXT_BLOCK` (500 repeats, line 7) is **defined but never
  referenced**. No context is passed anywhere. The in-code comment is explicit
  that the context is "mocked here as a sleep penalty."
- The script imports `run_simple_backtest` as a top-level module, so it only
  ever ran with `orchestration/` as the working directory. After the move to
  `benchmarks/v1_characterization/` that import will not resolve. This is left
  as-is deliberately — the file is a record, not a runnable target.

**Any wall-clock or throughput number from this script is a property of
`time.sleep`, not of hardware.**

### `real_gpu_stress_test.py` — concurrent, non-streaming

| | |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, `torch.float16`, `device_map="auto"` |
| Load | 105 tasks, `ThreadPoolExecutor(max_workers=15)` |
| Generation | `max_new_tokens=5`, `do_sample=False`, `use_cache=True` |
| Context | `"--- ANNUAL EARNINGS REPORT & FORWARD GUIDANCE ---\n" * 200` (~2600 tokens, estimated — see caveat below) |
| Streaming | **None** |
| Trace output | `/content/tb_logs` |

Because there is no streamer, this script **cannot report TTFT or ITL**. It
measures only end-to-end wall-clock latency per task, from before tokenization
to after `generate` returns.

Peak memory is reported as `torch.cuda.max_memory_allocated()` after
`reset_peak_memory_stats()` at the start of `main()`. That is **one global
allocator high-water mark for the whole process** — it is not per-thread, not
per-request, and cannot be divided by 15 to get a per-request footprint.

Three properties of this script materially affect how its numbers should be read:

1. **The profiler section does not profile the benchmark.** `run_profiler()` is
   called at the *end* of `main()`, after the concurrent section has finished.
   It is single-threaded, `active=2` steps — and it uses a **different prompt
   entirely**: `"Task profiler: Analyze and output."`, a handful of tokens
   rather than the ~2600-token block. So the profiled path differs from the
   benchmarked path in both concurrency *and* context length.
2. **The 15 threads share one model instance** with no batching layer. Python
   threads calling HuggingFace `generate` on a single model contend on the GIL
   and serialize at the allocator; this is thread-level oversubscription of one
   model, not concurrent serving in the sense a real inference server means it.
3. **There is no warmup.** The first tasks absorb lazy CUDA context init and
   kernel autotuning, which inflates the reported latency `Min`/`Max` spread.

### `advanced_gpu_stress_test.py` — sequential, streaming — **this is the analyzed trace**

| | |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, `torch.float16`, `device_map="auto"` |
| Load | **One request at a time**, sequential loop. No thread pool. |
| Generation | `max_new_tokens=15`, `do_sample=False` |
| Context | `"--- ANNUAL EARNINGS REPORT ---\n" * 100` + one trailing instruction (~1300 tokens, estimated — see caveat below) |
| Streaming | `TextIteratorStreamer` (a `Thread` is used to drive `generate` so the streamer can be consumed — this is a streaming mechanism, not concurrency) |
| Trace output | `/content/tb_logs_advanced` |
| Profiler schedule | 2 untraced warmups, then `wait=1, warmup=1, active=2, repeat=1` over 4 iterations |

**The torch profiler trace that was analyzed came from this script.**

---

## The thing that must not be conflated

> **The analyzed trace came from a single-request sequential run, not from the
> 105-task / 15-thread concurrent run.**

The three scripts differ along every axis that matters for an inference
measurement:

| | concurrent_stress | real_gpu_stress | advanced_gpu_stress |
|---|---|---|---|
| Real inference | **no** | yes | yes |
| Concurrency | 15 threads (of `sleep`) | 15 threads | **1 (sequential)** |
| Requests | 105 | 105 | 4 traced (+2 warmup) |
| `max_new_tokens` | n/a | **5** | **15** |
| Context repeats | 500 (unused) | **200** | **100** |
| Streaming / TTFT | n/a | **no** | yes |
| Trace dir | n/a | `/content/tb_logs` | `/content/tb_logs_advanced` |

Consequences to respect when writing up v1:

- A kernel-level or timeline finding from the `advanced` trace **describes a
  single-stream decode of 15 tokens**. It cannot explain throughput, latency
  spread, or memory pressure observed in the 105-task/15-thread `real_gpu` run.
- The two GPU scripts do not even share a context length. They also do not share
  a context *string* — `"...REPORT & FORWARD GUIDANCE ---\n"` vs
  `"...REPORT ---\n"` — so the ratio between them is **not** simply 200:100.
  Prefill cost is not comparable between the two without re-measuring.
- `max_new_tokens` differs 5 vs 15. Any per-token metric is dominated by prefill
  at these lengths, and differently so in each script.

### Caveat on the token counts

The ~2600 / ~1300 figures are **estimates carried over from the original
analysis and have not been re-verified against the Qwen2.5 tokenizer** (doing so
requires downloading the tokenizer, which is out of scope for a no-GPU
scaffolding phase). Treat them as approximate. The *structural* point above
holds regardless of the exact counts: the two contexts are built from different
strings at different repeat counts.

### Known measurement bug in `advanced_gpu_stress_test.py`

The reported "TPOT" is not inter-token latency:

```python
tpot = (end_time - first_token_time) / token_count
```

Two defects:

1. It divides the post-first-token window by the **total** token count rather
   than `token_count - 1`. With `max_new_tokens=15` that is a ~7% systematic
   understatement even before the next point.
2. `token_count` only increments when `new_text.strip()` is truthy, so
   whitespace-only streamer chunks are dropped from the denominator while the
   time they took stays in the numerator — inflating the per-token figure by an
   amount that depends on the output's whitespace.

Any per-token number quoted from this script inherits both. This is recorded
here rather than fixed: the script is preserved as the artifact that produced
the existing numbers.

---

## Status

Superseded. Do not extend these scripts. New measurements belong in the harness
described in [`../README.md`](../README.md), which emits a run manifest
([`../RUN_MANIFEST_SCHEMA.md`](../RUN_MANIFEST_SCHEMA.md)) binding every number
to a specific GPU, config, fixture, and upstream commit.
