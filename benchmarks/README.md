# ATL GPU Serving Benchmark

A before/after study of LLM inference serving for Agentic Trading Lab: what the
platform's agent workload costs on a hosted API, on a naive self-hosted stack,
and on an optimized self-hosted stack.

**Phase 4 — analysis and figures.** Everything from phase 3 plus an
`analysis/` package that turns run artifacts into a matched comparison, six
figures, and a generated `RESULTS.md`. Arm A (hosted API) is still outstanding.

Start here: [COLAB.md](COLAB.md) has the exact cell sequence.

## Why this is top-level

`benchmarks/` is a peer of `dashboard/`, `orchestration/`, `docs/`, and
`packaging/` — not a subdirectory of any of them.

**It belongs to neither subsystem.** What is under measurement is a *serving
stack* (vLLM vs. HuggingFace `generate` vs. a hosted API endpoint), driven by a
workload derived from ATL's agent traffic. `dashboard/` is the FastAPI product
that *consumes* inference; `orchestration/` is imported research code that is
not wired into the dashboard at all. A benchmark of the layer underneath both is
not a component of either, and filing it inside one would misrepresent what it
measures.

**It keeps CUDA off the deploy path.** This is the operative constraint:

- `.github/workflows/ci.yml` installs the root `requirements.txt` on **every**
  push and PR, and Render builds the production service from that same file.
  ML dependencies therefore live in [`requirements.txt`](requirements.txt) in
  *this* directory and are never added to the root one.
- CI runs `pytest dashboard/backend/tests/` with no path filter. Anything
  test-shaped under that tree runs on a GPU-less runner. Benchmark code lives
  here and is not collected by that invocation.

A `benchmarks/` directory under `dashboard/` would have put both of those one
careless import away from breaking prod deploys.

## Layout

```
benchmarks/
├── COLAB.md              # THE ENTRY POINT — exact cell sequence for the A100 runtime
├── colab_bootstrap.py    # Cell 2 — one-command environment setup; idempotent
├── session_start.py      # Cell 4 — per-session provenance; shouts if the GPU changed
├── probe_environment.py  # Cell 5 — writes ENVIRONMENT.md, gates the profiling layers
├── calibration_run.sh    # Cell 7 — fixed run measuring the noise floor (--analyze)
├── common/               # Shared instrumentation, used identically by every arm
│   ├── config.py         #   loads configs/workload.yaml; hashes the RESOLVED config
│   ├── fixtures.py       #   exact-length deterministic prompt sets
│   ├── metrics.py        #   per-request records -> TTFT / ITL / e2e / throughput
│   ├── monitor.py        #   20 Hz NVML + psutil sampler
│   ├── calibration.py    #   calibration log: within-card vs between-card spread
│   ├── trace_analysis.py #   torch.profiler chrome trace -> JSON summary
│   └── manifest.py       #   run manifest emission
├── configs/              # workload.yaml — all controlled variables
├── fixtures/             # Built prompt sets (*.meta.json committed, *.json not)
├── runners/              # Arm drivers — one per serving stack
│   ├── bench_hf_baseline.py    # Arm B — threads + transformers
│   └── bench_vllm_optimized.py # Arm C — asyncio + vLLM
├── profiling/            # run_nsys.sh (Layer 2), run_ncu.sh (Layer 4)
├── tests/                # Unit tests — no GPU, no torch. `pytest benchmarks/tests/`
├── analysis/            # Phase 4 — reads results, refuses unmatched comparisons
│   ├── loader.py        #   provenance guard: HARD REFUSAL on mismatched runs
│   ├── compare_arms.py  #   matched table (md + csv); VRAM kept in its own section
│   ├── raw_stats.py     #   per-request: ITL trajectory, starvation, ignore_eos
│   ├── plots.py         #   six figures, PNG + SVG, each carrying its provenance
│   ├── make_results.py  #   generates RESULTS.md from the JSONs, one command
│   └── RESULTS.md       #   GENERATED — never hand-edited
├── ablations/            # Single-variable isolations; each condition a fresh subprocess
│   ├── prefix_cache.py         #   caching ON/OFF x both fixtures -> a bracket
│   ├── continuous_batching.py  #   max_num_seqs 1 vs default -> a curve
│   └── gil_attribution.py      #   threads vs processes vs sequential
├── results/              # Committed: derived metrics + one manifest per run
├── traces/               # Local scratch for raw traces — gitignored, not committed
└── v1_characterization/  # Superseded first-pass scripts, preserved as evidence
```

## Profiling layers

Four layers, **each its own run**. `nsys` and `torch.profiler` both subscribe to
CUPTI and conflict if run together, and any CUPTI instrumentation perturbs the
timings Layer 1 exists to measure.

| Layer | Tool | Produces | Gate |
|---|---|---|---|
| 1 | app timing + NVML + psutil | latency (TTFT/ITL/e2e), throughput, VRAM, CPU | none |
| 2 | `nsys`, full workload | GPU utilization, SM activity, thread states | probe tier (a); SM sampling needs (b) |
| 3 | `torch.profiler`, bounded window | `cudaLaunchKernel`/sync/memcpy breakdown, kernel timeline | probe tier (a) |
| 4 | `ncu`, short slice | achieved occupancy, Tensor Core utilization | probe tier (c) |

Layers 2 and 4 answer different questions and both get reported: nsys gives
duration-weighted SM activity across the whole run; ncu gives exact per-kernel
achieved occupancy on a slice. "The GPU was busy 95% of the time" and "those
kernels ran at 8% occupancy" are simultaneously true, and together they are the
finding.

Run [`probe_environment.py`](probe_environment.py) before anything else — it
determines which of these the host will actually permit, and prints a
measurable-vs-not table to reconcile the study plan against.

## Session setup

[`COLAB.md`](COLAB.md) is the entry point: eight cells, in order. Three of them
exist because of failures that already cost real time.

- **[`colab_bootstrap.py`](colab_bootstrap.py)** — the whole dependency
  resolution in one idempotent command. The chain is narrow (vLLM 0.26 needs a
  cu13 torch; no working torchvision exists for it; removing torchvision can
  take torch with it) and was re-derived by hand three times before being
  captured here. It batches every package change into **one pass** because a
  runtime restart can make Colab hand back a different physical GPU.
- **[`session_start.py`](session_start.py)** — appends GPU UUID, versions,
  `pip_freeze_sha256` and branch SHA to a Drive log, and flags loudly when the
  card changed. Manifests already record `gpu_uuid`, so a change is recoverable
  after the fact; this surfaces it before the hours are spent.
- **[`calibration_run.sh`](calibration_run.sh)** — a short fixed arm-B run at
  the start of every session, so arm-to-arm deltas are reported against a
  *measured* noise floor. `--analyze` separates within-card spread from
  between-card spread; if a metric is flagged `HARDWARE-DOMINATED`, comparing it
  across a GPU change is a hardware comparison wearing a software label.

## Arms

| Arm | Stack | Question it answers |
|---|---|---|
| **A** | Hosted API (the provider adapters already in `dashboard/backend/infrastructure/llm/providers/`) | What does ATL pay today, in latency and cost, with zero infrastructure? |
| **B** | Naive self-hosted — HuggingFace `generate`, one model, no batching | What does "just run it yourself" actually get you? This is the v1 configuration, done properly. |
| **C** | Optimized self-hosted — vLLM, continuous batching, paged attention | What does a real serving runtime recover over arm B? |

The study's headline is the **B → C** delta. Arm A is the cost/latency baseline
the platform runs on today and sets the bar both self-hosted arms must clear to
be worth operating.

Each arm is swept across concurrency levels. **One run = one arm at one
concurrency**, emitting one manifest. A sweep is N runs, never one run with
internal phases — that conflation is precisely what made the v1 numbers
unusable.

**`concurrency` means offered load in every arm** — the number of requests
outstanding simultaneously. Arm B reaches it with `ThreadPoolExecutor(N)`, arm C
with `asyncio.Semaphore(N)` around submission. The mechanism differs; the
offered load does not. An arm-specific definition of concurrency would make
every B→C delta meaningless.

One asymmetry to respect when reading results: **arm C's NVML VRAM reflects
config, not demand.** vLLM pre-allocates its KV pool to
`gpu_memory_utilization` at startup, so the process figure barely moves with
load. `notes.kv_cache.usage_perc_peak` is arm C's demand signal, and the
manifest carries `gpu_memory_utilization` so the NVML number can be read in
context.

## Ablations

| Ablation | Isolates | Output shape |
|---|---|---|
| `prefix_cache.py` | vLLM prefix caching, across both fixtures | a **bracket** — `shared_prefix` (99.4% common prefix) is the upper bound, `low_overlap` (0.3%) the floor, real traffic between |
| `continuous_batching.py` | scheduler batching, via `--max-num-seqs 1` vs default, caching held off | a **curve** — benefit vs offered load |
| `gil_attribution.py` | one shared GIL vs one per process vs sequential | GIL cost **conditional on** assumed per-token Python fraction, plus a real-model anchor |

Each condition runs as a fresh subprocess: new CUDA context, new KV pool, new
prefix cache. Running two conditions in one process would measure the second
against state the first warmed.

## Analysis

[`analysis/`](analysis/) turns run artifacts into the write-up. One command
regenerates everything:

```bash
python -m analysis.make_results --arm-b <B>_summary.json --arm-c <C>_summary.json \
  --trace-analysis <L3>_trace.json --out analysis/RESULTS.md \
  --figures-dir analysis/figures --render-figures
```

Three properties are deliberate:

- **The provenance guard is a refusal, not a warning.** `compare_arms` and
  `make_results` exit non-zero and write nothing when runs disagree on
  `gpu_uuid`, `fixture_sha256`, `config_sha256`, `max_new_tokens` or
  `pip_freeze_sha256`. `--allow-mismatch <field>` waives it, and the waiver is
  printed in the document.
- **VRAM never appears in the comparison table.** Arm B's NVML peak is demand;
  arm C's is `gpu_memory_utilization` pre-allocating the KV pool. They live in
  separate sections with the reason attached, because adjacent columns would
  read as "arm C uses more memory" — which is false.
- **`RESULTS.md` is generated.** Every number is read from a summary JSON and
  attributed to a `run_id`; caveats are derived from the artifacts (a null
  `stats_source`, a level shorter than the monitor's warm-up window) rather
  than written by hand.

## Traces are not committed

A full profiled run produces roughly **1.5 GB** of raw trace. That does not go
in git.

- **Committed:** derived metrics under `results/`, and one manifest per run.
- **Not committed:** raw `torch.profiler` / Nsight output. Stored externally and
  referenced from the manifest by `trace_url` + `trace_sha256`, so a trace can
  be located and its integrity verified without the repo carrying it.

[`.gitignore`](.gitignore) enforces this for `traces/`. Model weights (`*.pt`,
`*.safetensors`) are ignored for the same reason.

Note that the repo already carries ~32 MB of git history and has a track record
of committed build artifacts; a few gigabytes of traces would be unrecoverable
without a history rewrite.

## Provenance

Two documents make results citable, and both are prerequisites rather than
paperwork:

- **[ENVIRONMENT.md](ENVIRONMENT.md)** — the Phase 0 environment capture.
  Includes two checks that can invalidate a planned arm before any code is
  written: whether `ncu` returns `ERR_NVGPUCTRPERM`, and whether installing
  vLLM disturbs Colab's torch build.
- **[RUN_MANIFEST_SCHEMA.md](RUN_MANIFEST_SCHEMA.md)** — the JSON every run
  emits. Note `gpu_uuid`: Colab reallocates physical hardware between sessions,
  and a B-vs-C comparison across two different A100s measures the hardware as
  much as the optimization.

The benchmark drives a pinned `upstream_sha`, not a moving branch. Upstream
`main` auto-deploys on merge and took 581 commits in July 2026 alone.

## Prior work

[`v1_characterization/`](v1_characterization/) holds the three original stress
scripts, moved here with `git mv` so their history survives. **Read
[its README](v1_characterization/README.md) before quoting any v1 number** — the
three scripts differ in concurrency, context length, and generation length, one
of them performs no inference at all, and the trace that was analyzed came from
a single-request sequential run rather than the 105-task concurrent one.

## Setup

Only on a GPU host. Never on the deploy path, and never in root `requirements.txt`:

```bash
pip install -r benchmarks/requirements.txt
```

Versions are unpinned until Phase 0 freezes them from the live Colab image.
