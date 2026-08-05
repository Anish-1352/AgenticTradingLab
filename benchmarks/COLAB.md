# Running the benchmark on Colab

Exact cell sequence for an A100 runtime. Run the cells in order — the ordering
is not cosmetic, and two of the constraints below will silently corrupt results
if ignored.

## The two rules

**1. Layers 2 and 3 are SEPARATE RUNS.** `nsys` and `torch.profiler` both
subscribe to CUPTI. Running them at once either errors or, worse, yields a
silently truncated capture. Never pass `--profile` to a runner invoked through
`run_nsys.sh`.

**2. Layer 1 is also its own run.** CUPTI instrumentation perturbs the very
timings Layer 1 exists to measure, so latency and throughput numbers come from
an *unprofiled* run. `--profile` is off by default for exactly this reason.

So one arm at one concurrency produces up to four runs, one per layer, and each
emits its own manifest recording which layer it was.

## Preemption

Colab preempts. Two consequences:

- **`--out-dir` must point at mounted Drive.** The local disk does not survive.
- The runner writes results after **every** concurrency level and resumes by
  `--run-id`. Re-invoke with the same `--run-id` and completed levels are
  skipped rather than re-run.

---

## Cell 1 — GPU check

```python
!nvidia-smi
```

Stop here if this is not an A100. Note the reported memory, but do not trust it
as the study's figure — cell 5 measures it from the device.

## Cell 2 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import os
OUT = '/content/drive/MyDrive/atl-gpu-bench'
os.makedirs(OUT, exist_ok=True)
print(OUT)
```

## Cell 3 — Clone and check out the branch

```python
!git clone https://github.com/Anish-1352/AgenticTradingLab.git /content/atl
%cd /content/atl
!git checkout feature/gpu-serving-benchmark
!git rev-parse HEAD
```

Record that SHA. It is `branch_sha` in every manifest, and a dirty tree marks
results non-citable.

## Cell 4 — Install

```python
!pip install -q -r benchmarks/requirements.txt
```

### The CUDA reconciliation, in order

A fresh Colab instance reproduces this exactly. Do not improvise around it —
each step exists because the previous one breaks something specific.

**1. vLLM 0.26.0 needs a cu13 torch; Colab ships cu128.** Installing vLLM on
the stock image and importing it gives:

```
ImportError: libcudart.so.13: cannot open shared object file
```

**2. Remove the cu128 stack and let vLLM pull its own torch.** A plain
`pip install -U vllm` on top of the existing torch does not fix it — the old
CUDA runtime packages stay resolved:

```python
!pip uninstall -y vllm torch nvidia-cuda-runtime
!pip install -q vllm
!python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
```

This resolves to **torch 2.11.0+cu130, CUDA 13.0, vLLM 0.26.0**.

**3. Now transformers breaks.** `torchvision`/`torchaudio` are still the cu128
builds, and transformers imports torchvision on the way to the model class:

```
RuntimeError: operator torchvision::nms does not exist
... failed to import Qwen2ForCausalLM
```

**4. Do NOT install a cu130 torchvision — that wheel is broken.** It carries the
same missing `torchvision::nms` extension, so "upgrading" torchvision
reproduces the identical error. **transformers works fine without any of them**
for a text-only causal LM, so remove them:

```python
!pip uninstall -y torchvision torchaudio torchcodec
!python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
```

Do not reintroduce `torchvision`, `torchaudio`, or `torchcodec` later — a
transitive pull from some other package will resurrect the same failure.

**5. Re-run the probe and re-freeze after EVERY environment change.** Steps 2
and 4 both change `pip freeze`, which changes `pip_freeze_sha256`, which is what
makes two runs comparable. A run whose environment moved after the probe carries
a manifest that describes a machine that no longer existed.

```python
!python benchmarks/probe_environment.py
```

The frozen target both arms must run on:

| Package | Version |
|---|---|
| torch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| vLLM | 0.26.0 |
| transformers | 5.13.1 |
| torchvision / torchaudio / torchcodec | **removed** |

**Do not install pynvml.** `benchmarks/requirements.txt` pins `nvidia-ml-py`
instead. The import name is the same (`import pynvml`), but the two
distributions conflict and torch warns that the `pynvml` package is deprecated.
NVML sampling underpins every VRAM number in the study. If the image already
ships the old one:

```python
!pip uninstall -y pynvml
!pip install -q nvidia-ml-py
```

### Nsight: do not assume apt, and do not assume PATH

**`ncu` is usually already present** — typically `/usr/local/cuda/bin/ncu`.

**`nsys` is the awkward one.** On the observed Colab image it is *not* on PATH
and *not* installable from apt (`nsight-systems-cli` does not resolve). It
ships **bundled inside the Nsight Compute tree**:

```
/opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys
```

Do not hardcode that version — it changes between images, and lab hardware will
differ again. **Cell 5 finds both tools by searching and records their absolute
paths**, and `run_nsys.sh` / `run_ncu.sh` invoke the recorded paths. So there is
usually nothing to install here. Have a look first:

```python
!which nsys ncu || true
!ls -d /opt/nvidia/nsight-compute/*/ 2>/dev/null || true
```

Only if the probe in cell 5 reports `nsys` NOT FOUND is an install needed. Try
apt, and if the package does not resolve, install Nsight Compute (which carries
nsys inside it) and re-run the probe:

```python
!apt-get update -qq && apt-get install -y -qq nsight-systems-cli || \
  echo "nsight-systems-cli unavailable — install nsight-compute instead; it bundles nsys"
```

If the probe still cannot find it, add the layout to `NSYS_SEARCH_PATTERNS` in
`probe_environment.py` rather than hardcoding a path in the shell scripts.

If `pip install vllm` moved torch, note the new version — cell 5 records it, and
a torch change mid-study means arm C is not directly comparable to arms A and B.

## Cell 5 — Probe the environment (RUN THIS FIRST)

```python
!python benchmarks/probe_environment.py --json-out {OUT}/probe.json
```

This resolves the tool paths, tests the three permission tiers, writes
`benchmarks/ENVIRONMENT.md`, and prints a **measurable vs not measurable**
table. Read it before running anything else.

Specifically check:

- **Resolved tool paths.** The table names the absolute path and version found
  for `nsys` and `ncu`, and lists every candidate considered. If a tool shows
  **NOT FOUND**, that is an install/path problem — distinct from found-but-
  blocked, which is a permission problem with a different fix.
- **Total memory in bytes.** The probe states whether it matches the 80GB
  (85,094,825,984) or 40GB (42,949,672,960) reference, or neither. The observed
  Colab A100 is the **80GB** part at 81920 MiB. Use the measured value anyway —
  the next session may not be the same card.
- **Tier (c) `ncu_counters`.** If `BLOCKED` with `ERR_NVGPUCTRPERM`, achieved
  occupancy and Tensor Core utilization are **not obtainable on this host** and
  Layer 4 is off the table. Adjust the study now, not in the write-up. On the
  observed image this tier was **OBTAINABLE** — real occupancy came back with no
  permission error.
- **Tier (b) `nsys_gpu_metrics`.** If `BLOCKED`, Layer 2 still runs but without
  SM activity sampling. `run_nsys.sh` handles this automatically, and also reads
  back *which* flag spelling this nsys accepts — `--gpu-metrics-devices`
  (plural) is current, `--gpu-metrics-device` is deprecated in 2025.x.
- **GPU metric sets.** The probe records the output of
  `--gpu-metrics-set=help`. The default is *General Metrics*; if a set with
  better Tensor-pipe coverage is listed, choose it **before** the real runs — the
  set is baked into a capture and cannot be changed afterwards. Pass it through
  as `GPU_METRICS_SET=<name>` when invoking `run_nsys.sh`.

An `efa_metrics` warning (`Executable path does not exist:
.../plugins/efa_metrics/nic_sampler`) is **benign** — an AWS network-adapter
sampler absent from bundled Nsight builds, irrelevant to GPU profiling. It is
recorded in `ENVIRONMENT.md` so it does not get re-investigated.

Commit the regenerated `ENVIRONMENT.md` with the session's results.

## Cell 6 — Build fixtures with the real tokenizer

```python
!python -m benchmarks.common.fixtures --model Qwen/Qwen2.5-7B-Instruct \
    --n-requests 105 --context-tokens 2620
```

Prints, per fixture, the sha256 and the **measured common prefix**. Sanity
check: `shared_prefix` should show a common prefix near the full context;
`low_overlap` should be near zero. If they are similar, the prefix-caching
comparison is measuring nothing.

## Cell 7 — Smoke test

```python
!python benchmarks/runners/bench_hf_baseline.py --dry-run
!python benchmarks/runners/bench_hf_baseline.py \
    --concurrency 1 --n-requests 2 --out-dir {OUT} --run-id smoke-B
```

Then the same for arm C. Prefix caching has **no default** — one of the two
flags is required, because it is the variable the prefix-cache ablation
isolates and an implicit value would silently decide that result:

```python
!python benchmarks/runners/bench_vllm_optimized.py --dry-run --no-prefix-caching
!python benchmarks/runners/bench_vllm_optimized.py \
    --no-prefix-caching --concurrency 1 --n-requests 2 \
    --out-dir {OUT} --run-id smoke-C
```

`--dry-run` builds the workload and prints token counts without loading a model
or constructing an engine. The second command in each pair runs two requests end
to end. Confirm the summary shows `input_tok_per_s` and `output_tok_per_s`
separately and a plausible TTFT before committing to a full sweep.

For arm C also check the console line `KV cache ... (source: ...)`. If
`stats_source` is `None`, the vLLM stats path did not resolve on this build —
see `stats_probe_attempts` in the manifest. **Do not read a missing KV number
as zero usage**; the prefix-cache ablation depends on it.

---

# Layer 1 — timing + NVML (every run)

The unprofiled sweep. This is where latency and throughput numbers come from.

```python
!python benchmarks/runners/bench_hf_baseline.py \
    --fixture shared_prefix --out-dir {OUT} --run-id L1-B-shared
```

Then the other fixture — **both are required**; the gap between them is the
prefix-caching result:

```python
!python benchmarks/runners/bench_hf_baseline.py \
    --fixture low_overlap --out-dir {OUT} --run-id L1-B-lowoverlap
```

Preempted mid-sweep? Re-run the identical command. Completed levels are skipped.

## Arm C — vLLM

Identical flags to arm B for everything shared, so the two sweeps differ only in
the serving stack:

```python
!python benchmarks/runners/bench_vllm_optimized.py \
    --fixture shared_prefix --enable-prefix-caching \
    --out-dir {OUT} --run-id L1-C-shared-cacheON

!python benchmarks/runners/bench_vllm_optimized.py \
    --fixture low_overlap --enable-prefix-caching \
    --out-dir {OUT} --run-id L1-C-lowoverlap-cacheON
```

**`concurrency` means the same thing in both arms**: the number of requests
outstanding simultaneously. Arm B gets there with `ThreadPoolExecutor(N)`; arm C
with `asyncio.Semaphore(N)` around submission, which the vLLM scheduler then
batches. The mechanism differs, the offered load does not — and that is what
makes a B-vs-C delta interpretable.

**Arm C's NVML VRAM is not comparable to arm B's at face value.** vLLM
pre-allocates its KV pool to `--gpu-memory-utilization` (default 0.90) at
startup, so NVML reports the *config*, not demand. Use
`notes.kv_cache.usage_perc_peak` for demand, and read NVML only alongside
`gpu_memory_utilization` from the manifest.

# Ablations

Each condition runs as a **fresh subprocess** — new CUDA context, new KV pool,
new prefix cache — so no condition can contaminate another.

```python
# Prefix caching: 4 runs = {shared_prefix, low_overlap} x {ON, OFF}
!python benchmarks/ablations/prefix_cache.py \
    --concurrency 15 --n-requests 105 --out-dir {OUT}

# Continuous batching: --max-num-seqs 1 vs engine default, across levels.
# Prefix caching is forced OFF in both, so caching cannot confound the result.
!python benchmarks/ablations/continuous_batching.py \
    --concurrency 1 4 8 15 32 64 --n-requests 105 --out-dir {OUT}

# GIL attribution: threadpool vs processpool vs sequential.
!python benchmarks/ablations/gil_attribution.py \
    --concurrency 15 --n-requests 30 --out-dir {OUT} --with-pyspy

# ...and the real-model anchor. Concurrency 2-3 ONLY: ProcessPoolExecutor
# loads one ~15GB model replica per worker.
!python benchmarks/ablations/gil_attribution.py \
    --real --real-concurrency 2 --real-requests 4 --out-dir {OUT}
```

Reading them:

- **prefix_cache** prints a **bracket**, not a number. `shared_prefix` (99.4%
  common prefix) is the upper bound and `low_overlap` (0.3%) the floor; real
  agent traffic sits between. Quote the interval.
- **continuous_batching** prints a **curve**. The benefit should grow with
  offered load — that growth, and where it saturates, is the result. If it does
  not grow, check the per-level `errored` counts before reporting: a partially
  OOMed level inflates apparent throughput.
- **gil_attribution** prints GIL cost as a function of assumed Python fraction,
  because that fraction is not known a priori. Run `--real` too: it anchors
  which fraction is plausible. A **negative** GIL cost at low concurrency is a
  real result — process-pool overhead exceeding lock contention — not a bug.

# Layer 2 — nsys, full workload

```python
!RUN_ID=L2-B-shared RESULTS_DIR={OUT} TRACES_DIR={OUT}/traces \
  bash benchmarks/profiling/run_nsys.sh \
    benchmarks/runners/bench_hf_baseline.py \
    --fixture shared_prefix --concurrency 15 --out-dir {OUT}
```

Note: **no `--profile`**. The script reads the resolved `nsys` absolute path
from `ENVIRONMENT.md` (it is not on PATH), appends the GPU-metrics flag only if
the probe marked tier (b) OBTAINABLE — using whichever spelling the probe found
this binary accepts — and emits `gpukernsum` / `cudaapisum` CSVs into
`RESULTS_DIR` so the numbers are readable without the Nsight GUI.

To pin a non-default metric set (see cell 5's list):

```python
!GPU_METRICS_SET=ga100 RUN_ID=L2-B-shared RESULTS_DIR={OUT} TRACES_DIR={OUT}/traces \
  bash benchmarks/profiling/run_nsys.sh \
    benchmarks/runners/bench_hf_baseline.py \
    --fixture shared_prefix --concurrency 15 --out-dir {OUT}
```

# Layer 3 — torch.profiler, bounded window

A separate invocation. `--profile` on, no nsys:

```python
!python benchmarks/runners/bench_hf_baseline.py \
    --fixture shared_prefix --concurrency 15 --profile \
    --out-dir {OUT} --traces-dir {OUT}/traces --run-id L3-B-shared
```

The profiler wraps the **concurrent section**, stepped by request completions
inside it. The manifest records `profiled_window_fraction` — the share of the
section's wall time the active window covered. Quote that fraction whenever a
trace-derived number appears; a bounded window is fine, an unlabelled one is
how v1 went wrong.

Then reduce the trace:

```python
!python -m benchmarks.common.trace_analysis \
    {OUT}/traces/L3-B-shared_c15_torch.json.gz \
    --out {OUT}/L3-B-shared_trace_summary.json
```

Pass `--decode-start-us` for an exact prefill/decode split; without it the split
falls back to a name heuristic and labels itself as such.

# Layer 4 — ncu, short slice (only if tier (c) OBTAINABLE)

```python
!RUN_ID=L4-B TRACES_DIR={OUT}/traces \
  bash benchmarks/profiling/run_ncu.sh \
    benchmarks/runners/bench_hf_baseline.py --out-dir {OUT}
```

The script re-checks `ENVIRONMENT.md` and exits with an explanation if counters
are unavailable. It forces `--concurrency 1 --n-requests 3`: ncu replays every
kernel many times, so pointing it at the 105-request sweep would take hours.

nsys and ncu answer different questions and both get reported: nsys gives
duration-weighted SM activity across the whole run, ncu gives exact per-kernel
achieved occupancy on a slice. "The GPU was busy 95% of the time" and "those
kernels ran at 8% occupancy" are both true at once, and together they are the
finding.

---

## Collecting results

```python
!ls -la {OUT}
!du -sh {OUT}/traces
```

Per run you should have `<run_id>_raw.json`, `<run_id>_summary.json`,
`<run_id>_manifest.json`, `<run_id>_pip_freeze.txt`, and one
`<run_id>_c<N>_monitor.csv` per level.

**Commit the summaries, manifests and `ENVIRONMENT.md`. Do not commit traces** —
they are ~1.5 GB per full run and `benchmarks/.gitignore` excludes them. Upload
them somewhere durable and record `trace_url` + `trace_sha256` in the manifest.

## Before quoting any number

- `gpu_uuid` identical across the runs being compared? Colab reallocates
  hardware between sessions; different UUIDs means the delta contains a
  hardware term.
- `config_sha256` identical across arms? If not, they are not comparable.
- `branch_sha` free of the `-dirty` suffix?
- `max_new_tokens` the same across arms? It is a control variable.
- Trace-derived claims accompanied by their `profiled_window_fraction`?
