# Running the benchmark on Colab

Eight cells, in order. Cells 1–7 are setup and take a few minutes; cell 8
onwards is the actual study.

**Do not improvise the setup.** The dependency situation on this image is
narrow and non-obvious — it was re-derived by hand across three sessions before
being captured in `colab_bootstrap.py`, and one of those attempts cost a GPU
reallocation mid-study. The [Why this is necessary](#why-this-is-necessary)
section at the bottom explains every constraint the script encodes, so nobody
has to work it out a fourth time.

---

## Cell 0 — Drive, and HF_HOME **before anything else**

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/atl_bench/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.makedirs('/content/drive/MyDrive/atl_bench/results', exist_ok=True)
print('HF_HOME =', os.environ['HF_HOME'])
```

**`HF_HOME` must point at Drive before any model load.** Qwen2.5-7B is ~15 GB.
On the local disk it re-downloads after every preemption, which costs ten
minutes each time and gives the cache a chance to come back holding a different
revision — a silent change to a controlled variable. Set it here, in the first
cell, before anything imports transformers or vLLM.

Everything the study produces should live on Drive for the same reason: the
local disk does not survive preemption.

---

## Cell 1 — Get the code

```python
%cd /content
!git clone https://github.com/Anish-1352/AgenticTradingLab.git 2>/dev/null || true
%cd /content/AgenticTradingLab
!git fetch origin
!git checkout feature/gpu-serving-benchmark
!git reset --hard origin/feature/gpu-serving-benchmark
!git log --oneline -3
```

`reset --hard` rather than `pull`: the branch has been force-pushed before (to
drop an upstream merge), and a plain `pull` in a stale clone will try to merge
the discarded history back in.

---

## Cell 2 — Bootstrap the environment

```python
!python benchmarks/colab_bootstrap.py
```

Idempotent. If the stack is already correct it prints `ALREADY CORRECT`, touches
nothing, and exits 0 — re-running it is free.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Correct. **No restart.** Skip cell 3. |
| 75 | Changes applied. **Restart** (cell 3), then re-run this cell. |
| 1 | Failed / half-resolved. The output says exactly what is wrong. **Stop.** |

---

## Cell 3 — Restart **only if cell 2 said 75**

```python
# Only if cell 2 exited 75. Then re-run cell 0 (HF_HOME) and cell 2.
import os
os.kill(os.getpid(), 9)
```

> **A restart can move you to a different physical GPU.** Colab reallocated
> `GPU-2ddfc69f…` → `GPU-afb936de…` at a restart mid-study, which put two arms
> on two different cards. `colab_bootstrap.py` batches every package change into
> one pass specifically so this happens **at most once** per session. Cell 4
> checks whether it happened.
>
> After restarting, re-run **cell 0** — `os.environ` does not survive.

---

## Cell 4 — Session provenance

```python
!python benchmarks/session_start.py
```

Appends one line to `/content/drive/MyDrive/atl_bench/session_log.jsonl` and
compares against the previous session. Exit 76 means **the GPU changed**.

That is not fatal, but it decides what you do next:

- Re-run calibration (cell 7) — the old noise floor was measured on the other
  card and does not transfer.
- Any comparison spanning the change is **cross-hardware**. Either re-run both
  arms on this card, or state it in the write-up.

Run manifests already carry `gpu_uuid`, so a change is *recoverable* after the
fact. This surfaces it in the first ten seconds instead of after the sweep.

---

## Cell 5 — Probe the profiling tiers

```python
!python benchmarks/probe_environment.py
```

Writes `benchmarks/ENVIRONMENT.md` and prints a **measurable vs not measurable**
table. Read it before designing a sweep around a metric this host will not
produce.

It also resolves the absolute paths of `nsys` and `ncu` by searching — **neither
is on PATH**, and `nsys` ships bundled inside the Nsight Compute tree rather
than as its own package. `run_nsys.sh` and `run_ncu.sh` read those paths from
the file, so this cell is a prerequisite for Layers 2 and 4.

**Re-run this after every environment change**, including the bootstrap. It
records `pip_freeze_sha256`, and a probe taken before the vLLM reinstall
describes an environment none of your runs actually used.

An `efa_metrics` warning is benign — an AWS network-adapter sampler absent from
bundled Nsight builds, unrelated to GPU profiling.

---

## Cell 6 — Build fixtures (only if absent)

```python
!ls benchmarks/fixtures/*.json 2>/dev/null || \
  python -m common.fixtures --model Qwen/Qwen2.5-7B-Instruct --n-requests 105
```

Run from `benchmarks/` (or with `PYTHONPATH=benchmarks`). Expect:

| Fixture | Common prefix | Role |
|---|---|---|
| `shared_prefix` | 2603 / 2620 = **99.4%** | deliberate upper bound for prefix caching |
| `low_overlap` | 8 / 2620 = **0.3%** | the floor |

Check the printed `sha256` against `fixtures/<name>.meta.json`. A mismatch means
the tokenizer moved, and the fixture is no longer the one earlier runs used.

---

## Cell 7 — Calibration

```python
!bash benchmarks/calibration_run.sh
```

A short fixed configuration — arm B, `shared_prefix`, C=15, 20 requests — run at
the **start of every session**. It measures the noise floor so an arm-to-arm
delta can be reported against something measured rather than assumed.

Spread across all recorded runs, grouped by card:

```python
!bash benchmarks/calibration_run.sh --analyze
```

If a metric comes back `HARDWARE-DOMINATED`, its variance tracks *which card you
landed on* rather than the run, and any cross-session comparison of it is a
hardware comparison wearing a software label.

Do not tune the calibration configuration. Its only job is to be identical every
time; changing it resets the history.

---

## Cell 8+ — The runs

Point `--out-dir` at Drive. Preemption takes the local disk with it, and every
runner is resumable per level only if its results survive.

```python
OUT = '/content/drive/MyDrive/atl_bench/results'
```

### Arm B — naive HuggingFace

```python
!python benchmarks/runners/bench_hf_baseline.py \
    --fixture shared_prefix --out-dir {OUT}
```

### Arm C — vLLM

Identical flags for everything shared with arm B. Prefix caching must be stated
explicitly — it is never defaulted, because it is the variable one ablation
isolates.

```python
!python benchmarks/runners/bench_vllm_optimized.py \
    --fixture shared_prefix --enable-prefix-caching --out-dir {OUT}
```

```python
# Same offered load, caching off — the other half of the bracket.
!python benchmarks/runners/bench_vllm_optimized.py \
    --fixture low_overlap --no-prefix-caching --out-dir {OUT}
```

Check the `KV cache … (source: …)` line on the first arm-C run. If
`stats_source` is `None`, the vLLM stats path did not resolve on this build and
the KV/hit-rate numbers are absent rather than zero — the prefix-cache ablation
depends on them.

### Ablations

```python
# Prefix caching: 4 fresh subprocesses -> a bracket, not a single number.
!python benchmarks/ablations/prefix_cache.py --out-dir {OUT}

# Continuous batching: max_num_seqs 1 vs default, caching held off -> a curve.
!python benchmarks/ablations/continuous_batching.py --out-dir {OUT}

# GIL attribution: threads vs processes vs sequential.
!python benchmarks/ablations/gil_attribution.py --synthetic --out-dir {OUT}
!python benchmarks/ablations/gil_attribution.py --real --out-dir {OUT}
```

### Profiling layers

Each is a **separate run** — `nsys` and `torch.profiler` both subscribe to CUPTI
and conflict, and any CUPTI instrumentation perturbs the Layer 1 timings.

```python
# Layer 2 — nsys, full workload
!RESULTS_DIR={OUT} TRACES_DIR={OUT}/traces bash benchmarks/profiling/run_nsys.sh \
    benchmarks/runners/bench_hf_baseline.py --concurrency 15 --out-dir {OUT}

# Layer 3 — torch.profiler, bounded window DURING the concurrent section
!python benchmarks/runners/bench_hf_baseline.py \
    --concurrency 15 --profile --out-dir {OUT}

# Layer 4 — ncu, short slice (forces concurrency 1, 3 requests)
!TRACES_DIR={OUT}/traces bash benchmarks/profiling/run_ncu.sh \
    benchmarks/runners/bench_hf_baseline.py --out-dir {OUT}
```

---

## Why this is necessary

Everything below was established empirically. `colab_bootstrap.py` encodes it.
It is written down so it does not get re-derived a fourth time.

### The dependency chain

1. **Colab ships torch 2.11.0+cu128**, and a `pynvml` distribution that
   conflicts with `nvidia-ml-py`. Both provide the `pynvml` import name; with
   both installed, NVML calls are unreliable — and NVML underpins every VRAM
   number in the study.

2. **vLLM 0.26.0 requires a cu13 torch.** On cu128 it fails at import:

   ```
   ImportError: libcudart.so.13: cannot open shared object file
   ```

3. **`pip install vllm` (after removing torch) resolves torch to 2.11.0+cu130.**
   That is the supported way to get a cu13 build here; installing a specific
   torch first and vLLM second does not converge.

4. **torchvision then mismatches, and both wheels are dead ends:**

   | Wheel | Failure |
   |---|---|
   | cu130 | broken compiled extension — `operator torchvision::nms does not exist` at import |
   | cu128 | rejected by torch's `_check_cuda_version()` — *"PyTorch has CUDA Version=13.0 and torchvision has CUDA Version=12.8"* |

   **There is no installable torchvision for torch 2.11.0+cu130 on this image.**
   This is why the bootstrap leaves it uninstalled rather than trying to fix it.

5. **transformers 5.13.1 works fine without torchvision.** Arm B is unaffected.

6. **vLLM does not.** `kernel_warmup` unconditionally imports MiniMax-M3 vision
   code that needs `torchvision.transforms.InterpolationMode`, even for a
   text-only Qwen2 model. That is what the torchvision shim exists for — the
   bootstrap deliberately leaves torchvision absent and expects the shim to
   satisfy vLLM's import.

7. **Removing `torchvision`/`torchaudio`/`torchcodec` has been observed to take
   torch and vllm with it.** The bootstrap therefore removes them **last**, then
   re-verifies the entire target state and fails loudly rather than leaving a
   stack that imports but is wrong.

### Target state

```
torch          2.11.0+cu130     (cu13 build is what matters)
vllm           0.26.0
transformers   5.13.1
nvidia-ml-py   present
pynvml         ABSENT           (conflicts with nvidia-ml-py)
torchvision    ABSENT           (no working wheel; shim covers vLLM)
torchaudio     ABSENT
torchcodec     ABSENT
```

### Restarts are the thing to minimise

A runtime restart can make Colab hand back a **different physical GPU**. That
already happened once mid-study (`GPU-2ddfc69f…` → `GPU-afb936de…`), splitting
arms across two cards and putting an unknown hardware term inside the B→C delta
the study exists to measure.

So: `colab_bootstrap.py` batches **every** package mutation into a single pass
and asks for **at most one** restart, instead of the three-restart hand sequence
it replaces. If nothing needs changing, it asks for none. `session_start.py`
then checks whether the card moved anyway, before any time is spent.

### Re-freeze after every environment change

`probe_environment.py` records `pip_freeze_sha256`, and every run manifest
carries it. Runs taken either side of an environment change are not strictly
comparable, so re-run cells 4 and 5 after anything that touches packages — the
record is only useful if it describes the environment the runs actually used.
