# Benchmark Environment

**Status: TEMPLATE — unfilled. Complete this on the Colab instance in Phase 0,
before any measurement run.**

Every field below is a precondition for the study, not documentation written
after the fact. Two of them (`ncu` permissions, vLLM installability) can
invalidate an entire planned arm, so they are checked first, while there is
still time to change the plan.

Fill this in from a live session and commit it. Re-verify — and append a new
dated block rather than overwriting — for any session that produces committed
results, because Colab does not guarantee you the same hardware twice.

---

## Session

| Field | Value | How to obtain |
|---|---|---|
| Date / time (UTC) | _TBD_ | `date -u` |
| Colab tier | _TBD_ | free / Pro / Pro+ — affects timeout and GPU class |
| Notebook URL | _TBD_ | |

## GPU

| Field | Value | How to obtain |
|---|---|---|
| GPU name | _TBD_ | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| **GPU UUID** | _TBD_ | `nvidia-smi --query-gpu=uuid --format=csv,noheader` |
| Total VRAM (MB) | _TBD_ | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` |
| Driver version | _TBD_ | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA runtime version | _TBD_ | `nvcc --version`, and `torch.version.cuda` |
| Compute capability | _TBD_ | `torch.cuda.get_device_capability()` |

The GPU UUID is the field that makes runs comparable. See
[RUN_MANIFEST_SCHEMA.md](RUN_MANIFEST_SCHEMA.md#why-gpu_uuid-matters).

## Blocking capability checks

These gate the study design. Answer them before writing any runner.

### `ncu` (Nsight Compute) — does it work?

| Field | Value |
|---|---|
| `ncu` present | _TBD_ |
| Runs without error | _TBD_ |
| Returns `ERR_NVGPUCTRPERM` | _TBD_ |

```bash
which ncu && ncu --version
ncu --target-processes all python -c "import torch; torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')"
```

`ERR_NVGPUCTRPERM` means GPU performance counters are locked to root. It is the
**expected** result on hosted Colab and is not fixable from inside the notebook
(the fix is a host-level `nvidia` module option or running as root). If it
appears, kernel-level counter analysis is off the table and the study must rely
on `torch.profiler` timeline data plus NVML sampling. **Record the outcome
either way** — "we chose not to use ncu" and "ncu was unavailable" are
different claims, and the presentation should make the true one.

### vLLM — does it install against Colab's torch?

| Field | Value |
|---|---|
| vLLM version installed | _TBD_ |
| Forced a torch reinstall | _TBD_ (yes = major risk) |
| Resulting torch version | _TBD_ |
| `LLM(...)` loads and serves a request | _TBD_ |

```bash
pip install vllm
python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
```

vLLM pins narrow torch ranges. If `pip install vllm` pulls its own torch build,
the environment every earlier measurement was taken in has changed underneath
you — which silently invalidates cross-arm comparison. If this happens, arm C
needs its own environment and its own `pip_freeze_sha256`, and that must be
stated in the writeup.

## Session limits

| Field | Value | Notes |
|---|---|---|
| Idle timeout | _TBD_ | |
| Max session length | _TBD_ | Must exceed the longest single run |
| Disk available (GB) | _TBD_ | `df -h /content` — traces are ~1.5 GB/run |
| Host RAM (GB) | _TBD_ | `free -g` |

If max session length is shorter than a full sweep, the sweep must be
checkpointed per arm — a run split across two sessions is a run split across
two GPUs unless the UUID is re-checked and matches.

## Provenance

| Field | Value | How to obtain |
|---|---|---|
| Pinned upstream SHA | _TBD_ | `git rev-parse upstream/main` |
| Benchmark branch SHA | _TBD_ | `git rev-parse HEAD` |
| `pip freeze` SHA-256 | _TBD_ | `pip freeze \| sha256sum` |
| `pip freeze` artifact | _TBD_ | commit the full output alongside this file |

The upstream SHA is pinned because the benchmark measures a serving stack
against a moving application. Upstream `main` auto-deploys on every merge and
took 581 commits in July 2026 alone; "measured against ATL" is not a
reproducible statement without a SHA.

---

## Filled sessions

Append one block per session that produced committed results. Do not overwrite
the template above.

<!--
### Session YYYY-MM-DD
- GPU name / UUID:
- Driver / CUDA:
- pip_freeze_sha256:
- ncu available:
- Runs produced:
-->
