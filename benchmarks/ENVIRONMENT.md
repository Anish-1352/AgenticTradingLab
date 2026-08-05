# Benchmark Environment

**NOT YET PROBED.** This file is a placeholder. It is *machine-generated* —
run the probe on the GPU host and commit the result:

```bash
python benchmarks/probe_environment.py
```

Do not hand-fill it. Phase 1 shipped a manual template here; it was replaced by
`probe_environment.py` so that the answers come from the device rather than
from recollection. The two facts most often misremembered — how much VRAM the
card has, and whether hardware counters are readable — are exactly the ones a
hand-filled template gets wrong.

## What the probe determines

**Device.** Name, SM count, compute capability, GPU UUID, driver and CUDA
runtime versions, and the **measured** total memory in bytes. Earlier write-ups
asserted 40GB; the expected part is the 80GB A100 (85,094,825,984 bytes). The
probe reports the actual value and states explicitly which reference — if
either — it matches. Assume neither until it has run.

**Three profiling permission tiers, tested separately.** They gate on different
things, and passing one does not imply the others:

| Tier | What it needs |
|---|---|
| (a) `nsys` CUDA trace | CUPTI tracing. Usually available unprivileged. |
| (b) `nsys --gpu-metrics-device` | Hardware counter *sampling*. |
| (c) `ncu` counters | Full per-kernel counter collection with replay. |

A host commonly passes (a) and fails (b) and (c). `ERR_NVGPUCTRPERM` means the
driver restricts counters to administrators; on Colab the NVIDIA kernel module
is loaded by the host, so this cannot be changed from inside the session.

If (c) is blocked, **achieved occupancy and Tensor Core utilization are not
obtainable in this environment.** There is no substitute — NVML
`utilization.gpu` is kernel residency, and a kernel name matching `s16816gemm`
shows a tensor-core GEMM ran, not how well it used the pipes. The study plan
must be reconciled against this before a sweep is run, not after.

**Also:** whether vLLM imports and whether installing it moved torch underneath
the earlier arms; free disk against the ~1.5 GB per traced run; and the
`pip_freeze_sha256` that every run manifest references.

## Machine-readable block

The generated file ends with a `PROBE_RESULTS_BEGIN/END` block that
`profiling/run_nsys.sh` and `profiling/run_ncu.sh` parse to decide whether to
pass `--gpu-metrics-device=0` and whether to run at all. Those scripts fail
closed when this file is absent or unparsed, so **the probe must run first**.

## Re-run it per session

Colab reallocates hardware between sessions. A GPU UUID, a driver version, or a
counter-permission result from a previous session is not evidence about the
current one. Re-run the probe on every session that produces committed results,
and commit the regenerated file alongside them.
