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

**Tool paths, by searching.** `nsys` and `ncu` are **not assumed to be on
PATH**. On the observed Colab image `ncu` sits at `/usr/local/cuda/bin/ncu`
while `nsys` is not on PATH at all and is not installable from apt — it ships
bundled inside the Nsight Compute tree
(`/opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys`). The probe
tries `shutil.which` first, then globs known layouts and takes the **highest
version**, and records the resolved absolute path plus the version the binary
reports. Versions are never hardcoded: the image changes between sessions and
lab hardware will differ again. `run_nsys.sh` and `run_ncu.sh` invoke the
recorded paths.

**Device.** Name, SM count, compute capability, GPU UUID, driver and CUDA
runtime versions, and the **measured** total memory in bytes. Earlier write-ups
asserted 40GB; the observed Colab A100 is the **80GB** part
(85,094,825,984 bytes / 81920 MiB, CC 8.0). The probe still reports the actual
value and states which reference — if either — it matches, because the next
session may not be the same card.

**Three profiling permission tiers, tested separately.** They gate on different
things, and passing one does not imply the others:

| Tier | What it needs |
|---|---|
| (a) `nsys` CUDA trace | CUPTI tracing. Usually available unprivileged. |
| (b) `nsys` GPU metrics sampling | Hardware counter *sampling*. |
| (c) `ncu` counters | Full per-kernel counter collection with replay. |

A host commonly passes (a) and fails (b) and (c) — though on the observed image
**all three were OBTAINABLE**, with `ncu` returning real occupancy and no
`ERR_NVGPUCTRPERM`. Where that error does appear, the driver restricts counters
to administrators, and if the NVIDIA kernel module is loaded by the host rather
than the session it cannot be changed from inside.

*Tool not found* and *tool found but blocked* are recorded as distinct states.
They have different fixes — install versus permission — and collapsing them
wastes a session.

**The GPU-metrics flag spelling.** `--gpu-metrics-device` (singular) is
deprecated in nsys 2025.x; `--gpu-metrics-devices` (plural) is unrecognised by
older builds. The probe tries the plural form and falls back to the singular
only when the tool rejects the option itself, then records which one worked so
`run_nsys.sh` never guesses.

**Available `--gpu-metrics-set` values.** The default is *General Metrics*. The
set is baked into a capture and cannot be changed afterwards, so if one with
better Tensor-pipe coverage exists it must be chosen before the real runs.

## Benign warnings

**`efa_metrics` — do not re-investigate.** A message like
`Executable path does not exist: .../plugins/efa_metrics/nic_sampler` refers to
a sampler for AWS Elastic Fabric Adapter network interfaces. It is absent from
bundled or partial Nsight builds, has nothing to do with GPU profiling, and does
not affect CUDA tracing, GPU metrics sampling, or the resulting report. The
generated file repeats this note so it stays attached to the evidence.

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
