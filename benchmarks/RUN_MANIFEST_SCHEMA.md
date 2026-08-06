# Run Manifest Schema

Every benchmark run emits exactly one manifest JSON alongside its metrics, at
`results/<run_id>_manifest.json`. Emitted by `common/manifest.py`; the sibling
artifacts from the same run are `<run_id>_raw.json`, `<run_id>_summary.json`,
`<run_id>_pip_freeze.txt`, and one `<run_id>_c<N>_monitor.csv` per concurrency
level.

The rule this enforces: **a number that cannot be traced to a manifest does not
go in the presentation.** The v1 characterization phase produced figures whose
provenance had to be reconstructed by re-reading the scripts (see
[`v1_characterization/README.md`](v1_characterization/README.md)) — three
scripts with different context lengths, generation lengths, and concurrency
models, whose outputs were not distinguishable after the fact. The manifest
exists so that cannot recur.

## Fields

All fields are required. Emit `null` only where explicitly noted, and never
omit a key — a missing key and a null value are different failures.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Unique id for this run. `<utc_compact>-<arm>-c<concurrency>-<short_hash>`, e.g. `20260812T141203Z-B-c16-a3f9c1`. Also the `results/` subdirectory name. |
| `utc_timestamp` | string | ISO 8601 UTC, start of run: `2026-08-12T14:12:03Z`. |
| `arm` | string | `A` \| `B` \| `C`. See [README.md](README.md#arms). |
| `concurrency` | integer | Offered concurrent request count for this run. One value per run — a sweep is N runs, not one run with N sections. |
| `upstream_sha` | string | Full 40-char SHA of the pinned `upstream/main` the harness drove. |
| `branch_sha` | string | Full 40-char SHA of the benchmark branch commit that ran. Must be a clean tree; append `-dirty` if not, and treat any `-dirty` result as non-citable. |
| `gpu_name` | string | e.g. `NVIDIA A100-SXM4-80GB`. |
| `gpu_uuid` | string | e.g. `GPU-6d8f...`. See below. |
| `total_vram_mb` | integer | From NVML, not from a config file. |
| `driver_version` | string | e.g. `535.104.05`. |
| `cuda_version` | string | Runtime CUDA the process actually linked, i.e. `torch.version.cuda`. |
| `pip_freeze_sha256` | string | SHA-256 of the full `pip freeze` output. The raw freeze is committed under `results/<run_id>/pip_freeze.txt`. |
| `config_sha256` | string | SHA-256 of the resolved config file from `configs/` used by this run — after any overrides are applied, not the on-disk template. |
| `fixture_sha256` | string | SHA-256 of the request fixture from `fixtures/` (prompts, context, arrival pattern). |
| `torchvision_shim` | bool \| null | `true` when the minimal torchvision stub was supplied so vLLM could import (see below). `null` for arms that never touch vLLM. |
| `trace_url` | string \| null | External location of the raw trace. `null` when the run was executed with tracing disabled — which is the normal case for throughput runs. |
| `trace_sha256` | string \| null | SHA-256 of the raw trace artifact. Null iff `trace_url` is null. |

### Controlled variables and profiling provenance

Added in Phase 2. Every one of these is a quantity that, if it silently differed
between two runs, would make their comparison meaningless while leaving both
looking valid.

| Field | Type | Description |
|---|---|---|
| `max_new_tokens` | integer | Generation length. **A control variable, not an implementation detail.** v1 used 5 against a ~2600-token context, making the run 99.8% prefill, and its throughput figure was then read as decode performance. Two runs with different values are not comparable. |
| `profiling_layer` | 1\|2\|3\|4 | Which instrumentation layer produced this run. 1 = app timing + NVML, 2 = nsys, 3 = torch.profiler, 4 = ncu. Layers are **separate runs** — nsys and torch.profiler both subscribe to CUPTI and conflict, and any CUPTI instrumentation perturbs the Layer 1 timings. |
| `profiled_window_fraction` | float \| null | Fraction of the run's concurrent-section wall time covered by the profiler's active window. `null` when `profiling_layer == 1`. A bounded window is unavoidable (a full trace is ~1.5 GB); an *unlabelled* bounded window is how v1 came to explain a 105-task concurrent run with a trace of a sequential single-request one. Quote this fraction wherever a trace-derived number appears. |
| `fixture_name` | string | `shared_prefix` or `low_overlap`. The gap between them is the prefix-caching result; a number from one is not a number from the other. |
| `context_tokens` | integer | Exact prompt length. Enforced by the fixture builder, not approximated. |
| `model` | string | Model id actually loaded. |

### Collection integrity

| Field | Type | Description |
|---|---|---|
| `upstream_ref` | string \| null | Which ref `upstream_sha` was read from — `upstream/main`, `origin/main`, or `main`, in that preference order. A fork clone often has no `upstream` remote, and silently substituting `origin/main` without recording it would misstate what the run was pinned to. |
| `manifest_complete` | bool | False if any field could not be collected. |
| `collection_errors` | string[] | Why. Empty when complete. |
| `extra` | object | Run-specific context (concurrency sweep, measured `common_prefix_tokens`, requests per level). Not schema-fixed. |

Collection is best-effort and never fatal: a run that produced real measurements
must not be discarded because NVML hiccuped. But a manifest with holes must
never be mistaken for a complete one, which is what `manifest_complete` is for.

## Why `gpu_uuid` matters

Colab **reallocates physical hardware between sessions.** Two runs that both
report `NVIDIA A100-SXM4-80GB` may have executed on two different physical
cards, in different hosts, with different thermal state, different neighbours
on the same board, and — in a virtualized allocation — a different share of it.

That makes `gpu_name` insufficient for the central claim of a before/after
study. If arm B ran on one A100 and arm C on another, the measured delta
contains an unknown hardware term, and the study's headline number is not a
property of the optimization.

`gpu_uuid` is the only field that identifies the actual card. The rules:

- **Record it on every run**, not once per session.
- **A cross-arm comparison is valid only when the UUIDs match.** If they do not,
  the comparison is reported as cross-hardware and its uncertainty stated — or
  the runs are redone in one session.
- **Re-read it after any reconnect.** A session that dropped and resumed may
  have moved. Do not carry a UUID forward from the top of the notebook.

The same reasoning is why `pip_freeze_sha256` is per-run rather than per-study:
installing vLLM can pull its own torch build (see
[ENVIRONMENT.md](ENVIRONMENT.md#vllm--does-it-install-against-colabs-torch)),
which changes the environment mid-study. A differing freeze hash between arms is
a finding to disclose, not a detail to smooth over.

## Example

Illustrative. The hardware values below reflect what the Colab A100 session
actually probed — **A100-SXM4-80GB, 81920 MiB**, settling the open question the
study inherited (earlier write-ups asserted 40GB; the device reports the 80GB
part).

That does not make them constants. `gpu_name`, `total_vram_mb`,
`driver_version` and `cuda_version` are read from NVML **at run time, on every
run**. Colab reallocates hardware between sessions and lab machines will differ
again, so a manifest that carries a value copied from this example is
worthless — the field exists precisely to detect that the hardware moved.

```json
{
  "run_id": "20260812T141203Z-B-c16-a3f9c1",
  "utc_timestamp": "2026-08-12T14:12:03Z",
  "arm": "B",
  "concurrency": 16,
  "upstream_sha": "c8aa7012efcc3bacb8c16d96642f84daf3856fd5",
  "upstream_ref": "upstream/main",
  "branch_sha": "a3f9c1e0000000000000000000000000000000ex",
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_uuid": "GPU-6d8f2b1a-0000-0000-0000-000000000000",
  "total_vram_mb": 81920,
  "driver_version": "580.82.07",
  "cuda_version": "13.0",
  "pip_freeze_sha256": "9f2c...",
  "config_sha256": "1b77...",
  "fixture_sha256": "c40a...",
  "max_new_tokens": 256,
  "profiling_layer": 3,
  "profiled_window_fraction": 0.061,
  "fixture_name": "shared_prefix",
  "context_tokens": 2620,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "trace_url": null,
  "trace_sha256": null,
  "manifest_complete": true,
  "collection_errors": []
}
```

## Why `torchvision_shim` is recorded

vLLM 0.26 reaches torchvision from at least two directions, neither guarded by
a feature check: `kernel_warmup` imports MiniMax-M3 vision code (needing
`transforms.InterpolationMode`) even for a text-only Qwen2 model, and
`transformers_utils.config` chains into `transformers/image_utils.py` (needing
`io.ImageReadMode` and `io.decode_image`).

No torchvision is installable on torch 2.11.0+cu130 (the cu130 wheel's compiled
extension is broken; the cu128 wheel is rejected by torch's CUDA version check),
so arm C runs against an **import-only stub**: any `torchvision.*` submodule
resolves, the two enums are exact, and every other symbol raises `RuntimeError`
if called. The stub ships no distribution metadata, so
`is_torchvision_available()` still reports False and transformers' *optional*
vision paths stay switched off.

The flag is in the manifest because it describes the **import environment the
run actually executed in**, which is not otherwise recoverable from
`pip_freeze_sha256` — the stub is materialised at runtime and never appears in
`pip freeze`. A reader comparing two arm-C runs where one used the shim and the
other had a real torchvision is comparing two different environments.

`false`/`null` on a vLLM run means a real torchvision imported successfully, so
the shim no-oped. See `common/torchvision_shim.py`.

## Emission rules

- Write the manifest **at run start**, with the outcome fields filled in on
  completion. A crashed run must still leave a manifest — an unexplained gap in
  `results/` is indistinguishable from a deleted bad result.
- Hardware fields come from NVML at runtime. Never from a config file, and
  never from a previous run.
- The manifest is committed. The trace it points at is not.
