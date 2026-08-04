# Run Manifest Schema

Every benchmark run emits exactly one manifest JSON alongside its metrics, at
`results/<run_id>/manifest.json`.

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
| `gpu_name` | string | e.g. `NVIDIA A100-SXM4-40GB`. |
| `gpu_uuid` | string | e.g. `GPU-6d8f...`. See below. |
| `total_vram_mb` | integer | From NVML, not from a config file. |
| `driver_version` | string | e.g. `535.104.05`. |
| `cuda_version` | string | Runtime CUDA the process actually linked, i.e. `torch.version.cuda`. |
| `pip_freeze_sha256` | string | SHA-256 of the full `pip freeze` output. The raw freeze is committed under `results/<run_id>/pip_freeze.txt`. |
| `config_sha256` | string | SHA-256 of the resolved config file from `configs/` used by this run — after any overrides are applied, not the on-disk template. |
| `fixture_sha256` | string | SHA-256 of the request fixture from `fixtures/` (prompts, context, arrival pattern). |
| `trace_url` | string \| null | External location of the raw trace. `null` when the run was executed with tracing disabled — which is the normal case for throughput runs. |
| `trace_sha256` | string \| null | SHA-256 of the raw trace artifact. Null iff `trace_url` is null. |

## Why `gpu_uuid` matters

Colab **reallocates physical hardware between sessions.** Two runs that both
report `NVIDIA A100-SXM4-40GB` may have executed on two different physical
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

```json
{
  "run_id": "20260812T141203Z-B-c16-a3f9c1",
  "utc_timestamp": "2026-08-12T14:12:03Z",
  "arm": "B",
  "concurrency": 16,
  "upstream_sha": "c8aa7010000000000000000000000000000000ex",
  "branch_sha": "a3f9c1e0000000000000000000000000000000ex",
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "gpu_uuid": "GPU-6d8f2b1a-0000-0000-0000-000000000000",
  "total_vram_mb": 40960,
  "driver_version": "535.104.05",
  "cuda_version": "12.1",
  "pip_freeze_sha256": "9f2c...",
  "config_sha256": "1b77...",
  "fixture_sha256": "c40a...",
  "trace_url": null,
  "trace_sha256": null
}
```

## Emission rules

- Write the manifest **at run start**, with the outcome fields filled in on
  completion. A crashed run must still leave a manifest — an unexplained gap in
  `results/` is indistinguishable from a deleted bad result.
- Hardware fields come from NVML at runtime. Never from a config file, and
  never from a previous run.
- The manifest is committed. The trace it points at is not.
