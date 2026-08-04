# ATL GPU Serving Benchmark

A before/after study of LLM inference serving for Agentic Trading Lab: what the
platform's agent workload costs on a hosted API, on a naive self-hosted stack,
and on an optimized self-hosted stack.

**Phase 1 — scaffolding only.** This directory currently contains structure,
documentation, and the preserved v1 evidence. No benchmark implementation yet.

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
├── configs/              # Per-arm run configs (hashed into every manifest)
├── fixtures/             # Frozen request workloads: prompts, context, arrival pattern
├── runners/              # Arm drivers — one per serving stack
├── ablations/            # Single-variable isolations off the main arms
├── results/              # Committed: derived metrics + one manifest per run
├── traces/               # Local scratch for raw traces — gitignored, not committed
└── v1_characterization/  # Superseded first-pass scripts, preserved as evidence
```

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
