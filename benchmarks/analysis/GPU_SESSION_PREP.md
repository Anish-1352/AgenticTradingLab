# GPU session prep — atl_realistic on both arms

**The four runs did not happen. This machine has no NVIDIA GPU.**

```
$ nvidia-smi          -> not found
$ uname -sm           -> Darwin arm64
$ python -c "import torch; torch.cuda.is_available()"   -> False
$ python -c "import vllm"                                -> ModuleNotFoundError
$ python session_start.py
    GPU query error: nvidia-smi not found
    OSError: [Errno 30] Read-only file system: '/content'
```

`session_start.py` fails at its first gate and its default log path is a Colab
path. Arms B and C need a CUDA device; arm C additionally needs vLLM. Nothing
in the matrix is executable here, and no partial substitute is worth having —
a CPU-run or a different-hardware run would produce exactly the second
non-citable dataset the brief says to abort for.

What follows is everything that could be done without the card, all of which
removes a way the session can be wasted.

---

## Preconditions verified

### 1. shared_prefix is intact

```
on-disk sha256: a69c2be743189489d0ab5b4a2f7fe09c163fdf3cd034429a904840e810135968
matches armB_shared / armC_shared manifests: True
n_requests: 105   context_tokens: 2620   common prefix: 99.4%
```

This mattered: an arm C smoke test at `--n-requests 4` previously overwrote
this file, and a 32-request rebuild hashes to `ad924245…` instead. The fixture
currently on disk is the one the existing runs used, so the clean re-runs will
be directly comparable to the dirty ones.

### 2. atl_realistic is built, at n=105

```
sha256: 626e7494a669c01feba48d4f06a3922940d04d1b3b7ba23769204fdd5cfaaaa4
n_requests: 105   context_tokens: 2620
common prefix: 268 tokens (10.2% of context)
```

Built at 105, not 32, for the reason the brief gives. Only the `.meta.json`
is committed — `fixtures/*.json` is gitignored, and the full fixture
regenerates exactly from (tokenizer, seed, context_tokens), so a rebuild whose
hash differs is a real signal rather than a lost artifact.

**10.2% against shared_prefix's 99.4% is the contrast the 2x2 exists to
measure.**

### 3. The provenance guard fires

Tested with real summaries and one field changed:

| Case | Result |
|---|---|
| Same everything, different `gpu_uuid` | **exit 3, hard refusal**, names the field |
| Same card, different fixture | **exit 3** |
| Different fixture, `--allow-mismatch fixture_name fixture_sha256` | exit 0, and the output carries `PROVENANCE GUARD WAIVED` |

Two things follow for the session:

- The cross-hardware check the brief asks for is real. If Colab reallocates
  mid-session, `compare_arms` refuses rather than silently mixing cards.
- **The cross-fixture comparison — which is the whole point of the 2x2 —
  requires an explicit waiver.** shared vs realistic differ on `fixture_name`
  and `fixture_sha256` by construction. Pass
  `--allow-mismatch fixture_name fixture_sha256`; the waiver is recorded in the
  output, so the comparison stays legible as a deliberate cross-condition one.

### 4. Clean tree

`HEAD` is committed with no working-tree changes, so a manifest generated now
records a bare SHA rather than `…-dirty`. Verify this again inside the session
before the first run — the brief's abort condition.

---

## The runs, ready to paste

Order is arm B first: it is the long pole (~25 min vs ~5 min), so a preemption
costs least if the expensive runs are already banked.

```bash
cd benchmarks
python session_start.py                      # log gpu_uuid BEFORE anything

python runners/bench_hf_baseline.py    --fixture shared_prefix  --concurrency 1 8 32 --n-requests 32 --run-id armB_shared_clean
python runners/bench_hf_baseline.py    --fixture atl_realistic  --concurrency 1 8 32 --n-requests 32 --run-id armB_realistic
python runners/bench_vllm_optimized.py --fixture shared_prefix  --concurrency 1 8 32 --n-requests 32 --run-id armC_shared_clean
python runners/bench_vllm_optimized.py --fixture atl_realistic  --concurrency 1 8 32 --n-requests 32 --run-id armC_realistic
```

Do not restart the runtime between them. Confirm all four manifests share one
`gpu_uuid` before generating any comparison.

Then, if units remain — in this order, and skipping Layer 3:

1. `ncu` on arm B: achieved occupancy, Tensor Core utilisation, memory
   bandwidth. Three metrics still open from the original list. ~10 min.
2. Arm C on atl_realistic at C=64, 128, 256. The 1,392 agents/card figure is
   extrapolated from C=32 and neither arm has ever been run to saturation.

---

## What the 2x2 will answer

The prefix-cache question, by a different route. The ablation was built to read
a hit rate and vLLM 0.26 reported `stats_source: null`, so no KV or hit-rate
signal exists on this build. **The throughput delta between the two fixtures is
the observable that survives that gap** — arm C at 99.4% overlap against arm C
at 10.2%, caching enabled in both.

Report it as a delta. The hit rate remains unmeasurable here, and a delta is
not a hit rate.

## What is at risk, and what is not

**At risk** — every figure derived from shared_prefix throughput:

- 9.28 req/s at C=32, and therefore 1,392 agents/card and the 4.6x headroom
- the 173x arm C / arm B ratio
- every crossover and GPU ceiling in `CADENCE_RESCOPE.md`, which are computed
  from both

**Not at risk** — mechanism findings, which do not depend on prompt overlap:

- arm B is launch-bound: 2.57M `cudaLaunchKernel` calls costing 2.6x the GPU's
  busy time, single stream, 79% idle
- arm B's throughput inverts with concurrency (0.109 → 0.054 req/s)
- pipeline steps are sequential with a real data dependency

## Not done, and why

`ADVISOR_REPORT.md` and the Phase 11 cost model are **not** regenerated. Both
were to be rerun with realistic throughput; there is no realistic throughput
yet. Regenerating them now would change nothing but the timestamp, and
back-filling the realistic column with anything other than a measurement is the
failure mode this project has spent eleven phases avoiding.

The dirty-tree runs stay exactly where they are. When the clean runs land they
become additional rows, not replacements, so the clean/dirty comparison stays
visible.
