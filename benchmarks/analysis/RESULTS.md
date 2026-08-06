# GPU serving benchmark — results

**NOT YET GENERATED.** This file is a placeholder.

`RESULTS.md` is *generated, never hand-written* — every number in it is read
from a `*_summary.json` and attributed to a `run_id`. A hand-maintained results
document drifts from the data the moment one run is re-done, and the v1 phase of
this study is a record of what that costs.

The analysis code is complete and tested; it just has not been pointed at the
run artifacts yet. They live on Drive at
`/content/drive/MyDrive/atl_bench/results/` and were not present locally when
this branch was written.

## Generate it

From `benchmarks/`, with the result JSONs available:

```bash
python -m analysis.make_results \
  --arm-b   <results>/armB_shared_summary.json \
  --arm-c   <results>/armC_shared_summary.json \
  --trace-analysis <results>/armB_L3_trace.json \
  --out     analysis/RESULTS.md \
  --figures-dir analysis/figures \
  --render-figures \
  --concurrency 32 \
  --exclude "<results>/smoke_c1_summary.json=superseded: ran on a different \
stack AND a different GPU; excluded rather than silently dropped"
```

The Layer 3 trace summary is produced first, with the prefill/decode boundary
**measured** rather than guessed:

```bash
python -m common.trace_analysis <results>/armB_L3_c8_torch.json.gz \
  --raw <results>/armB_L3_raw.json \
  --out <results>/armB_L3_trace.json
```

Without `--raw`, the split falls back to the kernel-name heuristic, whose own
caveat says not to quote it.

## What will refuse to run

`make_results` and `compare_arms` both call the provenance guard first and exit
non-zero — writing nothing — if the runs disagree on `gpu_uuid`,
`fixture_sha256`, `config_sha256`, `max_new_tokens` or `pip_freeze_sha256`.
That is deliberate: a silent cross-card or cross-fixture comparison is the exact
failure this study exists to avoid, so it is a refusal rather than a warning.

If a difference is intentional, `--allow-mismatch <field>` waives it and the
waiver is printed in the generated document, where the reader sees it too.
