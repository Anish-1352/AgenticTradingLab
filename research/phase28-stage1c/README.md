# Stage 1C evidence boundaries

`colab-6144-evidence.json` describes the user-executed external Google Colab L4
run. The downloaded originals were located, copied unchanged outside Git and
verified by SHA-256. They establish 12/12 valid requests, zero HTTP/inference
errors, Qwen2.5-3B-Instruct at the recorded model revision, vLLM 0.31.0, FP16,
and a 6,144-token context. Codex did not execute or independently reproduce it.

The original Colab runtime was reset. `atl_source_commit: UNKNOWN`,
`source_snapshot: NOT_CAPTURED`, `source_provenance: INCOMPLETE`. The run is
not attributed to cf13152b or local uncommitted diagnostics. Earlier local
prompt-hash reconstruction is a compatibility check, not source provenance.
The corrected repeat tests, patched GPU diagnostics and real-model historical
ATL backtest have no GPU verification from this original smoke result.
Original equality flags are not determinism evidence because prompts changed.
Raw reports, server logs and model caches remain outside Git.
