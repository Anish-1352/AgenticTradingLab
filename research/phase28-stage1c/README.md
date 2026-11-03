# Stage 1C evidence boundaries

`colab-6144-evidence.json` describes a user-executed external Google Colab L4
run. Codex located the downloaded originals, preserved their exact bytes outside
Git, verified their hashes and checked the reported 12/12 successful decisions.
Codex did not execute or independently reproduce that GPU test.

The original artifacts do not record the exact Colab Git SHA. The manifest
identifies the compatible earlier cf13152b source files and matches all twelve
prompt hashes and the configuration hash. This establishes compatibility, not
proof of the precise runtime checkout. Exact source attribution remains pending.
The uncommitted diagnostic patch, corrected repeat tests and historical ATL
backtest were not tested by this original GPU run. Changed timestamps also
invalidate the original repeated-response equality flags as determinism evidence.
Raw reports, logs and model caches are intentionally outside this repository.
