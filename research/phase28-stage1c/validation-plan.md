# Stage 1C integration boundary and bounded validation

This branch preserves the independently supplied Google Colab result in
`colab-6144-evidence.json`. Its source commit is UNKNOWN, source snapshot
NOT_CAPTURED, and source provenance INCOMPLETE after the original runtime reset.
It is a 12/12 GPU-serving pass, not a patched-diagnostics, repeatability, trading
execution or historical-backtest pass. No GPU rerun is needed merely to replace
that missing provenance. Stage 2 and model training are outside this work.

## Exact runtime path

`AgentRunner` creates a protocol run, polls `/api/v1/runs/{id}/steps/next`, calls
`on_step` and `decide`, and submits the returned Decision using the existing
idempotent HTTP contract. `VLLMTradingAgent` builds the allowlisted snapshot,
calls the pinned model, then applies strict orders-v1 parsing. An empty parsed
orders list is a genuine HOLD. Transport errors, timeouts, malformed JSON and
unauthorized symbols create a separately labeled fallback HOLD. A response from
a different model is a failed response, never a trade.

The adapter's parser checks schema, allowed symbols, order count, whole positive
shares and long-only sells against current holdings. It does not enforce buying
power or position weights. Those remain ATL's. Protocol `submit_decision` applies
the existing max-position-weight and per-order share gates, then delegates to
`ExternalBacktestSession.submit_decisions`, the common validator, execution
translation and `PortfolioManager.execute_actions`. Cash and the configured
execution quote determine actual fills. A valid reply, risk-approved proposal,
accepted batch and filled trade are different stages. Reports retain each,
including partial batches where only some orders fill.

Source bars are open-stamped five-minute OHLCV; completed hourly observations
are available at the decision boundary. The normal fill is that boundary's next
source open. The final partial hourly decision fills at the known regular-session
closing price. The correction in this stage drains completed source closes using
pre-fill holdings, then executes and marks at the actual fill event. It does not
rewrite earlier history or use an unfinished five-minute close. Post-decision
portfolio responses retain the same contemporaneous marks even when finalization
subsequently drains the rest of the tape. Legacy already-close-stamped hourly
mode retains its close-fill convention. This correction is limited to the
external historical path; it is not a blanket audit of every other ATL engine.

## Frozen historical experiment

`dashboard/examples/vllm_agent/stage1c-spec.json` fixes AAPL and KO on March 2–3,
2026, $1,000 starting capital, native 5m-source/hourly decisions, unchanged risk
limits and long-only execution. Costs are 1 bp commission and 5 bp adverse
slippage on each side, cent-rounded fees and $0.01 tick rounding. They are fixed
research assumptions. HOLD and a small fixed one-share entry/next-step exit
reference run twice. The model receives the existing system prompt and action
contract; no prompt, risk limit or strategy is tuned to results.

The frozen provider CSVs and source manifest live outside Git. The runner verifies
original file hashes, selects the native indicator warm-up pad and bounded window,
and supplies them through `market_data_store.get_dataset(loader_factory=...)`.
Native aggregation, indicators, source/open fill planning and portfolio accounting
remain in ATL. In-process FastAPI HTTP transport replaces a socket, not the API,
authentication, risk controls, execution or database. The trusted local client
sets the existing transaction-cost profile before the first decision; ordinary
production runs are not assigned new costs. Unrelated asynchronous buyhold/DJIA
comparison jobs are omitted; matched-cost baselines use the same actual engine.

The runner saves model text only in local research artifacts, parsed orders,
protocol-risk-approved orders, accept/reject responses, fills and positions. It
independently reconciles cash, quantities, terminal available-price marks and
persisted quote/fee/slippage fields against the frozen policy. Run manifests use
ATL's existing database table. Results remain retrievable after live-session
objects are removed. Gross P&L in the report means net P&L plus recorded fees and
adverse slippage, including any terminal marked open positions. A completed round
trip means a sale returns that symbol's position to zero. Turnover sums both sides'
executed notional divided by initial cash. Tiny two-day Sharpe estimates are not
used to claim performance or significance.

```bash
PYTHONPATH=packaging/agentictrading/src python3 -m \
  dashboard.examples.vllm_agent.stage1c_historical \
  --source-manifest /path/to/frozen/source-manifest.json \
  --out /path/to/new/local-baseline-output
```

On the existing configured L4/vLLM host, append `--qwen` to run actual pinned-model
inference after the same deterministic baselines. There is deliberately no mock
model option in this historical runner. A real-model PASS additionally requires
valid inferences, at least one model-proposed order reaching a fill, and reconciled
accounting. A completed run with only rejected model orders is not a proved
model-to-fill acceptance pass. Synthetic test fixtures are labeled as such and
are never reported as historical performance.

## Next external run provenance

Both smoke runners and the Stage 1C runner automatically record the full source
Git SHA, dirty-working-tree status, SHA-256 of the tracked working-tree patch,
and hashes of untracked source files. Raw untracked contents/credentials are not
copied. The inference smoke refuses to start without source Git identity. A
post-run capture identifies source changes during execution. Reports retain the
model revision, exact request configuration and inference/request/content hashes.
`artifact-checksums.json` hashes exported report/log bytes after they are written.
The historical run also records server vLLM version, measured inference usage,
latency, TTFT where available, and observed GPU memory when on the GPU host.

Use a versioned checkout and a fresh output directory. The prepared Colab delivery
uses a small Git bundle to preserve the validated commit exactly, a hashed real
CSV slice with parent source fingerprints, and a separate CPU/backend virtual
environment. It reuses the pinned vLLM GPU environment instead of replacing its
Torch/CUDA dependencies. Any unavailable package/runtime or failed check is a
blocker, not a reason to substitute canned model responses.

## Interpreting the twelve BUY results

All original external records are `status=valid`, `parsed=true`, `reason=null`,
with one parsed BUY order. `_action` derives BUY from the orders' actual sides;
there is no parser default BUY, coercion from HOLD to BUY, or error fallback BUY.
The schema permits an empty orders array and sell orders; the held-position
scenario also makes SELL legal. Cash and position limits are not a BUY generator.
The original smoke stops at adapter validation and never submits orders to ATL.
It therefore cannot establish whether those BUY orders would pass portfolio risk.

Names such as `short_neutral_hold` and `overbought_sell` describe synthetic scenarios,
not supervised ground-truth actions. Twelve BUY replies can be behavior of this
base model under this prompt and these few synthetic states. Raw original model
text was not saved, and there is no controlled prompt ablation, so attributing the
bias specifically to model weights versus prompt framing is not justified. No
prompt improvement is applied here. Future investigation must keep a frozen
baseline and measure behavior separately from infrastructure correctness.
