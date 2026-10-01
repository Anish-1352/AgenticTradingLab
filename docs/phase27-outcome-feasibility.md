# Phase 27: frozen outcome-supervision feasibility

This is an offline research harness, not a trading strategy recommendation. It
uses ATL's public protocol, SDK runner, existing execution engine, cost calculator,
and SQLite run/result/manifest tables. No separate portfolio simulator or model
training framework was introduced. Phase 28 is not implemented.

## Reproduction

From the repository root, using the Python environment described in the final
report:

```sh
PYTHONPATH=packaging/agentictrading/src${PYTHONPATH:+:$PYTHONPATH} python -m dashboard.scripts.phase27_experiment \
  --source-manifest /absolute/path/source/source-manifest.json \
  --output /absolute/path/new-experiment-directory
```

The manifest must sit beside `AAPL-5m.csv`; its SHA256 is checked before use.
The CLI never downloads bars or submits brokerage orders. It chooses a fresh
local SQLite path before importing ATL, disables remote database configuration,
and uses FastAPI's in-process HTTP transport. SDK lifecycle, API authentication,
validation, execution, and persistence remain real. Reproduce in a fresh output
directory. Keep the frozen data locally: a future provider download may differ.

## One fixed specification

`dashboard/backend/domain/research/phase27-v0.json` is the authoritative v0.
AAPL, 2023, New York regular sessions, complete 60-minute decision bars formed
from 5-minute SIP bars; one-hour long-only opportunities, one share, $3,000.
Decisions at 10:30–14:30 local have an exit at 11:30–15:30. The final half-hour
is excluded, as are exchange holidays and early-close sessions. Six completed
hourly bars warm up the feature history, which carries across sessions. A lag
across an overnight gap is a previous-observation return, not a constant elapsed
one-hour return. The source CSV also contains extended-hours bars; they are
excluded by the builder.

Each input contains record ID, symbol, decision and feature-availability time,
and six features: completed close, one- and three-observation simple returns,
six-observation standard deviation of log returns (ddof=0), current volume divided
by its six-observation mean, and estimated round-trip cost at the current close.
The model receives only timestamp, symbol, and these six named numbers.

Outcomes live in a separate file, joined by record ID only for training or after
inference has finished. Let E be the next 5-minute open at the completed decision
bar's boundary and X the open exactly 60 minutes later:

- forward return = X/E − 1;
- net return = (ATL buy net cash impact + ATL sell net cash impact)/(E × quantity);
- MAE = min(0, min future low/E − 1, X/E − 1);
- MFE = max(0, max future high/E − 1, X/E − 1);
- future volatility = square root of the sum of squared log increments along
  E, each of the 12 future 5-minute closes, X; it is not annualized;
- TRADE/LONG iff net return > 0.001, otherwise HOLD/NONE. Equality is HOLD.

Large negative returns are HOLD because this environment cannot execute SHORT.
Direction accuracy conditional on a chosen LONG uses these thresholded labels,
so it equals TRADE precision in this long-only experiment; it is not a separate
measure of predicting positive versus negative returns.

Costs are ATL's tick-rounded fills, 5 basis points of adverse slippage per side,
1 basis point commission per side, no minimum commission, $0.01 tick, no stamp
or transfer fees. Estimated input costs use the current completed close. Hidden
outcome costs use actual future entry/exit references and the same quote function.
These are fixed research assumptions, not a claim about actual Alpaca charges.

## Splits and model boundary

Expanding training, one validation month, and July/September/November tests are
fixed in the spec. Bounds are UTC half-open. A label whose availability reaches
or exceeds its partition end is purged. The first 24 calendar hours of validation
and test are embargoed; weekends or holidays can make the embargo remove zero
rows. The manifests record both excluded and selected IDs. Later folds may train
on earlier completed test months. Final test months are mutually disjoint.

The logistic classifier uses only training IDs, training mean/standard deviation,
600 full-batch steps, learning rate 0.1 and L2 0.01, with zero initial weights.
No random split, oversampling, class weighting, validation tuning, or threshold
search is performed. Validation rows are reserved and counted; v0 does not use
them for model selection. Test labels are joined after prediction files are saved.

The required baselines are constant HOLD (P=0) and this logistic gate (threshold
0.5). A separate `momentum_smoke` diagnostic implements P=1 iff previous return
>0, otherwise P=0, to prove real historical trading and costs even if the fitted
classifier abstains. It is not chosen or optimized for test profitability.

`run_window(..., predict=...)` is the future model substitution point. The
callback accepts the strict input payload and returns exactly
`{"trade_probability": float, "direction": "LONG" or "NONE"}`. Probability must
be finite in [0,1], and direction must agree with the fixed threshold. Unknown
fields and SHORT are rejected. A future HTTP-served open model can implement
this callback without changing execution or evaluation. No model serving or
base-open-model advantage has been demonstrated in Phase 27.

The policy closes the preceding one-hour position before entering a new chosen
opportunity, including consecutive LONGs. Exit-only control steps flatten at the
fixed horizon. Every round trip pays both sides' costs. There are no overnight
positions. This preserves the label contract; it is not a HOLD-means-liquidate
claim about ATL generally.

## Responsibilities and evidence

ATL owns clock, protocol observations, orders, risk checks, fills, positions,
valuation and stored results. The builder owns compact inputs, hidden outcomes,
calendar filtering and splits. The model owns P(TRADE) and direction. The
experiment evaluator owns classification metrics and joins predictions to hidden
outcomes only after execution. No model sees a full raw tape or outcome structure.

A trusted process-local replay registration supplies the frozen `MarketDataset`
and explicit cost profile to the existing external engine. It is not exposed as
an HTTP file-loading endpoint; unknown replay IDs or mismatched dates/universe
fail closed. Ordinary runs retain their existing data path and cost behavior.
Frozen runs omit the unrelated background buy-and-hold/DJIA comparison jobs;
matched-cost experimental baselines run separately through the same protocol.

Each window/policy runs twice. Fingerprints include ordered predictions, submitted
orders, persisted trades and fees, all equity points, and financial metrics;
random protocol IDs and wall-clock creation times are excluded. Each run keeps
raw results, execution responses, prediction hashes, model state, source/code/
split/spec identifiers, and a manifest in ATL's existing run-manifest table.
The evaluator also verifies flat terminal cash against the sum of chosen hidden
net outcomes. Failures produce a file with stage counts and exception type,
without credentials or unfiltered transport errors.

The SDK previously rejected all explicit capital except $1,000 although the
backend now accepts $0–$3,000. It now forwards finite nonnegative explicit capital,
leaving environment maximum enforcement to the backend. Omitting capital still
uses the backend default. Earlier causal repairs are documented separately in
`phase27-causal-clock.md`.

## Limits on interpretation

The frozen Alpaca SIP response was retrieved in 2026 for 2023 bars. The request
used the documented raw adjustment default and no fallback to IEX. It is a
reproducible snapshot, not a historical-vintage/as-published archive. Provider
corrections, survivorship from selecting AAPL today, and availability revisions
cannot be ruled out. Event-time tests prove causality relative to the frozen
snapshot; they do not prove that exact bytes were available in 2023.

Immediate next-open fills assume zero decision latency, no queue, no spread
beyond fixed slippage, and no market impact. Horizon completeness is a retrospective
sample-availability rule; missing future bars could select the sample. For this
snapshot the reported regular-session grid has no missing source bars. The
calendar is specifically 2023, not a general exchange-calendar implementation.

There are three disjoint test months in one stock and one year, not hundreds of
independent experiments. Features overlap and intraday observations share market
regimes. One-hour outcome intervals are adjacent (shared endpoints) rather than
independent. No significance, profitability, generalization, or LLM superiority
claim follows from harness feasibility. The classifier may correctly expose a
weak baseline by abstaining everywhere. Numerical probability metrics remain
measurable; selected-trade precision, direction accuracy and conditional return
are null when no trades are selected. Log loss clips probabilities to [1e-15,
1−1e-15]. Financial returns and Sharpe are secondary.
