# Phase 27 causal event clock

Scope: external historical runs and `HourlyBacktester.run_agent_backtest`.
This change repairs causal accounting; it does not implement the Phase 27
dataset, classifier, evaluation folds or Phase 28 training.

## Information availability

Alpaca source bars are open-stamped. A five-minute bar stamped 14:30 contains
an open known at 14:30; its final close, high, low and volume become known at
14:35. Aggregated decision bars are close-stamped. The 14:30 hourly observation
contains only the completed interval ending at 14:30. Its full OHLCV is legal.
Turnover uses completed source bars, and indicators use these completed
decision bars. Optional external metadata retains its separate as-of contract.

Legacy unaggregated, close-stamped inputs retain the existing decision-close
fill convention. They must not be passed open-stamped bars while claiming
close-stamped semantics.

## Timeline

1. The decision bar closes at t. Its OHLCV and causal indicators are available.
2. ATL publishes that observation and the portfolio with current holdings.
3. The external agent or dashboard strategy submits its decision at t.
4. Before applying a fill at f, ATL drains source-close events available by f,
   using pre-fill cash and holdings. A source close is timestamped at the end
   of its interval, not its opening timestamp.
5. The order fills at f using the execution plan's price field. Cash, holdings
   and execution costs transition here, through the existing PortfolioManager.
6. ATL marks the new state with the contemporaneous source open (or close for
   a planned close fill). The protocol post-fill portfolio uses the same marks.
7. The just-opened bar's close becomes available only at f + source interval.
   ATL can then mark with that close. Intervening closes are processed before
   any subsequent fill.
8. The next completed decision bar supplies the next observation.

The existing session-final convention fills at the last regular-hours close.
It is an idealized zero-latency close execution, not a claim that a real order
submitted after observing the close can obtain that price. It is preserved
here and must be an explicit assumption in future research or replaced by a
separately tested next-session execution policy.

## Historical accounting

`EventTimeValuator` is a shared incremental mark scheduler for the existing
PortfolioManager, not another simulator. It retains only prices already
observed. Missing current quotes carry the last known mark, not a future close.
It emits an initial open anchor for minute-source runs and completed-close
marks thereafter. At a shared close/open/fill timestamp it publishes the final
post-fill state as one record. Only that exact timestamp may be replaced;
records strictly earlier than a subsequent fill remain unchanged. It rejects
backward-time marks. Engine reporting-currency conversion remains delegated
to the existing reporting context.

The source cursor advances before execution. Finalization drains only remaining
closes and persists the same corrected curve through the existing stores.
No execution prices, strategy choices or transaction-cost parameters are tuned.

## Feature warm-up and errors

Whole-frame mean/minimum/maximum fallbacks are replaced by expanding prefix
statistics. SMA warm-up uses expanding mean; Bollinger warm-up uses expanding
minimum/maximum. RSI is neutral until 15 closes; MACD and its signal are neutral
until 34 closes. Normal mature indicators retain their existing parameters.
The same defaults fill early missing values on long frames, so extending a
frame across a library minimum-length threshold cannot revise its prefix.
Indicator failure defaults are causal too. Nothing is hidden only at the API.

## Evidence and limits

The original three failing HTTP/accounting assertions are retained unchanged.
Additional tests compare legal feature prefixes, force indicator failures,
perturb unfinished source OHLCV, check a later sale and API marks, and exercise
the dashboard engine's first fill. HOLD tests check deterministic histories
and persisted result retrieval.

This is not a certification of every ATL data source or strategy. News and
optional fundamentals require their own availability audits. Source clocks,
corporate actions, revision policies and fixed-frequency performance-metric
assumptions still matter. The US external path still defaults to zero execution
costs; explicit research costs and purged walk-forward evaluation remain later
Phase 27 work. No final dataset or training run is produced by this repair.
