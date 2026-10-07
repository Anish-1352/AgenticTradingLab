"""Phase 27B evaluation through ATL's public API and the unmodified SDK loop.

Run as a module. The CLI sets an isolated SQLite path before loading the app.
No data acquisition, no training framework and no execution simulator live here:
fills, costs, positions and valuation are ATL's.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]


def validate_prediction(value):
    """Phase 27's probability contract, or Phase 27D's continuous one -- nothing else.

    A probability output keeps Phase 27's rule (LONG iff p >= 0.5); a predicted
    net return carries its gate's direction. Either schema must match exactly.
    """
    from dashboard.backend.domain.research.phase27d import validate_regression_prediction
    from dashboard.scripts.phase27_experiment import validate_prediction as v27
    if isinstance(value, dict) and "predicted_net_return" in value:
        return validate_regression_prediction(value)
    return v27(value)


CLASSIFIER_BASELINES = ("always_hold", "momentum_20", "logistic")
REGRESSOR_BASELINES = ("train_mean", "linear_gt0", "linear_top20")


def fold_gates(names, inputs, outcomes, fold, by_symbol, spec):
    """(name, predict, state) for each named baseline, every model fit on training rows only.

    The top-20% gate ranks the fold's test rows at each decision timestamp from
    their model-visible features -- what a decision at that moment can see.
    """
    from dashboard.backend.domain.research.phase27d import (
        fit_linear_regressor, fit_train_mean, top_fraction_selection)
    unknown = [n for n in names if n not in CLASSIFIER_BASELINES + REGRESSOR_BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline: {unknown}")
    threshold = spec["probability_threshold"]
    feature = lambda payload: [[payload["features"][f] for f in spec["feature_fields"]]]
    gates = []
    if "always_hold" in names:
        gates.append(("always_hold", always_hold, {"constant_probability": 0.0}))
    if "momentum_20" in names:
        gates.append(("momentum_20", momentum_rule, spec["momentum_rule"]))
    if "logistic" in names:
        model = fit_pooled_gate(inputs, outcomes, fold["train_ids"], spec)

        def logistic(payload, model=model):
            p = float(model.predict(feature(payload))[0])
            return {"trade_probability": p, "direction": "LONG" if p >= threshold else "NONE"}
        gates.append(("logistic", logistic, model.state()))
    if "train_mean" in names:
        mean = fit_train_mean(outcomes, fold["train_ids"])
        gates.append(("train_mean", lambda payload, m=mean: {
            "predicted_net_return": m, "direction": "LONG" if m > 0 else "NONE"}, {"train_mean": mean}))
    if {"linear_gt0", "linear_top20"} & set(names):
        from dashboard.backend.domain.research.phase27b import model_payload
        regressor = fit_linear_regressor(inputs, outcomes, fold["train_ids"], spec)
        predict = lambda payload, r=regressor: float(r.predict(feature(payload))[0])
        if "linear_gt0" in names:
            gates.append(("linear_gt0", lambda payload: (lambda v: {
                "predicted_net_return": v, "direction": "LONG" if v > 0 else "NONE"})(predict(payload)),
                regressor.state()))
        if "linear_top20" in names:
            rows = [r for group in by_symbol.values() for r in group]
            scores = {r["record_id"]: predict(model_payload(r, spec)) for r in rows}
            chosen = {(r["timestamp"], r["symbol"]) for r in rows
                      if r["record_id"] in top_fraction_selection(rows, scores, spec["ranking_gate"]["fraction"])}
            gates.append(("linear_top20", lambda payload: {
                "predicted_net_return": predict(payload),
                "direction": "LONG" if (payload["timestamp"], payload["symbol"]) in chosen else "NONE"},
                {**regressor.state(), "ranking_gate": spec["ranking_gate"]}))
    return gates


def momentum_rule(payload):
    """Baseline 1, frozen in the spec: TRADE/LONG iff return_20 > 0."""
    from dashboard.backend.domain.research.phase27b import load_spec
    rule = load_spec()["momentum_rule"]
    p = float(payload["features"][rule["feature"]] > rule["threshold"])
    return {"trade_probability": p, "direction": "LONG" if p else "NONE"}


def always_hold(payload):
    """Baseline 0: no model, no orders."""
    return {"trade_probability": 0.0, "direction": "NONE"}


def fit_pooled_gate(inputs, outcomes, train_ids, spec):
    """Baseline 2: one logistic gate across symbols, fitted on training rows only."""
    from dashboard.backend.domain.research.phase27 import LogisticGate
    from dashboard.backend.domain.research.phase27b import model_payload
    rows = {r["record_id"]: r for r in inputs}
    train = set(train_ids)
    labels = {o["record_id"]: o["trade_worthy"] for o in outcomes if o["record_id"] in train}
    x = [[model_payload(rows[rid], spec)["features"][f] for f in spec["feature_fields"]] for rid in train_ids]
    cfg = spec["classifier"]
    return LogisticGate(cfg["iterations"], cfg["learning_rate"], cfg["l2"]).fit(
        x, [labels[rid] for rid in train_ids])


def exit_timestamp(t, calendar, spec):
    """The fixed five-session exit, from the calendar alone -- never from outcomes."""
    import pandas as pd
    day = pd.Timestamp(t).tz_convert(spec["timezone"]).date()
    exit_day = calendar[calendar.index(day) + spec["horizon_sessions"]]
    return pd.Timestamp(f"{exit_day} {spec['decision_bar_end_local']}", tz=spec["timezone"]).tz_convert("UTC")


def run_symbol_window(http, directory, raw, inputs, calendar, spec, symbol, start, end, predict, metadata,
                      prepared=None):
    """Inference and execution for one symbol and one window. Outcomes cannot enter.

    ``predict`` is the replacement point for a future open model. It sees the
    allowlisted payload -- timestamp, symbol, named features -- and nothing else.
    """
    import pandas as pd
    from agentictrading import ATLClient, AgentRunner
    from dashboard.backend.domain.research.phase27 import canonical, digest
    from dashboard.backend.domain.research.phase27b import model_payload
    from dashboard.backend.domain.research import phase27b_replay
    from dashboard.backend.domain.backtesting.replay import register_replay
    from dashboard.backend.domain.backtesting import external_run_service as ebs
    from dashboard.backend.domain.runs import service as runs
    from dashboard.backend.database import db

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (directory / name).write_text(canonical(value) + "\n")

    class LocalClient(ATLClient):
        # Only the HTTP transport changes. SDK schemas and lifecycle are real.
        def _request(self, method, path, *, body=None, params=None, timeout=None):
            response = http.request(method, path, json=body, params=params, headers={"X-API-Key": self._api_key})
            response.raise_for_status()
            return response.json()

    # ``prepared`` carries the whole-tape derivations computed once per symbol;
    # without it the run derives them itself, with identical results.
    if prepared is None:
        prepared = phase27b_replay.prepare_symbol_tape(raw, symbol, spec)
    prices = prepared["sessions"]
    observed_close = {row.decision_timestamp: row.price for row in prices.itertuples()}
    entries = [pd.Timestamp(r["timestamp"]) for r in inputs]
    exits = [exit_timestamp(t, calendar, spec) for t in entries]

    class Policy:
        def __init__(self):
            self.rows = {pd.Timestamp(r["timestamp"]): r for r in inputs}
            self.predictions, self.decisions, self.executions = [], [], []
            self.expected_fills, self.previous = 0, None

        def decide(self, obs):
            bar = obs.market["bars"][symbol]
            t = pd.Timestamp(bar["timestamp"])
            if self.previous is not None and t <= self.previous:
                raise ValueError("non-increasing decision clock")
            self.previous = t
            held = sum(p["quantity"] for p in obs.positions if p["symbol"] == symbol)
            long_ = False
            if t in self.rows:
                row = self.rows[t]
                if abs(bar["close"] - observed_close[t]) > 1e-9:
                    raise ValueError("ATL observation and frozen session price disagree")
                prediction = validate_prediction(predict(model_payload(row, spec)))
                self.predictions.append({"record_id": row["record_id"], "timestamp": row["timestamp"],
                                         "symbol": symbol, **prediction})
                long_ = prediction["direction"] == "LONG"
            # A LONG on a held position holds through: ATL's position cap counts
            # this decision's buys but not its sells, so selling and rebuying a
            # name above ~$375 is judged as two shares and rejected.
            orders = []
            if held and not long_:
                orders.append({"symbol": symbol, "side": "sell", "quantity": held})
            elif long_ and not held:
                orders.append({"symbol": symbol, "side": "buy", "quantity": spec["quantity"]})
            self.expected_fills = len(orders)
            self.decisions.append({"timestamp": t.isoformat(), "orders": orders})
            return {"orders": orders, "confidence": 1.0}

        def on_execution_result(self, result):
            if result.rejections or len(result.fills) != self.expected_fills:
                raise ValueError("rejected or missing execution")
            self.executions.append(result.raw)

    policy = Policy()
    try:
        replay_id = "phase27b-" + uuid.uuid4().hex
        register_replay(replay_id, phase27b_replay.make_symbol_replay(raw, symbol, entries, exits, spec,
                                                                      start, end, prepared=prepared))
        response = http.post("/api/v1/agents", json={"name": "phase27b-offline"},
                             headers={"X-Session-Id": uuid.uuid4().hex})
        response.raise_for_status()
        registered = response.json()
        key, agent_id = registered["api_key"], registered["agent"]["agent_id"]
        response = http.post(f"/api/v1/agents/{agent_id}/versions", headers={"X-API-Key": key},
                             json={"version": "phase27b-v1",
                                   "model_backbones": [metadata.get("baseline", "deterministic-gate")]})
        response.raise_for_status()
        sdk = LocalClient("http://testserver", key)
        result = AgentRunner(sdk, policy).run_backtest(
            response.json()["agent_version"]["agent_version_id"], environment_id="us-equity-hourly-v1",
            start_date=start, end_date=end, symbols=[symbol], initial_cash=spec["initial_cash"],
            config={"replay_id": replay_id}, poll_interval=0.001)
        if len(policy.predictions) != len(inputs) or len(policy.executions) != len(policy.decisions):
            raise ValueError("missing prediction or auto-held decision")
        if result.metrics.get("timeout_holds", 0):
            raise ValueError("timed-out decision")
        (directory / "predictions.jsonl").write_text("".join(canonical(p) + "\n" for p in policy.predictions))
        save("decisions.json", policy.decisions)
        save("result.json", result.raw)
        # Prove persisted retrieval without the live execution/session objects.
        runs._runs.pop(result.run_id, None)
        for identifier, session in list(ebs._sessions.items()):
            if session.run_id == result.result_run_id:
                ebs._sessions.pop(identifier, None)
        if sdk.get_run_result(result.run_id).raw != result.raw:
            raise ValueError("persisted result differs from completed run")
        fields = ("timestamp", "symbol", "side", "quantity", "price", "total_fees",
                  "slippage_amount", "reference_price")
        stable = {"predictions": policy.predictions, "decisions": policy.decisions,
                  "trades": [{k: t.get(k) for k in fields} for t in result.trades],
                  "equity_curve": result.equity_curve, "metrics": result.metrics}
        fingerprint = digest(stable)
        manifest = {**metadata, "spec_hash": digest(spec), "symbol": symbol, "run_id": result.run_id,
                    "result_run_id": result.result_run_id, "fingerprint": fingerprint,
                    "predictions_sha256": digest((directory / "predictions.jsonl").read_bytes())}
        db.insert_run_manifest(result.result_run_id, manifest)
        save("manifest.json", manifest)
        return {**stable, "fingerprint": fingerprint, "run_id": result.run_id,
                "result_run_id": result.result_run_id}
    except Exception as error:
        # No exception text: a transport error could carry an authenticated URL.
        save("failure.json", {"error_type": type(error).__name__, "symbol": symbol,
                              "prediction_count": len(policy.predictions),
                              "decision_count": len(policy.decisions)})
        raise


def portfolio_financials(runs, spec):
    """Sum the per-symbol $initial_cash accounts into one book. Secondary metrics.

    Sharpe uses the combined curve sampled at the weekly decision steps and is
    annualised by sqrt(52). One quarter of weekly steps is ~12 returns, so it is
    reported for completeness, not as evidence.
    """
    import numpy as np
    import pandas as pd
    initial = spec["initial_cash"] * len(runs)
    if not runs:
        return {"accounts": 0}
    curves = []
    steps = set()
    for run in runs.values():
        c = pd.Series({pd.Timestamp(p["timestamp"]): p["equity"] for p in run["equity_curve"]})
        curves.append(c[~c.index.duplicated(keep="last")].sort_index())
        steps |= {pd.Timestamp(d["timestamp"]) for d in run["decisions"]}
    index = sorted(set().union(*[c.index for c in curves]))
    book = sum(c.reindex(index).ffill().fillna(spec["initial_cash"]) for c in curves)
    drawdown = float((book / book.cummax() - 1).min()) if len(book) else 0.0
    sampled = book.reindex(sorted(steps), method="ffill").pct_change().dropna()
    sd = float(sampled.std(ddof=1)) if len(sampled) > 1 else 0.0
    trades = [t for run in runs.values() for t in run["trades"]]
    final = float(sum(run["metrics"]["final_equity"] for run in runs.values()))
    return {"accounts": len(runs), "initial_cash": initial, "final_equity": final,
            "cumulative_return": final / initial - 1, "fills": len(trades),
            "fees": float(sum(t["total_fees"] or 0 for t in trades)),
            "slippage": float(sum(t["slippage_amount"] or 0 for t in trades)),
            "turnover": float(sum(t["quantity"] * t["price"] for t in trades)) / initial,
            "max_drawdown": drawdown,
            "sharpe_weekly_sqrt52": float(sampled.mean() / sd * np.sqrt(52)) if sd > 0 else None,
            "weekly_steps": int(len(sampled))}


def evaluate(http, output, raws, market_raw, source, spec, code_hash, fold_ids=None):
    """Every fold x baseline x symbol through ATL, twice; labels joined after sealing."""
    import pandas as pd
    from dashboard.backend.domain.research.phase27 import canonical, digest
    from dashboard.backend.domain.research import phase27b_replay
    from dashboard.backend.domain.research.phase27b import (
        build_dataset, build_splits, calendar_sessions, session_frame, spell_cash, write_dataset)
    from dashboard.backend.domain.research.phase27d import regression_metrics
    from dashboard.backend.domain.research.phase27b_metrics import (
        cost_sensitivity, probability_buckets, rank_signal, score_window)
    from dashboard.backend.database import db

    output = Path(output)
    traded = {s: raws[s] for s in spec["universe"] if s in raws}
    inputs, outcomes, quality = build_dataset(traded, market_raw, spec)
    market = session_frame(market_raw, spec)
    calendar = calendar_sessions(market.index[0], market.index[-1], spec)
    manifest = write_dataset(output / "dataset", inputs, outcomes, quality, calendar, spec, source, code_hash)
    folds = build_splits(inputs, outcomes, calendar, spec)
    if fold_ids is not None:
        # A worker runs a subset of the folds of the SAME spec and dataset, so
        # workers' windows join on spec_hash and dataset_id.
        unknown = sorted(set(fold_ids) - {f["id"] for f in folds})
        if unknown:
            raise ValueError(f"unknown folds: {unknown}")
        folds = [f for f in folds if f["id"] in set(fold_ids)]
    indexed = {r["record_id"]: r for r in inputs}
    hidden = {o["record_id"]: o for o in outcomes}
    prepared = {}
    threshold = spec["probability_threshold"]
    experiment_id = digest([manifest["dataset_id"], spec, code_hash])
    report = {"experiment_id": experiment_id, "dataset_id": manifest["dataset_id"], "code_hash": code_hash,
              "source_sha256": source.get("sha256"), "spec_hash": digest(spec), "windows": [], "pooled": {},
              "pooled_by_class": {}, "fold_ids": [f["id"] for f in folds]}
    pooled, pooled_by_class = {}, {}

    for fold in folds:
        if not fold["train_ids"] or not fold["test_ids"]:
            raise ValueError(f"empty split in {fold['id']}")
        by_symbol = {}
        for rid in fold["test_ids"]:
            by_symbol.setdefault(indexed[rid]["symbol"], []).append(indexed[rid])
        start = fold["bounds"]["test"][0]
        end = (pd.Timestamp(fold["bounds"]["test"][1]) - pd.Timedelta(days=1)).date().isoformat()
        for name, gate, state in fold_gates(spec.get("baselines", CLASSIFIER_BASELINES),
                                            inputs, outcomes, fold, by_symbol, spec):
            root = output / "runs" / fold["id"] / name
            per_symbol, sealed = {}, []
            for symbol in sorted(by_symbol):
                rows = sorted(by_symbol[symbol], key=lambda r: r["timestamp"])
                metadata = {"experiment_id": experiment_id, "dataset_id": manifest["dataset_id"],
                            "code_hash": code_hash, "fold": fold["id"], "baseline": name,
                            "model_state": state, "seed": spec["seed"]}
                if symbol not in prepared:
                    prepared[symbol] = phase27b_replay.prepare_symbol_tape(traded[symbol], symbol, spec)
                repeats = [run_symbol_window(http, root / symbol / str(k), traded[symbol], rows, calendar,
                                             spec, symbol, start, end, gate, {**metadata, "repeat": k},
                                             prepared=prepared[symbol])
                           for k in range(2)]
                if repeats[0]["fingerprint"] != repeats[1]["fingerprint"]:
                    raise ValueError(f"non-reproducible replay: {fold['id']} {name} {symbol}")
                per_symbol[symbol] = repeats[0]
                sealed.extend(repeats[0]["predictions"])
            sealed.sort(key=lambda p: (p["timestamp"], p["symbol"]))
            root.mkdir(parents=True, exist_ok=True)
            (root / "predictions.jsonl").write_text("".join(canonical(p) + "\n" for p in sealed))

            # ---- test outcomes are joined only after every prediction is on disk ----
            outs = [hidden[p["record_id"]] for p in sealed]
            regressor = name in REGRESSOR_BASELINES
            # A gate's trades are its LONG directions; for a probability model
            # that is p >= threshold, as the contract enforces. Classification
            # metrics see probabilities, or the 0/1 trade decision of a regressor.
            selected = [1.0 if p["direction"] == "LONG" else 0.0 for p in sealed]
            scores = [p["predicted_net_return" if regressor else "trade_probability"] for p in sealed]
            probs = selected if regressor else scores
            labels = [o["trade_worthy"] for o in outs]
            net = [o["cost_adjusted_forward_return"] for o in outs]
            for symbol, run in per_symbol.items():
                expected = spec["initial_cash"] + spell_cash(
                    [hidden[p["record_id"]] for p in run["predictions"] if p["direction"] == "LONG"],
                    spec)
                if abs(run["metrics"]["final_equity"] - expected) > 1e-6:
                    raise ValueError(f"ATL cash disagrees with hidden outcomes: {fold['id']} {name} {symbol}")
            window_class = fold["bounds"].get("window_class", "unclassified")
            entry = {"fold": fold["id"], "baseline": name, "window_class": window_class,
                     "classification": score_window(labels, probs, net,
                                                    [o["maximum_adverse_excursion"] for o in outs],
                                                    [o["maximum_favorable_excursion"] for o in outs], threshold),
                     "ranking": rank_signal(scores, labels, net),
                     "probability_buckets": None if regressor else probability_buckets(probs, net),
                     "regression": regression_metrics(scores, net) if regressor else None,
                     "cost_sensitivity": cost_sensitivity(outs, probs, spec),
                     "financial": portfolio_financials(per_symbol, spec),
                     "per_symbol": {s: {"predictions": len(r["predictions"]), "fills": len(r["trades"]),
                                        "final_equity": r["metrics"]["final_equity"],
                                        "atl_max_drawdown": r["metrics"].get("max_drawdown"),
                                        "atl_sharpe_ratio": r["metrics"].get("sharpe_ratio")}
                                    for s, r in per_symbol.items()},
                     "reproducible": True, "cash_matches_outcomes": True,
                     "train_rows": len(fold["train_ids"]), "validation_rows_reserved": len(fold["validation_ids"]),
                     "test_rows": len(fold["test_ids"]), "purged": len(fold["purged_ids"]),
                     "embargoed": len(fold["embargoed_ids"]),
                     "run_ids": {s: r["run_id"] for s, r in per_symbol.items()}}
            for run in per_symbol.values():
                stored = db.get_run_manifest(run["result_run_id"]) or {}
                db.insert_run_manifest(run["result_run_id"], {**stored, "evaluation_window": fold["id"]})
            (root / "evaluation.json").write_text(canonical(entry) + "\n")
            report["windows"].append(entry)
            # Seen and unseen evidence never share an accumulator: "pooled"
            # spans every window, "pooled_by_class" keeps each class apart.
            for acc in (pooled.setdefault(name, {k: [] for k in ("y", "p", "net", "mae", "mfe")}),
                        pooled_by_class.setdefault(window_class, {}).setdefault(
                            name, {k: [] for k in ("y", "p", "net", "mae", "mfe")})):
                acc["y"] += labels; acc["p"] += probs; acc["net"] += net
                acc["mae"] += [o["maximum_adverse_excursion"] for o in outs]
                acc["mfe"] += [o["maximum_favorable_excursion"] for o in outs]
            print(f"Completed {fold['id']} {name}: {len(sealed)} predictions, "
                  f"{entry['financial']['fills']} fills; repeat identical", flush=True)

    def summarise(acc):
        return {**score_window(acc["y"], acc["p"], acc["net"], acc["mae"], acc["mfe"], threshold),
                "ranking": rank_signal(acc["p"], acc["y"], acc["net"]),
                "probability_buckets": probability_buckets(acc["p"], acc["net"])}

    report["pooled"] = {name: summarise(acc) for name, acc in pooled.items()}
    report["pooled_by_class"] = {cls: {name: summarise(acc) for name, acc in by_name.items()}
                                 for cls, by_name in pooled_by_class.items()}
    (output / "experiment.json").write_text(canonical(report) + "\n")
    return report


def configure_offline(output):
    """Select local storage before importing any ATL store or app module."""
    os.environ["DATABASE_PATH"] = str(Path(output) / "atl-research.db")
    for name in ("CONTENT_DATABASE_URL", "USERS_DATABASE_URL", "AGENT_RUNS_DATABASE_URL", "DATABASE_URL"):
        os.environ[name] = ""
    os.environ["ATL_BAR_CACHE"] = "0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", default="phase27b-v1.json",
                        help="frozen spec file name in domain/research/")
    parser.add_argument("--folds", type=lambda v: v.split(","), default=None,
                        help="comma-separated fold ids to run; default all")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_offline(output)
    from dashboard.backend.domain.research.phase27 import canonical, digest
    from dashboard.backend.domain.research.phase27b import load_spec, read_sources
    from dashboard.backend.app import app
    from fastapi.testclient import TestClient
    paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, text=True).splitlines()
    files = {p: digest((REPO / p).read_bytes()) for p in sorted(set(paths)) if (REPO / p).is_file()}
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "files": files, "source_tree_hash": digest(files), "python": sys.version}
    (output / "code-manifest.json").write_text(canonical(provenance) + "\n")
    spec = load_spec(args.spec)
    raws, meta = read_sources(args.source_manifest, spec["universe"], spec["market_symbol"])
    source = {"sha256": digest(meta), "manifest": meta}
    market = raws.pop(spec["market_symbol"])
    try:
        # Not used as a context manager: entering it would fire the app's
        # startup workers, one of which fetches live bars.
        evaluate(TestClient(app), output, raws, market, source, spec, provenance["source_tree_hash"],
                 fold_ids=args.folds)
    except Exception as error:
        (output / "failure.json").write_text(canonical({"error_type": type(error).__name__}) + "\n")
        raise


if __name__ == "__main__":
    main()
