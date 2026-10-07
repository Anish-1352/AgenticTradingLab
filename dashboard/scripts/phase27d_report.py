"""Phase 27D report: the continuous target against Phase 27C's binary one, row for row.

    python -m dashboard.scripts.phase27d_report --spec phase27d-v1.json \
        --phase27c-workers /abs/c/w0 ... --phase27d-workers /abs/d/w0 ... --output /abs/report.json

Phase 27C's sealed LogisticGate, momentum and always-HOLD predictions are reused
unchanged. The two runs are joined only when every dataset file is
byte-identical and every fold's held-out rows are the same set, so each
comparison is on identical rows. Outcomes come from the separate outcomes file
and are joined to predictions that were sealed first.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.backend.domain.research.phase27c_stats import DRAWS, LEVEL, SEED
from dashboard.backend.domain.research.phase27d import TARGET, regression_metrics
from dashboard.backend.domain.research.phase27d_stats import (
    _spearman_with_ci, beats_binary, continuous_signal, leave_one_out_spearman, paired_difference_with_ci,
    quintile_spread_with_ci, quintile_table, readiness)

DATASET_FILES = ("inputs.jsonl", "outcomes.jsonl", "splits.json", "sessions.json")
SCORE = {"logistic": "trade_probability", "momentum_20": "trade_probability", "always_hold": "trade_probability",
         "train_mean": "predicted_net_return", "linear_gt0": "predicted_net_return",
         "linear_top20": "predicted_net_return"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _side(roots, spec):
    experiments = [json.loads((Path(r) / "experiment.json").read_text()) for r in roots]
    owner = {}
    for root, e in zip(roots, experiments):
        for fold in e["fold_ids"]:
            if fold in owner:
                raise ValueError(f"fold {fold} ran twice")
            owner[fold] = Path(root)
    missing = sorted({f["id"] for f in spec["folds"]} - set(owner))
    if missing:
        raise ValueError(f"missing folds: {missing}")
    return owner, [w for e in experiments for w in e["windows"]]


def load_runs(phase27c_roots, phase27d_roots, spec):
    """Every prediction from both runs, joined to hidden outcomes, refusing any mismatch."""
    roots = [*phase27c_roots, *phase27d_roots]
    hashes = [{f: _sha(Path(r) / "dataset" / f) for f in DATASET_FILES} for r in roots]
    if any(h != hashes[0] for h in hashes):
        raise ValueError("dataset files differ between runs")
    classes = {f["id"]: f["window_class"] for f in spec["folds"]}
    frames, windows, held_out = [], [], {}
    for side_roots in (phase27c_roots, phase27d_roots):
        owner, side_windows = _side(side_roots, spec)
        windows += side_windows
        for fold, root in sorted(owner.items()):
            for path in sorted((root / "runs" / fold).glob("*/predictions.jsonl")):
                baseline = path.parent.name
                preds = pd.read_json(path, lines=True)
                ids = frozenset(preds["record_id"])
                if held_out.setdefault(fold, ids) != ids:
                    raise ValueError(f"held-out rows differ in {fold} ({baseline})")
                preds["score"] = preds[SCORE[baseline]].astype(float)
                preds["fold"], preds["baseline"], preds["window_class"] = fold, baseline, classes[fold]
                frames.append(preds)
    outcomes = pd.read_json(Path(roots[0]) / "dataset" / "outcomes.jsonl", lines=True).set_index("record_id")
    outcomes = outcomes.rename(columns={"direction": "label_direction"}).drop(columns=["timestamp", "symbol"])
    rows = pd.concat(frames, ignore_index=True).join(outcomes, on="record_id", how="left")
    if rows[TARGET].isna().any():
        raise ValueError("a sealed prediction has no outcome row")
    return {"rows": rows, "windows": windows, "folds": sorted(held_out),
            "dataset_files_identical": True, "held_out_rows_identical": True, "dataset_sha256": hashes[0]}


def _distribution(values, frame):
    v = pd.Series(np.asarray(values, float))
    pct = {f"p{q:02d}": float(v.quantile(q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)}
    ordered = frame.assign(value_=v.to_numpy()).sort_values("value_", kind="stable")
    pick = lambda part: [{"symbol": r.symbol, "timestamp": str(r.timestamp), "net": float(r.value_)}
                         for r in part.itertuples()]
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(v.median()), "std": float(v.std(ddof=0)),
            "skewness": float(v.skew()), **pct, "smallest": pick(ordered.head(5)), "largest": pick(ordered.tail(5))}


def _group_stats(frame, key):
    g = frame.groupby(key)[TARGET]
    return {str(k): {"count": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
                     "std": float(s.std(ddof=0))} for k, s in g}


def _economics(rows, windows):
    selected = rows["direction"] == "LONG"
    net = rows[TARGET].to_numpy(float)
    chosen = net[selected.to_numpy()]
    fin = {"initial_cash": 0.0, "pnl": 0.0, "fills": 0, "fees": 0.0, "turnover_dollars": 0.0,
           "worst_window_drawdown": 0.0}
    sharpes, costs = [], {}
    for w in windows:
        f = w["financial"]
        fin["initial_cash"] += f["initial_cash"]; fin["pnl"] += f["final_equity"] - f["initial_cash"]
        fin["fills"] += f["fills"]; fin["fees"] += f["fees"]; fin["turnover_dollars"] += f["turnover"] * f["initial_cash"]
        fin["worst_window_drawdown"] = min(fin["worst_window_drawdown"], f["max_drawdown"])
        if f.get("sharpe_weekly_sqrt52") is not None:
            sharpes.append(f["sharpe_weekly_sqrt52"])
        for c in w["cost_sensitivity"]:
            a = costs.setdefault(str(c["multiplier"]), {"selected": 0, "sum": 0.0})
            a["selected"] += c["selected"]; a["sum"] += c["sum_net_return_selected"]
    return {"rows": int(len(rows)), "coverage": float(selected.mean()) if len(rows) else None,
            "selected": int(selected.sum()),
            "selected_mean_net": float(chosen.mean()) if chosen.size else None,
            "selected_median_net": float(np.median(chosen)) if chosen.size else None,
            "selected_positive_fraction": float((chosen > 0).mean()) if chosen.size else None,
            "edge_vs_all_rows": float(chosen.mean() - net.mean()) if chosen.size else None,
            "mean_mae_selected": float(rows.loc[selected, "maximum_adverse_excursion"].mean()) if chosen.size else None,
            "mean_mfe_selected": float(rows.loc[selected, "maximum_favorable_excursion"].mean()) if chosen.size else None,
            **fin, "median_window_sharpe": float(np.median(sharpes)) if sharpes else None, "windows": len(windows),
            "cost_sensitivity": {m: {"selected": a["selected"],
                                     "mean_net_return_selected": a["sum"] / a["selected"] if a["selected"] else None}
                                 for m, a in sorted(costs.items())}}


def _ranking(frame, u):
    d, p, net = frame["timestamp"].to_numpy(), frame["score"].to_numpy(float), frame[TARGET].to_numpy(float)
    w = frame["fold"].to_numpy()
    quint = quintile_table(w, p, d, net, frame["maximum_adverse_excursion"], frame["maximum_favorable_excursion"])
    return {"rows": int(len(frame)), "dates": int(frame["timestamp"].nunique()),
            "spearman": _spearman_with_ci(d, p, net, u["draws"], u["seed"], u["level"]),
            "quintiles": quint, "top_quintile_mean": quint[-1]["mean_net_return"],
            "bottom_quintile_mean": quint[0]["mean_net_return"],
            "spread": quintile_spread_with_ci(w, p, d, net, u["draws"], u["seed"], u["level"])}


def _robustness(frame, u):
    d, p, net = frame["timestamp"].to_numpy(), frame["score"].to_numpy(float), frame[TARGET].to_numpy(float)
    out = {name: leave_one_out_spearman(frame[col].to_numpy(), d, p, net, u["draws"], u["seed"], u["level"])
           for name, col in (("leave_one_year_out", "year"), ("leave_one_quarter_out", "fold"),
                             ("leave_one_symbol_out", "symbol"))}
    out["without_2020"] = out["leave_one_year_out"].get("2020")
    out["minimum"] = {k: min((v["rho"] for v in out[k].values() if v["rho"] is not None), default=None)
                      for k in ("leave_one_year_out", "leave_one_quarter_out", "leave_one_symbol_out")}
    return out


def build_report(joined, spec, phase_pass=False):
    u = {"draws": DRAWS, "seed": SEED, "level": LEVEL, **spec.get("uncertainty", {})}
    rows = joined["rows"].assign(year=lambda f: f["fold"].str[:4])
    primary = spec.get("primary_classes")
    if primary:
        rows = rows[rows["window_class"].isin(primary)]
    view = lambda b: rows[rows["baseline"] == b].sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    linear, logistic, momentum = view("linear_gt0"), view("logistic"), view("momentum_20")
    top20, mean = view("linear_top20"), view("train_mean")
    if not (linear["record_id"].tolist() == logistic["record_id"].tolist() == mean["record_id"].tolist()):
        raise ValueError("models were not scored on the same rows")
    gate_runs_agree = bool(np.array_equal(linear["score"].to_numpy(), top20["score"].to_numpy()))
    windows = lambda b: [w for w in joined["windows"] if w["baseline"] == b
                         and (not primary or w["window_class"] in primary)]

    comparison = {"linear": _ranking(linear, u), "logistic": _ranking(logistic, u),
                  "momentum_20": {"spearman": _spearman_with_ci(momentum["timestamp"], momentum["score"],
                                                                momentum[TARGET], u["draws"], u["seed"], u["level"])}}
    paired = paired_difference_with_ci(linear["fold"], linear["timestamp"], linear["score"], logistic["score"],
                                       linear[TARGET], u["draws"], u["seed"], u["level"])
    robustness = {"linear": _robustness(linear, u), "logistic": _robustness(logistic, u)}
    regression = {"linear": regression_metrics(linear["score"], linear[TARGET], reference=mean["score"]),
                  "train_mean": regression_metrics(mean["score"], mean[TARGET], reference=mean["score"])}
    by_class = {cls: {m: {"rows": int(len(f)), "dates": int(f["timestamp"].nunique()),
                          "spearman": _spearman_with_ci(f["timestamp"], f["score"], f[TARGET],
                                                        u["draws"], u["seed"], u["level"]),
                          "spread": quintile_spread_with_ci(f["fold"], f["score"], f["timestamp"], f[TARGET],
                                                            u["draws"], u["seed"], u["level"])}
                      for m, f in (("linear", linear[linear.window_class == cls]),
                                   ("logistic", logistic[logistic.window_class == cls]))}
                for cls in sorted(linear["window_class"].unique())}
    economics = {b: _economics(view(b), windows(b))
                 for b in ("linear_gt0", "linear_top20", "train_mean", "logistic", "momentum_20", "always_hold")}

    beats = beats_binary(paired["delta_spearman"], paired["delta_spearman_ci_low"],
                         paired["delta_spearman_ci_high"], paired["delta_spread"])
    rank, spread, mins = comparison["linear"]["spearman"], comparison["linear"]["spread"], robustness["linear"]["minimum"]
    signal, checks = continuous_signal(rank["rho"], rank["ci_low"], spread["spread"], spread["ci_low"],
                                       mins["leave_one_year_out"], mins["leave_one_quarter_out"],
                                       mins["leave_one_symbol_out"], beats)
    return {"folds": joined["folds"], "dataset_sha256": joined["dataset_sha256"],
            "dataset_files_identical": joined["dataset_files_identical"],
            "held_out_rows_identical": joined["held_out_rows_identical"], "gate_runs_agree": gate_runs_agree,
            "target_distribution": {"all_evaluated": _distribution(linear[TARGET], linear),
                                    "by_year": _group_stats(linear, "year"),
                                    "by_symbol": _group_stats(linear, "symbol")},
            "comparison": comparison, "paired_linear_minus_logistic": paired, "robustness": robustness,
            "regression": regression, "by_window_class": by_class, "economics": economics,
            "rule_checks": checks, "CONTINUOUS_SIGNAL": signal, "CONTINUOUS_BEATS_BINARY": beats,
            **readiness(phase_pass, signal, "NO")}


def main():
    from dashboard.backend.domain.research.phase27b import load_spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--phase27c-workers", nargs="+", required=True)
    parser.add_argument("--phase27d-workers", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-pass", action="store_true",
                        help="set only after the PASS conditions were verified")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    report = build_report(load_runs(args.phase27c_workers, args.phase27d_workers, spec), spec, args.phase_pass)
    Path(args.output).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("CONTINUOUS_SIGNAL", "CONTINUOUS_BEATS_BINARY", "MODEL_TARGET_READY",
                                              "SEALED_CONFIRMED", "PHASE_28_READY")}))


if __name__ == "__main__":
    main()
