"""Phase 27C report from the sealed predictions of one or more experiment workers.

    python -m dashboard.scripts.phase27c_report --spec phase27c-v1.json \
        --workers /abs/w1 /abs/w2 ... --output /abs/report.json

Workers run fold subsets of one spec over one dataset. They are joined only if
every worker reports the same dataset id and spec hash and wrote byte-identical
dataset files, which is also the dataset-rebuild determinism check. Outcomes
are read from the dataset's separate outcomes file and joined to predictions
that were sealed before any outcome was read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.backend.domain.research.phase27b_metrics import score_window
from dashboard.backend.domain.research.phase27c_stats import (
    auc_with_ci, bucket_table, dependence, leave_one_out, phase28_ready, sealed_confirmation,
    spearman_with_ci, task_signal, window_quintile_table)

BASELINES = ("always_hold", "momentum_20", "logistic")
DATASET_FILES = ("inputs.jsonl", "outcomes.jsonl", "splits.json", "sessions.json")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_workers(roots, spec):
    """Every fold exactly once, from workers that agree on dataset and spec."""
    experiments = [json.loads((Path(r) / "experiment.json").read_text()) for r in roots]
    if len({(e["dataset_id"], e["spec_hash"]) for e in experiments}) != 1:
        raise ValueError("workers disagree on dataset id or spec hash")
    hashes = [{f: _sha(Path(r) / "dataset" / f) for f in DATASET_FILES} for r in roots]
    identical = all(h == hashes[0] for h in hashes)
    if not identical:
        raise ValueError("workers wrote different dataset files")
    owner = {}
    for root, e in zip(roots, experiments):
        for fold in e["fold_ids"]:
            if fold in owner:
                raise ValueError(f"fold {fold} ran in two workers")
            owner[fold] = Path(root)
    missing = sorted({f["id"] for f in spec["folds"]} - set(owner))
    if missing:
        raise ValueError(f"missing folds: {missing}")
    outcomes = pd.read_json(Path(roots[0]) / "dataset" / "outcomes.jsonl", lines=True).set_index("record_id")
    # The hidden label's direction shares its name with the model's sealed
    # decision; the prediction keeps "direction", the label is renamed.
    outcomes = outcomes.rename(columns={"direction": "label_direction"})
    classes = {f["id"]: f["window_class"] for f in spec["folds"]}
    rows = []
    for fold, root in sorted(owner.items()):
        for baseline in BASELINES:
            preds = pd.read_json(root / "runs" / fold / baseline / "predictions.jsonl", lines=True)
            preds["fold"], preds["baseline"], preds["window_class"] = fold, baseline, classes[fold]
            rows.append(preds)
    predictions = pd.concat(rows, ignore_index=True)
    joined = predictions.join(outcomes.drop(columns=["timestamp", "symbol"]), on="record_id", how="left")
    if joined["trade_worthy"].isna().any():
        raise ValueError("a sealed prediction has no outcome row")
    windows = [w for e in experiments for w in e["windows"]]
    return {"rows": joined, "windows": windows, "folds": sorted(owner),
            "dataset_identical": identical, "dataset_sha256": hashes[0],
            "dataset_id": experiments[0]["dataset_id"], "spec_hash": experiments[0]["spec_hash"]}


def _block(rows, windows, spec, with_diagnostics=False):
    """Every preregistered number for one baseline over one set of windows."""
    u = spec["uncertainty"]
    draws, seed, level = u["draws"], u["seed"], u["level"]
    threshold = spec["probability_threshold"]
    y, p = rows["trade_worthy"].astype(int).to_numpy(), rows["trade_probability"].to_numpy(float)
    net = rows["cost_adjusted_forward_return"].to_numpy(float)
    mae, mfe = rows["maximum_adverse_excursion"].to_numpy(float), rows["maximum_favorable_excursion"].to_numpy(float)
    d = rows["timestamp"].to_numpy()
    selected = p >= threshold
    out = {
        "rows": int(len(rows)), "dates": int(rows["timestamp"].nunique()),
        "trade_prevalence": float(y.mean()) if y.size else None,
        "classification": score_window(y, p, net, mae, mfe, threshold),
        "edge_vs_all_rows": (float(net[selected].mean() - net.mean()) if selected.any() else None),
        "auc": auc_with_ci(d, y, p, draws=draws, seed=seed, level=level),
        "spearman": spearman_with_ci(d, p, net, draws=draws, seed=seed, level=level),
        "fixed_buckets": bucket_table(p, d, y, net, mae, mfe, spec.get("probability_ranking", {}).get(
            "fixed_buckets", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        "window_quintiles": window_quintile_table(rows["fold"].to_numpy(), p, d, y, net, mae, mfe),
    }
    costs, fin = {}, {"initial_cash": 0.0, "pnl": 0.0, "fills": 0, "fees": 0.0, "turnover_dollars": 0.0,
                      "worst_window_drawdown": 0.0, "window_sharpes": []}
    for w in windows:
        for c in w["cost_sensitivity"]:
            acc = costs.setdefault(c["multiplier"], {"selected": 0, "sum": 0.0, "positive": 0.0,
                                                     "label_rows": 0.0, "rows": 0})
            acc["selected"] += c["selected"]; acc["sum"] += c["sum_net_return_selected"]
            acc["positive"] += (c.get("fraction_selected_positive") or 0.0) * c["selected"]
            n = w["classification"]["n"]
            acc["label_rows"] += (c.get("trade_label_rate") or 0.0) * n; acc["rows"] += n
        f = w["financial"]
        fin["initial_cash"] += f["initial_cash"]; fin["pnl"] += f["final_equity"] - f["initial_cash"]
        fin["fills"] += f["fills"]; fin["fees"] += f["fees"]
        fin["turnover_dollars"] += f["turnover"] * f["initial_cash"]
        fin["worst_window_drawdown"] = min(fin["worst_window_drawdown"], f["max_drawdown"])
        if f.get("sharpe_weekly_sqrt52") is not None:
            fin["window_sharpes"].append(f["sharpe_weekly_sqrt52"])
    out["cost_sensitivity"] = {m: {"selected": a["selected"],
                                   "mean_net_return_selected": a["sum"] / a["selected"] if a["selected"] else None,
                                   "fraction_selected_positive": a["positive"] / a["selected"] if a["selected"] else None,
                                   "trade_label_rate": a["label_rows"] / a["rows"] if a["rows"] else None}
                               for m, a in sorted(costs.items())}
    sharpes = fin.pop("window_sharpes")
    fin["median_window_sharpe"] = float(np.median(sharpes)) if sharpes else None
    fin["windows"] = len(windows)
    out["financial"] = fin
    if with_diagnostics:
        out["dependence"] = dependence(d, rows["symbol"].to_numpy(), y, net)
        out["leave_one_symbol_out"] = leave_one_out(rows["symbol"].to_numpy(), d, y, p, draws=draws, seed=seed)
        out["leave_one_quarter_out"] = leave_one_out(rows["fold"].to_numpy(), d, y, p, draws=draws, seed=seed)
    return out


def build_report(joined, spec):
    rows, windows = joined["rows"], joined["windows"]
    pick = lambda classes, b: rows[rows["window_class"].isin(classes) & (rows["baseline"] == b)].sort_values(
        ["timestamp", "symbol"], kind="stable")
    wins = lambda classes, b: [w for w in windows if w["window_class"] in classes and w["baseline"] == b]
    by_class = {}
    for cls in sorted(rows["window_class"].unique()):
        by_class[cls] = {b: _block(pick([cls], b), wins([cls], b), spec, with_diagnostics=(b == "logistic"))
                         for b in BASELINES}
    primary_classes = spec["evidence_sets"]["primary"]
    logistic = _block(pick(primary_classes, "logistic"), wins(primary_classes, "logistic"), spec,
                      with_diagnostics=True)
    primary = {"classes": primary_classes, "logistic": logistic,
               "momentum_20": _block(pick(primary_classes, "momentum_20"), wins(primary_classes, "momentum_20"), spec),
               "always_hold": _block(pick(primary_classes, "always_hold"), wins(primary_classes, "always_hold"), spec),
               "leave_one_symbol_out": {k: v["auc"] for k, v in logistic["leave_one_symbol_out"].items()},
               "leave_one_quarter_out": {k: v["auc"] for k, v in logistic["leave_one_quarter_out"].items()}}
    quint = logistic["window_quintiles"]
    top_minus_bottom = (quint[-1]["mean_net_return"] - quint[0]["mean_net_return"]
                        if quint[-1]["mean_net_return"] is not None and quint[0]["mean_net_return"] is not None
                        else None)
    loo_min = lambda d: min((v for v in d.values() if v is not None), default=None)
    signal, checks = task_signal(logistic["auc"], loo_min(primary["leave_one_symbol_out"]),
                                 loo_min(primary["leave_one_quarter_out"]), logistic["spearman"]["rho"],
                                 top_minus_bottom)
    sealed = by_class.get("sealed", {}).get("logistic", {}).get("auc")
    return {"dataset_id": joined["dataset_id"], "spec_hash": joined["spec_hash"], "folds": joined["folds"],
            "dataset_identical_across_workers": joined["dataset_identical"],
            "dataset_sha256": joined["dataset_sha256"],
            "by_class": by_class, "primary": {**primary, "top_minus_bottom_quintile_net": top_minus_bottom,
                                              "rule_checks": checks},
            "TASK_SIGNAL": signal, "SEALED_CONFIRMATION": sealed_confirmation(sealed),
            "sealed_auc": sealed["auc"] if sealed else None}


def main():
    from dashboard.backend.domain.research.phase27b import load_spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-pass", action="store_true",
                        help="set only after the PASS conditions were verified")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    report = build_report(load_workers(args.workers, spec), spec)
    report["PHASE_28_READY"] = phase28_ready(args.phase_pass, report["TASK_SIGNAL"], report["sealed_auc"])
    Path(args.output).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("TASK_SIGNAL", "SEALED_CONFIRMATION", "PHASE_28_READY")}))


if __name__ == "__main__":
    main()
