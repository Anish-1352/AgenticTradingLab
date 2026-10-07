"""Phase 27D report: continuous vs Phase 27C binary, on identical held-out rows."""
import json

import numpy as np
import pytest

from dashboard.scripts.phase27d_report import build_report, load_runs

FOLDS = ["2019Q1", "2020Q1", "2021Q1", "2022Q1"]
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _dataset(root, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for f_i, fold in enumerate(FOLDS):
        for day in range(15):
            for sym in SYMBOLS:
                score = rng.normal()
                rows.append({"record_id": f"{fold}-{day}-{sym}", "fold": fold, "symbol": sym, "score": score,
                             "timestamp": f"{2019 + f_i}-02-{day + 1:02d}T20:30:00+00:00",
                             "noise": rng.normal()})
    return rows


def _write(root, rows, baselines, signal, outcome_noise_seed=0, folds=FOLDS):
    rng = np.random.default_rng(outcome_noise_seed)
    (root / "dataset").mkdir(parents=True)
    outcomes = []
    for r in rows:
        net = 0.01 * (signal * r["score"] + r["noise"])
        outcomes.append({"record_id": r["record_id"], "timestamp": r["timestamp"], "symbol": r["symbol"],
                         "cost_adjusted_forward_return": net, "forward_return": net + 0.0012,
                         "trade_worthy": net > 0.0025, "direction": "LONG" if net > 0.0025 else "NONE",
                         "maximum_adverse_excursion": min(0.0, net) - 0.01,
                         "maximum_favorable_excursion": max(0.0, net) + 0.01, "future_volatility": 0.03,
                         "entry_price": 100.0, "exit_price": 100.0 * (1 + net)})
    (root / "dataset" / "outcomes.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes))
    for name in ("inputs.jsonl", "splits.json", "sessions.json"):
        (root / "dataset" / name).write_text("same\n")
    windows = []
    for fold in folds:
        for b in baselines:
            d = root / "runs" / fold / b
            d.mkdir(parents=True)
            preds = []
            for r in (x for x in rows if x["fold"] == fold):
                s = r["score"]
                if b in ("logistic", "momentum_20", "always_hold"):
                    p = {"logistic": float(1 / (1 + np.exp(-0.05 * s - rng.normal(scale=2.0)))),
                         "momentum_20": float(s > 0), "always_hold": 0.0}[b]
                    pred = {"trade_probability": p, "direction": "LONG" if p >= 0.5 else "NONE"}
                else:
                    v = {"train_mean": 0.001, "linear_gt0": 0.01 * s, "linear_top20": 0.01 * s}[b]
                    pred = {"predicted_net_return": v, "direction": "LONG" if v > 0 else "NONE"}
                preds.append({"record_id": r["record_id"], "timestamp": r["timestamp"], "symbol": r["symbol"],
                              **pred})
            (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
            windows.append({"fold": fold, "baseline": b, "window_class": "unclassified",
                            "financial": {"initial_cash": 15000.0, "final_equity": 15010.0, "fills": 4,
                                          "fees": 0.5, "turnover": 0.1, "max_drawdown": -0.01,
                                          "sharpe_weekly_sqrt52": 0.5},
                            "cost_sensitivity": [{"multiplier": m, "selected": 2, "sum_net_return_selected": 0.01,
                                                  "trade_label_rate": 0.5, "fraction_selected_positive": 0.5}
                                                 for m in (0.5, 1.0, 2.0)],
                            "classification": {"n": 75}})
    (root / "experiment.json").write_text(json.dumps({"dataset_id": "d", "spec_hash": root.name,
                                                      "fold_ids": folds, "windows": windows}))
    return root


def _spec():
    return {"folds": [{"id": f, "window_class": "new_historical" if f < "2021" else "sealed"} for f in FOLDS],
            "uncertainty": {"draws": 200, "seed": 2727, "level": 0.95}, "primary_classes": None}


def _pair(tmp_path, signal=2.0, seed=1):
    rows = _dataset(tmp_path, seed)
    c = _write(tmp_path / "c", rows, ["always_hold", "momentum_20", "logistic"], signal)
    d = _write(tmp_path / "d", rows, ["train_mean", "linear_gt0", "linear_top20"], signal)
    return c, d


def test_runs_join_only_on_identical_dataset_files_and_held_out_rows(tmp_path):
    c, d = _pair(tmp_path)
    joined = load_runs([c], [d], _spec())
    assert joined["held_out_rows_identical"] and joined["dataset_files_identical"]
    (d / "dataset" / "inputs.jsonl").write_text("different\n")
    with pytest.raises(ValueError, match="dataset"):
        load_runs([c], [d], _spec())


def test_a_missing_held_out_row_is_refused(tmp_path):
    c, d = _pair(tmp_path)
    p = d / "runs" / "2019Q1" / "linear_gt0" / "predictions.jsonl"
    p.write_text("".join(p.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(ValueError, match="rows"):
        load_runs([c], [d], _spec())


def test_a_strong_continuous_signal_beats_a_weak_binary_one(tmp_path):
    c, d = _pair(tmp_path, signal=2.0)
    report = build_report(load_runs([c], [d], _spec()), _spec())
    assert report["CONTINUOUS_BEATS_BINARY"] == "YES"
    assert report["CONTINUOUS_SIGNAL"] == "PROMISING"
    assert report["SEALED_CONFIRMED"] == "NO" and report["PHASE_28_READY"] == "NO"
    t = report["comparison"]
    assert t["linear"]["spearman"]["rho"] > t["logistic"]["spearman"]["rho"]
    assert set(report["robustness"]["linear"]["leave_one_year_out"]) == {"2019", "2020", "2021", "2022"}
    assert report["robustness"]["linear"]["without_2020"]["rows"] == 225
    assert report["target_distribution"]["all_evaluated"]["count"] == 300
    assert {"skewness", "p01", "p99", "largest", "smallest"} <= set(report["target_distribution"]["all_evaluated"])
    assert report["regression"]["linear"]["oos_r2_vs_reference"] is not None


def test_noise_is_not_promising(tmp_path):
    c, d = _pair(tmp_path, signal=0.0, seed=3)
    report = build_report(load_runs([c], [d], _spec()), _spec())
    assert report["CONTINUOUS_SIGNAL"] != "PROMISING"
