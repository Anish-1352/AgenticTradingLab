"""Phase 27C report: joins parallel workers, keeps window classes apart, applies the rule."""
import json

import numpy as np
import pytest

from dashboard.scripts.phase27c_report import build_report, load_workers

FOLDS = [("2018Q1", "new_historical"), ("2018Q2", "new_historical"),
         ("2024Q1", "legacy"), ("2025Q4", "sealed")]


def _worker(root, folds, dataset_id="d1", spec_hash="s1", signal=1.5, seed=0):
    rng = np.random.default_rng(seed)
    (root / "dataset").mkdir(parents=True)
    inputs, outcomes = [], []
    windows = []
    for f_i, (fold, cls) in enumerate(FOLDS):
        for day in range(12):
            for sym in ("AAA", "BBB", "CCC", "DDD"):
                rid = f"{fold}-{day}-{sym}"
                score = rng.normal()
                net = 0.01 * (signal * score + rng.normal())
                ts = f"20{18 + f_i}-01-{day + 1:02d}T20:30:00+00:00"
                inputs.append({"record_id": rid, "timestamp": ts, "symbol": sym})
                outcomes.append({"record_id": rid, "timestamp": ts, "symbol": sym,
                                 "cost_adjusted_forward_return": net, "trade_worthy": int(net > 0.0025),
                                 # The real outcome schema also carries the label's direction,
                                 # which shares its name with the model's sealed direction.
                                 "direction": "LONG" if net > 0.0025 else "NONE",
                                 "maximum_adverse_excursion": min(0, net) - 0.01,
                                 "maximum_favorable_excursion": max(0, net) + 0.01,
                                 "entry_price": 100.0, "exit_price": 100.0 * (1 + net), "_score": score})
    for name in ("inputs", "outcomes"):
        rows = inputs if name == "inputs" else [{k: v for k, v in o.items() if k != "_score"} for o in outcomes]
        (root / "dataset" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_fold = {}
    for o in outcomes:
        by_fold.setdefault(o["record_id"].split("-")[0], []).append(o)
    for fold, cls in FOLDS:
        if fold not in folds:
            continue
        for baseline in ("always_hold", "momentum_20", "logistic"):
            d = root / "runs" / fold / baseline
            d.mkdir(parents=True)
            preds = []
            for o in by_fold[fold]:
                p = {"always_hold": 0.0, "momentum_20": float(o["_score"] > 0),
                     "logistic": float(1 / (1 + np.exp(-o["_score"])))}[baseline]
                preds.append({"record_id": o["record_id"], "timestamp": o["timestamp"], "symbol": o["symbol"],
                              "trade_probability": p, "direction": "LONG" if p >= 0.5 else "NONE"})
            d.joinpath("predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
            windows.append({"fold": fold, "baseline": baseline, "window_class": cls,
                            "cost_sensitivity": [{"multiplier": m, "selected": 1, "sum_net_return_selected": 0.0,
                                                  "trade_label_rate": 0.5, "fraction_selected_positive": 1.0}
                                                 for m in (0.5, 1.0, 2.0)],
                            "financial": {"initial_cash": 3000.0, "final_equity": 3000.0, "fills": 0,
                                          "fees": 0.0, "turnover": 0.0, "max_drawdown": 0.0,
                                          "sharpe_weekly_sqrt52": None},
                            "classification": {"n": len(preds)}})
    (root / "experiment.json").write_text(json.dumps({
        "dataset_id": dataset_id, "spec_hash": spec_hash, "fold_ids": folds, "windows": windows}))
    return root


def _spec():
    return {"folds": [{"id": f, "window_class": c} for f, c in FOLDS], "probability_threshold": 0.5,
            "uncertainty": {"draws": 200, "seed": 2727, "level": 0.95},
            "evidence_sets": {"primary": ["new_historical", "sealed"]}}


def test_workers_join_only_on_one_dataset_and_spec(tmp_path):
    a = _worker(tmp_path / "a", ["2018Q1", "2018Q2"])
    b = _worker(tmp_path / "b", ["2024Q1", "2025Q4"])
    joined = load_workers([a, b], _spec())
    assert sorted(joined["folds"]) == [f for f, _ in FOLDS] and joined["dataset_identical"]
    c = _worker(tmp_path / "c", ["2024Q1", "2025Q4"], dataset_id="other")
    with pytest.raises(ValueError, match="dataset"):
        load_workers([a, c], _spec())
    with pytest.raises(ValueError, match="missing"):
        load_workers([a], _spec())


def test_report_keeps_classes_apart_and_applies_the_rule(tmp_path):
    a = _worker(tmp_path / "a", ["2018Q1", "2018Q2"])
    b = _worker(tmp_path / "b", ["2024Q1", "2025Q4"])
    report = build_report(load_workers([a, b], _spec()), _spec())
    assert set(report["by_class"]) == {"new_historical", "legacy", "sealed"}
    primary = report["primary"]
    assert primary["classes"] == ["new_historical", "sealed"]
    assert primary["logistic"]["auc"]["rows"] == 3 * 48       # legacy excluded
    assert primary["logistic"]["auc"]["dates"] == 36
    assert report["by_class"]["legacy"]["logistic"]["auc"]["rows"] == 48
    assert set(primary["leave_one_symbol_out"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert set(primary["leave_one_quarter_out"]) == {"2018Q1", "2018Q2", "2025Q4"}
    assert report["TASK_SIGNAL"] in {"PROMISING", "WEAK", "INCONCLUSIVE"}
    assert report["SEALED_CONFIRMATION"] in {"PASS", "INSUFFICIENT"}
    assert report["by_class"]["sealed"]["logistic"]["window_quintiles"][0]["count"] > 0
    assert report["by_class"]["new_historical"]["always_hold"]["classification"]["trade_coverage"] == 0.0


def test_a_strong_synthetic_signal_reads_promising_and_noise_does_not(tmp_path):
    strong = [_worker(tmp_path / "s1", ["2018Q1", "2018Q2"], signal=3.0, seed=1),
              _worker(tmp_path / "s2", ["2024Q1", "2025Q4"], signal=3.0, seed=1)]
    assert build_report(load_workers(strong, _spec()), _spec())["TASK_SIGNAL"] == "PROMISING"
    noise = [_worker(tmp_path / "n1", ["2018Q1", "2018Q2"], signal=0.0, seed=2),
             _worker(tmp_path / "n2", ["2024Q1", "2025Q4"], signal=0.0, seed=2)]
    assert build_report(load_workers(noise, _spec()), _spec())["TASK_SIGNAL"] != "PROMISING"


def test_the_hidden_label_direction_never_overwrites_the_model_direction(tmp_path):
    a = _worker(tmp_path / "a", ["2018Q1", "2018Q2"])
    b = _worker(tmp_path / "b", ["2024Q1", "2025Q4"])
    rows = load_workers([a, b], _spec())["rows"]
    hold = rows[rows["baseline"] == "always_hold"]
    assert (hold["direction"] == "NONE").all()              # the model's decision, sealed
    assert set(hold["label_direction"]) == {"LONG", "NONE"}   # the hidden label, joined after
