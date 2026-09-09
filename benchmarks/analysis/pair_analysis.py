#!/usr/bin/env python3
"""What the captured pairs say about training against the schema.

    python benchmarks/analysis/pair_analysis.py results/pairs_*.jsonl \
        --out results/pair_analysis.json

The question this serves is Phase 23's third deliverable: how much of a
response is schema-mandated, and how much is output the parser never reads.
That difference is what tuning a small model would remove, so it has to be a
number rather than an impression.

THREE LAYERS OF WASTE, MEASURED SEPARATELY
-------------------------------------------
1. Reasoning tokens the response never shows. ``output_tokens`` bills every
   token the model produced, including the thinking block; the visible text is
   usually far smaller. This is the layer Phase 22 showed can exhaust the
   ceiling and produce no answer at all.
2. JSON the parser reads. ``symbol``, ``side``/``action``, ``qty``, ``reason``.
3. JSON the parser ignores. Fields the shipped templates ask for and
   ``pipeline_output_to_decision`` never looks at -- ``order_type`` and
   ``limit_price`` -- plus anything else the model volunteers.

Only (2) is a training target. (1) and (3) are the removable part.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

# Fields pipeline_output_to_decision actually reads out of an order object.
# Confirmed by schema_probe against origin/main, not assumed.
FIELDS_READ = {"symbol", "side", "action", "qty", "quantity", "position_size",
               "confidence", "reason", "rationale", "stop_loss_price",
               "take_profit_price"}
# Asked for by every shipped template's final step; never read.
FIELDS_ASKED_BUT_IGNORED = {"order_type", "limit_price"}

# ~4 chars/token is the usual English/JSON approximation. It is only used for
# the visible-text share; every billed figure below comes from the provider's
# own usage numbers, so the estimate never enters a cost claim.
CHARS_PER_TOKEN = 4.0


def load(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def visible_token_estimate(text: Optional[str]) -> int:
    return int(round(len(text) / CHARS_PER_TOKEN)) if text else 0


def field_usage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which JSON keys the model actually emitted, and whether anything reads them."""
    emitted: Counter = Counter()
    for r in rows:
        txt = r.get("response_text")
        if not txt:
            continue
        try:
            obj = json.loads(txt)
        except Exception:  # noqa: BLE001
            continue
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    emitted[k] += 1
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
    read = {k: v for k, v in emitted.items() if k in FIELDS_READ}
    ignored_asked = {k: v for k, v in emitted.items()
                     if k in FIELDS_ASKED_BUT_IGNORED}
    return {
        "keys_emitted": dict(emitted.most_common()),
        "keys_read_by_parser": read,
        "keys_asked_for_but_ignored": ignored_asked,
        "n_distinct_keys": len(emitted),
    }


def token_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    billed = sum(r["output_tokens"] for r in rows)
    visible = sum(visible_token_estimate(r.get("response_text")) for r in rows)
    hidden = max(0, billed - visible)
    return {
        "n_pairs": len(rows),
        "billed_output_tokens": billed,
        "visible_text_tokens_est": visible,
        "not_shown_in_the_response_est": hidden,
        "hidden_share_est": hidden / billed if billed else None,
        "note": ("hidden = billed minus an estimate of the visible text; on a "
                 "reasoning model this is dominated by the thinking block"),
    }


def by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Dict[str, Any]] = {}
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get(key)].append(r)
    for k, g in groups.items():
        n = len(g)
        out[str(k)] = {
            "n": n,
            "parsed_json": sum(1 for r in g if r["parsed_json"]),
            "parse_rate": sum(1 for r in g if r["parsed_json"]) / n,
            "hit_ceiling": sum(1 for r in g if r["hit_output_ceiling"]),
            "ceiling_rate": sum(1 for r in g if r["hit_output_ceiling"]) / n,
            "no_text": sum(1 for r in g if not r.get("response_text")),
            "mean_output_tokens": sum(r["output_tokens"] for r in g) / n,
            "mean_prompt_chars": sum(r["prompt_chars"] for r in g) / n,
        }
    return out


def final_step_conversion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Only the LAST pipeline step is meant to convert to a decision.

    Intermediate steps emit ``facts`` and ``signals``, which
    ``pipeline_output_to_decision`` correctly declines. Counting them as
    failures would manufacture a defect that is not there.
    """
    per_bar: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["path"] == "pipeline":
            per_bar[(r["regime"], r["bar_index"])].append(r)
    finals, converted = 0, 0
    for _, g in per_bar.items():
        last = max(g, key=lambda r: (r["step_index"] or 0))
        finals += 1
        converted += 1 if last["converts_to_decision"] else 0
    singles = [r for r in rows if r["path"] == "single_prompt"]
    return {
        "pipeline_final_steps": finals,
        "pipeline_final_converted": converted,
        "pipeline_final_conversion_rate": converted / finals if finals else None,
        "single_prompt_calls": len(singles),
        "single_prompt_converted": sum(1 for r in singles
                                       if r["converts_to_decision"]),
    }


def crossed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Regime x path, because the two are otherwise confounded.

    The first capture ran the pipeline on flat and downtrend windows and the
    single-prompt path on flat and uptrend ones. Read as a regime column that
    says "uptrend parses 29%", which is really "the single-prompt path parses
    29%". Only a filled grid separates them.
    """
    cells: Dict[str, Dict[str, Any]] = {}
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("regime"), r.get("path"))].append(r)
    for (regime, path), g in groups.items():
        n = len(g)
        cells[f"{regime}|{path}"] = {
            "regime": regime, "path": path, "n": n,
            "parse_rate": sum(1 for r in g if r["parsed_json"]) / n,
            "ceiling_rate": sum(1 for r in g if r["hit_output_ceiling"]) / n,
            "no_text": sum(1 for r in g if not r.get("response_text")),
            "mean_output_tokens": sum(r["output_tokens"] for r in g) / n,
            "mean_prompt_chars": sum(r["prompt_chars"] for r in g) / n,
        }
    regimes = sorted({r.get("regime") for r in rows})
    paths = sorted({r.get("path") for r in rows})
    filled = sum(1 for rg in regimes for p in paths
                 if f"{rg}|{p}" in cells)
    return {"cells": cells, "regimes": regimes, "paths": paths,
            "cells_filled": filled, "cells_possible": len(regimes) * len(paths),
            "fully_crossed": filled == len(regimes) * len(paths)}


def build(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_pairs": len(rows),
        "by_path": by(rows, "path"),
        "by_regime": by(rows, "regime"),
        "by_step_label": by(rows, "step_label"),
        "crossed": crossed(rows),
        "token_split": token_split(rows),
        "field_usage": field_usage(rows),
        "conversion": final_step_conversion(rows),
        "content_type_shapes": dict(Counter(
            str(r.get("content_types")) for r in rows).most_common()),
        "no_text_responses": sum(1 for r in rows if not r.get("response_text")),
        "ceiling_hits": sum(1 for r in rows if r["hit_output_ceiling"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows = load(args.paths)
    if not rows:
        print("no pairs found", file=sys.stderr)
        return 2
    d = build(rows)

    print(f"{d['n_pairs']} pairs")
    ts = d["token_split"]
    print(f"\ntokens: {ts['billed_output_tokens']} billed, "
          f"~{ts['visible_text_tokens_est']} visible in the text, "
          f"~{ts['hidden_share_est']:.0%} never shown")
    print(f"\nno-text responses: {d['no_text_responses']}   "
          f"ceiling hits: {d['ceiling_hits']}")
    print(f"content shapes: {d['content_type_shapes']}")
    for name in ("by_path", "by_regime", "by_step_label"):
        print(f"\n{name}:")
        for k, v in sorted(d[name].items()):
            print(f"  {str(k)[:28]:28s} n={v['n']:3d} parse={v['parse_rate']:.0%} "
                  f"ceiling={v['ceiling_rate']:.0%} "
                  f"out_tok={v['mean_output_tokens']:.0f} "
                  f"prompt_chars={v['mean_prompt_chars']:.0f}")
    cr = d["crossed"]
    print(f"\nregime x path ({cr['cells_filled']}/{cr['cells_possible']} cells "
          f"filled, fully_crossed={cr['fully_crossed']}):")
    print(f"  {'regime':11s} {'path':14s} {'n':>4s} {'parse':>7s} {'ceiling':>8s} "
          f"{'out_tok':>8s} {'prompt_ch':>10s}")
    for k in sorted(cr["cells"]):
        v = cr["cells"][k]
        print(f"  {v['regime']:11s} {v['path']:14s} {v['n']:4d} "
              f"{v['parse_rate']:7.0%} {v['ceiling_rate']:8.0%} "
              f"{v['mean_output_tokens']:8.0f} {v['mean_prompt_chars']:10.0f}")

    c = d["conversion"]
    print(f"\nfinal-step conversion: {c['pipeline_final_converted']}/"
          f"{c['pipeline_final_steps']} pipeline bars; "
          f"{c['single_prompt_converted']}/{c['single_prompt_calls']} "
          f"single-prompt calls")
    fu = d["field_usage"]
    print(f"\nkeys emitted: {fu['n_distinct_keys']} distinct")
    print(f"  read by the parser : {fu['keys_read_by_parser']}")
    print(f"  asked for, ignored : {fu['keys_asked_for_but_ignored']}")

    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(
            _BENCH_ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1, default=str)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
