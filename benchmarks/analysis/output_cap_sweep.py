#!/usr/bin/env python3
"""Measure what an output-token cap costs and what it breaks.

    python benchmarks/analysis/output_cap_sweep.py --dry-run
    python benchmarks/analysis/output_cap_sweep.py --integration openrouter \
        --models nvidia/nemotron-3-nano-30b-a3b --caps 2000 1000 500 250 -n 12

WHY CAPS AND NOT SOMETHING ELSE
---------------------------------
Output is priced ~4x input across the roster, and the cost structure INVERTS
between tiers: input is 61.5% of Nemotron's per-call cost but output is 77.4%
of GPT-5.5's [MEASURED, load_measured]. So a cap attacks the dominant term
exactly where the money is.

The underlying driver is output VOLUME, not tier. Input is roughly constant
across the roster (3895-5502 tokens/call) while output spans 860 to 5005 — a
5.8x range. The verbose models happen to be the expensive ones, which is what
produces the inversion.

WHAT A CAP ACTUALLY DOES
-------------------------
``max_tokens`` truncates. It does not persuade a model to be concise. A
truncated JSON object does not parse, and ``pipeline_runner`` aborts the whole
decision on an unparseable step with no retry. So the measurement that matters
is not "how many tokens did we save" but "at what cap does the parse rate fall
off a cliff", and the two must be read together.

This sweep reuses the production parser and prompt builder by import. Measuring
with a different parser would produce a number about this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.infrastructure.llm.backtest_harness import (  # noqa: E402
    extract_response_text, extract_token_usage, parse_llm_response,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (  # noqa: E402
    PIPELINE_SYSTEM_PROMPT, _build_step_prompt,
)

MARKETPLACE = os.path.join(_REPO_ROOT, "dashboard", "config", "marketplace.json")
DEFAULT_TEMPLATE = "pipeline-analyst"
DEFAULT_CAPS = (2000, 1000, 500, 250)

# Same fixed snapshot the Phase 16 harness used, so cap results are comparable
# to the uncapped baseline it measured rather than to a different workload.
REFERENCE_SNAPSHOT: Dict[str, Any] = {
    "timestamp": "2026-04-16T14:00:00",
    "portfolio": {"cash": 4820.55, "positions_value": 5310.20,
                  "total_equity": 10130.75, "num_positions": 3},
    "current_holdings": {
        "AAPL": {"shares": 8, "avg_price": 214.30, "current_price": 219.885,
                 "position_value": 1759.08, "pnl_pct": 2.61},
        "MSFT": {"shares": 4, "avg_price": 402.10, "current_price": 397.22,
                 "position_value": 1588.88, "pnl_pct": -1.21},
        "JPM": {"shares": 9, "avg_price": 218.44, "current_price": 217.36,
                "position_value": 1956.24, "pnl_pct": -0.49}},
    "recent_trades": [{"symbol": "AAPL", "side": "buy", "shares": 3,
                       "price": 216.4, "timestamp": "2026-04-16T10:00:00"}],
    "top_signals": {
        "AAPL": {"price": 219.885, "rsi": 61.4, "sma_20": 213.2,
                 "volume_ratio": 1.18, "momentum_pct": 2.9},
        "MSFT": {"price": 397.22, "rsi": 44.1, "sma_20": 404.8,
                 "volume_ratio": 0.92, "momentum_pct": -1.6},
        "JPM": {"price": 217.36, "rsi": 51.8, "sma_20": 216.1,
                "volume_ratio": 1.04, "momentum_pct": 0.4},
        "NVDA": {"price": 118.72, "rsi": 68.9, "sma_20": 109.4,
                 "volume_ratio": 1.63, "momentum_pct": 6.1},
        "XOM": {"price": 112.05, "rsi": 38.2, "sma_20": 115.9,
                "volume_ratio": 1.11, "momentum_pct": -2.8}}}


def load_steps(template: str = DEFAULT_TEMPLATE) -> List[Dict[str, Any]]:
    with open(MARKETPLACE, "r", encoding="utf-8") as fh:
        for tpl in json.load(fh)["templates"]:
            if tpl.get("template_id") == template:
                return list(tpl.get("pipeline") or [])
    raise SystemExit(f"no template {template!r}")


def reference_context(steps) -> List[Dict[str, Any]]:
    out = []
    for i, step in enumerate(steps):
        fmt = (step.get("outputFormat") or "").strip()
        try:
            a, b = fmt.find("{"), fmt.rfind("}") + 1
            ph = json.loads(fmt[a:b]) if a >= 0 and b > a else None
        except Exception:  # noqa: BLE001
            ph = None
        out.append({"step": i + 1, "label": step.get("label"),
                    "presetKey": step.get("presetKey"), "id": step.get("id"),
                    "output": ph if isinstance(ph, dict) else {"summary": "x"}})
    return out


def classify(text: str, parsed) -> str:
    """Truncation is the failure a cap causes; it must be named separately."""
    s = (text or "").strip()
    if not s:
        return "empty"
    if parsed is not None:
        return "ok"
    if "{" in s and "}" not in s:
        return "truncated"
    if "{" not in s:
        return "no_json"
    return "malformed_json"


def sweep(client, steps, model: str, caps, attempts: int,
          step_indices=(0, 1), verbose: bool = True,
          on_row=None) -> List[Dict[str, Any]]:
    """One row per (model, step, cap).

    Only steps 1-2 by default: step 3 returns a valid empty-orders no-trade
    that production treats as a failure (Phase 17), so its parse rate measures
    that defect rather than the cap.
    """
    ref = reference_context(steps)
    rows = []
    for idx in step_indices:
        prompt = _build_step_prompt(
            step_index=idx, step=steps[idx], market_snapshot=REFERENCE_SNAPSHOT,
            prior_outputs=ref[:idx], is_last=False)
        for cap in caps:
            cats, outs, ins = Counter(), [], []
            for _ in range(attempts):
                try:
                    resp = client.messages.create(
                        model=model, max_tokens=cap,
                        system=PIPELINE_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}])
                except Exception as exc:  # noqa: BLE001
                    cats[f"api_error:{type(exc).__name__}"] += 1
                    continue
                i_tok, o_tok = extract_token_usage(resp)
                ins.append(i_tok)
                outs.append(o_tok)
                try:
                    text = extract_response_text(resp)
                except AttributeError:
                    text = ""
                cats[classify(text, parse_llm_response(text) if text else None)] += 1
            n_ok = cats.get("ok", 0)
            rows.append({
                "model": model, "step_index": idx,
                "step_label": steps[idx].get("label"), "max_tokens": cap,
                "attempts": attempts, "n_ok": n_ok,
                "parse_rate": n_ok / attempts if attempts else None,
                "mean_output_tokens": (sum(outs) / len(outs)) if outs else None,
                "mean_input_tokens": (sum(ins) / len(ins)) if ins else None,
                "categories": dict(cats)})
            if verbose:
                print(f"    step{idx + 1} cap={cap:<5} {n_ok}/{attempts} "
                      f"out={rows[-1]['mean_output_tokens'] or 0:.0f}  {dict(cats)}",
                      flush=True)
            # Flush after every cell: a sweep is minutes of paid calls and an
            # interruption at cell 15 of 16 should not discard the other 14.
            if on_row is not None:
                on_row(rows[-1])
    return rows


def estimate_ceiling(steps, models, caps, attempts, step_indices=(0, 1)) -> Dict[str, Any]:
    """Upper bound on spend: every call assumed to hit its cap."""
    from analysis.cost_model_lib import PRICE_BY_DB_MODEL  # noqa: PLC0415

    by_slug = {v["slug"]: v for v in PRICE_BY_DB_MODEL.values()}
    ref = reference_context(steps)
    in_tok = {}
    for idx in step_indices:
        p = _build_step_prompt(step_index=idx, step=steps[idx],
                               market_snapshot=REFERENCE_SNAPSHOT,
                               prior_outputs=ref[:idx], is_last=False)
        in_tok[idx] = int((len(p) + len(PIPELINE_SYSTEM_PROMPT)) / 3.8)

    rows, total = [], 0.0
    for model in models:
        price = by_slug.get(model)
        if price is None:
            rows.append({"model": model, "priced": False, "max_cost_usd": None})
            continue
        c = 0.0
        for idx in step_indices:
            for cap in caps:
                c += attempts * ((in_tok[idx] / 1e6) * price["in"]
                                 + (cap / 1e6) * price["out"])
        rows.append({"model": model, "priced": True, "max_cost_usd": c})
        total += c
    return {"rows": rows, "total_max_cost_usd": total,
            "calls": len(models) * len(step_indices) * len(caps) * attempts,
            "assumption": "every call assumed to emit its full cap — pessimistic"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Output-token cap sweep.")
    ap.add_argument("--models", nargs="+",
                    default=["nvidia/nemotron-3-nano-30b-a3b"])
    ap.add_argument("--caps", nargs="+", type=int, default=list(DEFAULT_CAPS))
    ap.add_argument("-n", "--attempts", type=int, default=12)
    ap.add_argument("--integration", default="openrouter")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 2],
                    help="1-based step numbers to sweep (default 1 2; step 3 "
                         "returns a valid empty-orders no-trade that "
                         "production treats as a failure, so its parse rate "
                         "measures that defect rather than the cap)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.attempts > 50:
        print(f"refusing -n {args.attempts}: this sweep is sized for tens.",
              file=sys.stderr)
        return 2
    steps = load_steps(args.template)
    est = estimate_ceiling(steps, args.models, args.caps, args.attempts,
                           tuple(i - 1 for i in args.steps))

    print(f"CAP SWEEP  template={args.template}  caps={args.caps}  "
          f"n={args.attempts}  calls={est['calls']}")
    for r in est["rows"]:
        c = f"{r['max_cost_usd']:.4f}" if r["priced"] else "unpriced"
        print(f"  {r['model'][:44]:46s} max ${c}")
    print(f"  TOTAL max ${est['total_max_cost_usd']:.4f}  ({est['assumption']})")

    if args.dry_run:
        print("\n--dry-run: nothing called, nothing spent.")
        return 0
    if not args.yes:
        if input(f"\nSpend up to ${est['total_max_cost_usd']:.4f}? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            print("Aborted; nothing spent.")
            return 1

    from dashboard.backend.infrastructure.llm.providers import (  # noqa: PLC0415
        make_llm_client)
    client = make_llm_client(args.integration)
    if client is None:
        print(f"\nno client for integration={args.integration!r}: its API key "
              f"is not set, so no call was attempted.", file=sys.stderr)
        return 3

    step_indices = tuple(i - 1 for i in args.steps)
    all_rows: List[Dict[str, Any]] = []

    def _flush(row=None):
        # The row is appended HERE rather than relying on the extend() below:
        # sweep() accumulates into its own list and only returns at the end, so
        # flushing all_rows alone would write an empty file for the whole run.
        if row is not None:
            all_rows.append(row)
        if not args.out:
            return
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"template": args.template, "caps": args.caps,
                       "attempts": args.attempts, "integration": args.integration,
                       "steps": args.steps, "cost_ceiling": est,
                       "rows": all_rows}, fh, indent=1)

    for model in args.models:
        print(f"\n  {model}")
        sweep(client, steps, model, args.caps, args.attempts,
              step_indices=step_indices, on_row=_flush)
    _flush()
    if args.out:
        print(f"\n  wrote {args.out} ({len(all_rows)} cells)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
