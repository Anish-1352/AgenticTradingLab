#!/usr/bin/env python3
"""What the decision schema actually is, established by exercising it.

    DATABASE_PATH=$PWD/local_atl.db python benchmarks/analysis/schema_probe.py \
        --out results/schema_probe.json

Phase 23 asked for a specification. Writing one by reading the code would
produce a tidy document that may not match the parser; this calls the parser
instead and records what it does, so every row is MEASURED.

Read-only: it imports the parsers and the shipped template config and calls
them on constructed inputs. It issues no network calls and writes nothing under
``dashboard/``.

THERE IS NOT ONE PARSER
------------------------
Three code paths turn model output into a decision, and they do not agree:

* ``pipeline_output_to_decision`` -- multi-step pipeline. Permissive and
  coercive: unknown ``side`` becomes ``hold``, an unparseable ``qty`` becomes
  ``0``.
* ``portfolio_manager.make_trading_decision_with_llm`` -- single-prompt path.
  Consumes ``actions`` directly and applies its own rules downstream.
* ``parse_actions_payload`` / ``LLMTradingDecision`` -- external agents and the
  AI-hedge-fund adapter. A pydantic model that rejects rather than coerces.

A training target has to name which of these it is aimed at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED_DB = os.path.join(_REPO_ROOT, "dashboard", "storage", "data", "backtest.db")
SEED_DB_SHA256 = "414bf53cb056b2c60bbdf4d963dffd1ffd7998fcb6235cf9bda963bd52504c80"
MARKETPLACE = os.path.join(_REPO_ROOT, "dashboard", "config", "marketplace.json")


def tree_revision() -> Dict[str, Any]:
    """Which tree this ran against.

    The first run of this probe used the local checkout, 533 commits behind
    ``origin/main``, and reported the pre-fix parser behaviour as current. The
    revision goes in the output so a reader can tell which code a row describes.
    """
    import subprocess
    def _g(*a):
        r = subprocess.run(["git", "-C", _REPO_ROOT, *a],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    head = _g("rev-parse", "HEAD")
    return {
        "head": head,
        "head_subject": _g("log", "-1", "--format=%s"),
        "is_origin_main": head == _g("rev-parse", "origin/main"),
        "behind_origin_main": _g("rev-list", "--count", "HEAD..origin/main"),
        "marketplace_path": MARKETPLACE,
    }


def parser_agreement(pipeline_rows, pydantic_rows) -> Dict[str, Any]:
    """Where the permissive and strict parsers disagree on the same payload."""
    by_case = {r["case"]: r for r in pydantic_rows}
    both, only_pipeline, only_pydantic, neither = [], [], [], []
    for r in pipeline_rows:
        o = by_case.get(r["case"])
        if o is None:
            continue
        if r["accepted"] and o["accepted"]:
            both.append(r["case"])
        elif r["accepted"]:
            only_pipeline.append(r["case"])
        elif o["accepted"]:
            only_pydantic.append(r["case"])
        else:
            neither.append(r["case"])
    n = len(both) + len(only_pipeline) + len(only_pydantic) + len(neither)
    return {
        "compared": n,
        "accepted_by_both": len(both),
        "accepted_by_pipeline_only": len(only_pipeline),
        "accepted_by_pydantic_only": len(only_pydantic),
        "rejected_by_both": len(neither),
        "agreement_rate": (len(both) + len(neither)) / n if n else None,
        "pipeline_only_cases": only_pipeline,
        "pydantic_only_cases": only_pydantic,
    }


def envelope_output_shapes(rows) -> Dict[str, Any]:
    """The same parser emits different action shapes per envelope.

    ``actions`` is returned verbatim; ``orders`` and ``risk_actions`` are
    normalised into a seven-field shape. A consumer -- or a model being tuned
    to satisfy one -- cannot rely on a single output contract.
    """
    out = {}
    for r in rows:
        if r["group"] != "envelope" or not r["accepted"]:
            continue
        acts = (r["parsed"] or {}).get("actions") or []
        if acts and isinstance(acts[0], dict):
            out[r["case"]] = sorted(acts[0].keys())
    shapes = {tuple(v) for v in out.values()}
    return {"per_case_keys": out, "distinct_shapes": len(shapes)}


def guard_seed_db() -> None:
    target = os.getenv("DATABASE_PATH")
    if not target:
        raise SystemExit(
            "DATABASE_PATH is not set. Importing the backend runs migrations "
            "at import time against the default path, which is the committed "
            "seed DB.\n  export DATABASE_PATH=\"$PWD/local_atl.db\"")
    if os.path.abspath(target) == os.path.abspath(SEED_DB):
        raise SystemExit("DATABASE_PATH points at the committed seed DB.")


def seed_db_sha256() -> str:
    import hashlib
    h = hashlib.sha256()
    with open(SEED_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- the cases --
# Each case is (group, name, payload). Grouping keeps the report readable and
# separates "does this envelope parse" from "what does it do to my fields".
def cases() -> List[tuple]:
    ok = {"symbol": "AAPL", "side": "buy", "qty": 3}
    return [
        # envelopes
        ("envelope", "actions, non-empty", {"actions": [{"action": "buy", "symbol": "AAPL"}]}),
        ("envelope", "orders, non-empty", {"orders": [ok]}),
        ("envelope", "risk_actions, non-empty", {"risk_actions": [{"symbol": "AAPL", "action": "stop_loss"}]}),
        ("envelope", "actions, empty", {"actions": []}),
        ("envelope", "orders, empty", {"orders": []}),
        ("envelope", "risk_actions, empty", {"risk_actions": []}),
        ("envelope", "unknown key only", {"decisions": [ok]}),
        ("envelope", "empty object", {}),
        ("envelope", "a list, not an object", ["x"]),
        ("envelope", "orders is not a list", {"orders": {"symbol": "AAPL"}}),
        ("envelope", "orders of non-objects", {"orders": [1, 2, 3]}),
        ("envelope", "template shape verbatim", {"orders": [{
            "symbol": "AAPL", "side": "buy", "qty": 2, "order_type": "market",
            "limit_price": None, "reason": "trend"}]}),
        # side / action coercion
        ("side", "side=buy", {"orders": [dict(ok, side="buy")]}),
        ("side", "side=SELL (uppercase)", {"orders": [dict(ok, side="SELL")]}),
        ("side", "side=hold", {"orders": [dict(ok, side="hold")]}),
        ("side", "side=short (unknown)", {"orders": [dict(ok, side="short")]}),
        ("side", "side=liquidate everything", {"orders": [dict(ok, side="liquidate everything")]}),
        ("side", "side missing", {"orders": [{"symbol": "AAPL", "qty": 3}]}),
        ("side", "action= used instead of side=", {"orders": [{"symbol": "AAPL", "action": "sell", "qty": 3}]}),
        ("side", "side=None", {"orders": [dict(ok, side=None)]}),
        # qty coercion
        ("qty", "qty=3", {"orders": [dict(ok, qty=3)]}),
        ("qty", "qty='7' (string)", {"orders": [dict(ok, qty="7")]}),
        ("qty", "qty=2.9 (float)", {"orders": [dict(ok, qty=2.9)]}),
        ("qty", "qty=-5 (negative)", {"orders": [dict(ok, qty=-5)]}),
        ("qty", "qty=1e9 (huge)", {"orders": [dict(ok, qty=10**9)]}),
        ("qty", "qty='many' (unparseable)", {"orders": [dict(ok, qty="many")]}),
        ("qty", "qty missing", {"orders": [{"symbol": "AAPL", "side": "buy"}]}),
        ("qty", "quantity= alias", {"orders": [{"symbol": "AAPL", "side": "buy", "quantity": 4}]}),
        ("qty", "position_size= alias", {"orders": [{"symbol": "AAPL", "side": "buy", "position_size": 5}]}),
        # symbol
        ("symbol", "symbol missing", {"orders": [{"side": "buy", "qty": 1}]}),
        ("symbol", "symbol=None", {"orders": [dict(ok, symbol=None)]}),
        ("symbol", "symbol not in DJIA", {"orders": [dict(ok, symbol="NOTAREALTICKER")]}),
        ("symbol", "symbol lowercase", {"orders": [dict(ok, symbol="aapl")]}),
        ("symbol", "symbol is a number", {"orders": [dict(ok, symbol=42)]}),
        # confidence
        ("confidence", "confidence missing", {"orders": [ok]}),
        ("confidence", "confidence=2.5 (out of range)", {"orders": [dict(ok, confidence=2.5)]}),
        ("confidence", "confidence=-1", {"orders": [dict(ok, confidence=-1)]}),
        ("confidence", "confidence='high'", {"orders": [dict(ok, confidence="high")]}),
        # extras
        ("extras", "unknown extra field", {"orders": [dict(ok, wibble="ignored?")]}),
        ("extras", "order_type / limit_price (template asks for these)",
         {"orders": [dict(ok, order_type="limit", limit_price=123.4)]}),
    ]


def run_pipeline_parser() -> List[Dict[str, Any]]:
    from dashboard.backend.infrastructure.llm.pipeline_runner import (
        pipeline_output_to_decision as P)
    rows = []
    for group, name, payload in cases():
        try:
            got = P(payload)
            err = None
        except Exception as exc:  # noqa: BLE001
            got, err = None, f"{type(exc).__name__}: {exc}"
        rows.append({
            "group": group, "case": name, "input": payload,
            "parsed": got, "error": err,
            "accepted": got is not None,
            "n_actions": len(got["actions"]) if isinstance(got, dict) else None,
        })
    return rows


def run_pydantic_parser() -> List[Dict[str, Any]]:
    """The strict path, for contrast. Takes only the ``actions`` envelope."""
    from dashboard.backend.infrastructure.llm.validator import (
        parse_actions_payload)
    rows = []
    for group, name, payload in cases():
        if not isinstance(payload, dict):
            continue
        # Translate an orders envelope into the actions envelope this parser
        # expects, so the two are compared on the same content.
        p = payload
        if "orders" in payload and isinstance(payload["orders"], list):
            p = {"actions": [
                {"action": o.get("side") or o.get("action"),
                 "symbol": o.get("symbol"),
                 "confidence": o.get("confidence", 0.75),
                 "reasoning": o.get("reason", ""),
                 "position_size": o.get("qty", o.get("quantity", 0))}
                if isinstance(o, dict) else o
                for o in payload["orders"]]}
        try:
            decisions, err = parse_actions_payload(p)
        except Exception as exc:  # noqa: BLE001
            decisions, err = None, f"{type(exc).__name__}: {exc}"
        rows.append({
            "group": group, "case": name,
            "accepted": decisions is not None,
            "error": err,
            "n_actions": len(decisions) if decisions is not None else None,
        })
    return rows


def template_formats() -> Dict[str, Any]:
    with open(MARKETPLACE, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out = []
    for t in cfg.get("templates", []):
        steps = t.get("pipeline") or []
        rows = []
        for i, s in enumerate(steps):
            fmt = (s.get("outputFormat") or "").strip()
            env = next((k for k in ("orders", "actions", "risk_actions",
                                    "signals", "facts")
                        if f'"{k}"' in fmt), None)
            rows.append({"index": i, "label": s.get("label"),
                         "envelope": env, "outputFormat": fmt})
        out.append({"name": t.get("name"), "n_steps": len(steps),
                    "steps": rows,
                    "final_envelope": rows[-1]["envelope"] if rows else None})
    finals = [t["final_envelope"] for t in out if t["final_envelope"]]
    return {"templates": out,
            "distinct_final_envelopes": sorted(set(finals)),
            "all_templates_agree_on_final": len(set(finals)) <= 1}


def ignored_fields() -> Dict[str, Any]:
    """Fields the shipped templates ask the model for that no parser reads."""
    from dashboard.backend.infrastructure.llm import pipeline_runner as pr
    import inspect
    src = inspect.getsource(pr.pipeline_output_to_decision)
    asked = ["symbol", "side", "qty", "order_type", "limit_price", "reason"]
    return {f: (f in src) for f in asked}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    guard_seed_db()
    before = seed_db_sha256()

    payload = {
        "tree": tree_revision(),
        "seed_db_sha256_before": before,
        "pipeline_parser": run_pipeline_parser(),
        "pydantic_parser": run_pydantic_parser(),
        "templates": template_formats(),
        "template_fields_read_by_pipeline_parser": ignored_fields(),
    }
    payload["parser_agreement"] = parser_agreement(
        payload["pipeline_parser"], payload["pydantic_parser"])
    payload["envelope_output_shapes"] = envelope_output_shapes(
        payload["pipeline_parser"])

    after = seed_db_sha256()
    if after != before or after != SEED_DB_SHA256:
        raise AssertionError("seed DB changed during the probe")
    payload["seed_db_sha256_after"] = after

    pp = payload["pipeline_parser"]
    print(f"pipeline parser: {sum(1 for r in pp if r['accepted'])}/{len(pp)} "
          f"cases accepted")
    for r in pp:
        mark = "accept" if r["accepted"] else "REJECT"
        extra = ""
        if r["accepted"] and r["n_actions"]:
            a = r["parsed"]["actions"][0]
            if isinstance(a, dict):
                extra = (f"  -> action={a.get('action')!r} "
                         f"symbol={a.get('symbol')!r} "
                         f"size={a.get('position_size')!r}")
        print(f"  [{mark}] {r['group']:10s} {r['case']:44s}{extra}")

    ag = payload["parser_agreement"]
    print(f"\nparser agreement: {ag['agreement_rate']:.0%} over "
          f"{ag['compared']} payloads "
          f"({ag['accepted_by_pipeline_only']} accepted by the permissive "
          f"parser only)")
    es = payload["envelope_output_shapes"]
    print(f"distinct action shapes out of ONE parser: {es['distinct_shapes']}")
    for k, v in es["per_case_keys"].items():
        print(f"  {k:26s} {v}")

    tr = payload["tree"]
    print(f"\ntree: {tr['head'][:8]} (origin/main={tr['is_origin_main']}, "
          f"behind={tr['behind_origin_main']})")

    t = payload["templates"]
    print(f"\ntemplates: {len(t['templates'])}; distinct final envelopes "
          f"{t['distinct_final_envelopes']}; agree={t['all_templates_agree_on_final']}")
    print(f"template fields the pipeline parser reads: "
          f"{payload['template_fields_read_by_pipeline_parser']}")

    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(
            _BENCH_ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
