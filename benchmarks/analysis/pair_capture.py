#!/usr/bin/env python3
"""Capture real prompt/response pairs from ATL's own LLM traffic.

    DATABASE_PATH=$PWD/local.db python benchmarks/analysis/pair_capture.py \
        --dry-run
    DATABASE_PATH=$PWD/local.db python benchmarks/analysis/pair_capture.py \
        --yes --mode pipeline --start 2026-04-01 --end 2026-04-03 \
        --out results/pairs_pipeline.jsonl

WHY NOT THE EXISTING FIXTURE
-----------------------------
``atl_realistic`` records in its own metadata that it was built from default
proportions and is "NOT derived from a measurement". A training-target study
cannot rest on assumed traffic, so this records what the pipeline actually
sends and actually gets back.

WHAT A PAIR CARRIES
--------------------
The prompt text as sent, the raw response as received, the step's configured
``outputFormat``, whether each parser accepted it and why not, token counts,
and whether the reply hit the output ceiling -- which Phase 22 established is
the signature of the dominant failure (all 32 observed failures returned at
exactly ``max_tokens``).

Read-only against ``dashboard/``: everything here is in-process monkeypatching,
and the seed DB is asserted unchanged before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED_DB = os.path.join(_REPO_ROOT, "dashboard", "storage", "data", "backtest.db")
SEED_DB_SHA256 = "414bf53cb056b2c60bbdf4d963dffd1ffd7998fcb6235cf9bda963bd52504c80"
MARKETPLACE = os.path.join(_REPO_ROOT, "dashboard", "config", "marketplace.json")


def guard_seed_db() -> None:
    target = os.getenv("DATABASE_PATH")
    if not target:
        raise SystemExit(
            "DATABASE_PATH is not set; the backend migrates at import time "
            "against the committed seed DB.")
    if os.path.abspath(target) == os.path.abspath(SEED_DB):
        raise SystemExit("DATABASE_PATH points at the committed seed DB.")


def seed_db_sha256() -> str:
    import hashlib
    h = hashlib.sha256()
    with open(SEED_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_template(name: str) -> List[Dict[str, Any]]:
    with open(MARKETPLACE, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for t in cfg.get("templates", []):
        if (t.get("name") or "").lower() == name.lower():
            return t.get("pipeline") or []
    raise SystemExit(
        f"template {name!r} not found; have: "
        + ", ".join(repr(t.get("name")) for t in cfg.get("templates", [])))


_PATCHES: List[tuple] = []


def _set(owner, attr, value):
    _PATCHES.append((owner, attr, getattr(owner, attr)))
    setattr(owner, attr, value)


def restore() -> None:
    for owner, attr, original in reversed(_PATCHES):
        setattr(owner, attr, original)
    _PATCHES.clear()


class Capture:
    def __init__(self, mode: str, regime: str, sink) -> None:
        self.mode = mode
        self.regime = regime
        self.sink = sink
        self.pairs: List[Dict[str, Any]] = []
        # prompt text -> the step that produced it, so a response can be
        # attributed to a step without threading state through the runner.
        self._step_by_prompt: Dict[str, Dict[str, Any]] = {}
        self.bar_index = -1

    def note_step(self, prompt: str, step: Dict[str, Any], index: int) -> None:
        self._step_by_prompt[prompt] = {
            "step_index": index,
            "step_label": step.get("label"),
            "step_preset": step.get("presetKey"),
            "output_format": (step.get("outputFormat") or "").strip(),
        }

    def record(self, *, prompt: str, response, seconds: float,
               max_tokens: Optional[int], path: str) -> None:
        from dashboard.backend.infrastructure.llm.pipeline_runner import (
            _hit_output_ceiling, response_text_or_none, truncation_reason,
            pipeline_output_to_decision)
        from dashboard.backend.infrastructure.llm.backtest_harness import (
            DEFAULT_MAX_OUTPUT_TOKENS, extract_token_usage, parse_llm_response)

        try:
            in_tok, out_tok = extract_token_usage(response)
        except Exception:  # noqa: BLE001
            in_tok, out_tok = 0, 0
        text = response_text_or_none(response)
        cap = max_tokens or DEFAULT_MAX_OUTPUT_TOKENS

        parsed = None
        parse_error = None
        if text is not None:
            try:
                parsed = parse_llm_response(text)
            except Exception as exc:  # noqa: BLE001
                parse_error = f"{type(exc).__name__}: {exc}"

        decision = None
        decision_error = None
        if isinstance(parsed, dict):
            try:
                decision = pipeline_output_to_decision(parsed)
            except Exception as exc:  # noqa: BLE001
                decision_error = f"{type(exc).__name__}: {exc}"

        content_types = [getattr(b, "type", None)
                         for b in (getattr(response, "content", None) or [])]

        meta = self._step_by_prompt.get(prompt, {})
        row = {
            "mode": self.mode,
            "regime": self.regime,
            "path": path,
            "bar_index": self.bar_index,
            "step_index": meta.get("step_index"),
            "step_label": meta.get("step_label"),
            "output_format": meta.get("output_format"),
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "response_text": text,
            "response_chars": len(text) if text else 0,
            "content_types": content_types,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "max_output_tokens": cap,
            "hit_output_ceiling": bool(_hit_output_ceiling(response, out_tok, cap)),
            "truncation_reason": truncation_reason(response, out_tok, text or "")
            if text is not None else None,
            "parsed_json": parsed is not None,
            "parse_error": parse_error,
            "converts_to_decision": decision is not None,
            "decision_error": decision_error,
            "n_actions": len(decision["actions"]) if isinstance(decision, dict) else None,
            "seconds": seconds,
        }
        self.pairs.append(row)
        if self.sink:
            self.sink.write(json.dumps(row, default=str) + "\n")
            self.sink.flush()


def instrument(cap: Capture, deadline: Optional[float] = None) -> Dict[str, bool]:
    from dashboard.backend.infrastructure.llm import pipeline_runner as pr
    from dashboard.backend.domain.backtesting import portfolio_manager as pm

    attached: Dict[str, bool] = {}

    # --- pipeline path -----------------------------------------------------
    orig_build = pr._build_step_prompt
    def build(*, step_index, step, market_snapshot, prior_outputs, is_last):
        prompt = orig_build(step_index=step_index, step=step,
                            market_snapshot=market_snapshot,
                            prior_outputs=prior_outputs, is_last=is_last)
        cap.note_step(prompt, step, step_index)
        return prompt
    _set(pr, "_build_step_prompt", build)
    attached["build_step_prompt"] = True

    orig_create = pr._create_pipeline_response
    def create(client, *, model, prompt, **kw):
        if deadline and time.perf_counter() > deadline:
            raise RuntimeError("capture budget exhausted")
        t0 = time.perf_counter()
        resp = orig_create(client, model=model, prompt=prompt, **kw)
        cap.record(prompt=prompt, response=resp,
                   seconds=time.perf_counter() - t0,
                   max_tokens=kw.get("max_tokens"), path="pipeline")
        return resp
    _set(pr, "_create_pipeline_response", create)
    attached["pipeline_response"] = True

    # --- single-prompt path ------------------------------------------------
    orig_req = pm._request_trading_decision
    def req(client, *, prompt, model=None, max_tokens=None, temperature=None,
            market_context=None):
        if deadline and time.perf_counter() > deadline:
            raise RuntimeError("capture budget exhausted")
        t0 = time.perf_counter()
        resp = orig_req(client, prompt=prompt, model=model,
                        max_tokens=max_tokens, temperature=temperature,
                        market_context=market_context)
        cap.record(prompt=prompt, response=resp,
                   seconds=time.perf_counter() - t0,
                   max_tokens=max_tokens, path="single_prompt")
        return resp
    _set(pm, "_request_trading_decision", req)
    attached["single_prompt_request"] = True

    orig_state = pm.PortfolioManager.get_portfolio_state
    def state(self, *a, **kw):
        cap.bar_index += 1
        return orig_state(self, *a, **kw)
    _set(pm.PortfolioManager, "get_portfolio_state", state)
    attached["bar_boundary"] = True
    return attached


def estimate(mode: str, start: str, end: str, model: str, n_steps: int) -> Dict[str, Any]:
    from datetime import date
    from analysis.cost_model_lib import PRICE_BY_DB_MODEL
    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    span = (date(y1, m1, d1) - date(y0, m0, d0)).days or 1
    bars = max(1, int(span * 5 / 7)) * 7
    calls = bars * (n_steps if mode == "pipeline" else 1)
    p = {v["slug"]: v for v in PRICE_BY_DB_MODEL.values()}.get(model)
    # Pessimistic: assume every call retries to the loop's cap.
    worst = calls * 5
    cost = (worst * ((6000 / 1e6) * p["in"] + (2500 / 1e6) * p["out"])
            if p else None)
    return {"approx_bars": bars, "calls_if_no_retries": calls,
            "worst_case_calls": worst, "max_cost_usd": cost,
            "priced": p is not None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Capture real prompt/response pairs.")
    ap.add_argument("--mode", choices=("pipeline", "single"), default="pipeline")
    ap.add_argument("--template", default="Three-Step Analyst")
    ap.add_argument("--model", default="nvidia/nemotron-3-nano-30b-a3b")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-04-03")
    ap.add_argument("--symbols", default="AAPL,MSFT")
    ap.add_argument("--regime", default="unlabelled",
                    help="label for this window; measured separately, not asserted here")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout-seconds", type=float, default=120.0)
    ap.add_argument("--budget-seconds", type=float, default=1800.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    steps = load_template(args.template) if args.mode == "pipeline" else []
    est = estimate(args.mode, args.start, args.end, args.model, len(steps) or 1)

    print(f"PAIR CAPTURE  mode={args.mode}  model={args.model}  "
          f"{args.start}..{args.end}  symbols={args.symbols}")
    if args.mode == "pipeline":
        print(f"  template {args.template!r}: {len(steps)} step(s) "
              + ", ".join(repr(s.get('label')) for s in steps))
    print(f"  ~{est['approx_bars']} bars -> {est['calls_if_no_retries']} calls "
          f"if nothing retries; worst case {est['worst_case_calls']}")
    print("  max cost " + (f"${est['max_cost_usd']:.4f}" if est["priced"]
                           else "unpriced"))
    if args.dry_run:
        print("\n--dry-run: nothing run, nothing spent.")
        return 0
    if not args.yes:
        if input(f"\nSpend up to ${est['max_cost_usd']:.4f}? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            print("Aborted; nothing spent.")
            return 1

    guard_seed_db()
    before = seed_db_sha256()

    # The SDK default is read=600s with max_retries=2; without this a stalled
    # request hangs the capture for ten minutes per attempt.
    import anthropic, httpx
    orig_init = anthropic.Anthropic.__init__
    def patched(self, *a, **kw):
        kw.setdefault("timeout", httpx.Timeout(args.timeout_seconds,
                                               connect=min(10.0, args.timeout_seconds)))
        return orig_init(self, *a, **kw)
    _set(anthropic.Anthropic, "__init__", patched)

    sink = open(args.out, "w", encoding="utf-8") if args.out else None
    cap = Capture(args.mode, args.regime, sink)
    attached = instrument(cap, deadline=time.perf_counter() + args.budget_seconds)
    run_id = None
    try:
        from dashboard.backend.domain.backtesting.engine import HourlyBacktester
        engine = HourlyBacktester(
            args.start, args.end, "pair-capture", use_llm=True,
            model=args.model, symbols=args.symbols.split(","),
            pipeline=steps or None)
        engine.load_data()
        engine.calculate_indicators()
        run_id, _ = engine.run_agent_backtest()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  RUN STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        restore()
        if sink:
            sink.close()

    after = seed_db_sha256()
    if after != before or after != SEED_DB_SHA256:
        raise AssertionError("seed DB changed during the capture")

    n = len(cap.pairs)
    parsed = sum(1 for p in cap.pairs if p["parsed_json"])
    conv = sum(1 for p in cap.pairs if p["converts_to_decision"])
    ceil = sum(1 for p in cap.pairs if p["hit_output_ceiling"])
    print(f"\n  captured {n} pairs  run_id={run_id}")
    print(f"  parsed as JSON      {parsed}/{n}")
    print(f"  converted to decision {conv}/{n}")
    print(f"  hit output ceiling  {ceil}/{n}")
    print(f"  timers attached: {attached}")
    if args.out:
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
