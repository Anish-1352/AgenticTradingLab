#!/usr/bin/env python3
"""Parse reliability per model per pipeline step. Measure this BEFORE cost.

    # always start here — no calls, no spend
    python dashboard/scripts/measure_step_reliability.py --dry-run

    python dashboard/scripts/measure_step_reliability.py \
        --models nvidia/nemotron-3-nano-30b-a3b anthropic/claude-sonnet-4-6 \
        --attempts 20 --integration openrouter

WHY THIS COMES FIRST
---------------------
``pipeline_runner`` aborts the ENTIRE decision when any step returns
unparseable JSON (``pipeline_runner.py:479``) and there is no retry on that
path. Downstream, ``portfolio_manager`` either raises ``LLMDecisionError``
under ``strict_llm`` or falls back to rule-based trading — so on a leaderboard
run a parse failure does not produce a slightly worse decision, it produces a
decision that is not the model's at all.

A 40% cost reduction that drops 10% of decisions is not a saving. Cost is
arithmetic once reliability is known; reliability is the measurement.

TWO GATES, NOT ONE
-------------------
Every step must return parseable JSON. The FINAL step must additionally
satisfy ``pipeline_output_to_decision``, which returns None unless the parsed
object carries a non-empty ``actions``/``orders``/``risk_actions`` list. A
model can parse perfectly at every step and still fail the last one, so the
two gates are counted separately and the final step is reported with both.

THE PARSER IS THE PRODUCTION PARSER
------------------------------------
``parse_llm_response`` and ``pipeline_output_to_decision`` are imported, not
reimplemented. Measuring with a more or less tolerant parser than the one that
runs in production would produce a number about this script rather than about
the pipeline. Note that the production parser is fairly forgiving: it strips
code fences and slices from the first ``{`` to the last ``}``, so a prose
wrapper usually SURVIVES. Response shape is therefore categorised separately
from parse success — a wrapped answer that parsed today is still a signal.

MODES
-----
``--mode isolated`` (default) measures each step against an IDENTICAL frozen
upstream context, so a per-step comparison across models is clean: every model
sees the same input for step 3 regardless of how it did at step 1. This is the
measurement the routing decision needs.

``--mode endtoend`` runs whole pipelines and reports the abort rate and the
compounding token effect. It cannot attribute a failure to a step cleanly,
because a step-1 failure means steps 2..N are never attempted — that is
precisely the production behaviour, and why the isolated mode exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import ensure_repo_root  # noqa: E402

ensure_repo_root()

from dashboard.backend.infrastructure.llm.backtest_harness import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    extract_response_text,
    extract_token_usage,
    parse_llm_response,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (  # noqa: E402
    PIPELINE_SYSTEM_PROMPT,
    _build_step_prompt,
    pipeline_output_to_decision,
    split_pipeline,
)
from dashboard.backend.infrastructure.llm.routing_cost import (  # noqa: E402
    PRICE_KNOWN,
    price_provenance,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKETPLACE = os.path.join(REPO_ROOT, "dashboard", "config", "marketplace.json")
DEFAULT_TEMPLATE = "pipeline-analyst"

# Failure taxonomy. Counting failures without categorising them cannot
# distinguish "the model needs more max_tokens" from "the model cannot follow
# the schema", and those have opposite fixes.
FAIL_EMPTY = "empty"                    # no text came back at all
FAIL_NO_JSON = "no_json"                # no braces anywhere; pure prose
FAIL_TRUNCATED = "truncated"            # JSON opened and never closed
FAIL_MALFORMED = "malformed_json"       # braces present, unrepairable
FAIL_NOT_OBJECT = "not_an_object"       # parsed, but not a dict
FAIL_SCHEMA = "wrong_schema"            # final step: parsed, no usable actions
# Distinct from wrong_schema and much more interesting: the model returned the
# RIGHT key with an empty list — a well-formed "make no trades this hour",
# which is the correct answer most hours. pipeline_output_to_decision returns
# None for it, so production treats a valid no-trade decision as an abort.
# Conflating the two would blame the model for a pipeline limitation.
FAIL_EMPTY_ACTIONS = "empty_actions_no_trade"
OK_CLEAN = "ok"
OK_WRAPPED = "ok_prose_wrapped"         # parsed, but the model added prose
OK_FENCED = "ok_code_fenced"            # parsed, but wrapped in ``` fences


# --------------------------------------------------------------------------
# a market snapshot of the real shape
# --------------------------------------------------------------------------
# Schema copied field-for-field from portfolio_manager.py:297 so the prompt the
# model sees has production structure. The VALUES are a fixture — a fixed,
# plausible hour — because every model and attempt must receive byte-identical
# input or the comparison measures the snapshot instead of the model.

REFERENCE_SNAPSHOT: Dict[str, Any] = {
    "timestamp": "2026-04-16T14:00:00",
    "portfolio": {
        "cash": 4820.55,
        "positions_value": 5310.20,
        "total_equity": 10130.75,
        "num_positions": 3,
    },
    "current_holdings": {
        "AAPL": {"shares": 8, "avg_price": 214.30, "current_price": 219.885,
                 "position_value": 1759.08, "pnl_pct": 2.61},
        "MSFT": {"shares": 4, "avg_price": 402.10, "current_price": 397.22,
                 "position_value": 1588.88, "pnl_pct": -1.21},
        "JPM": {"shares": 9, "avg_price": 218.44, "current_price": 217.36,
                "position_value": 1956.24, "pnl_pct": -0.49},
    },
    "recent_trades": [
        {"symbol": "AAPL", "side": "buy", "shares": 3, "price": 216.4,
         "timestamp": "2026-04-16T10:00:00"},
    ],
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
                "volume_ratio": 1.11, "momentum_pct": -2.8},
    },
}


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def classify_response(text: str, parsed: Optional[Dict[str, Any]],
                      *, is_last: bool) -> Tuple[bool, str, Optional[str]]:
    """``(passed, category, detail)`` for one response.

    ``passed`` means the pipeline would have CONTINUED — i.e. parsed, and for
    the last step also converted into actions. The category distinguishes a
    clean answer from one that only survived the parser's tolerance.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, FAIL_EMPTY, None

    has_open = "{" in stripped
    has_close = "}" in stripped

    if parsed is None:
        if not has_open:
            return False, FAIL_NO_JSON, stripped[:160]
        if not has_close:
            return False, FAIL_TRUNCATED, stripped[-160:]
        return False, FAIL_MALFORMED, stripped[:160]

    if not isinstance(parsed, dict):
        return False, FAIL_NOT_OBJECT, type(parsed).__name__

    if is_last:
        # The second gate. Parsing was necessary but is not sufficient here.
        if pipeline_output_to_decision(parsed) is None:
            for key in ("actions", "orders", "risk_actions"):
                if isinstance(parsed.get(key), list) and not parsed[key]:
                    return False, FAIL_EMPTY_ACTIONS, key
            return False, FAIL_SCHEMA, ", ".join(sorted(parsed)[:8]) or "empty object"

    # It passed. Record HOW cleanly, because a model that needs the parser's
    # tolerance today is a model that fails when a response is slightly longer.
    if "```" in stripped:
        return True, OK_FENCED, None
    first, last = stripped.find("{"), stripped.rfind("}")
    if first > 0 or last < len(stripped) - 1:
        return True, OK_WRAPPED, None
    return True, OK_CLEAN, None


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class Attempt:
    ok: bool
    category: str
    detail: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class StepResult:
    model: str
    step_index: int
    step_label: str
    step_id: Optional[str]
    is_last: bool
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def n_ok(self) -> int:
        return sum(1 for a in self.attempts if a.ok)

    @property
    def parse_rate(self) -> Optional[float]:
        return (self.n_ok / self.n) if self.n else None

    @property
    def mean_output_tokens(self) -> Optional[float]:
        vals = [a.output_tokens for a in self.attempts if a.ok and a.output_tokens]
        return (sum(vals) / len(vals)) if vals else None

    @property
    def mean_input_tokens(self) -> Optional[float]:
        vals = [a.input_tokens for a in self.attempts if a.input_tokens]
        return (sum(vals) / len(vals)) if vals else None

    def categories(self) -> Dict[str, int]:
        return dict(Counter(a.category for a in self.attempts))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "step_index": self.step_index,
            "step_label": self.step_label,
            "step_id": self.step_id,
            "is_last": self.is_last,
            "n_attempts": self.n,
            "n_passed": self.n_ok,
            "parse_rate": self.parse_rate,
            "categories": self.categories(),
            "mean_output_tokens": self.mean_output_tokens,
            "mean_input_tokens": self.mean_input_tokens,
            "attempts": [asdict(a) for a in self.attempts],
        }


# --------------------------------------------------------------------------
# pipeline loading
# --------------------------------------------------------------------------


def load_pipeline(source: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    """A real pipeline: a marketplace template id, or a path to a JSON file.

    Defaults to the shipped ``pipeline-analyst`` template — three steps that
    real users run — rather than a pipeline invented for this script.
    """
    source = source or DEFAULT_TEMPLATE
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        steps = payload if isinstance(payload, list) else payload.get("pipeline")
        if not steps:
            raise SystemExit(f"{source} contains no pipeline steps")
        return list(steps), os.path.basename(source)

    with open(MARKETPLACE, "r", encoding="utf-8") as fh:
        templates = json.load(fh)["templates"]
    for tpl in templates:
        if tpl.get("template_id") == source:
            steps = tpl.get("pipeline") or []
            if not steps:
                raise SystemExit(
                    f"template {source!r} has no pipeline (it is a single-prompt agent). "
                    f"Multi-step templates: "
                    f"{[t['template_id'] for t in templates if t.get('pipeline') and len(t['pipeline']) > 1]}")
            return list(steps), source
    raise SystemExit(
        f"no template {source!r} and no such file. Available: "
        f"{[t.get('template_id') for t in templates]}")


def build_reference_context(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Frozen upstream outputs so every model sees the same input at step k.

    Derived from each step's own declared ``outputFormat``, which is what the
    step promises downstream. This is a STAND-IN for a real upstream answer:
    it has the right keys and shape but not a real model's content or length,
    so absolute input-token counts here are lower than production. It is the
    right control for COMPARING models at one step, and the wrong tool for
    measuring the compounding effect — use ``--mode endtoend`` for that.
    """
    context: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        fmt = (step.get("outputFormat") or "").strip()
        placeholder: Any
        try:
            start, end = fmt.find("{"), fmt.rfind("}") + 1
            placeholder = json.loads(fmt[start:end]) if start >= 0 and end > start else None
        except Exception:  # noqa: BLE001
            placeholder = None
        if not isinstance(placeholder, dict):
            placeholder = {"summary": f"upstream output of step {i + 1}"}
        context.append({
            "step": i + 1,
            "label": step.get("label") or f"Step {i + 1}",
            "presetKey": step.get("presetKey"),
            "id": step.get("id"),
            "output": placeholder,
        })
    return context


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _prompt_for(steps: List[Dict[str, Any]], index: int,
                prior: List[Dict[str, Any]]) -> str:
    """The production prompt builder, not a copy of it."""
    return _build_step_prompt(
        step_index=index,
        step=steps[index],
        market_snapshot=REFERENCE_SNAPSHOT,
        prior_outputs=prior,
        is_last=(index == len(steps) - 1),
    )


def run_isolated(client, steps: List[Dict[str, Any]], model: str,
                 attempts: int, *, sleep_s: float = 0.0,
                 verbose: bool = True) -> List[StepResult]:
    """Every step, ``attempts`` times each, against a frozen upstream context."""
    reference = build_reference_context(steps)
    results: List[StepResult] = []

    for index, step in enumerate(steps):
        is_last = index == len(steps) - 1
        prompt = _prompt_for(steps, index, reference[:index])
        res = StepResult(
            model=model, step_index=index,
            step_label=step.get("label") or f"Step {index + 1}",
            step_id=step.get("id"), is_last=is_last)

        for attempt_no in range(attempts):
            started = time.perf_counter()
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    system=PIPELINE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:  # noqa: BLE001
                res.attempts.append(Attempt(
                    ok=False, category="api_error",
                    detail=f"{type(exc).__name__}: {exc}"[:200],
                    latency_s=time.perf_counter() - started,
                    error=str(exc)[:200]))
                continue

            latency = time.perf_counter() - started
            in_tok, out_tok = extract_token_usage(response)
            try:
                text = extract_response_text(response)
            except AttributeError:
                text = ""
            parsed = parse_llm_response(text) if text else None
            ok, category, detail = classify_response(text, parsed, is_last=is_last)
            res.attempts.append(Attempt(
                ok=ok, category=category, detail=detail,
                input_tokens=in_tok, output_tokens=out_tok, latency_s=latency))
            if sleep_s:
                time.sleep(sleep_s)

        if verbose:
            print(f"    step {index + 1} {res.step_label[:28]:28s} "
                  f"{res.n_ok}/{res.n}  {res.categories()}")
        results.append(res)
    return results


def run_endtoend(client, steps: List[Dict[str, Any]], model: str,
                 attempts: int, *, sleep_s: float = 0.0,
                 verbose: bool = True) -> Dict[str, Any]:
    """Whole pipelines. Reports the abort rate and where aborts happen.

    A step-1 failure means steps 2..N are never attempted, so per-step rates
    from this mode are conditional on reaching the step. That is production
    behaviour, not a flaw — but it is why isolated mode exists.
    """
    completed = 0
    aborted_at: Counter = Counter()
    input_tokens_by_step: Dict[int, List[int]] = {}
    output_tokens_by_step: Dict[int, List[int]] = {}
    runs: List[Dict[str, Any]] = []

    for _ in range(attempts):
        prior: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        failed_at = None
        for index in range(len(steps)):
            is_last = index == len(steps) - 1
            prompt = _prompt_for(steps, index, prior)
            try:
                response = client.messages.create(
                    model=model, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    system=PIPELINE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}])
            except Exception as exc:  # noqa: BLE001
                failed_at = (index, f"api_error: {type(exc).__name__}")
                break
            in_tok, out_tok = extract_token_usage(response)
            input_tokens_by_step.setdefault(index, []).append(in_tok)
            output_tokens_by_step.setdefault(index, []).append(out_tok)
            try:
                text = extract_response_text(response)
            except AttributeError:
                text = ""
            parsed = parse_llm_response(text) if text else None
            ok, category, _detail = classify_response(text, parsed, is_last=is_last)
            trace.append({"step": index + 1, "ok": ok, "category": category,
                          "input_tokens": in_tok, "output_tokens": out_tok})
            if not ok:
                failed_at = (index, category)
                break
            prior.append({
                "step": index + 1,
                "label": steps[index].get("label") or f"Step {index + 1}",
                "presetKey": steps[index].get("presetKey"),
                "id": steps[index].get("id"),
                "output": parsed,
            })
            if sleep_s:
                time.sleep(sleep_s)
        if failed_at is None:
            completed += 1
        else:
            aborted_at[f"step{failed_at[0] + 1}:{failed_at[1]}"] += 1
        runs.append({"completed": failed_at is None, "trace": trace})

    def _mean(v):
        return (sum(v) / len(v)) if v else None

    if verbose:
        print(f"    completed {completed}/{attempts}  aborts={dict(aborted_at)}")
    return {
        "model": model,
        "attempts": attempts,
        "completed": completed,
        "completion_rate": completed / attempts if attempts else None,
        "aborted_at": dict(aborted_at),
        "mean_input_tokens_by_step": {
            k: _mean(v) for k, v in sorted(input_tokens_by_step.items())},
        "mean_output_tokens_by_step": {
            k: _mean(v) for k, v in sorted(output_tokens_by_step.items())},
        "runs": runs,
    }


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


def estimate_run_cost(steps: List[Dict[str, Any]], models: List[str],
                      attempts: int, mode: str) -> Dict[str, Any]:
    """What this would spend, before it spends it.

    Input tokens are estimated from the real prompt text this run WILL send.
    Output tokens are assumed at the ``max_tokens`` cap, which is deliberately
    pessimistic: the true figure is the thing being measured, so the estimate
    must not quietly assume the answer.
    """
    reference = build_reference_context(steps)
    per_step_chars = []
    for index in range(len(steps)):
        prior = reference[:index] if mode == "isolated" else reference[:index]
        per_step_chars.append(len(_prompt_for(steps, index, prior))
                              + len(PIPELINE_SYSTEM_PROMPT))

    rows = []
    total = 0.0
    for model in models:
        prov, (in_price, out_price) = price_provenance(model)
        model_cost = 0.0
        for chars in per_step_chars:
            in_tok = int(chars / 3.8)
            model_cost += attempts * (
                (in_tok / 1_000_000) * in_price
                + (DEFAULT_MAX_OUTPUT_TOKENS / 1_000_000) * out_price)
        rows.append({
            "model": model,
            "input_price_per_mtok": in_price,
            "output_price_per_mtok": out_price,
            "calls": len(steps) * attempts,
            "max_cost_usd": model_cost,
            "price_provenance": prov,
            "priced": prov == PRICE_KNOWN,
        })
        total += model_cost
    return {"rows": rows, "total_max_cost_usd": total,
            "total_calls": len(steps) * attempts * len(models),
            "assumption": (
                f"output tokens assumed at the {DEFAULT_MAX_OUTPUT_TOKENS} "
                f"max_tokens cap — the true value is what this run measures")}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_report(all_results: List[StepResult], attempts: int) -> str:
    lines = ["", "=" * 78,
             f"PARSE RELIABILITY — {attempts} attempts per model per step",
             "=" * 78]
    by_model: Dict[str, List[StepResult]] = {}
    for r in all_results:
        by_model.setdefault(r.model, []).append(r)

    for model, results in by_model.items():
        lines.append(f"\n{model}")
        lines.append(f"  {'step':<32} {'pass':>9}  {'out tok':>8}  categories")
        lines.append("  " + "-" * 74)
        for r in sorted(results, key=lambda x: x.step_index):
            rate = f"{r.n_ok}/{r.n}" if r.n else "—"
            pct = f"({r.parse_rate:.0%})" if r.parse_rate is not None else ""
            tok = f"{r.mean_output_tokens:.0f}" if r.mean_output_tokens else "—"
            cats = ", ".join(f"{k}={v}" for k, v in sorted(r.categories().items()))
            tag = " *" if r.is_last else "  "
            lines.append(f"  {r.step_label[:30]:<30}{tag} {rate:>5}{pct:>5}  "
                         f"{tok:>8}  {cats}")
        worst = min((r.parse_rate for r in results
                     if r.parse_rate is not None), default=None)
        if worst is not None:
            lines.append(f"  -> weakest step: {worst:.0%}. A pipeline completes only "
                         f"if EVERY step parses.")
            product = 1.0
            for r in results:
                if r.parse_rate is not None:
                    product *= r.parse_rate
            lines.append(f"  -> implied end-to-end completion (product of steps): "
                         f"{product:.0%}")
    lines.append("\n  * = final step; it must also produce usable actions, "
                 "not merely valid JSON.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Measure per-model per-step JSON parse reliability.")
    ap.add_argument("--pipeline", default=None,
                    help=f"marketplace template id or path to a pipeline JSON "
                         f"(default: {DEFAULT_TEMPLATE})")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model ids to test (default: the agent's configured model)")
    ap.add_argument("--attempts", type=int, default=20,
                    help="attempts per model per step. Keep this in the tens.")
    ap.add_argument("--integration", default=None,
                    help="provider: openrouter / commonstack / anthropic")
    ap.add_argument("--mode", default="isolated", choices=["isolated", "endtoend"])
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls, for rate limits")
    ap.add_argument("--out", default=None, help="write full JSON results here")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the cost ceiling; call nothing")
    ap.add_argument("--yes", action="store_true",
                    help="skip the spend confirmation prompt")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    steps, source = load_pipeline(args.pipeline)

    if args.attempts > 100:
        print(f"Refusing --attempts {args.attempts}: this harness is sized for "
              f"tens of attempts per model per step. Raise the cap deliberately "
              f"in the source if you really want hundreds.", file=sys.stderr)
        return 2

    decision_steps, post_trade = split_pipeline(steps)
    models = args.models or []

    print("=" * 78)
    print(f"PIPELINE: {source}  ({len(decision_steps)} decision steps"
          + (f", {len(post_trade)} post-trade ignored)" if post_trade else ")"))
    print("=" * 78)
    for i, s in enumerate(decision_steps):
        print(f"  {i + 1}. {s.get('label')}"
              f"   [id={s.get('id')} presetKey={s.get('presetKey')}]")

    if not models:
        print("\nNo --models given. Nothing to measure.", file=sys.stderr)
        return 2

    est = estimate_run_cost(decision_steps, models, args.attempts, args.mode)
    print(f"\nCOST CEILING ({args.attempts} attempts x {len(decision_steps)} steps "
          f"x {len(models)} models = {est['total_calls']} calls)")
    print(f"  {'model':<42} {'calls':>6} {'max $':>10}")
    print("  " + "-" * 62)
    for row in est["rows"]:
        marker = "" if row["priced"] else f"   [{row['price_provenance']}]"
        print(f"  {row['model'][:40]:<42} {row['calls']:>6} "
              f"{row['max_cost_usd']:>10.4f}{marker}")
    print(f"  {'TOTAL':<42} {est['total_calls']:>6} "
          f"{est['total_max_cost_usd']:>10.4f}")
    print(f"  assumption: {est['assumption']}")
    if any(r["price_provenance"] == "defaulted" for r in est["rows"]):
        print("  [defaulted] = not in the pricing table; the figure is a guess "
              "at Claude-Haiku rates, not a price.")
    if any(r["price_provenance"] == "local" for r in est["rows"]):
        print("  [local] = self-hosted, so no per-token API cost. Real cost is "
              "GPU time, which this does not model.")

    if args.dry_run:
        print("\n--dry-run: no API calls made, nothing spent.")
        return 0

    if not args.yes:
        reply = input(f"\nSpend up to ${est['total_max_cost_usd']:.4f}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted; nothing spent.")
            return 1

    from dashboard.backend.infrastructure.llm.providers import (  # noqa: PLC0415
        make_llm_client,
    )

    client = make_llm_client(args.integration)
    if client is None:
        print(f"\nNo LLM client for integration={args.integration!r}. "
              f"The API key for that provider is not set, so no call was "
              f"attempted.", file=sys.stderr)
        return 3

    payload: Dict[str, Any] = {
        "pipeline_source": source,
        "mode": args.mode,
        "attempts": args.attempts,
        "integration": args.integration,
        "steps": [{"index": i, "label": s.get("label"), "id": s.get("id"),
                   "presetKey": s.get("presetKey")}
                  for i, s in enumerate(decision_steps)],
        "cost_ceiling": est,
    }

    if args.mode == "isolated":
        all_results: List[StepResult] = []
        for model in models:
            print(f"\n  {model}")
            all_results.extend(run_isolated(
                client, decision_steps, model, args.attempts, sleep_s=args.sleep))
        print(format_report(all_results, args.attempts))
        payload["results"] = [r.to_dict() for r in all_results]
    else:
        payload["results"] = []
        for model in models:
            print(f"\n  {model}")
            payload["results"].append(run_endtoend(
                client, decision_steps, model, args.attempts, sleep_s=args.sleep))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
