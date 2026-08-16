"""Drive ATL's real pipeline runner with a stub client — no key, no network.

    python -m analysis.atl_pipeline_probe configs/atl_pipelines/pipeline_3step.json
    python -m analysis.atl_pipeline_probe configs/atl_pipelines/*.json --json-out …

WHAT THIS IS FOR
----------------
``run_pipeline_decision`` has no recorded run anywhere: all seven seed runs used
the single-call path, so calls-per-decision has never been observed on the
multi-step path. Measuring it for real needs an API key and market data. This
module gets the *structural* half of that answer without either, by calling the
genuine ``pipeline_runner`` with a client that records prompts and returns
canned JSON.

WHAT IT ESTABLISHES (key-free, deterministic)
----------------------------------------------
* **calls == configured decision steps**, on the real code path. This is the
  ``b`` derivation in ``atl_token_extract.py`` confirmed against execution
  rather than against a metadata field.
* **the pipeline file is well-formed** — it splits as intended, and the final
  step's ``outputFormat`` actually converts to a trading decision. A pipeline
  that fails this would abort mid-run and silently understate calls per
  decision, which is the ``a < b`` disagreement case.
* **where prompt bulk sits.** The market snapshot is injected into step 1 ONLY;
  later steps instead carry ``prior_outputs``, which accumulates. So the static
  prefix does not repeat across steps, and prompt growth is driven entirely by
  upstream output.

WHAT IT CANNOT ESTABLISH
------------------------
* **Any token count.** The stub's outputs are a few dozen characters; a real
  model emits 860–5,005 tokens per call. Every size here is therefore a LOWER
  BOUND, and the growth ratio in particular understates the real one badly —
  ``prior_outputs`` carries model output, so the gap widens with verbosity.
* **Retries.** A stub that always parses never triggers the retry surfaces, so
  this run cannot produce the ``a > b`` disagreement. Only a real run can.
* **Cost, latency, or anything model-specific.**

Structural claims from this module are citable. Numeric ones are not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
for _p in (_BENCH_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["StubClient", "probe_pipeline", "format_report"]

# A minimal snapshot with the fields _build_step_prompt and the execution rules
# reference. Deliberately small: its size must not be mistaken for a measurement
# of ATL's real market context.
DEFAULT_SNAPSHOT: Dict[str, Any] = {
    "symbols": ["AAPL", "MSFT"],
    "cash": 100000.0,
    "current_holdings": {"AAPL": 10},
    "bars": {"AAPL": {"close": 200.0}, "MSFT": {"close": 400.0}},
}


class StubClient:
    """Anthropic-shaped client that records prompts and returns canned JSON.

    Returns a final-step payload for the last step and a generic one otherwise,
    so the pipeline runs to completion exactly as it would against a model that
    always answers in the requested format.
    """

    def __init__(self, final_index: int):
        self.final_index = final_index
        self.prompts: List[str] = []
        self.models: List[str] = []
        self.max_tokens: List[Any] = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw: Any) -> Any:
        prompt = kw["messages"][0]["content"]
        self.prompts.append(prompt)
        self.models.append(kw.get("model"))
        self.max_tokens.append(kw.get("max_tokens"))
        index = len(self.prompts) - 1
        if index == self.final_index:
            payload: Dict[str, Any] = {"actions": [{
                "action": "hold", "symbol": "AAPL", "position_size": 0,
                "confidence": 0.5, "reasoning": "stub",
            }]}
        else:
            payload = {"output": f"stub step {index + 1}"}
        text = json.dumps(payload)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=text)],
            usage=types.SimpleNamespace(
                # Chars/4 is a stand-in so the plumbing is exercised; it is NOT
                # a tokenizer and no token figure here should be quoted.
                input_tokens=len(prompt) // 4, output_tokens=len(text) // 4),
        )


def probe_pipeline(pipeline: List[Dict[str, Any]],
                   snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one pipeline against the stub and report its structure."""
    from dashboard.backend.infrastructure.llm import pipeline_runner as pr

    decision_steps, post_trade_steps = pr.split_pipeline(pipeline)
    if not decision_steps:
        return {"available": False,
                "reason": "pipeline has no decision steps — it would return "
                          "None before issuing any call"}

    client = StubClient(final_index=len(decision_steps) - 1)
    decision, (t_in, t_out), calls, outputs = pr.run_pipeline_decision(
        client, pipeline=pipeline, market_snapshot=snapshot or DEFAULT_SNAPSHOT,
        model="stub-model")

    sizes = [len(p) for p in client.prompts]
    return {
        "available": True,
        "configured_decision_steps": len(decision_steps),
        "configured_post_trade_steps": len(post_trade_steps),
        "llm_calls_issued": calls,
        "calls_equal_steps": calls == len(decision_steps),
        "decision_produced": decision is not None,
        "step_outputs_recorded": len(outputs),
        "step_labels": [s.get("label") for s in decision_steps],
        "prompt_chars_per_step": sizes,
        "prompt_growth_ratio": (sizes[-1] / sizes[0]) if sizes and sizes[0] else None,
        "snapshot_in_step": ["MARKET SNAPSHOT" in p for p in client.prompts],
        "upstream_outputs_in_step": [
            "UPSTREAM PIPELINE OUTPUTS" in p for p in client.prompts],
        "max_tokens_per_call": sorted(set(client.max_tokens)),
        "stub_token_totals": {
            "input": t_in, "output": t_out,
            "WARNING": "chars/4 from a stub, NOT a tokenizer and NOT a model. "
                       "Never quote these as token counts.",
        },
        "caveats": [
            "Prompt sizes are LOWER BOUNDS: prior_outputs carries real model "
            "output (860-5,005 tokens/call in the seed data), so the true "
            "growth across steps is far larger than shown here.",
            "The stub always parses, so no retry surface fires. This run "
            "cannot produce the observed>configured disagreement.",
        ],
    }


def format_report(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A("ATL PIPELINE PROBE — real runner, stub client, no key and no network")
    A("=" * 78)
    for name, r in results:
        A("")
        A(f"  {name}")
        if not r.get("available"):
            A(f"    UNAVAILABLE: {r.get('reason')}")
            continue
        A(f"    configured decision steps : {r['configured_decision_steps']}"
          f"   post-trade: {r['configured_post_trade_steps']}")
        A(f"    LLM calls issued          : {r['llm_calls_issued']}"
          f"   {'== steps ✓' if r['calls_equal_steps'] else '!= steps ✗'}")
        A(f"    decision produced         : {r['decision_produced']}")
        A(f"    steps                     : {r['step_labels']}")
        A(f"    prompt chars per step     : {r['prompt_chars_per_step']}")
        if r.get("prompt_growth_ratio"):
            A(f"    last/first prompt         : {r['prompt_growth_ratio']:.2f}x "
              f"(LOWER BOUND — stub outputs are tiny)")
        A(f"    market snapshot in step   : {r['snapshot_in_step']}")
        A(f"    upstream outputs in step  : {r['upstream_outputs_in_step']}")
        A(f"    max_tokens per call       : {r['max_tokens_per_call']}")
    A("")
    A("  The snapshot enters step 1 ONLY; later steps carry prior_outputs, which")
    A("  accumulates. The static prefix therefore does NOT repeat across steps,")
    A("  so prompt growth is driven entirely by upstream model output — and a")
    A("  verbose model inflates its own later prompts.")
    A("")
    A("  NOT ESTABLISHED HERE: token counts, retries, cost, latency.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Structurally probe an ATL pipeline file without a key.")
    ap.add_argument("pipeline_files", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    results = []
    for path in args.pipeline_files:
        try:
            with open(path) as fh:
                pipeline = json.load(fh)
        except (OSError, ValueError) as exc:
            results.append((os.path.basename(path),
                            {"available": False, "reason": str(exc)}))
            continue
        results.append((os.path.basename(path), probe_pipeline(pipeline)))

    print(format_report(results))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({name: r for name, r in results}, fh, indent=2, default=str)
        print(f"\n[probe] {args.json_out}")
    return 0 if all(r.get("available") for _, r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
