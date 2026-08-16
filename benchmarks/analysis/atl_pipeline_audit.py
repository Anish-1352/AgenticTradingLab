"""Static audit of ATL's real decision loop. Reads the code; runs nothing.

    python -m analysis.atl_pipeline_audit --json-out results/atl_audit.json

Every cost figure in ``arm_a_vs_bc_model.py`` rests on one call per decision at
2,620 input / 256 output tokens. That came from a synthetic fixture, not from
ATL. This walks ``dashboard/backend`` with ``ast`` and reports what the pipeline
actually does — how many LLM calls a decision costs, whether that count is
fixed, what goes into each prompt, and which parts of it could ever be shared
across agents.

READ-ONLY AND NON-EXECUTING. Nothing under ``dashboard/`` is imported or
modified; the pipeline needs a model, a market feed and a database, and running
it to count its calls would be both expensive and beside the point.

WHY THE STATIC/PER-AGENT/PER-DECISION SPLIT IS THE POINT
--------------------------------------------------------
The fixtures bracket prefix overlap at 99.4% (``shared_prefix``) and 0.3%
(``low_overlap``) — both chosen, neither measured. Where reality sits is decided
by prompt *composition*: text that is identical across every agent can be
cached or shared, text that varies per agent cannot, and text that varies per
decision defeats caching entirely. Classifying each prompt segment is what
turns those bounds into an estimate.

WHAT STATIC ANALYSIS CANNOT SETTLE
-----------------------------------
Reported as unknown rather than guessed, because unknowns are the deliverable:

* **Step count.** The pipeline is user-configured data, not code. The audit can
  say "one call per configured step" and find where the count comes from; only a
  real run's ``metadata.initial_pipeline`` says what a given agent used.
* **Token counts.** Prompt size depends on the market snapshot's runtime size.
* **Retry frequency.** Retry paths are found and reported as multipliers; how
  often they fire is a runtime property.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

__all__ = ["LLM_CALL_PATTERNS", "find_llm_calls", "classify_prompt_segments",
           "audit_tree", "audit_paths", "format_report"]

# Attribute chains that constitute an outbound LLM call. Matched on the dotted
# suffix so `client.messages.create` and `self._client.messages.create` both hit.
LLM_CALL_PATTERNS = (
    "messages.create",
    "chat.completions.create",
    "completions.create",
)

# Callables whose presence in a prompt-building expression means the value is
# computed at runtime rather than written in the source.
DYNAMIC_MARKERS = ("json.dumps", "str", "format", "repr", "join")


def _dotted(node: ast.AST) -> str:
    """Render an attribute chain as a dotted string, or '' if it is not one."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _kwarg(call: ast.Call, name: str) -> Optional[str]:
    for kw in call.keywords:
        if kw.arg == name:
            try:
                return ast.unparse(kw.value)
            except Exception:  # pragma: no cover - ast.unparse is 3.9+
                return type(kw.value).__name__
    return None


class _CallFinder(ast.NodeVisitor):
    """Locate LLM calls and record the control flow enclosing each one."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: List[Dict[str, Any]] = []
        self._func_stack: List[str] = []
        self._loop_stack: List[Tuple[str, int]] = []
        self._try_depth = 0
        self._async_func = False

    # -- scope tracking --

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        prev, self._async_func = self._async_func, False
        self.generic_visit(node)
        self._async_func = prev
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        prev, self._async_func = self._async_func, True
        self.generic_visit(node)
        self._async_func = prev
        self._func_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        try:
            target = ast.unparse(node.iter)
        except Exception:  # pragma: no cover
            target = "<iter>"
        self._loop_stack.append((f"for … in {target}", node.lineno))
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self._loop_stack.append(("while", node.lineno))
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_Try(self, node: ast.Try) -> None:
        self._try_depth += 1
        self.generic_visit(node)
        self._try_depth -= 1

    # -- the calls themselves --

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        if any(dotted.endswith(p) for p in LLM_CALL_PATTERNS):
            self.calls.append({
                "file": self.path,
                "line": node.lineno,
                "callee": dotted,
                "function": self._func_stack[-1] if self._func_stack else "<module>",
                "function_stack": list(self._func_stack),
                # A call inside a loop is a call whose COUNT is data-dependent.
                # That is the difference between "one call per decision" and
                # "one call per configured step".
                "in_loop": bool(self._loop_stack),
                "loops": [{"kind": k, "line": ln} for k, ln in self._loop_stack],
                "in_try": self._try_depth > 0,
                "is_async": self._async_func,
                "max_tokens": _kwarg(node, "max_tokens"),
                "model": _kwarg(node, "model"),
                "system": _kwarg(node, "system"),
                "temperature": _kwarg(node, "temperature"),
            })
        self.generic_visit(node)


def find_llm_calls(source: str, path: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"file": path, "parse_error": str(exc)}]
    finder = _CallFinder(path)
    finder.visit(tree)
    return finder.calls


# --------------------------------------------------------------------------
# prompt composition
# --------------------------------------------------------------------------


def classify_prompt_segments(source: str, func_name: str) -> Dict[str, Any]:
    """Classify the pieces a prompt-building function assembles.

    Literal strings are **static** — identical for every agent and every
    decision, so they are exactly the text a prefix cache can share. Anything
    interpolated is dynamic, and the variable name is reported so a human can
    decide whether it varies per agent (a configured prompt) or per decision (a
    market snapshot). The tool does not guess which; naming the source is the
    useful part.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"available": False, "reason": f"parse error: {exc}"}

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == func_name:
            target = node
            break
    if target is None:
        return {"available": False, "reason": f"{func_name} not found"}

    static: List[str] = []
    dynamic: List[Dict[str, Any]] = []
    conditional: List[Dict[str, Any]] = []

    # Segments guarded by an `if` are conditional: present in some prompts and
    # not others, which breaks a shared prefix at the point they appear.
    conditional_lines = set()
    for node in ast.walk(target):
        if isinstance(node, ast.If):
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    conditional_lines.add(child.lineno)
            try:
                cond = ast.unparse(node.test)
            except Exception:  # pragma: no cover
                cond = "<test>"
            conditional.append({"line": node.lineno, "condition": cond})

    for node in ast.walk(target):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if text and not text.startswith('"""'):
                static.append(text[:120])
        elif isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if any(m in dotted for m in DYNAMIC_MARKERS):
                try:
                    if node.args:
                        arg = ast.unparse(node.args[0])
                    elif isinstance(node.func, ast.Attribute):
                        # A no-arg method like `.strip()` — the interesting part
                        # is the RECEIVER, which names where the text came from
                        # (e.g. step.get('prompt') is per-agent configuration).
                        arg = ast.unparse(node.func.value)
                    else:
                        arg = ""
                except Exception:  # pragma: no cover
                    arg = ""
                dynamic.append({
                    "line": node.lineno,
                    "via": dotted,
                    "source_expression": arg[:160],
                    "conditional": node.lineno in conditional_lines,
                })
        elif isinstance(node, ast.JoinedStr):  # f-string
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    try:
                        expr = ast.unparse(value.value)
                    except Exception:  # pragma: no cover
                        expr = "<expr>"
                    dynamic.append({
                        "line": node.lineno, "via": "f-string",
                        "source_expression": expr[:160],
                        "conditional": node.lineno in conditional_lines,
                    })

    return {
        "available": True,
        "function": func_name,
        "static_segment_count": len(static),
        "static_segments": static[:30],
        "dynamic_segment_count": len(dynamic),
        "dynamic_segments": dynamic[:30],
        "conditional_branches": conditional,
        "interpretation": (
            "Static segments are candidates for a shared prefix. Dynamic ones "
            "are not, and a dynamic segment that appears EARLY in the prompt "
            "truncates the shareable prefix at that point regardless of how "
            "much static text follows it."
        ),
    }


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def audit_tree(root: str, subdirs: Sequence[str] = ("domain", "infrastructure")) -> Dict[str, Any]:
    """Walk the backend and collect every LLM call site."""
    calls: List[Dict[str, Any]] = []
    files_scanned = 0
    errors: List[Dict[str, Any]] = []

    for sub in subdirs:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            errors.append({"path": base, "error": "not a directory"})
            continue
        for dirpath, _dirs, filenames in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, _REPO_ROOT)
                try:
                    with open(full, encoding="utf-8") as fh:
                        source = fh.read()
                except OSError as exc:
                    errors.append({"path": rel, "error": str(exc)})
                    continue
                files_scanned += 1
                for call in find_llm_calls(source, rel):
                    if "parse_error" in call:
                        errors.append(call)
                    else:
                        calls.append(call)

    return {"files_scanned": files_scanned, "calls": calls, "errors": errors}


def summarise_calls(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn call sites into a per-decision cost characterisation."""
    fixed = [c for c in calls if not c["in_loop"]]
    looped = [c for c in calls if c["in_loop"]]

    by_function: Dict[str, List[Dict[str, Any]]] = {}
    for c in calls:
        by_function.setdefault(c["function"], []).append(c)

    return {
        "total_call_sites": len(calls),
        "call_sites_in_loops": len(looped),
        "call_sites_fixed": len(fixed),
        "calls_per_decision_is_fixed": not looped,
        "by_function": {
            fn: [{"file": c["file"], "line": c["line"],
                  "in_loop": c["in_loop"], "max_tokens": c["max_tokens"],
                  "loops": c["loops"]} for c in cs]
            for fn, cs in sorted(by_function.items())
        },
        "sequential_vs_concurrent": {
            "async_call_sites": sum(1 for c in calls if c["is_async"]),
            "sync_call_sites": sum(1 for c in calls if not c["is_async"]),
            "note": (
                "A synchronous call inside a for-loop is a SEQUENTIAL chain: "
                "per-decision latency is the sum of every step, not the max. "
                "No call site here is dispatched concurrently."
            ),
        },
        "max_tokens_settings": sorted({
            str(c["max_tokens"]) for c in calls if c["max_tokens"]
        }),
        "retry_surfaces": [
            {"file": c["file"], "line": c["line"], "function": c["function"],
             "loops": c["loops"],
             "why": "call sits inside a loop AND a try block — a retry path "
                    "here multiplies calls per decision"}
            for c in calls if c["in_loop"] and c["in_try"]
        ],
    }


def audit_paths(
    backend_root: Optional[str] = None,
    prompt_builders: Sequence[Tuple[str, str]] = (),
) -> Dict[str, Any]:
    root = backend_root or os.path.join(_REPO_ROOT, "dashboard", "backend")
    tree = audit_tree(root)
    summary = summarise_calls(tree["calls"])

    prompts: Dict[str, Any] = {}
    for rel_path, func in prompt_builders:
        full = os.path.join(_REPO_ROOT, rel_path)
        try:
            with open(full, encoding="utf-8") as fh:
                prompts[f"{rel_path}::{func}"] = classify_prompt_segments(
                    fh.read(), func)
        except OSError as exc:
            prompts[f"{rel_path}::{func}"] = {"available": False,
                                              "reason": str(exc)}

    return {
        "backend_root": os.path.relpath(root, _REPO_ROOT),
        "files_scanned": tree["files_scanned"],
        "call_sites": tree["calls"],
        "summary": summary,
        "prompt_composition": prompts,
        "errors": tree["errors"],
        "unknowns": [
            {
                "unknown": "calls per decision (absolute number)",
                "why": (
                    "The pipeline is user-configured DATA, not code: one LLM "
                    "call is made per configured step. Static analysis can say "
                    "'one per step' and locate the loop; only a real run's "
                    "metadata.initial_pipeline says how many steps an agent had."
                ),
                "how_to_resolve": (
                    "Read metadata.initial_pipeline from agent_runs, or "
                    "llm_calls / number-of-decisions for the observed average. "
                    "atl_token_extract.py does both."
                ),
            },
            {
                "unknown": "input tokens per call",
                "why": (
                    "Prompt size is dominated by the market snapshot and the "
                    "accumulated upstream outputs, both runtime values."
                ),
                "how_to_resolve": (
                    "agent_runs.input_tokens / agent_runs.llm_calls gives the "
                    "MEAN. The distribution is not recoverable — see "
                    "atl_token_extract.py."
                ),
            },
            {
                "unknown": "how often retry paths fire",
                "why": "Retries are triggered by malformed model output at runtime.",
                "how_to_resolve": "Instrument the retry branch with a counter.",
            },
        ],
    }


def format_report(audit: Dict[str, Any]) -> str:
    s = audit["summary"]
    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A("ATL DECISION LOOP — STATIC AUDIT")
    A("=" * 78)
    A(f"  scanned {audit['files_scanned']} files under {audit['backend_root']}")
    A(f"  LLM call sites: {s['total_call_sites']} "
      f"({s['call_sites_in_loops']} inside loops)")
    A("")
    A(f"  Calls per decision FIXED? {'yes' if s['calls_per_decision_is_fixed'] else 'NO'}")
    if not s["calls_per_decision_is_fixed"]:
        A("    At least one call site is inside a loop, so the count is "
          "data-dependent, not a constant.")
    A("")
    A("  Call sites by function:")
    for fn, cs in s["by_function"].items():
        A(f"    {fn}")
        for c in cs:
            loop = f"  [in {c['loops'][0]['kind']}]" if c["loops"] else ""
            A(f"      {c['file']}:{c['line']}  max_tokens={c['max_tokens']}{loop}")
    A("")
    A(f"  max_tokens settings seen: {s['max_tokens_settings']}")
    A(f"  {s['sequential_vs_concurrent']['note']}")

    if s["retry_surfaces"]:
        A("")
        A("  Retry surfaces (multiply calls per decision):")
        for r in s["retry_surfaces"]:
            A(f"    {r['file']}:{r['line']} in {r['function']}")

    for key, comp in audit["prompt_composition"].items():
        A("")
        A(f"  Prompt composition — {key}")
        if not comp.get("available"):
            A(f"    unavailable: {comp.get('reason')}")
            continue
        A(f"    static segments : {comp['static_segment_count']}")
        A(f"    dynamic segments: {comp['dynamic_segment_count']}")
        for d in comp["dynamic_segments"][:8]:
            flag = " (conditional)" if d.get("conditional") else ""
            A(f"      line {d['line']}: {d['via']}({d['source_expression']}){flag}")

    A("")
    A("  UNKNOWNS — not determinable from the code alone:")
    for u in audit["unknowns"]:
        A(f"    - {u['unknown']}")
        A(f"        why: {u['why']}")
        A(f"        fix: {u['how_to_resolve']}")
    return "\n".join(lines)


DEFAULT_PROMPT_BUILDERS = (
    ("dashboard/backend/infrastructure/llm/pipeline_runner.py", "_build_step_prompt"),
    ("dashboard/backend/infrastructure/llm/pipeline_runner.py", "_build_post_trade_prompt"),
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Static audit of ATL's decision loop.")
    ap.add_argument("--backend-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    audit = audit_paths(args.backend_root, DEFAULT_PROMPT_BUILDERS)
    print(format_report(audit))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(audit, fh, indent=2)
        print(f"\n[audit] {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
