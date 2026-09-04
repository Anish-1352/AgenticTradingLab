#!/usr/bin/env python3
"""Check which of this project's findings upstream has overtaken.

    python benchmarks/analysis/upstream_preflight.py --out results/preflight.json

Two prior findings went stale within days of being written, and every
``dashboard/`` line citation in the Phase 21 report drifted once upstream
landed 533 commits. This runs the checks that catch that, against
``origin/main`` rather than the local tree, and emits JSON so the report
built from it cannot claim something the repository no longer says.

It reads git only. It never checks anything out and never writes to the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_ROOT, ".."))

UPSTREAM = "origin/main"

# Each probe is a claim this project has made, plus the anchor whose presence
# or absence upstream decides whether the claim still holds. The anchor is a
# string from the code, not a line number: line numbers are exactly what drifts.
PROBES = [
    {
        "id": "per_call_usage_table",
        "claim": "no per-call usage table exists",
        "paths": ["dashboard/"],
        "pattern": r"llm_call_usage",
        "holds_when": "absent",
    },
    {
        "id": "attempt_level_column",
        "claim": "nothing records which attempt a call was",
        "paths": ["dashboard/"],
        "pattern": r"attempt_index|retry_count|attempt_number",
        "holds_when": "absent",
    },
    {
        "id": "llm_decisions_persisted",
        "claim": "llm_decisions is never written to agent_runs",
        "paths": ["dashboard/backend/database.py"],
        "pattern": r"llm_decisions",
        "holds_when": "absent",
    },
    {
        "id": "retry_loop_unchanged",
        "claim": "the no-text retry loop is still five attempts on one trigger",
        "paths": ["dashboard/backend/domain/backtesting/portfolio_manager.py"],
        "pattern": r"no_text_retries = 4",
        "holds_when": "present",
    },
    {
        "id": "retry_trigger_unchanged",
        "claim": "the only trigger is an AttributeError carrying 'No text content'",
        "paths": ["dashboard/backend/domain/backtesting/portfolio_manager.py"],
        "pattern": r'"No text content" not in str',
        "holds_when": "present",
    },
    {
        "id": "reasoning_off_rescue",
        "claim": "the fifth attempt rescues by forcing reasoning off",
        "paths": ["dashboard/backend/domain/backtesting/portfolio_manager.py"],
        "pattern": r'OPENROUTER_REASONING_EFFORT"\] = "none"',
        "holds_when": "present",
    },
    {
        "id": "backtest_client_has_no_timeout",
        "claim": "the backtest LLM client is built with no timeout or max_retries",
        "paths": ["dashboard/backend/infrastructure/llm/providers/openrouter.py"],
        "pattern": r"timeout|max_retries",
        "holds_when": "absent",
    },
    {
        "id": "pipeline_decisions_not_persisted",
        "claim": "the pipeline runtime writes no per-decision rows",
        "paths": ["dashboard/backend/domain/backtesting/engine.py"],
        "pattern": r"if self\.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:",
        "holds_when": "present",
    },
    {
        "id": "hourly_interval_hardcoded",
        "claim": "the bar interval is hardcoded to one hour at the fetch site",
        "paths": ["dashboard/backend/infrastructure/market_data/alpaca_bars.py"],
        "pattern": r"timeframe=self\.TimeFrame\.Hour",
        "holds_when": "present",
    },
    {
        "id": "structured_outputs_unused",
        "claim": "the backtest path requests no JSON schema / structured output",
        "paths": ["dashboard/backend/infrastructure/llm/"],
        "pattern": r"response_format|json_schema|guided_json",
        "holds_when": "absent",
    },
]

# Anchors whose upstream line number the reports cite. The code moving is
# fine; the citation pointing at the wrong line is not.
CITATION_ANCHORS = [
    ("dashboard/backend/domain/backtesting/portfolio_manager.py",
     "no_text_retries = 4", "the retry loop"),
    ("dashboard/backend/domain/backtesting/portfolio_manager.py",
     "retry {attempt + 1}/{no_text_retries}", "the retry print"),
    ("dashboard/backend/domain/backtesting/portfolio_manager.py",
     "Final rescue", "the rescue call"),
    ("dashboard/backend/database.py",
     "llm_calls INTEGER DEFAULT 0", "the billing counter column"),
    ("dashboard/backend/database.py",
     "CREATE TABLE IF NOT EXISTS agent_runs", "the run table"),
    ("dashboard/backend/domain/backtesting/engine.py",
     "db.insert_decisions", "the decision write site"),
]


def _git(*args: str) -> subprocess.CompletedProcess:
    # errors="replace": merge-tree streams binary blobs (built frontend
    # assets) and would otherwise raise on the first non-UTF-8 byte.
    return subprocess.run(["git", "-C", _REPO_ROOT, *args],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _rev(ref: str) -> Optional[str]:
    r = _git("rev-parse", ref)
    return r.stdout.strip() if r.returncode == 0 else None


def run_probe(p: Dict[str, Any]) -> Dict[str, Any]:
    r = _git("grep", "-nE", p["pattern"], UPSTREAM, "--", *p["paths"])
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    # A match inside the upstream test suite proves the behaviour is asserted,
    # not that production code grew the feature; keep them apart.
    prod = [h for h in hits if "/tests/" not in h and "test_" not in h]
    found = bool(prod)
    still_holds = (not found) if p["holds_when"] == "absent" else found
    return {
        "id": p["id"], "claim": p["claim"], "holds_when": p["holds_when"],
        "found_upstream": found, "still_holds": still_holds,
        "n_matches": len(prod), "sample": prod[:4],
    }


def upstream_line(path: str, anchor: str) -> Optional[int]:
    r = _git("show", f"{UPSTREAM}:{path}")
    if r.returncode != 0:
        return None
    for i, line in enumerate(r.stdout.splitlines(), 1):
        if anchor in line:
            return i
    return None


_CIT = re.compile(r"`([A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`")


def check_report_citations(report_dir: str) -> List[Dict[str, Any]]:
    """Every `file.py:line` in the generated reports, checked for range."""
    out: List[Dict[str, Any]] = []
    tracked = _git("ls-tree", "-r", "--name-only", UPSTREAM).stdout.splitlines()
    by_base: Dict[str, List[str]] = {}
    for t in tracked:
        if t.endswith(".py"):
            by_base.setdefault(os.path.basename(t), []).append(t)
    lengths: Dict[str, int] = {}
    for fn in sorted(os.listdir(report_dir)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(report_dir, fn), encoding="utf-8") as fh:
            text = fh.read()
        for m in _CIT.finditer(text):
            cited, lo, hi = m.group(1), int(m.group(2)), m.group(3)
            cands = by_base.get(os.path.basename(cited), [])
            path = next((c for c in cands if c.endswith(cited)),
                        cands[0] if cands else None)
            if path is None:
                out.append({"report": fn, "citation": m.group(0),
                            "status": "file_absent_upstream"})
                continue
            if path not in lengths:
                r = _git("show", f"{UPSTREAM}:{path}")
                lengths[path] = len(r.stdout.splitlines())
            top = int(hi) if hi else lo
            out.append({
                "report": fn, "citation": m.group(0), "path": path,
                "upstream_lines": lengths[path],
                "status": "in_range" if top <= lengths[path] else "out_of_range",
            })
    return out


def branch_merges_cleanly(branch: str) -> Dict[str, Any]:
    base = _git("merge-base", branch, UPSTREAM).stdout.strip()
    if not base:
        return {"branch": branch, "resolved": False}
    r = _git("merge-tree", base, branch, UPSTREAM)
    both = [l for l in r.stdout.splitlines() if l.startswith("changed in both")]
    files = re.findall(r"^\s+base\s+\d+\s+\w+\s+(\S+)$", r.stdout, re.M)
    return {"branch": branch, "resolved": True, "merge_base": base[:8],
            "files_changed_in_both": len(both),
            "conflict_candidates": sorted(set(files))[:12],
            "merges_cleanly": len(both) == 0}


def build(branches: List[str]) -> Dict[str, Any]:
    return {
        "upstream": UPSTREAM,
        "upstream_head": _rev(UPSTREAM),
        "local_head": _rev("HEAD"),
        "commits_behind": int(_git("rev-list", "--count",
                                   f"HEAD..{UPSTREAM}").stdout.strip() or 0),
        "probes": [run_probe(p) for p in PROBES],
        "citation_anchors": [
            {"path": p, "anchor": a, "describes": d,
             "upstream_line": upstream_line(p, a)}
            for p, a, d in CITATION_ANCHORS],
        "report_citations": check_report_citations(_HERE),
        "branches": [branch_merges_cleanly(b) for b in branches],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--branch", action="append", default=[])
    args = ap.parse_args(argv)
    if _rev(UPSTREAM) is None:
        print(f"{UPSTREAM} not present; run `git fetch origin` first",
              file=sys.stderr)
        return 2
    data = build(args.branch or ["feature/serving-cost-reduction"])

    print(f"upstream {data['upstream_head'][:8]}  "
          f"local {data['local_head'][:8]}  "
          f"behind {data['commits_behind']}\n")
    for p in data["probes"]:
        mark = "holds" if p["still_holds"] else "SUPERSEDED"
        print(f"  [{mark:10s}] {p['claim']}")
        if not p["still_holds"] and p["sample"]:
            print(f"               upstream: {p['sample'][0][:96]}")
    bad = [c for c in data["report_citations"] if c["status"] != "in_range"]
    print(f"\n  citations: {len(data['report_citations'])} checked, "
          f"{len(bad)} out of range")
    for c in bad:
        print(f"    {c['report']}: {c['citation']} {c['status']}")
    for b in data["branches"]:
        if not b.get("resolved"):
            continue
        if b["merges_cleanly"]:
            print(f"\n  {b['branch']}: merges cleanly")
        else:
            print(f"\n  {b['branch']}: does NOT merge cleanly — "
                  f"{b['files_changed_in_both']} files changed in both")
            for f in b["conflict_candidates"]:
                print(f"    {f}")
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(_BENCH_ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        print(f"\n  wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
