"""Shared plumbing for the ablations: subprocess launch, result loading, tables.

WHY EVERY ARM OF AN ABLATION IS A FRESH SUBPROCESS
--------------------------------------------------
An ablation compares two engine configurations. Running both in one process
would share a CUDA context, an allocator, and — decisively — a KV cache pool
that vLLM pre-allocates at startup and a prefix cache that persists. The second
configuration would then be measured against state the first one warmed, and
the difference attributed to the variable under test would partly be
contamination.

Each condition therefore runs as its own ``python runners/...`` invocation:
fresh CUDA context, fresh KV pool, fresh prefix cache. It costs an engine
startup per condition and buys a result that means what it says.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_ROOT, ".."))

ARM_C = os.path.join(BENCH_ROOT, "runners", "bench_vllm_optimized.py")
ARM_B = os.path.join(BENCH_ROOT, "runners", "bench_hf_baseline.py")

__all__ = [
    "BENCH_ROOT",
    "REPO_ROOT",
    "ARM_B",
    "ARM_C",
    "run_condition",
    "load_summary",
    "level_for",
    "pct_delta",
    "fmt",
    "render_table",
    "write_comparison",
]


def run_condition(
    script: str,
    run_id: str,
    extra_args: Sequence[str],
    out_dir: str,
    dry_run: bool = False,
    echo: bool = True,
) -> Dict[str, Any]:
    """Run one condition as a fresh subprocess. Returns a record of the attempt.

    Failures are captured, not raised: one condition OOMing at high concurrency
    is a result worth reporting next to the conditions that survived, and losing
    an entire ablation to it would be worse than reporting the gap.
    """
    cmd = [
        sys.executable, script,
        "--run-id", run_id,
        "--out-dir", out_dir,
        *extra_args,
    ]
    if dry_run:
        cmd.append("--dry-run")

    if echo:
        print(f"\n$ {' '.join(cmd)}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if echo:
        tail = (proc.stdout or "").strip().splitlines()
        for line in tail[-18:]:
            print(f"  | {line}")
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            for line in err[-12:]:
                print(f"  ! {line}")

    return {
        "run_id": run_id,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_s": elapsed,
        "summary_path": os.path.join(out_dir, f"{run_id}_summary.json"),
        "manifest_path": os.path.join(out_dir, f"{run_id}_manifest.json"),
        "stdout_tail": "\n".join((proc.stdout or "").strip().splitlines()[-40:]),
        "stderr_tail": "\n".join((proc.stderr or "").strip().splitlines()[-40:]),
    }


def load_summary(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def level_for(summary: Optional[Dict[str, Any]], concurrency: int) -> Optional[Dict[str, Any]]:
    """Pull one concurrency level out of a run summary."""
    if not summary:
        return None
    for lv in summary.get("levels", []):
        if int(lv.get("concurrency", -1)) == int(concurrency):
            return lv
    return None


def pct_delta(new: Optional[float], base: Optional[float]) -> Optional[float]:
    """Percent change of ``new`` relative to ``base``. None when undefined."""
    if new is None or base is None:
        return None
    if base == 0:
        return None
    return (new - base) / base * 100.0


def fmt(value: Any, unit: str = "", nd: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{nd}f}{unit}"
    return str(value)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Fixed-width text table, so output is readable in a Colab cell."""
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i])) if i < len(row) else 0)

    def line(cells: Sequence[Any]) -> str:
        return "  ".join(
            str(cells[i] if i < len(cells) else "").ljust(widths[i]) for i in range(cols)
        ).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def write_comparison(out_dir: str, name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path
