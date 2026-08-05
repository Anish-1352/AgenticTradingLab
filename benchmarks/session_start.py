#!/usr/bin/env python3
"""Record what this session is running on, and shout if the GPU changed.

    python benchmarks/session_start.py

Run at the top of **every** session, after ``colab_bootstrap.py``. Appends one
JSON line to a persistent log on Drive and prints a comparison against the
previous session.

WHY A SESSION-LEVEL LOG WHEN RUNS ALREADY CARRY A UUID
------------------------------------------------------
Every run manifest already records ``gpu_uuid``, so a hardware change is
*recoverable* from the results. The problem is **when** you find out: after the
sweep, with the hours already spent and two arms sitting on different cards.

This puts the same fact in front of you in the first ten seconds of the
session, before anything expensive starts. Colab reallocates hardware across
sessions and across restarts — that is exactly how ``GPU-2ddfc69f…`` became
``GPU-afb936de…`` mid-study.

A UUID change is not fatal. It means: **re-run the calibration** so the new
card's noise floor is measured, and treat any comparison spanning the change as
cross-hardware, stating it as such. What is fatal is not noticing.

EXIT CODES
----------
    0   Recorded. GPU matches the previous session (or this is the first one).
    76  Recorded, but the **GPU CHANGED** since the last session.
    1   Could not establish the environment (no nvidia-smi, no CUDA, …).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from common.manifest import _git, sha256_text  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_GPU_CHANGED = 76

DEFAULT_LOG = "/content/drive/MyDrive/atl_bench/session_log.jsonl"
TRACKED_PACKAGES = ("torch", "vllm", "transformers", "nvidia-ml-py")


def _pkg_version(name: str) -> Optional[str]:
    """Metadata lookup — deliberately does not import the package."""
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover
        return None


def query_gpu(
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Optional[str]]:
    """GPU UUID and name straight from nvidia-smi.

    nvidia-smi rather than NVML-through-Python: it is the one source that works
    before any CUDA context exists and cannot be perturbed by a half-resolved
    torch install, which is precisely the situation this script runs in.
    """
    try:
        proc = runner(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        if getattr(proc, "returncode", 1) != 0:
            return {"uuid": None, "name": None, "memory_total": None,
                    "driver_version": None,
                    "error": (getattr(proc, "stderr", "") or "").strip()[:400]}
        line = (proc.stdout or "").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        while len(parts) < 4:
            parts.append(None)
        return {"uuid": parts[0], "name": parts[1],
                "memory_total": parts[2], "driver_version": parts[3], "error": None}
    except FileNotFoundError:
        return {"uuid": None, "name": None, "memory_total": None,
                "driver_version": None, "error": "nvidia-smi not found"}
    except Exception as exc:  # noqa: BLE001
        return {"uuid": None, "name": None, "memory_total": None,
                "driver_version": None, "error": f"{type(exc).__name__}: {exc}"}


def pip_freeze_sha256(
    runner: Callable[..., Any] = subprocess.run,
) -> Dict[str, Optional[str]]:
    try:
        proc = runner([sys.executable, "-m", "pip", "freeze"],
                      capture_output=True, text=True, timeout=300)
        if getattr(proc, "returncode", 1) != 0:
            return {"sha256": None, "package_count": None}
        text = proc.stdout or ""
        return {
            "sha256": sha256_text(text),
            "package_count": len([ln for ln in text.splitlines() if ln.strip()]),
        }
    except Exception:  # noqa: BLE001
        return {"sha256": None, "package_count": None}


def build_record(
    gpu_fn: Callable[[], Dict[str, Optional[str]]] = query_gpu,
    freeze_fn: Callable[[], Dict[str, Optional[str]]] = pip_freeze_sha256,
    version_fn: Callable[[str], Optional[str]] = _pkg_version,
) -> Dict[str, Any]:
    gpu = gpu_fn()
    freeze = freeze_fn()
    branch_sha = _git("rev-parse", "HEAD")
    if branch_sha and _git("status", "--porcelain"):
        branch_sha = f"{branch_sha}-dirty"

    return {
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_uuid": gpu.get("uuid"),
        "gpu_name": gpu.get("name"),
        "gpu_memory_total": gpu.get("memory_total"),
        "driver_version": gpu.get("driver_version"),
        "gpu_error": gpu.get("error"),
        "packages": {name: version_fn(name) for name in TRACKED_PACKAGES},
        "pip_freeze_sha256": freeze.get("sha256"),
        "pip_package_count": freeze.get("package_count"),
        "branch_sha": branch_sha,
        "hf_home": os.environ.get("HF_HOME"),
    }


def read_log(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line is expected if a session was preempted
                # mid-write. Skip it rather than losing the whole history.
                continue
    return out


def append_log(path: str, record: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def compare_to_previous(
    record: Dict[str, Any], history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Diff this session against the most recent one that saw a GPU."""
    prev = None
    for entry in reversed(history):
        if entry.get("gpu_uuid"):
            prev = entry
            break

    if prev is None:
        return {"first_session": True, "gpu_changed": False, "changes": []}

    changes: List[str] = []
    gpu_changed = bool(
        record.get("gpu_uuid") and prev.get("gpu_uuid")
        and record["gpu_uuid"] != prev["gpu_uuid"]
    )
    if gpu_changed:
        changes.append(f"gpu_uuid: {prev['gpu_uuid']} -> {record['gpu_uuid']}")
    for key in ("gpu_name", "driver_version", "pip_freeze_sha256", "branch_sha"):
        if prev.get(key) != record.get(key):
            changes.append(f"{key}: {prev.get(key)} -> {record.get(key)}")
    for pkg in TRACKED_PACKAGES:
        a = (prev.get("packages") or {}).get(pkg)
        b = (record.get("packages") or {}).get(pkg)
        if a != b:
            changes.append(f"{pkg}: {a} -> {b}")

    return {
        "first_session": False,
        "gpu_changed": gpu_changed,
        "previous_timestamp": prev.get("utc_timestamp"),
        "changes": changes,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Record session provenance.")
    ap.add_argument("--log", default=os.environ.get("ATL_SESSION_LOG", DEFAULT_LOG))
    ap.add_argument("--no-append", action="store_true",
                    help="print only; do not write to the log")
    args = ap.parse_args(argv)

    print("=" * 68)
    print("SESSION START — provenance record")
    print("=" * 68)

    record = build_record()
    history = read_log(args.log)
    cmp = compare_to_previous(record, history)

    print(f"\n  timestamp     {record['utc_timestamp']}")
    print(f"  GPU           {record['gpu_name'] or 'UNKNOWN'}")
    print(f"  GPU UUID      {record['gpu_uuid'] or 'UNKNOWN'}")
    print(f"  memory        {record['gpu_memory_total'] or 'UNKNOWN'}")
    print(f"  driver        {record['driver_version'] or 'UNKNOWN'}")
    for pkg in TRACKED_PACKAGES:
        print(f"  {pkg:<13} {record['packages'].get(pkg) or 'ABSENT'}")
    print(f"  pip freeze    {record['pip_freeze_sha256'] or 'UNKNOWN'}")
    print(f"  branch        {record['branch_sha'] or 'UNKNOWN'}")
    print(f"  HF_HOME       {record['hf_home'] or 'NOT SET'}")

    if record.get("gpu_error"):
        print(f"\n  GPU query error: {record['gpu_error']}")

    if not args.no_append:
        path = append_log(args.log, record)
        print(f"\n  logged to     {path}  ({len(history) + 1} session(s))")
    else:
        print("\n  --no-append: not written")

    if record.get("hf_home") is None:
        print("\n  WARNING: HF_HOME is not set. Set it to a Drive path BEFORE any")
        print("           model load or ~15 GB of weights re-download on every")
        print("           preemption.")

    if cmp["first_session"]:
        print("\n  First recorded session — nothing to compare against.")
    elif cmp["gpu_changed"]:
        print("\n" + "!" * 68)
        print("!!  GPU CHANGED SINCE THE PREVIOUS SESSION")
        print("!" * 68)
        for c in cmp["changes"]:
            print(f"  {c}")
        print("")
        print("  This is a DIFFERENT PHYSICAL CARD. Consequences:")
        print("    - Re-run calibration_run.sh now; the previous noise floor")
        print("      was measured on the other card and does not transfer.")
        print("    - Any comparison spanning this boundary is CROSS-HARDWARE.")
        print("      Say so in the write-up, or re-run both arms on this card.")
        print("    - Check manifests: runs before and after carry different")
        print("      gpu_uuid and are not directly comparable.")
        print("!" * 68)
        return EXIT_GPU_CHANGED
    else:
        print(f"\n  Same GPU as the previous session "
              f"({cmp['previous_timestamp']}). Comparable.")
        if cmp["changes"]:
            print("\n  But these changed since then:")
            for c in cmp["changes"]:
                print(f"    - {c}")
            print("\n  A pip_freeze_sha256 change means the software stack moved;")
            print("  runs either side of it are not strictly comparable.")

    if record.get("gpu_uuid") is None:
        print("\n  Could not read a GPU UUID — is this a GPU runtime?")
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
