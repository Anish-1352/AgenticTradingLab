#!/usr/bin/env python3
"""Take a fresh Colab A100 to a verified working state in one command.

    python benchmarks/colab_bootstrap.py

Idempotent. If the stack is already correct it reports and exits 0 **without
touching pip** — re-running it is free, and that matters because the
alternative is re-deriving this by hand, which has now happened three times.

EXIT CODES
----------
    0   Stack is correct. No restart needed. Continue to session_start.py.
    75  Changes were applied. **RESTART THE RUNTIME**, then re-run this script
        (it will exit 0) and continue.
    1   Failed, or left in a half-resolved state. The message says exactly what
        is wrong. Do not proceed — a half-resolved stack produces runs that
        look fine and are not comparable to anything.

WHY RESTARTS ARE THE THING TO MINIMISE
--------------------------------------
A runtime restart can make Colab hand back a **different physical GPU**. That
already happened mid-study (``GPU-2ddfc69f…`` -> ``GPU-afb936de…``), which
splits arms across two cards and puts an unknown hardware term inside the
B->C delta the study exists to measure.

So this script batches **every** package mutation into one pass and asks for
**at most one** restart, rather than the three-restart hand sequence it
replaces. If nothing needs changing, it asks for none.

THE DEPENDENCY SITUATION (established empirically — do not re-derive)
---------------------------------------------------------------------
1. Colab ships torch 2.11.0+**cu128** and a ``pynvml`` distribution that
   conflicts with ``nvidia-ml-py``.
2. vLLM 0.26.0 needs a **cu13** torch. On cu128 it dies at import with
   ``ImportError: libcudart.so.13``.
3. ``pip install vllm`` (after removing torch) resolves torch to 2.11.0+cu130.
4. torchvision then mismatches, and **both** wheels are dead ends:
     * cu130 — broken compiled extension: ``operator torchvision::nms does not
       exist`` at import.
     * cu128 — torch's ``_check_cuda_version()`` rejects it: *"PyTorch has CUDA
       Version=13.0 and torchvision has CUDA Version=12.8"*.
   There is **no installable torchvision** for torch 2.11.0+cu130 here.
5. transformers 5.13.1 works fine without torchvision. Arm B is unaffected.
6. vLLM does **not**: ``kernel_warmup`` unconditionally imports MiniMax-M3
   vision code needing ``torchvision.transforms.InterpolationMode``, even for a
   text-only Qwen2 model. That is what the torchvision shim exists for; this
   script deliberately leaves torchvision **absent** and expects the shim to
   cover vLLM's import.
7. Removing torchvision/torchaudio/torchcodec has been observed to take torch
   and vllm with it. This script therefore removes them **last** and then
   **verifies** the whole target state, failing loudly rather than leaving a
   stack that imports but is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_RESTART_REQUIRED = 75

# --- target state ---------------------------------------------------------
# torch is matched on the CUDA local version rather than an exact pin: what
# actually matters is that it is a cu13 build, which is what vLLM 0.26 links
# against. Pinning 2.11.0 exactly would make this script fail the day pip
# resolves a compatible 2.11.1.
REQUIRED_TORCH_CUDA_TAG = "cu130"
EXPECTED_TORCH_VERSION = "2.11.0+cu130"
REQUIRED_PACKAGES = ("torch", "vllm", "transformers")
# Must NOT be installed. See points 4 and 6 above.
FORBIDDEN_PACKAGES = ("torchvision", "torchaudio", "torchcodec")
# pynvml and nvidia-ml-py provide the same import name and conflict.
CONFLICTING_NVML = "pynvml"
REQUIRED_NVML = "nvidia-ml-py"

HF_HOME_DEFAULT = "/content/drive/MyDrive/atl_bench/hf_cache"


# --------------------------------------------------------------------------
# detection (never imports the packages)
# --------------------------------------------------------------------------


def _installed_version(name: str) -> Optional[str]:
    """Version from package metadata. Deliberately does NOT import the package.

    Importing torch here would load CUDA into this process and cost seconds on
    every idempotency check; importing vllm would trip the very torchvision
    problem this script exists to arrange around.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - importlib.metadata is stdlib
        return None


def detect_state(version_fn: Callable[[str], Optional[str]] = _installed_version) -> Dict[str, Any]:
    """Snapshot the installed versions relevant to the target state."""
    names = list(REQUIRED_PACKAGES) + list(FORBIDDEN_PACKAGES) + [
        CONFLICTING_NVML, REQUIRED_NVML,
    ]
    versions = {name: version_fn(name) for name in names}
    torch_v = versions.get("torch")
    return {
        "versions": versions,
        "torch_is_cu130": bool(torch_v and REQUIRED_TORCH_CUDA_TAG in torch_v),
        "torch_version_expected": EXPECTED_TORCH_VERSION,
    }


# --------------------------------------------------------------------------
# planning (pure — no side effects, so tests can assert "does nothing")
# --------------------------------------------------------------------------


def plan_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Decide what must change. Pure: returns a list, touches nothing.

    An empty list means the stack is already correct and pip must not run —
    that is the property the idempotency test pins, because a bootstrap that
    reinstalls on every invocation would force a restart, and a restart can
    change the GPU underneath the study.

    Order is load-bearing:
      1. NVML swap first — cheap, independent of the torch mess.
      2. torch/vLLM resolution second — ``pip install vllm`` is what pulls a
         cu130 torch, and it may drag torchvision back in as a dependency.
      3. torchvision removal LAST — so nothing can reintroduce it afterwards.
    """
    v = state["versions"]
    actions: List[Dict[str, Any]] = []

    # 1. NVML: exactly one of the two distributions may be present.
    if v.get(CONFLICTING_NVML) is not None:
        actions.append({
            "id": "uninstall_pynvml",
            "why": f"{CONFLICTING_NVML} conflicts with {REQUIRED_NVML} (same import name)",
            "cmd": ["pip", "uninstall", "-y", CONFLICTING_NVML],
        })
    if v.get(REQUIRED_NVML) is None:
        actions.append({
            "id": "install_nvidia_ml_py",
            "why": f"{REQUIRED_NVML} provides NVML; every VRAM number depends on it",
            "cmd": ["pip", "install", "-q", REQUIRED_NVML],
        })

    # 2. torch + vLLM. Rebuild only if torch is not a cu13 build or vllm is
    #    missing — never "just to be sure", which would cost a restart.
    needs_torch_rebuild = (not state["torch_is_cu130"]) or v.get("vllm") is None
    if needs_torch_rebuild:
        actions.append({
            "id": "uninstall_torch_stack",
            "why": (
                f"torch={v.get('torch')} vllm={v.get('vllm')}: vLLM 0.26 needs a "
                f"{REQUIRED_TORCH_CUDA_TAG} torch or it fails with "
                f"ImportError: libcudart.so.13"
            ),
            "cmd": ["pip", "uninstall", "-y", "vllm", "torch", "nvidia-cuda-runtime"],
        })
        actions.append({
            "id": "install_vllm",
            "why": "installing vllm is what resolves torch to a cu130 build",
            "cmd": ["pip", "install", "-q", "vllm"],
        })

    # 3. Forbidden packages, removed last. Also re-checked after execution,
    #    because step 2 can reintroduce torchvision as a vllm dependency.
    present_forbidden = [p for p in FORBIDDEN_PACKAGES if v.get(p) is not None]
    if present_forbidden or needs_torch_rebuild:
        actions.append({
            "id": "uninstall_forbidden",
            "why": (
                "no installable torchvision exists for torch 2.11.0+cu130 "
                "(cu130 wheel has a broken extension; cu128 wheel is rejected by "
                "torch's CUDA version check). The shim covers vLLM's import."
            ),
            "cmd": ["pip", "uninstall", "-y", *FORBIDDEN_PACKAGES],
            # Safe when they are absent: pip exits 0 with "not installed".
            "tolerate_failure": True,
        })

    return actions


def verify_state(state: Dict[str, Any]) -> List[str]:
    """Return a list of problems. Empty means the stack is correct."""
    v = state["versions"]
    problems: List[str] = []

    for pkg in REQUIRED_PACKAGES:
        if v.get(pkg) is None:
            problems.append(
                f"{pkg} is NOT installed. Removing torchvision has been observed "
                f"to take torch and vllm with it — re-run this script."
            )
    if v.get("torch") and not state["torch_is_cu130"]:
        problems.append(
            f"torch is {v['torch']}, which is not a {REQUIRED_TORCH_CUDA_TAG} build. "
            f"vLLM 0.26 will fail with ImportError: libcudart.so.13."
        )
    for pkg in FORBIDDEN_PACKAGES:
        if v.get(pkg) is not None:
            problems.append(
                f"{pkg} {v[pkg]} is installed and must not be. There is no working "
                f"torchvision for torch+cu130 here; its presence breaks the import "
                f"path the shim covers."
            )
    if v.get(CONFLICTING_NVML) is not None:
        problems.append(
            f"{CONFLICTING_NVML} is installed and conflicts with {REQUIRED_NVML}. "
            f"NVML sampling underpins every VRAM measurement."
        )
    if v.get(REQUIRED_NVML) is None:
        problems.append(f"{REQUIRED_NVML} is NOT installed; NVML sampling will be unavailable.")
    return problems


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _default_runner(cmd: Sequence[str]) -> "subprocess.CompletedProcess":
    # 'pip' is rewritten to this interpreter's pip so the script cannot install
    # into a different Python than the one the notebook is running.
    argv = list(cmd)
    if argv and argv[0] == "pip":
        argv = [sys.executable, "-m", "pip", *argv[1:]]
    print(f"    $ {' '.join(argv)}")
    return subprocess.run(argv, text=True, capture_output=False)


def execute(
    actions: List[Dict[str, Any]],
    runner: Callable[[Sequence[str]], Any] = _default_runner,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, action in enumerate(actions, 1):
        print(f"\n[{i}/{len(actions)}] {action['id']}")
        print(f"    why: {action['why']}")
        proc = runner(action["cmd"])
        rc = getattr(proc, "returncode", 0)
        ok = rc == 0 or action.get("tolerate_failure", False)
        results.append({"id": action["id"], "returncode": rc, "ok": ok})
        if not ok:
            print(f"    FAILED (exit {rc})")
            break
        print("    ok")
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_state(state: Dict[str, Any], header: str) -> None:
    v = state["versions"]
    print(f"\n{header}")
    print("  " + "-" * 58)
    for pkg in REQUIRED_PACKAGES:
        got = v.get(pkg) or "ABSENT"
        mark = "ok " if v.get(pkg) else "!! "
        if pkg == "torch" and v.get(pkg):
            mark = "ok " if state["torch_is_cu130"] else "!! "
        print(f"  {mark}{pkg:<16} {got}")
    for pkg in FORBIDDEN_PACKAGES:
        got = v.get(pkg)
        print(f"  {'!! ' if got else 'ok '}{pkg:<16} "
              f"{got + '  (MUST BE ABSENT)' if got else 'absent (correct)'}")
    print(f"  {'!! ' if v.get(CONFLICTING_NVML) else 'ok '}{CONFLICTING_NVML:<16} "
          f"{v.get(CONFLICTING_NVML) or 'absent (correct)'}")
    print(f"  {'ok ' if v.get(REQUIRED_NVML) else '!! '}{REQUIRED_NVML:<16} "
          f"{v.get(REQUIRED_NVML) or 'ABSENT'}")


def check_hf_home() -> Optional[str]:
    """HF_HOME must point at Drive BEFORE any model load.

    On the local disk the ~15 GB of weights is re-downloaded after every
    preemption, and a mid-study re-download is both slow and a chance for the
    cache to end up holding a different revision.
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return (f"HF_HOME is not set. Set it to {HF_HOME_DEFAULT} BEFORE any "
                f"model load, or weights re-download on every preemption.")
    if "drive" not in hf_home.lower():
        return (f"HF_HOME={hf_home} does not look like a Drive path. Weights "
                f"will not survive preemption.")
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap a Colab A100 for the benchmark.")
    ap.add_argument("--check-only", action="store_true",
                    help="report state and what would change; never run pip")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    print("=" * 68)
    print("COLAB BOOTSTRAP — ATL GPU serving benchmark")
    print("=" * 68)

    before = detect_state()
    print_state(before, "Detected state:")

    actions = plan_actions(before)

    if not actions:
        problems = verify_state(before)
        if problems:
            # Nothing to do, yet still wrong: a state plan_actions cannot fix.
            print("\n" + "!" * 68)
            print("STACK IS INCONSISTENT and no automatic action applies:")
            for p in problems:
                print(f"  - {p}")
            print("!" * 68)
            return EXIT_FAILED
        print("\n" + "=" * 68)
        print("ALREADY CORRECT — nothing installed, no restart needed.")
        print("=" * 68)
        hf = check_hf_home()
        if hf:
            print(f"\n  WARNING: {hf}")
        print("\nNext: python benchmarks/session_start.py")
        return EXIT_OK

    print(f"\n{len(actions)} action(s) required:")
    for a in actions:
        print(f"  - {a['id']}: {a['why']}")

    if args.check_only:
        print("\n--check-only: nothing executed.")
        return EXIT_RESTART_REQUIRED

    print("\n" + "-" * 68)
    print("Applying ALL changes in one pass, so at most ONE restart is needed.")
    print("A restart can make Colab hand back a different physical GPU, which")
    print("splits the study across two cards. This is why the sequence is")
    print("batched rather than interleaved with imports.")
    print("-" * 68)

    results = execute(actions)
    failed = [r for r in results if not r["ok"]]

    after = detect_state()
    print_state(after, "Resulting state:")
    problems = verify_state(after)

    payload = {
        "before": before["versions"],
        "after": after["versions"],
        "actions": [a["id"] for a in actions],
        "results": results,
        "problems": problems,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)

    if failed or problems:
        print("\n" + "!" * 68)
        print("BOOTSTRAP FAILED — the stack is half-resolved. DO NOT PROCEED.")
        for r in failed:
            print(f"  - action {r['id']} exited {r['returncode']}")
        for p in problems:
            print(f"  - {p}")
        print("")
        print("  Re-running this script is safe and is usually the fix: it is")
        print("  idempotent and will only touch what is still wrong.")
        print("!" * 68)
        return EXIT_FAILED

    hf = check_hf_home()
    print("\n" + "=" * 68)
    print("CHANGES APPLIED SUCCESSFULLY.")
    print("")
    print("  >>> RESTART THE RUNTIME NOW <<<")
    print("      Runtime -> Restart session")
    print("")
    print("  Then re-run this cell. It will report ALREADY CORRECT and exit 0.")
    print("")
    print("  This is the ONLY restart the setup needs. After restarting, check")
    print("  the GPU UUID in session_start.py against the previous session —")
    print("  a restart can move you to a different physical A100.")
    print("=" * 68)
    if hf:
        print(f"\n  WARNING: {hf}")
    return EXIT_RESTART_REQUIRED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
