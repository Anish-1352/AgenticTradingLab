"""Minimal ``torchvision`` stub so vLLM 0.26 can start without torchvision.

THE PROBLEM
-----------
vLLM 0.26.0's kernel warmup imports MiniMax-M3 vision code unconditionally::

    vllm/model_executor/warmup/kernel_warmup.py
      -> vllm/models/minimax_m3/...
        -> vllm/transformers_utils/processors/minimax_m3.py:20
          -> from torchvision.transforms import InterpolationMode

That fires even for a text-only Qwen2 model and kills EngineCore at startup.

And torchvision **cannot be installed** on torch 2.11.0+cu130:

* the cu130 wheel has a broken compiled extension — ``operator
  torchvision::nms does not exist`` at import;
* the cu128 wheel is rejected by torch's ``_check_cuda_version()`` on the CUDA
  major mismatch.

Both confirmed empirically. This shim is therefore the resolution, not a
stopgap: it supplies the single symbol the Qwen2 path actually touches and
nothing else. It deliberately does **not** attempt to emulate torchvision — a
partial implementation of image ops would be far more dangerous than an
ImportError, because it would fail silently and wrongly on a real vision model.

THE HARD PART: THE CHILD PROCESS
--------------------------------
vLLM runs EngineCore in a **separate process** (the traceback names
``(EngineCore pid=…)``). Patching ``sys.modules`` in the parent does not reach
it, so an in-process shim is useless here.

Whether that child is *forked* or *spawned* depends on vLLM's configured
multiprocessing method, which varies by version and by whether CUDA is already
initialised — so this does not rely on knowing which:

    A real stub package on disk, with its directory prepended to BOTH
    ``sys.path`` and the ``PYTHONPATH`` environment variable.

* **fork** — the child inherits the parent's ``sys.path`` -> covered.
* **spawn** — the child is a fresh interpreter that reads ``PYTHONPATH``
  -> covered.
* **exec of a new python** (worker launchers, ``nsys``/``ncu`` wrappers)
  — inherits the environment -> covered.

A ``sitecustomize.py`` would also survive spawn, but it hijacks *every* Python
process started in the environment and silently loses to any pre-existing
``sitecustomize`` on the path. A named package directory affects exactly one
import name and is inspectable on disk.

NEVER SHADOWING A WORKING INSTALL
---------------------------------
The shim activates only when real torchvision fails to import, and that check
runs in a **subprocess** for two reasons: a failed torchvision import leaves a
broken partial module in ``sys.modules`` that would poison the parent, and the
check must see the interpreter's real import state rather than one this module
already modified. The probe explicitly strips the shim directory from
``PYTHONPATH`` so it cannot detect itself and conclude torchvision works.

If a working torchvision ever becomes installable, the probe succeeds and
``ensure()`` no-ops.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

__all__ = [
    "INTERPOLATION_MEMBERS",
    "SHIM_VERSION",
    "default_shim_root",
    "real_torchvision_importable",
    "materialize",
    "ensure",
    "status",
]

# torchvision's own values, so anything comparing against the string form
# behaves identically.
INTERPOLATION_MEMBERS = (
    ("NEAREST", "nearest"),
    ("BILINEAR", "bilinear"),
    ("BICUBIC", "bicubic"),
    ("BOX", "box"),
    ("HAMMING", "hamming"),
    ("LANCZOS", "lanczos"),
    ("NEAREST_EXACT", "nearest-exact"),
)

# Marks the stub as ours in `torchvision.__version__`, so a confused reader
# (or a later probe) can tell instantly that this is not a real install.
SHIM_VERSION = "0.0.0+atl-shim"

_ENV_MARKER = "ATL_TORCHVISION_SHIM_ROOT"

# The probe must not be able to detect the shim and conclude torchvision works.
# Stripping the shim root from PYTHONPATH is not sufficient on its own: a root
# from an EARLIER session (or a differently-configured one) can still be on the
# path, and filtering only the root we were handed would miss it. So the probe
# imports torchvision and explicitly REJECTS anything carrying the stub marker.
# That check holds no matter how the stub got onto the path.
_PROBE_SNIPPET = (
    "import sys, torchvision\n"
    "if getattr(torchvision, '__atl_shim__', False):\n"
    "    sys.stderr.write('ATL shim detected, not a real torchvision\\n')\n"
    "    raise SystemExit(3)\n"
    "from torchvision.transforms import InterpolationMode\n"
    "print(InterpolationMode.BILINEAR)\n"
)


def default_shim_root() -> str:
    """Where the stub package is materialised.

    A temp directory rather than a path inside the repo: a directory literally
    named ``torchvision`` sitting in the source tree is an accident waiting to
    happen for any tool that walks it, and the shim must never be importable
    except when ``ensure()`` has deliberately activated it.
    """
    return os.environ.get(
        _ENV_MARKER, os.path.join(tempfile.gettempdir(), "atl_torchvision_shim")
    )


# --------------------------------------------------------------------------
# generated stub source
# --------------------------------------------------------------------------


def _package_init_src() -> str:
    return f'''"""ATL benchmark stub — NOT the real torchvision.

Exists only so vLLM 0.26's unconditional MiniMax-M3 vision import resolves on
an environment where torchvision cannot be installed (torch 2.11.0+cu130: the
cu130 wheel's compiled extension is broken, the cu128 wheel is rejected by
torch's CUDA version check).

It provides `torchvision.transforms.InterpolationMode` and NOTHING ELSE. Any
other torchvision attribute raises AttributeError with this explanation — a
loud failure is correct here, because silently returning something plausible
for a real image operation would corrupt results instead of stopping.

Generated by benchmarks/common/torchvision_shim.py. Do not edit.
"""

__version__ = "{SHIM_VERSION}"
__atl_shim__ = True

from . import transforms  # noqa: F401


def __getattr__(name):
    raise AttributeError(
        f"torchvision.{{name}} is not available: this is the ATL benchmark stub, "
        f"which provides only transforms.InterpolationMode. Real torchvision "
        f"cannot be installed on torch 2.11.0+cu130. If you need real "
        f"torchvision functionality, this environment cannot support it."
    )
'''


def _transforms_init_src() -> str:
    members = "\n".join(f'    {name} = "{value}"' for name, value in INTERPOLATION_MEMBERS)
    return f'''"""ATL benchmark stub — only InterpolationMode. NOT real torchvision.transforms."""

from enum import Enum

__atl_shim__ = True


class InterpolationMode(str, Enum):
    """Mirrors torchvision.transforms.InterpolationMode.

    A str Enum with torchvision's own values, so equality against the plain
    string form behaves the same way it would with the real package.
    """

{members}


__all__ = ["InterpolationMode"]


def __getattr__(name):
    raise AttributeError(
        f"torchvision.transforms.{{name}} is not available: this is the ATL "
        f"benchmark stub, which provides only InterpolationMode."
    )
'''


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def _env_without_shim(shim_root: str, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Copy of the environment with the shim directory removed from PYTHONPATH."""
    base = dict(os.environ if env is None else env)
    raw = base.get("PYTHONPATH", "")
    if raw:
        target = os.path.abspath(shim_root)
        kept = [
            p for p in raw.split(os.pathsep)
            if p and os.path.abspath(p) != target
        ]
        if kept:
            base["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            base.pop("PYTHONPATH", None)
    return base


def real_torchvision_importable(
    shim_root: Optional[str] = None,
    python: Optional[str] = None,
    runner: Any = subprocess.run,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Does a REAL torchvision import successfully?

    Runs in a subprocess with the shim stripped from ``PYTHONPATH``. Both
    details matter: in-process the failed import would leave a broken partial
    module behind, and without stripping, the shim would detect itself and
    conclude that torchvision works.
    """
    root = shim_root or default_shim_root()
    exe = python or sys.executable
    try:
        proc = runner(
            [exe, "-c", _PROBE_SNIPPET],
            capture_output=True, text=True, timeout=timeout,
            env=_env_without_shim(root),
        )
        rc = getattr(proc, "returncode", 1)
        err = (getattr(proc, "stderr", "") or "").strip()
        return {
            "importable": rc == 0,
            "returncode": rc,
            "error": None if rc == 0 else err[-600:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "returncode": -1,
                "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# materialisation / activation
# --------------------------------------------------------------------------


def materialize(shim_root: Optional[str] = None) -> str:
    """Write the stub package to disk. Idempotent; returns the root directory."""
    root = os.path.abspath(shim_root or default_shim_root())
    pkg = os.path.join(root, "torchvision")
    transforms = os.path.join(pkg, "transforms")
    os.makedirs(transforms, exist_ok=True)

    files = {
        os.path.join(pkg, "__init__.py"): _package_init_src(),
        os.path.join(transforms, "__init__.py"): _transforms_init_src(),
    }
    for path, src in files.items():
        # Rewrite only on change, so re-running does not churn mtimes and
        # invalidate the child interpreter's bytecode cache every session.
        existing = None
        if os.path.exists(path):
            with open(path) as fh:
                existing = fh.read()
        if existing != src:
            with open(path, "w") as fh:
                fh.write(src)
    return root


def _prepend_pythonpath(root: str) -> str:
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if not parts or os.path.abspath(parts[0]) != os.path.abspath(root):
        parts = [root] + [p for p in parts if os.path.abspath(p) != os.path.abspath(root)]
    value = os.pathsep.join(parts)
    os.environ["PYTHONPATH"] = value
    return value


def ensure(
    shim_root: Optional[str] = None,
    force: bool = False,
    probe: Any = real_torchvision_importable,
) -> Dict[str, Any]:
    """Activate the shim **only if** real torchvision does not import.

    Returns a report suitable for the run manifest. ``active`` is the value
    recorded as ``torchvision_shim``.
    """
    root = os.path.abspath(shim_root or default_shim_root())

    probe_result = {"importable": False, "returncode": None,
                    "error": "probe skipped (force=True)"}
    if not force:
        probe_result = probe(shim_root=root)
        if probe_result.get("importable"):
            return {
                "active": False,
                "reason": "real torchvision imports successfully — not shadowing it",
                "shim_root": None,
                "probe": probe_result,
                "start_method": _start_method(),
            }

    materialize(root)
    # sys.path covers fork and this process; PYTHONPATH covers spawn and any
    # freshly exec'd interpreter.
    if root not in sys.path:
        sys.path.insert(0, root)
    pythonpath = _prepend_pythonpath(root)
    os.environ[_ENV_MARKER] = root

    # Drop any half-initialised real torchvision so this process picks up the
    # stub too; the child processes get it from PYTHONPATH regardless.
    for name in [m for m in sys.modules if m == "torchvision" or m.startswith("torchvision.")]:
        if not getattr(sys.modules[name], "__atl_shim__", False):
            del sys.modules[name]

    return {
        "active": True,
        "reason": (
            "real torchvision does not import; supplying "
            "transforms.InterpolationMode only"
        ),
        "shim_root": root,
        "pythonpath": pythonpath,
        "probe": probe_result,
        "start_method": _start_method(),
        "note": (
            "Stub package on PYTHONPATH + sys.path so it reaches vLLM's "
            "EngineCore child process whether that child is forked or spawned."
        ),
    }


def _start_method() -> Optional[str]:
    """Observed multiprocessing start method — recorded, not relied upon."""
    try:
        import multiprocessing  # noqa: PLC0415

        return multiprocessing.get_start_method(allow_none=True) or \
            multiprocessing.get_start_method()
    except Exception:  # noqa: BLE001
        return None


def status(shim_root: Optional[str] = None) -> Dict[str, Any]:
    root = os.path.abspath(shim_root or default_shim_root())
    pkg = os.path.join(root, "torchvision", "__init__.py")
    return {
        "shim_root": root,
        "materialized": os.path.exists(pkg),
        "on_sys_path": root in [os.path.abspath(p) for p in sys.path if p],
        "on_pythonpath": root in [
            os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
        ],
        "real_torchvision": real_torchvision_importable(shim_root=root),
        "start_method": _start_method(),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="torchvision shim for vLLM 0.26.")
    ap.add_argument("--root", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--install", action="store_true",
                   help="materialize and activate if real torchvision is missing")
    ap.add_argument("--force", action="store_true",
                    help="install without probing (testing only)")
    args = ap.parse_args(argv)

    if args.status:
        print(json.dumps(status(args.root), indent=2))
        return 0

    report = ensure(shim_root=args.root, force=args.force)
    print(json.dumps(report, indent=2))
    if report["active"]:
        print("\nExport this so child processes inherit it:", file=sys.stderr)
        print(f'export PYTHONPATH="{report["shim_root"]}:$PYTHONPATH"', file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
