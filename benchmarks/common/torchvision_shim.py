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
stopgap.

vLLM's MiniMax import is not the only one. ``vllm.transformers_utils.config``
pulls in ``transformers.models.auto.image_processing_auto``, which reaches
``transformers/image_utils.py``::

    from torchvision.io import ImageReadMode, decode_image

and that import is **not** guarded by ``is_torchvision_available()``.

IMPORTS RESOLVE; USE RAISES
---------------------------
Chasing these one ImportError at a time is not a strategy — transformers'
import graph will keep producing them. So the stub fabricates **any**
``torchvision.*`` submodule on demand via a meta-path finder, and every symbol
it hands back is a poison object that raises ``RuntimeError`` the moment it is
called or instantiated.

``InterpolationMode`` and ``ImageReadMode`` are defined exactly (real values,
real enum semantics) because they are *read* at import time, not called.
Everything else exists only to let an import statement complete. Nothing in the
stub ever returns data a caller could mistake for a real image operation: a
plausible-looking tensor would be far worse than a raise, because it would flow
into real code and produce results that look valid.

AND IT IS STILL DETECTED AS UNAVAILABLE
---------------------------------------
Making the stub importable does **not** make feature detection think torchvision
is installed. ``transformers._is_package_available`` pairs ``find_spec`` with
``importlib.metadata.version``, and the stub deliberately ships **no
distribution metadata** — so the metadata lookup raises and
``is_torchvision_available()`` returns False. Verified directly; see
``tests/test_torchvision_shim.py``.

That is the reason the reported failure was not preventable by a marker: the
guard was already returning False, and the failing import simply is not behind
it. **Never add a .dist-info to the stub** — its absence is what keeps optional
vision paths switched off while unconditional imports still resolve.

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
    "IMAGE_READ_MODE_MEMBERS",
    "REAL_SUBMODULES",
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

# torchvision.io.ImageReadMode is an IntEnum; the integer values are mirrored
# so any code comparing or serialising them behaves identically.
IMAGE_READ_MODE_MEMBERS = (
    ("UNCHANGED", 0),
    ("GRAY", 1),
    ("GRAY_ALPHA", 2),
    ("RGB", 3),
    ("RGB_ALPHA", 4),
)

# Submodules backed by real generated files. Everything else under torchvision
# is fabricated on demand by the finder, so no future missing submodule can
# break an import.
REAL_SUBMODULES = (
    "torchvision._shim_support",
    "torchvision.transforms",
    "torchvision.io",
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
    real = ", ".join(f'"{m}"' for m in REAL_SUBMODULES)
    return f'''"""ATL benchmark stub — NOT the real torchvision.

Exists so vLLM 0.26 and transformers can complete their *imports* on an
environment where torchvision cannot be installed (torch 2.11.0+cu130: the
cu130 wheel's compiled extension is broken, the cu128 wheel is rejected by
torch's CUDA version check).

TWO PROPERTIES, DELIBERATELY IN TENSION
---------------------------------------
1. **Importable.** Any ``torchvision.*`` submodule resolves, and any attribute
   of one resolves. Unconditional imports therefore succeed.
2. **Detectably unavailable.** This package ships **no installed distribution
   metadata**, so ``importlib.metadata.version("torchvision")`` raises
   PackageNotFoundError. Feature-detection helpers that pair ``find_spec`` with
   a metadata lookup — including transformers' ``_is_package_available`` and
   therefore ``is_torchvision_available()`` — evaluate this as **False** and
   skip their optional vision paths.

   *Never add a .dist-info directory here.* Its absence is what keeps the stub
   from being mistaken for a working install.

NOTHING IS USABLE
-----------------
Every fabricated attribute is a poison object that raises RuntimeError the
moment it is **called or instantiated**. The stub resolves imports; it never
returns a value a caller could mistake for a real image operation. Failing at
use rather than at import is the point: import-time failure was producing an
endless one-symbol-at-a-time chase through transformers' import graph, while
use-time failure ends that chase without ever letting wrong data through.

Generated by benchmarks/common/torchvision_shim.py. Do not edit.
"""

import importlib.abc
import importlib.machinery
import sys

# Real file in this package, resolved by normal machinery before the finder
# below is installed.
from ._shim_support import ShimModule as _ShimModule
from ._shim_support import shim_getattr as _shim_getattr

__version__ = "{SHIM_VERSION}"
__atl_shim__ = True

# Backed by real files in this package; the finder must not shadow them.
_REAL_SUBMODULES = frozenset({{{real}}})


class _ShimLoader(importlib.abc.Loader):
    def create_module(self, spec):
        # ShimModule, not a plain module: it is callable, so a fabricated name
        # used as a class or function raises the stub's RuntimeError instead of
        # "module object is not callable".
        return _ShimModule(spec.name)

    def exec_module(self, module):
        # Empty __path__ marks it a package, so deeper submodules fabricate too
        # (e.g. torchvision.transforms.v2.functional).
        module.__path__ = []


class _ShimFinder(importlib.abc.MetaPathFinder):
    """Fabricates any torchvision submodule that is not backed by a real file.

    This is the answer to "adding symbols one failure at a time is not a
    strategy": rather than chasing each ImportError, every submodule under
    torchvision resolves, and the failure is deferred to use.
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("torchvision."):
            return None
        if fullname in _REAL_SUBMODULES:
            return None  # defer to the real file
        return importlib.machinery.ModuleSpec(fullname, _ShimLoader(), is_package=True)


if not any(isinstance(f, _ShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ShimFinder())

from . import io  # noqa: E402,F401
from . import transforms  # noqa: E402,F401

__getattr__ = _shim_getattr("torchvision")
'''


def _transforms_init_src() -> str:
    members = "\n".join(f'    {name} = "{value}"' for name, value in INTERPOLATION_MEMBERS)
    return f'''"""ATL benchmark stub — torchvision.transforms. NOT the real thing.

InterpolationMode is defined exactly; everything else resolves to a poison
object that raises when called. Submodules (functional, v2, v2.functional, …)
are fabricated by the finder in torchvision/__init__.py.

Generated by benchmarks/common/torchvision_shim.py. Do not edit.
"""

from enum import Enum

from .. import _shim_support as _s

__atl_shim__ = True


class InterpolationMode(str, Enum):
    """Mirrors torchvision.transforms.InterpolationMode.

    A str Enum with torchvision's own values, so equality against the plain
    string form behaves the same way it would with the real package.
    """

{members}


__all__ = ["InterpolationMode"]

__getattr__ = _s.shim_getattr("torchvision.transforms")
'''


def _io_init_src() -> str:
    members = "\n".join(
        f"    {name} = {value}" for name, value in IMAGE_READ_MODE_MEMBERS
    )
    return f'''"""ATL benchmark stub — torchvision.io. NOT the real thing.

Reached via transformers/image_utils.py:
    ``from torchvision.io import ImageReadMode, decode_image``
which is NOT guarded by is_torchvision_available(), so the symbols must exist
even though the stub is correctly detected as an unavailable package.

Generated by benchmarks/common/torchvision_shim.py. Do not edit.
"""

from enum import IntEnum

from .. import _shim_support as _s

__atl_shim__ = True


class ImageReadMode(IntEnum):
    """Mirrors torchvision.io.ImageReadMode, including its integer values."""

{members}


def decode_image(*args, **kwargs):
    """Present for import; raises if actually called.

    Returning a plausible tensor here would be far worse than raising: it would
    feed fabricated pixel data into a real code path and produce results that
    look valid. This benchmark drives a text-only Qwen2 model, so this function
    should never be reached — if it is, that is a genuine finding.
    """
    raise RuntimeError(
        "torchvision.io.decode_image was CALLED through the ATL benchmark stub. "
        "Real torchvision is not installable on this environment "
        "(torch 2.11.0+cu130), and the stub never returns image data. A code "
        "path that genuinely decodes images is being exercised — investigate "
        "rather than stubbing further."
    )


__all__ = ["ImageReadMode", "decode_image"]

__getattr__ = _s.shim_getattr("torchvision.io")
'''


def _shim_support_src() -> str:
    """Shared poison factory, importable by the real submodules."""
    return '''"""Internal support for the ATL torchvision stub. Not part of torchvision."""

import importlib
import types

__atl_shim__ = True

_MSG = (
    "{qualname} was reached through the ATL benchmark torchvision stub and "
    "cannot be executed. Real torchvision is not installable on this "
    "environment (torch 2.11.0+cu130). The stub exists so imports resolve; "
    "nothing in it performs real work. If this fires, a code path that "
    "genuinely needs torchvision is being exercised - investigate rather than "
    "stubbing further."
)


class _UnavailableMeta(type):
    """Metaclass so a poison symbol raises on call AND on instantiation."""

    def __call__(cls, *args, **kwargs):
        raise RuntimeError(cls._atl_message)

    def __repr__(cls):
        return f"<ATL torchvision stub: {cls.__name__} unavailable>"


def unavailable(qualname):
    """Poison stand-in for ``qualname``.

    A *type*, not a function: types are callable (instantiation raises), usable
    as base classes, and valid in isinstance checks - so a module that
    subclasses or annotates against a torchvision symbol still imports.
    """
    name = qualname.rsplit(".", 1)[-1]
    return _UnavailableMeta(
        name, (), {"_atl_message": _MSG.format(qualname=qualname),
                   "__atl_shim__": True, "__atl_qualname__": qualname}
    )


def _resolve(module_name, name):
    """Attribute lookup: prefer a submodule, fall back to a poison symbol.

    The submodule attempt is essential. CPython's `from X import Y` checks
    `hasattr(X, Y)` BEFORE trying to import `X.Y`, so a __getattr__ that
    immediately returned a poison symbol would shadow every submodule import -
    `from torchvision.transforms import functional` would hand back a type
    rather than a module, and the subsequent `functional.resize` would raise a
    confusing AttributeError instead of the stub's explanation.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    try:
        return importlib.import_module(f"{module_name}.{name}")
    except Exception:
        return unavailable(f"{module_name}.{name}")


class ShimModule(types.ModuleType):
    """A fabricated torchvision submodule.

    Callable so that a name imported as a submodule but *used* as a class or
    function - `from torchvision.transforms import Compose; Compose(...)` -
    still raises the stub's RuntimeError rather than a bare
    "module is not callable" TypeError.
    """

    __atl_shim__ = True

    def __call__(self, *args, **kwargs):
        raise RuntimeError(_MSG.format(qualname=self.__name__))

    def __getattr__(self, name):
        return _resolve(self.__name__, name)


def shim_getattr(module_name):
    def __getattr__(name):
        return _resolve(module_name, name)
    return __getattr__
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
    io_dir = os.path.join(pkg, "io")
    os.makedirs(transforms, exist_ok=True)
    os.makedirs(io_dir, exist_ok=True)

    # NOTE: no .dist-info is written, and none must ever be. Its absence makes
    # importlib.metadata.version("torchvision") raise, which is what causes
    # transformers' _is_package_available -> is_torchvision_available() to
    # report False and skip optional vision paths.
    files = {
        os.path.join(pkg, "_shim_support.py"): _shim_support_src(),
        os.path.join(pkg, "__init__.py"): _package_init_src(),
        os.path.join(transforms, "__init__.py"): _transforms_init_src(),
        os.path.join(io_dir, "__init__.py"): _io_init_src(),
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
            "real torchvision does not import; supplying an import-only stub "
            "(InterpolationMode + ImageReadMode exact, everything else poison)"
        ),
        "shim_root": root,
        "pythonpath": pythonpath,
        "probe": probe_result,
        "start_method": _start_method(),
        # No .dist-info is written, so importlib.metadata.version() raises and
        # transformers' is_torchvision_available() stays False. Reported so a
        # reader can confirm the stub is not masquerading as an install.
        "detectable_as_unavailable": not _has_distribution_metadata(),
        "real_submodules": list(REAL_SUBMODULES),
        "note": (
            "Stub package on PYTHONPATH + sys.path so it reaches vLLM's "
            "EngineCore child process whether that child is forked or spawned. "
            "Any torchvision.* submodule resolves; every symbol raises "
            "RuntimeError on use."
        ),
    }


def _has_distribution_metadata() -> bool:
    """True if something has registered torchvision as an installed dist.

    Must stay False for the stub. If it ever goes True, feature detection will
    start believing torchvision is installed and transformers will take vision
    code paths that then hit poison objects at runtime.
    """
    try:
        import importlib.metadata as md  # noqa: PLC0415

        md.version("torchvision")
        return True
    except Exception:
        return False


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
