"""Tests for the torchvision shim.

The load-bearing test here is ``test_child_process_sees_the_shim``: vLLM runs
EngineCore in a separate process, so a shim that only works in-process is
worthless for the actual failure. That test spawns a real interpreter and a
real ``multiprocessing`` spawn-context child and asserts the import resolves in
both.

No torch, no vllm, no GPU.
"""

import multiprocessing
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import torchvision_shim as tvs  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Restore PYTHONPATH and sys.path around every test.

    ``ensure()`` mutates both by design — that is how the shim reaches a child
    process — so without this a test that activates the shim leaks it into
    every test that follows, and the "is real torchvision installed" probe
    starts finding the leftover stub.
    """
    saved_env = os.environ.get("PYTHONPATH")
    saved_marker = os.environ.get(tvs._ENV_MARKER)
    saved_path = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for key, value in ((("PYTHONPATH"), saved_env), (tvs._ENV_MARKER, saved_marker)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# generated stub content
# --------------------------------------------------------------------------


def test_materialize_creates_the_package(tmp_path):
    root = tvs.materialize(str(tmp_path / "shim"))
    assert os.path.exists(os.path.join(root, "torchvision", "__init__.py"))
    assert os.path.exists(os.path.join(root, "torchvision", "transforms", "__init__.py"))


def test_materialize_is_idempotent(tmp_path):
    root = tvs.materialize(str(tmp_path / "shim"))
    path = os.path.join(root, "torchvision", "__init__.py")
    before = os.path.getmtime(path)
    tvs.materialize(root)
    # Unchanged content must not be rewritten: churning mtimes would invalidate
    # the child interpreter's bytecode cache on every session.
    assert os.path.getmtime(path) == before


def test_interpolation_members_match_torchvision_values():
    got = dict(tvs.INTERPOLATION_MEMBERS)
    assert got == {
        "NEAREST": "nearest",
        "BILINEAR": "bilinear",
        "BICUBIC": "bicubic",
        "BOX": "box",
        "HAMMING": "hamming",
        "LANCZOS": "lanczos",
        "NEAREST_EXACT": "nearest-exact",
    }


def _run_in_child(root, snippet):
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, env=env, timeout=60
    )


def test_stub_provides_all_seven_members(tmp_path):
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.transforms import InterpolationMode as M;"
        "print(','.join(sorted(m.name for m in M)))"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == (
        "BICUBIC,BILINEAR,BOX,HAMMING,LANCZOS,NEAREST,NEAREST_EXACT"
    )


def test_stub_is_a_str_enum(tmp_path):
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.transforms import InterpolationMode as M;"
        "print(M.BICUBIC == 'bicubic', isinstance(M.BICUBIC, str))"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True True"


def test_stub_is_identifiable_as_a_stub(tmp_path):
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, "import torchvision;print(torchvision.__version__, torchvision.__atl_shim__)")
    assert proc.returncode == 0, proc.stderr
    assert tvs.SHIM_VERSION in proc.stdout
    assert "True" in proc.stdout


def test_top_level_attributes_resolve_but_raise_on_use(tmp_path):
    """The contract changed deliberately in the second pass.

    v1 of this stub raised AttributeError on every unknown attribute, which is
    what produced an endless one-symbol-at-a-time chase through transformers'
    import graph. Now attribute access resolves and *use* raises — the failure
    still cannot pass fabricated data to a caller, it just happens one step
    later.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "import torchvision\n"
        "nms = torchvision.ops.nms          # resolves\n"
        "try:\n"
        "    nms(1, 2)                      # raises\n"
        "    print('DID NOT RAISE')\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError', 'ATL' in str(e))\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RuntimeError True"


def test_stub_symbols_import_but_raise_on_use(tmp_path):
    """Imports must resolve; USE must raise.

    Import-time failure is what produced the one-symbol-at-a-time chase through
    transformers' import graph. Deferring the failure to use ends that chase
    without ever letting fabricated data through.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.transforms import Resize\n"
        "try:\n"
        "    Resize(224)\n"
        "except RuntimeError as e:\n"
        "    print('RAISED', 'ATL' in str(e))\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RAISED True"


# --------------------------------------------------------------------------
# torchvision.io — the reported failure
# --------------------------------------------------------------------------


def test_io_provides_the_symbols_transformers_imports(tmp_path):
    """transformers/image_utils.py:54 — the exact failing import."""
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.io import ImageReadMode, decode_image;"
        "print(int(ImageReadMode.RGB), len(list(ImageReadMode)))"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "3 5"


def test_image_read_mode_members_and_values():
    assert dict(tvs.IMAGE_READ_MODE_MEMBERS) == {
        "UNCHANGED": 0, "GRAY": 1, "GRAY_ALPHA": 2, "RGB": 3, "RGB_ALPHA": 4,
    }


def test_decode_image_raises_when_called(tmp_path):
    """Present for import, never silently returning wrong data."""
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.io import decode_image\n"
        "try:\n"
        "    decode_image(b'not an image')\n"
        "    print('DID NOT RAISE')\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError', 'CALLED' in str(e))\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RuntimeError True"


# --------------------------------------------------------------------------
# no more whack-a-mole: arbitrary submodules resolve
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snippet", [
    "from torchvision.transforms import functional as F; F.resize",
    "from torchvision.transforms.v2 import functional as F; F.to_dtype",
    "from torchvision.transforms.v2 import Compose, Resize",
    "import torchvision.ops",
    "import torchvision.datasets",
    "import torchvision.models",
    "from torchvision.utils import make_grid",
    "from torchvision.io.image import ImageReadMode",
    "import torchvision.some.deeply.nested.module",
])
def test_arbitrary_submodules_resolve(tmp_path, snippet):
    """Adding symbols one failure at a time is not a strategy — this is why."""
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, snippet)
    assert proc.returncode == 0, f"{snippet!r} failed:\n{proc.stderr}"


@pytest.mark.parametrize("expr", [
    "__import__('torchvision.ops', fromlist=['nms']).nms(1, 2)",
    "__import__('torchvision.utils', fromlist=['make_grid']).make_grid(None)",
    "__import__('torchvision.transforms.v2', fromlist=['Resize']).Resize(224)",
    "__import__('torchvision.transforms', fromlist=['functional']).functional.resize(1, 2)",
])
def test_fabricated_symbols_raise_runtime_error_on_use(tmp_path, expr):
    """Every fabricated path must fail as RuntimeError, not TypeError/AttributeError.

    A module used as a class must raise the stub's explanation too, which is why
    fabricated modules are a callable ModuleType subclass.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        f"try:\n"
        f"    {expr}\n"
        f"    print('DID NOT RAISE')\n"
        f"except RuntimeError:\n"
        f"    print('RuntimeError')\n"
        f"except Exception as e:\n"
        f"    print(type(e).__name__)\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RuntimeError"


def test_from_import_of_a_submodule_yields_a_module_not_a_poison_type(tmp_path):
    """CPython checks hasattr() BEFORE importing a submodule for `from X import Y`.

    A __getattr__ that returned a poison symbol immediately would shadow every
    submodule import, and `functional.resize` would then raise a confusing
    AttributeError instead of the stub's explanation.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "from torchvision.transforms import functional as F;"
        "import types;"
        "print(isinstance(F, types.ModuleType))"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


# --------------------------------------------------------------------------
# detectable as unavailable — the second-order effect
# --------------------------------------------------------------------------


def test_stub_has_no_distribution_metadata(tmp_path):
    """The property that keeps transformers' vision paths switched off.

    transformers' _is_package_available pairs find_spec with
    importlib.metadata.version. The stub satisfies the first and must NEVER
    satisfy the second.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "import importlib.util, importlib.metadata as md\n"
        "found = importlib.util.find_spec('torchvision') is not None\n"
        "try:\n"
        "    md.version('torchvision'); meta = True\n"
        "except md.PackageNotFoundError:\n"
        "    meta = False\n"
        "print(found, meta)\n"
    ))
    assert proc.returncode == 0, proc.stderr
    # importable, but not a registered distribution
    assert proc.stdout.strip() == "True False"


def test_transformers_is_package_available_would_return_false(tmp_path):
    """Replicates transformers._is_package_available against the stub."""
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, (
        "import importlib.util, importlib.metadata as md\n"
        "def _is_package_available(name):\n"
        "    exists = importlib.util.find_spec(name) is not None\n"
        "    if exists:\n"
        "        try:\n"
        "            md.version(name)\n"
        "        except md.PackageNotFoundError:\n"
        "            exists = False\n"
        "    return exists\n"
        "print(_is_package_available('torchvision'))\n"
    ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_no_dist_info_is_written(tmp_path):
    """Guards the invariant directly against a future 'helpful' addition."""
    root = tvs.materialize(str(tmp_path / "shim"))
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        found += [d for d in dirnames if d.endswith((".dist-info", ".egg-info"))]
    assert found == [], f"stub must never register as an installed dist: {found}"


def test_ensure_reports_detectable_as_unavailable(tmp_path, monkeypatch):
    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report = tvs.ensure(shim_root=root, force=True)
    assert report["detectable_as_unavailable"] is True


def test_probe_rejects_the_stub_itself(tmp_path):
    """The stub on PYTHONPATH must NOT be mistaken for a working torchvision.

    Otherwise a shim root left over from an earlier session makes ensure()
    no-op, and the current session never activates one.
    """
    root = tvs.materialize(str(tmp_path / "shim"))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", tvs._PROBE_SNIPPET],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 3
    assert "ATL shim detected" in proc.stderr


def test_ensure_activates_even_with_a_stale_shim_root_on_pythonpath(tmp_path, monkeypatch):
    """A leftover shim from a previous session must not suppress a new one."""
    stale = tvs.materialize(str(tmp_path / "stale"))
    monkeypatch.setenv("PYTHONPATH", stale)
    fresh = str(tmp_path / "fresh")
    monkeypatch.setenv(tvs._ENV_MARKER, fresh)
    report = tvs.ensure(shim_root=fresh)
    assert report["active"] is True


# --------------------------------------------------------------------------
# THE point: does it reach a child process?
# --------------------------------------------------------------------------


def _child_probe(q):  # pragma: no cover - runs in a child process
    try:
        from torchvision.transforms import InterpolationMode

        q.put(("ok", str(InterpolationMode.BICUBIC.value)))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def test_child_process_sees_the_shim_via_subprocess(tmp_path):
    """A fresh interpreter (the `spawn` case) must resolve the import."""
    root = tvs.materialize(str(tmp_path / "shim"))
    proc = _run_in_child(root, "from torchvision.transforms import InterpolationMode;print(InterpolationMode.BICUBIC.value)")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "bicubic"


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="spawn start method unavailable",
)
def test_child_process_sees_the_shim_via_multiprocessing_spawn(tmp_path, monkeypatch):
    """vLLM launches EngineCore through multiprocessing.

    A spawn-context child is a brand-new interpreter: it does NOT inherit
    sys.modules or sys.path mutations, only the environment. This is the exact
    mechanism a sys.modules patch fails at, and the reason the shim is a real
    package on PYTHONPATH.
    """
    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    report = tvs.ensure(shim_root=root, force=True)
    assert report["active"] is True

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_probe, args=(q,))
    p.start()
    p.join(timeout=60)
    status, value = q.get(timeout=10)
    assert status == "ok", f"child could not import the shim: {value}"
    assert value == "bicubic"


def test_ensure_sets_both_sys_path_and_pythonpath(tmp_path, monkeypatch):
    """sys.path covers fork; PYTHONPATH covers spawn. Both are required."""
    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report = tvs.ensure(shim_root=root, force=True)
    assert os.path.abspath(root) in [os.path.abspath(p) for p in sys.path if p]
    assert os.path.abspath(root) in [
        os.path.abspath(p) for p in os.environ["PYTHONPATH"].split(os.pathsep) if p
    ]
    assert report["shim_root"] == os.path.abspath(root)


# --------------------------------------------------------------------------
# never shadow a working install
# --------------------------------------------------------------------------


def test_shim_absent_when_real_torchvision_imports(tmp_path):
    def probe_ok(shim_root=None):
        return {"importable": True, "returncode": 0, "error": None}

    report = tvs.ensure(shim_root=str(tmp_path / "shim"), probe=probe_ok)
    assert report["active"] is False
    assert report["shim_root"] is None
    assert "not shadowing" in report["reason"]
    # Nothing written to disk either.
    assert not os.path.exists(tmp_path / "shim" / "torchvision")


def test_shim_present_when_real_torchvision_fails(tmp_path, monkeypatch):
    def probe_bad(shim_root=None):
        return {"importable": False, "returncode": 1,
                "error": "operator torchvision::nms does not exist"}

    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report = tvs.ensure(shim_root=root, probe=probe_bad)
    assert report["active"] is True
    assert os.path.exists(os.path.join(root, "torchvision", "__init__.py"))
    assert "nms" in report["probe"]["error"]


def test_shim_no_ops_if_torchvision_becomes_installable(tmp_path):
    """Documented requirement: a future working torchvision disables the shim."""
    root = str(tmp_path / "shim")
    tvs.materialize(root)  # left over from a previous session

    def probe_ok(shim_root=None):
        return {"importable": True, "returncode": 0, "error": None}

    report = tvs.ensure(shim_root=root, probe=probe_ok)
    assert report["active"] is False


def test_probe_strips_the_shim_from_pythonpath(tmp_path, monkeypatch):
    """Otherwise the shim detects itself and concludes torchvision works."""
    root = os.path.abspath(str(tmp_path / "shim"))
    monkeypatch.setenv("PYTHONPATH", root + os.pathsep + "/some/other/path")
    env = tvs._env_without_shim(root)
    assert root not in [os.path.abspath(p) for p in env["PYTHONPATH"].split(os.pathsep)]
    assert "/some/other/path" in env["PYTHONPATH"]


def test_probe_drops_pythonpath_entirely_when_only_the_shim(tmp_path, monkeypatch):
    root = os.path.abspath(str(tmp_path / "shim"))
    monkeypatch.setenv("PYTHONPATH", root)
    assert "PYTHONPATH" not in tvs._env_without_shim(root)


def test_real_probe_reports_failure_on_this_machine():
    """No torchvision is installed here, so the probe must say so."""
    result = tvs.real_torchvision_importable()
    assert result["importable"] is False
    assert result["error"]


def test_probe_handles_subprocess_explosion():
    def boom(*a, **k):
        raise OSError("no exec for you")

    result = tvs.real_torchvision_importable(runner=boom)
    assert result["importable"] is False
    assert "OSError" in result["error"]


def test_ensure_end_to_end_on_this_machine(tmp_path, monkeypatch):
    """Real probe, real materialisation, real child — no mocks at all."""
    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report = tvs.ensure(shim_root=root)
    # torchvision genuinely is not installed here, so the shim must engage.
    assert report["active"] is True
    proc = _run_in_child(root, "from torchvision.transforms import InterpolationMode;print('ok')")
    assert proc.returncode == 0, proc.stderr


def test_status_reports_materialization(tmp_path, monkeypatch):
    root = str(tmp_path / "shim")
    monkeypatch.setenv(tvs._ENV_MARKER, root)
    assert tvs.status(root)["materialized"] is False
    tvs.materialize(root)
    assert tvs.status(root)["materialized"] is True
