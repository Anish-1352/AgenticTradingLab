"""Unit tests for nsys/ncu path resolution against a mocked filesystem.

The real tool layouts cannot be exercised on a machine with no GPU and no
Nsight install, so ``resolve_tool`` takes injectable ``which_fn``/``glob_fn``/
``version_fn`` and the selection logic is tested in isolation.

The cases that matter, all observed or plausible across images:
  * tool on PATH
  * tool absent from PATH but present under a versioned nsight-compute tree
    (this is how nsys ships when it has no standalone package)
  * several versions installed -> the highest must win, and version ordering
    must be numeric, not lexicographic
  * tool absent entirely -> found=False, which is a DIFFERENT state from
    found-but-blocked
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import probe_environment as pe  # noqa: E402


def make_fs(paths, on_path=None):
    """Build (which_fn, glob_fn) over a fake set of existing paths."""
    existing = set(paths)

    def which_fn(name):
        return on_path

    def glob_fn(pattern):
        # Minimal glob: '*' matches within one path segment, which is all the
        # real search patterns use.
        import fnmatch

        return sorted(p for p in existing if fnmatch.fnmatchcase(p, pattern))

    return which_fn, glob_fn


def resolve(paths, on_path=None, patterns=None, name="nsys"):
    which_fn, glob_fn = make_fs(paths, on_path)
    return pe.resolve_tool(
        name,
        patterns if patterns is not None else pe.NSYS_SEARCH_PATTERNS,
        which_fn=which_fn,
        glob_fn=glob_fn,
        version_fn=lambda p: "mock-version",
    )


# --------------------------------------------------------------------------
# version parsing
# --------------------------------------------------------------------------


def test_parse_path_version_extracts_dotted_component():
    v = pe.parse_path_version(
        "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
    )
    assert v == (2025, 1, 1)


def test_parse_path_version_ignores_non_numeric_segments():
    """'target-linux-x64' has digits but is not a dotted version."""
    assert pe.parse_path_version("/opt/x/target-linux-x64/nsys") == ()
    assert pe.parse_path_version("/usr/local/cuda/bin/ncu") == ()


def test_parse_path_version_handles_cuda_style_dirs():
    assert pe.parse_path_version("/usr/local/cuda-12.4/bin/nsys") == ()
    assert pe.parse_path_version("/opt/nvidia/nsight-compute/2024.3/ncu") == (2024, 3)


def test_version_ordering_is_numeric_not_lexicographic():
    """'2025.10' > '2025.9' numerically, but sorts lower as a string."""
    assert pe.parse_path_version("/a/2025.10/x") > pe.parse_path_version("/a/2025.9/x")


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_tool_on_path_wins():
    r = resolve(paths=[], on_path="/usr/bin/nsys")
    assert r["found"] is True
    assert r["path"] == "/usr/bin/nsys"
    assert r["source"] == "PATH"
    assert r["on_path"] == "/usr/bin/nsys"


def test_path_preferred_over_search_hits():
    r = resolve(
        paths=["/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"],
        on_path="/usr/bin/nsys",
    )
    assert r["path"] == "/usr/bin/nsys"
    assert r["source"] == "PATH"
    # The search hit is still recorded, so the report can show what else exists.
    assert r["candidates"] == [
        "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
    ]


def test_found_only_under_versioned_nsight_compute_tree():
    """The real Colab case: nsys bundled inside the Nsight Compute install."""
    p = "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
    r = resolve(paths=[p], on_path=None)
    assert r["found"] is True
    assert r["path"] == p
    assert r["source"] == "search"
    assert r["version"] == "mock-version"


def test_multiple_versions_highest_wins():
    paths = [
        "/opt/nvidia/nsight-compute/2024.3.1/host/target-linux-x64/nsys",
        "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys",
        "/opt/nvidia/nsight-compute/2023.2.0/host/target-linux-x64/nsys",
    ]
    r = resolve(paths=paths, on_path=None)
    assert r["path"] == "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
    assert len(r["candidates"]) == 3


def test_multiple_versions_double_digit_minor():
    """2025.10 must beat 2025.9 — the lexicographic trap."""
    paths = [
        "/opt/nvidia/nsight-compute/2025.9/host/target-linux-x64/nsys",
        "/opt/nvidia/nsight-compute/2025.10/host/target-linux-x64/nsys",
    ]
    r = resolve(paths=paths, on_path=None)
    assert r["path"] == "/opt/nvidia/nsight-compute/2025.10/host/target-linux-x64/nsys"


def test_versioned_install_beats_unversioned_when_both_present():
    paths = [
        "/usr/local/cuda/bin/nsys",
        "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys",
    ]
    r = resolve(paths=paths, on_path=None)
    assert r["path"] == "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"


def test_unversioned_install_used_when_it_is_the_only_candidate():
    r = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    assert r["found"] is True
    assert r["path"] == "/usr/local/cuda/bin/nsys"


def test_tool_absent_entirely():
    r = resolve(paths=[], on_path=None)
    assert r["found"] is False
    assert r["path"] is None
    assert r["source"] is None
    assert r["version"] is None
    assert r["candidates"] == []
    # The patterns searched are recorded so the failure is actionable.
    assert r["search_patterns"] == list(pe.NSYS_SEARCH_PATTERNS)


def test_ncu_resolution_under_cuda_bin():
    r = resolve(paths=["/usr/local/cuda/bin/ncu"], on_path=None,
                patterns=pe.NCU_SEARCH_PATTERNS, name="ncu")
    assert r["found"] is True
    assert r["path"] == "/usr/local/cuda/bin/ncu"


def test_ncu_resolution_under_versioned_tree():
    r = resolve(paths=["/opt/nvidia/nsight-compute/2025.1.1/ncu"], on_path=None,
                patterns=pe.NCU_SEARCH_PATTERNS, name="ncu")
    assert r["path"] == "/opt/nvidia/nsight-compute/2025.1.1/ncu"


def test_duplicate_hits_across_patterns_are_deduped():
    """A path matching two patterns must appear once."""
    r = resolve(
        paths=["/usr/local/cuda/bin/ncu"],
        on_path=None,
        patterns=("/usr/local/cuda/bin/ncu", "/usr/local/cuda/bin/*"),
        name="ncu",
    )
    assert r["candidates"] == ["/usr/local/cuda/bin/ncu"]


def test_glob_failure_does_not_abort_resolution():
    def bad_glob(pattern):
        if "nsight-compute" in pattern:
            raise OSError("permission denied")
        return ["/usr/local/cuda/bin/nsys"] if pattern == "/usr/local/cuda/bin/nsys" else []

    r = pe.resolve_tool("nsys", pe.NSYS_SEARCH_PATTERNS,
                        which_fn=lambda n: None, glob_fn=bad_glob,
                        version_fn=lambda p: "v")
    assert r["found"] is True
    assert r["path"] == "/usr/local/cuda/bin/nsys"


def test_version_probe_failure_is_not_fatal():
    def bad_version(path):
        raise RuntimeError("binary refused to run")

    r = pe.resolve_tool("nsys", pe.NSYS_SEARCH_PATTERNS,
                        which_fn=lambda n: "/usr/bin/nsys",
                        glob_fn=lambda p: [], version_fn=bad_version)
    assert r["found"] is True
    assert r["version"] is None


# --------------------------------------------------------------------------
# found-but-blocked vs not-found
# --------------------------------------------------------------------------


def test_not_found_result_is_distinct_from_blocked():
    """Two different problems with two different fixes; never collapse them."""
    tool = resolve(paths=[], on_path=None)
    res = pe._not_found_result(tool, "tier a")
    assert res["status"] == pe.BLOCKED
    assert res["tool_found"] is False
    assert res["tool_path"] is None
    assert res["permission_error"] is False
    assert "not found" in res["reason"]
    # The searched patterns end up in the report so the fix is obvious.
    assert "nsight-compute" in res["stderr_tail"]


def test_classify_permission_error_keeps_tool_path():
    tool = resolve(paths=["/usr/local/cuda/bin/ncu"], on_path=None,
                   patterns=pe.NCU_SEARCH_PATTERNS, name="ncu")
    res = pe._classify(1, "", "ERR_NVGPUCTRPERM: insufficient permissions",
                       None, tool)
    assert res["status"] == pe.BLOCKED
    assert res["permission_error"] is True
    assert res["tool_found"] is True          # found, but blocked
    assert res["tool_path"] == "/usr/local/cuda/bin/ncu"


def test_classify_success():
    tool = resolve(paths=["/usr/local/cuda/bin/ncu"], on_path=None,
                   patterns=pe.NCU_SEARCH_PATTERNS, name="ncu")
    res = pe._classify(0, "ok", "", None, tool)
    assert res["status"] == pe.OBTAINABLE
    assert res["permission_error"] is False


def test_classify_nonzero_exit_is_ran_but_failed_not_missing():
    tool = resolve(paths=["/usr/local/cuda/bin/ncu"], on_path=None,
                   patterns=pe.NCU_SEARCH_PATTERNS, name="ncu")
    res = pe._classify(2, "", "some other failure", None, tool)
    assert res["status"] == pe.BLOCKED
    assert res["tool_found"] is True
    assert "exited 2" in res["reason"]


# --------------------------------------------------------------------------
# gpu-metrics flag selection
# --------------------------------------------------------------------------


def test_gpu_metrics_prefers_plural_flag(monkeypatch, tmp_path):
    """Plural is tried first; singular is deprecated in nsys 2025.x."""
    calls = []

    def fake_run(cmd, timeout=300):
        calls.append(cmd)
        return 0, "ok", ""

    monkeypatch.setattr(pe, "_run", fake_run)
    monkeypatch.setattr(pe, "_artifact_exists", lambda prefix: True)

    tool = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    res = pe.probe_nsys_gpu_metrics(tool, str(tmp_path))

    assert res["status"] == pe.OBTAINABLE
    assert res["flag_used"] == "--gpu-metrics-devices"
    assert len(calls) == 1  # no need to try the singular form
    assert any("--gpu-metrics-devices=0" in a for a in calls[0])


def test_gpu_metrics_falls_back_to_singular_when_option_rejected(monkeypatch, tmp_path):
    """An older nsys rejects the plural spelling; retry the deprecated one."""
    seen = []

    def fake_run(cmd, timeout=300):
        seen.append(cmd)
        if any("--gpu-metrics-devices=" in a for a in cmd):
            return 1, "", "unrecognized option '--gpu-metrics-devices'"
        return 0, "ok", ""

    monkeypatch.setattr(pe, "_run", fake_run)
    monkeypatch.setattr(pe, "_artifact_exists", lambda prefix: True)

    tool = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    res = pe.probe_nsys_gpu_metrics(tool, str(tmp_path))

    assert res["status"] == pe.OBTAINABLE
    assert res["flag_used"] == "--gpu-metrics-device"
    assert len(seen) == 2
    assert len(res["attempts"]) == 2


def test_gpu_metrics_does_not_retry_on_permission_error(monkeypatch, tmp_path):
    """A counter-permission failure means the flag WAS understood.

    Retrying with a different spelling would relabel a permission problem as a
    flag problem and send the reader down the wrong path.
    """
    seen = []

    def fake_run(cmd, timeout=300):
        seen.append(cmd)
        return 1, "", "ERR_NVGPUCTRPERM"

    monkeypatch.setattr(pe, "_run", fake_run)
    tool = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    res = pe.probe_nsys_gpu_metrics(tool, str(tmp_path))

    assert res["status"] == pe.BLOCKED
    assert res["permission_error"] is True
    assert len(seen) == 1
    assert res["flag_used"] == "--gpu-metrics-devices"


def test_gpu_metrics_not_found_short_circuits(tmp_path):
    tool = resolve(paths=[], on_path=None)
    res = pe.probe_nsys_gpu_metrics(tool, str(tmp_path))
    assert res["status"] == pe.BLOCKED
    assert res["tool_found"] is False
    assert res["flag_used"] is None


# --------------------------------------------------------------------------
# metric sets
# --------------------------------------------------------------------------


def test_metric_sets_parsed_from_help_output(monkeypatch):
    help_text = (
        "Possible --gpu-metrics-set values are:\n"
        "\t[0] ga100 (General Metrics for NVIDIA GA100)\n"
        "\t[1] ga100-tensor (Tensor Core utilization for NVIDIA GA100)\n"
    )
    monkeypatch.setattr(pe, "_run", lambda cmd, timeout=300: (0, help_text, ""))

    tool = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    res = pe.probe_nsys_metric_sets(tool)

    assert res["available"] is True
    names = [s["name"] for s in res["sets"]]
    assert "ga100" in names and "ga100-tensor" in names
    assert "Tensor Core" in res["raw"]


def test_metric_sets_records_raw_even_when_unparseable(monkeypatch):
    monkeypatch.setattr(
        pe, "_run", lambda cmd, timeout=300: (0, "some unexpected format", "")
    )
    tool = resolve(paths=["/usr/local/cuda/bin/nsys"], on_path=None)
    res = pe.probe_nsys_metric_sets(tool)
    assert "some unexpected format" in res["raw"]


def test_metric_sets_skipped_when_nsys_absent():
    tool = resolve(paths=[], on_path=None)
    res = pe.probe_nsys_metric_sets(tool)
    assert res["available"] is False
    assert res["sets"] == []
