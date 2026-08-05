"""Unit tests for colab_bootstrap, session_start, and the calibration log.

No GPU, no torch, no vllm, and — critically — **no pip**. Version detection and
subprocess execution are both injected, and several tests assert that the
runner was never called at all.

The idempotency property is the one that matters most: a bootstrap that
reinstalls on every invocation forces a runtime restart, and a restart can make
Colab hand back a different physical GPU mid-study.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import colab_bootstrap as cb  # noqa: E402
import session_start as ss  # noqa: E402
from common import calibration as cal  # noqa: E402


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

CORRECT = {
    "torch": "2.11.0+cu130",
    "vllm": "0.26.0",
    "transformers": "5.13.1",
    "torchvision": None,
    "torchaudio": None,
    "torchcodec": None,
    "pynvml": None,
    "nvidia-ml-py": "13.580.65",
}

FRESH_COLAB = {
    "torch": "2.11.0+cu128",
    "vllm": None,
    "transformers": "5.13.1",
    "torchvision": "0.26.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "torchcodec": "0.9.0",
    "pynvml": "11.5.0",
    "nvidia-ml-py": None,
}


def state(versions):
    return cb.detect_state(lambda name: versions.get(name))


class ExplodingRunner:
    """Fails the test if anything tries to shell out."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        raise AssertionError(f"pip must not run in unit tests, got: {cmd}")


class RecordingRunner:
    def __init__(self, returncode=0):
        self.calls = []
        self.returncode = returncode

    def __call__(self, cmd):
        self.calls.append(list(cmd))

        class P:
            pass

        p = P()
        p.returncode = self.returncode
        return p


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_detect_state_flags_cu130():
    assert state(CORRECT)["torch_is_cu130"] is True
    assert state(FRESH_COLAB)["torch_is_cu130"] is False


def test_detect_state_handles_missing_torch():
    st = state({**CORRECT, "torch": None})
    assert st["torch_is_cu130"] is False


def test_detection_never_imports_the_packages():
    """Importing torch here would load CUDA; importing vllm would hit the very
    torchvision problem the bootstrap exists to arrange around."""
    before = set(sys.modules)
    state(CORRECT)
    new = set(sys.modules) - before
    assert "torch" not in new and "vllm" not in new and "torchvision" not in new


# --------------------------------------------------------------------------
# idempotency — the property that protects the GPU allocation
# --------------------------------------------------------------------------


def test_correct_stack_plans_nothing():
    assert cb.plan_actions(state(CORRECT)) == []


def test_correct_stack_has_no_problems():
    assert cb.verify_state(state(CORRECT)) == []


def test_correct_stack_never_shells_out():
    runner = ExplodingRunner()
    cb.execute(cb.plan_actions(state(CORRECT)), runner=runner)
    assert runner.calls == []


def test_rerun_after_success_is_a_no_op():
    """Second invocation must plan nothing, so no second restart is requested."""
    assert cb.plan_actions(state(CORRECT)) == []
    assert cb.plan_actions(state(CORRECT)) == []


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_fresh_colab_plan_is_ordered_correctly():
    ids = [a["id"] for a in cb.plan_actions(state(FRESH_COLAB))]
    assert ids == [
        "uninstall_pynvml",
        "install_nvidia_ml_py",
        "uninstall_torch_stack",
        "install_vllm",
        "uninstall_forbidden",
    ]


def test_forbidden_removal_comes_after_vllm_install():
    """`pip install vllm` can drag torchvision back in as a dependency, so the
    removal has to be the last mutation or it gets undone."""
    ids = [a["id"] for a in cb.plan_actions(state(FRESH_COLAB))]
    assert ids.index("uninstall_forbidden") > ids.index("install_vllm")


def test_cu128_torch_triggers_rebuild():
    ids = [a["id"] for a in cb.plan_actions(state({**CORRECT, "torch": "2.11.0+cu128"}))]
    assert "uninstall_torch_stack" in ids and "install_vllm" in ids


def test_missing_vllm_triggers_rebuild_even_on_cu130():
    ids = [a["id"] for a in cb.plan_actions(state({**CORRECT, "vllm": None}))]
    assert "install_vllm" in ids


def test_pynvml_alone_is_a_minimal_plan():
    """A conflicting NVML package must not drag in a torch rebuild."""
    ids = [a["id"] for a in cb.plan_actions(state({**CORRECT, "pynvml": "11.5.0"}))]
    assert ids == ["uninstall_pynvml"]
    assert "uninstall_torch_stack" not in ids


def test_stray_torchvision_alone_does_not_rebuild_torch():
    ids = [a["id"] for a in cb.plan_actions(state({**CORRECT, "torchvision": "0.26.0"}))]
    assert ids == ["uninstall_forbidden"]


def test_plan_actions_is_pure():
    st = state(FRESH_COLAB)
    snapshot = json.dumps(st, sort_keys=True)
    cb.plan_actions(st)
    assert json.dumps(st, sort_keys=True) == snapshot


def test_pip_commands_are_rewritten_to_this_interpreter(monkeypatch):
    """Never install into a different Python than the notebook is running."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv

        class P:
            returncode = 0

        return P()

    monkeypatch.setattr(cb.subprocess, "run", fake_run)
    cb._default_runner(["pip", "install", "-q", "vllm"])
    assert seen["argv"][:3] == [sys.executable, "-m", "pip"]


# --------------------------------------------------------------------------
# verification / failing loudly
# --------------------------------------------------------------------------


def test_missing_torch_after_uninstall_is_reported():
    """Removing torchvision has been observed to take torch with it."""
    problems = cb.verify_state(state({**CORRECT, "torch": None}))
    assert any("torch is NOT installed" in p for p in problems)


def test_missing_vllm_is_reported():
    assert any("vllm is NOT installed" in p
               for p in cb.verify_state(state({**CORRECT, "vllm": None})))


def test_cu128_torch_is_reported_with_the_symptom():
    problems = cb.verify_state(state({**CORRECT, "torch": "2.11.0+cu128"}))
    assert any("libcudart.so.13" in p for p in problems)


def test_present_torchvision_is_reported():
    problems = cb.verify_state(state({**CORRECT, "torchvision": "0.26.0+cu130"}))
    assert any("torchvision" in p and "must not be" in p for p in problems)


def test_pynvml_present_is_reported():
    problems = cb.verify_state(state({**CORRECT, "pynvml": "11.5.0"}))
    assert any("conflicts with nvidia-ml-py" in p for p in problems)


def test_execute_stops_at_first_hard_failure():
    runner = RecordingRunner(returncode=1)
    actions = [
        {"id": "a", "why": "", "cmd": ["pip", "install", "x"]},
        {"id": "b", "why": "", "cmd": ["pip", "install", "y"]},
    ]
    results = cb.execute(actions, runner=runner)
    assert len(results) == 1 and results[0]["ok"] is False
    assert len(runner.calls) == 1  # did not continue past the failure


def test_execute_tolerates_uninstall_of_absent_package():
    runner = RecordingRunner(returncode=1)
    actions = [{"id": "u", "why": "", "cmd": ["pip", "uninstall", "-y", "torchvision"],
                "tolerate_failure": True}]
    results = cb.execute(actions, runner=runner)
    assert results[0]["ok"] is True


def test_exit_codes_are_distinct():
    codes = {cb.EXIT_OK, cb.EXIT_FAILED, cb.EXIT_RESTART_REQUIRED}
    assert len(codes) == 3
    assert cb.EXIT_OK == 0


def _no_execute(*args, **kwargs):
    raise AssertionError("execute() must not run in these tests")


def test_check_only_never_runs_pip(monkeypatch, capsys):
    # Snapshot BEFORE patching: state() calls the real detect_state.
    fresh = state(FRESH_COLAB)
    monkeypatch.setattr(cb, "detect_state", lambda *a, **k: fresh)
    monkeypatch.setattr(cb, "execute", _no_execute)
    rc = cb.main(["--check-only"])
    assert rc == cb.EXIT_RESTART_REQUIRED
    assert "nothing executed" in capsys.readouterr().out


def test_main_exits_zero_and_silent_when_already_correct(monkeypatch, capsys):
    correct = state(CORRECT)
    monkeypatch.setattr(cb, "detect_state", lambda *a, **k: correct)
    monkeypatch.setattr(cb, "execute", _no_execute)
    rc = cb.main([])
    assert rc == cb.EXIT_OK
    assert "ALREADY CORRECT" in capsys.readouterr().out


def test_hf_home_warning():
    assert cb.check_hf_home() is not None or os.environ.get("HF_HOME")


def test_hf_home_accepts_drive_path(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/content/drive/MyDrive/atl_bench/hf_cache")
    assert cb.check_hf_home() is None


def test_hf_home_rejects_local_path(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    msg = cb.check_hf_home()
    assert msg and "preemption" in msg


# --------------------------------------------------------------------------
# session_start
# --------------------------------------------------------------------------


def fake_smi(uuid="GPU-2ddfc69f", name="NVIDIA A100-SXM4-80GB"):
    def runner(cmd, **kw):
        class P:
            returncode = 0
            stdout = f"{uuid}, {name}, 81920 MiB, 580.82.07\n"
            stderr = ""

        return P()

    return runner


def test_query_gpu_parses_nvidia_smi():
    got = ss.query_gpu(runner=fake_smi())
    assert got["uuid"] == "GPU-2ddfc69f"
    assert got["name"] == "NVIDIA A100-SXM4-80GB"
    assert got["memory_total"] == "81920 MiB"
    assert got["driver_version"] == "580.82.07"


def test_query_gpu_handles_missing_nvidia_smi():
    def boom(cmd, **kw):
        raise FileNotFoundError()

    got = ss.query_gpu(runner=boom)
    assert got["uuid"] is None and "not found" in got["error"]


def test_first_session_is_not_a_change():
    rec = {"gpu_uuid": "GPU-a"}
    cmp = ss.compare_to_previous(rec, [])
    assert cmp["first_session"] is True and cmp["gpu_changed"] is False


def test_gpu_change_is_detected():
    """The exact failure that split the study across two cards."""
    prev = {"gpu_uuid": "GPU-2ddfc69f", "packages": {}}
    rec = {"gpu_uuid": "GPU-afb936de", "packages": {}}
    cmp = ss.compare_to_previous(rec, [prev])
    assert cmp["gpu_changed"] is True
    assert any("GPU-2ddfc69f -> GPU-afb936de" in c for c in cmp["changes"])


def test_same_gpu_is_not_flagged():
    prev = {"gpu_uuid": "GPU-a", "packages": {}}
    rec = {"gpu_uuid": "GPU-a", "packages": {}}
    assert ss.compare_to_previous(rec, [prev])["gpu_changed"] is False


def test_package_drift_reported_even_on_same_gpu():
    prev = {"gpu_uuid": "GPU-a", "packages": {"vllm": "0.26.0"}}
    rec = {"gpu_uuid": "GPU-a", "packages": {"vllm": "0.27.0"}}
    cmp = ss.compare_to_previous(rec, [prev])
    assert cmp["gpu_changed"] is False
    assert any("vllm: 0.26.0 -> 0.27.0" in c for c in cmp["changes"])


def test_compare_skips_entries_without_a_uuid():
    history = [{"gpu_uuid": "GPU-a", "packages": {}}, {"gpu_uuid": None}]
    cmp = ss.compare_to_previous({"gpu_uuid": "GPU-a", "packages": {}}, history)
    assert cmp["gpu_changed"] is False


def test_log_roundtrip(tmp_path):
    path = str(tmp_path / "sessions.jsonl")
    ss.append_log(path, {"gpu_uuid": "GPU-a", "utc_timestamp": "t1"})
    ss.append_log(path, {"gpu_uuid": "GPU-b", "utc_timestamp": "t2"})
    log = ss.read_log(path)
    assert [e["gpu_uuid"] for e in log] == ["GPU-a", "GPU-b"]


def test_log_survives_a_truncated_final_line(tmp_path):
    """Preemption mid-write must not destroy the history."""
    path = tmp_path / "sessions.jsonl"
    path.write_text('{"gpu_uuid": "GPU-a"}\n{"gpu_uuid": "GPU-b"\n')
    assert [e["gpu_uuid"] for e in ss.read_log(str(path))] == ["GPU-a"]


def test_read_log_missing_file_is_empty():
    assert ss.read_log("/nonexistent/path/x.jsonl") == []


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def _summary(concurrency=15, ttft=0.1973, out_tps=26.7, vram=15648.0):
    return {
        "run_id": "cal-1",
        "arm": "B",
        "levels": [{
            "concurrency": concurrency,
            "requested": 20, "completed": 20, "errored": 0,
            "ttft_p50": ttft, "ttft_p95": ttft * 1.4, "e2e_p50": 9.6,
            "output_tok_per_s": out_tps, "completed_requests_per_s": 0.1,
            "resources": {"vram_peak_mb": vram, "vram_steady_state_mb": vram - 400},
        }],
    }


def test_record_from_summary_converts_to_ms(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_summary()))
    rec = cal.record_from_summary(str(p), gpu_uuid="GPU-a", concurrency=15)
    assert rec["ttft_p50_ms"] == pytest.approx(197.3)
    assert rec["output_tok_per_s"] == pytest.approx(26.7)
    assert rec["vram_peak_mb"] == pytest.approx(15648.0)
    assert rec["gpu_uuid"] == "GPU-a"


def test_record_from_summary_rejects_wrong_concurrency(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_summary(concurrency=15)))
    with pytest.raises(ValueError, match="no level at concurrency=8"):
        cal.record_from_summary(str(p), gpu_uuid="GPU-a", concurrency=8)


def test_record_from_summary_refuses_ambiguous_multilevel(tmp_path):
    """Silently picking a level would widen the measured noise floor."""
    data = _summary()
    data["levels"].append({**data["levels"][0], "concurrency": 32})
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pass --concurrency"):
        cal.record_from_summary(str(p), gpu_uuid="GPU-a")


def test_analyze_reports_spread():
    recs = [
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.0},
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 27.0},
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.5},
    ]
    a = cal.analyze(recs)
    m = a["metrics"]["output_tok_per_s"]["overall"]
    assert m["n"] == 3
    assert m["mean"] == pytest.approx(26.5)
    assert m["min"] == 26.0 and m["max"] == 27.0


def test_analyze_flags_hardware_dominated_metric():
    """Tight within each card, far apart between them -> hardware, not noise."""
    recs = [
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.0},
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.1},
        {"gpu_uuid": "GPU-b", "output_tok_per_s": 34.0},
        {"gpu_uuid": "GPU-b", "output_tok_per_s": 34.1},
    ]
    a = cal.analyze(recs)
    assert a["n_distinct_gpus"] == 2
    assert a["metrics"]["output_tok_per_s"]["hardware_dominated"] is True
    assert "HARDWARE-DOMINATED" in cal.format_analysis(a)


def test_analyze_does_not_flag_when_cards_agree():
    recs = [
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.0},
        {"gpu_uuid": "GPU-a", "output_tok_per_s": 27.0},
        {"gpu_uuid": "GPU-b", "output_tok_per_s": 26.4},
        {"gpu_uuid": "GPU-b", "output_tok_per_s": 26.6},
    ]
    assert cal.analyze(recs)["metrics"]["output_tok_per_s"]["hardware_dominated"] is False


def test_analyze_marks_noisy_metric():
    recs = [{"gpu_uuid": "GPU-a", "ttft_p50_ms": v} for v in (100.0, 200.0, 300.0)]
    assert cal.analyze(recs)["metrics"]["ttft_p50_ms"]["noisy"] is True


def test_analyze_single_record_is_reportable():
    a = cal.analyze([{"gpu_uuid": "GPU-a", "output_tok_per_s": 26.0}])
    assert a["n_records"] == 1
    assert "Fewer than 2 runs" in cal.format_analysis(a)


def test_analyze_empty_log():
    a = cal.analyze([])
    assert a["n_records"] == 0
    assert cal.format_analysis(a)


def test_calibration_log_roundtrip(tmp_path):
    path = str(tmp_path / "cal.jsonl")
    cal.append_record(path, {"gpu_uuid": "GPU-a", "output_tok_per_s": 26.0})
    cal.append_record(path, {"gpu_uuid": "GPU-b", "output_tok_per_s": 27.0})
    assert len(cal.read_log(path)) == 2
