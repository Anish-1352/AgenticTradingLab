"""Arm C and ablation tests. No GPU, no torch, no vllm — everything mocked.

vLLM is not installed in the test environment (and must not be: it drags in a
cu13 torch). Arm C keeps every ``import vllm`` inside a function for exactly
this reason, so the module imports and its logic is testable here.
"""

import os
import sys
import types

import pytest

_BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _BENCH)
sys.path.insert(0, os.path.join(_BENCH, "ablations"))

import _ablation_util as au  # noqa: E402
from common.metrics import Summary, attach_resources, console_lines, summarize  # noqa: E402
from common.monitor import ResourceMonitor, Sample  # noqa: E402
from runners import bench_hf_baseline as arm_b  # noqa: E402
from runners import bench_vllm_optimized as arm_c  # noqa: E402


def test_vllm_is_not_installed():
    """Guard the premise: these tests are worthless if vllm is importable."""
    with pytest.raises(ImportError):
        import vllm  # noqa: F401


# --------------------------------------------------------------------------
# CLI parity between arms
# --------------------------------------------------------------------------

SHARED_FLAGS = [
    "--config", "--concurrency", "--n-requests", "--run-id", "--fixture",
    "--out-dir", "--traces-dir", "--fixtures-dir", "--profile",
    "--max-new-tokens", "--dry-run", "--no-resume",
]


def _option_strings(parser):
    out = set()
    for action in parser._actions:
        out.update(action.option_strings)
    return out


def test_both_arms_accept_every_shared_flag():
    """A shared flag missing from one arm means the two cannot be invoked
    identically, and a sweep script would silently diverge between them."""
    b = _option_strings(arm_b.build_parser())
    c = _option_strings(arm_c.build_parser())
    for flag in SHARED_FLAGS:
        assert flag in b, f"arm B missing {flag}"
        assert flag in c, f"arm C missing {flag}"


def test_arm_c_has_its_own_flags():
    c = _option_strings(arm_c.build_parser())
    for flag in ("--enable-prefix-caching", "--no-prefix-caching",
                 "--max-num-seqs", "--gpu-memory-utilization"):
        assert flag in c


def test_prefix_caching_is_required_not_defaulted():
    """Prefix caching is the prefix_cache ablation's variable. An implicit
    default would silently decide the result."""
    parser = arm_c.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--concurrency", "1"])

    on = parser.parse_args(["--enable-prefix-caching"])
    off = parser.parse_args(["--no-prefix-caching"])
    assert on.prefix_caching is True
    assert off.prefix_caching is False


def test_shared_flags_parse_to_same_attribute_names():
    b = arm_b.build_parser().parse_args(
        ["--concurrency", "1", "8", "--n-requests", "5", "--fixture", "low_overlap"]
    )
    c = arm_c.build_parser().parse_args(
        ["--concurrency", "1", "8", "--n-requests", "5", "--fixture", "low_overlap",
         "--no-prefix-caching"]
    )
    for attr in ("concurrency", "n_requests", "fixture", "dry_run", "no_resume"):
        assert getattr(b, attr) == getattr(c, attr), attr


# --------------------------------------------------------------------------
# VllmIntrospector — mocked engines
# --------------------------------------------------------------------------


class _Metric:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _EngineWithGetMetrics:
    def __init__(self, pairs):
        self._pairs = pairs

    def get_metrics(self):
        return [_Metric(n, v) for n, v in self._pairs]


class _CacheConfig:
    def __init__(self, num_gpu_blocks=1000, block_size=16, gmu=0.9, prefix=True):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.gpu_memory_utilization = gmu
        self.enable_prefix_caching = prefix


def test_introspector_reads_get_metrics():
    engine = _EngineWithGetMetrics([
        ("vllm:gpu_cache_usage_perc", 0.42),
        ("vllm:gpu_prefix_cache_hit_rate", 0.87),
    ])
    got = arm_c.VllmIntrospector(engine).sample()
    assert got["kv_cache_usage_perc"] == pytest.approx(0.42)
    assert got["prefix_cache_hit_rate"] == pytest.approx(0.87)
    assert got["stats_source"] == "engine.get_metrics()"


def test_introspector_derives_hit_rate_from_counters():
    """Some versions expose hits/queries rather than a precomputed rate."""
    engine = _EngineWithGetMetrics([
        ("vllm:prefix_cache_hits", 30.0),
        ("vllm:prefix_cache_queries", 120.0),
    ])
    got = arm_c.VllmIntrospector(engine).sample()
    assert got["prefix_cache_hit_rate"] == pytest.approx(0.25)
    assert got["prefix_cache_hits"] == 30.0


def test_introspector_returns_none_not_zero_when_nothing_resolves():
    """A missing stats path must be distinguishable from genuine zero usage."""
    engine = types.SimpleNamespace()
    intro = arm_c.VllmIntrospector(engine)
    got = intro.sample()
    assert got["kv_cache_usage_perc"] is None
    assert got["prefix_cache_hit_rate"] is None
    assert got["stats_source"] is None
    # And the failure is explained, not silent.
    assert intro.report()["stats_probe_attempts"]


def test_introspector_cache_config_paths():
    engine = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(cache_config=_CacheConfig())
    )
    cc = arm_c.VllmIntrospector(engine).cache_config()
    assert cc["num_gpu_blocks"] == 1000
    assert cc["block_size"] == 16
    assert cc["kv_capacity_tokens"] == 16000  # so absolute KV bytes derive
    assert cc["gpu_memory_utilization"] == 0.9
    assert cc["source"] == "vllm_config.cache_config"


def test_introspector_cache_config_fallback_path():
    engine = types.SimpleNamespace(
        engine=types.SimpleNamespace(cache_config=_CacheConfig(num_gpu_blocks=7, block_size=8))
    )
    cc = arm_c.VllmIntrospector(engine).cache_config()
    assert cc["source"] == "engine.cache_config"
    assert cc["kv_capacity_tokens"] == 56


def test_introspector_cache_config_absent():
    cc = arm_c.VllmIntrospector(types.SimpleNamespace()).cache_config()
    assert cc["num_gpu_blocks"] is None
    assert cc["kv_capacity_tokens"] is None


def test_reset_prefix_cache_found_and_absent():
    called = {"n": 0}

    class _E:
        def reset_prefix_cache(self):
            called["n"] += 1
            return True

    assert arm_c.VllmIntrospector(_E()).reset_prefix_cache()["reset"] is True
    assert called["n"] == 1

    missing = arm_c.VllmIntrospector(types.SimpleNamespace()).reset_prefix_cache()
    assert missing["reset"] is False
    assert "no reset_prefix_cache path" in missing["error"]


def test_make_prompt_falls_back_to_mapping_without_vllm():
    """With vllm absent, the plain mapping form is used — still token ids."""
    p = arm_c.make_prompt([1, 2, 3])
    assert p == {"prompt_token_ids": [1, 2, 3]}


# --------------------------------------------------------------------------
# metrics additions (deliverable 5)
# --------------------------------------------------------------------------


def _record(**kw):
    from common.metrics import RequestRecord

    base = dict(
        request_id="r", prompt_tokens=2620, output_tokens=4, t_submit=0.0,
        t_first_token=1.0, per_token_timestamps=[1.0, 1.5, 2.0, 2.5], t_done=2.5,
    )
    base.update(kw)
    return RequestRecord(**base)


def test_summary_carries_resources():
    s = summarize("run", 1, [_record()])
    assert s.resources == {}
    attach_resources(s, {"vram_peak_mb": 15648.0, "vram_steady_state_mb": 15036.0})
    assert s.resources["vram_peak_mb"] == 15648.0
    assert "resources" in s.to_dict()


def test_console_lines_surface_every_requested_metric():
    """The advisor's list, all reachable from the console rollup."""
    s = summarize("run", 15, [_record()])
    attach_resources(s, {
        "vram_peak_mb": 15648.0, "vram_steady_state_mb": 15036.0,
        "util_gpu_mean_pct": 62.5, "util_gpu_max_pct": 99.0,
        "cpu_mean_pct_across_cores": 21.0, "cpu_peak_pct_across_cores": 55.0,
        "cpu_core_count": 12,
    })
    text = "\n".join(console_lines(s))
    for token in ("TTFT", "e2e", "ITL", "throughput", "requests/s",
                  "VRAM", "steady", "GPU util", "CPU util", "total"):
        assert token in text, token
    # The occupancy caveat travels with the number it qualifies.
    assert "NOT occupancy" in text


def test_console_lines_tolerate_missing_resources():
    s = summarize("run", 1, [_record(error="CUDA_OOM")])
    text = "\n".join(console_lines(s))
    assert "—" in text  # nulls render, nothing raises


def test_monitor_reports_cpu_peak_and_mean():
    m = ResourceMonitor(out_csv=None)
    m.samples = [
        Sample(t=0.0, wall=0.0, vram_process_mb=100.0, vram_device_mb=None,
               util_gpu_pct=10.0, util_mem_pct=5.0, cpu_per_core=[10.0, 20.0]),
        Sample(t=10.0, wall=0.0, vram_process_mb=300.0, vram_device_mb=None,
               util_gpu_pct=90.0, util_mem_pct=50.0, cpu_per_core=[80.0, 100.0]),
    ]
    s = m.summary()
    assert s["cpu_mean_pct_across_cores"] == pytest.approx(52.5)  # (15 + 90) / 2
    assert s["cpu_peak_pct_across_cores"] == pytest.approx(90.0)  # busiest instant
    assert s["cpu_max_single_core_pct"] == pytest.approx(100.0)
    assert s["vram_peak_mb"] == pytest.approx(300.0)


def test_monitor_steady_state_excludes_warmup():
    m = ResourceMonitor(out_csv=None, warmup_s=5.0)
    m.samples = [
        Sample(t=0.0, wall=0.0, vram_process_mb=1.0, vram_device_mb=None,
               util_gpu_pct=None, util_mem_pct=None, cpu_per_core=[]),
        Sample(t=9.0, wall=0.0, vram_process_mb=100.0, vram_device_mb=None,
               util_gpu_pct=None, util_mem_pct=None, cpu_per_core=[]),
        Sample(t=10.0, wall=0.0, vram_process_mb=200.0, vram_device_mb=None,
               util_gpu_pct=None, util_mem_pct=None, cpu_per_core=[]),
    ]
    s = m.summary()
    assert s["vram_peak_mb"] == pytest.approx(200.0)      # includes warmup
    assert s["vram_steady_state_mb"] == pytest.approx(150.0)  # median of post-warmup
    assert s["vram_steady_state_sample_count"] == 2


# --------------------------------------------------------------------------
# ablation utilities
# --------------------------------------------------------------------------


def test_pct_delta():
    assert au.pct_delta(120.0, 100.0) == pytest.approx(20.0)
    assert au.pct_delta(80.0, 100.0) == pytest.approx(-20.0)
    assert au.pct_delta(1.0, 0.0) is None
    assert au.pct_delta(None, 100.0) is None
    assert au.pct_delta(100.0, None) is None


def test_level_for_selects_the_right_concurrency():
    summary = {"levels": [{"concurrency": 1, "ttft_p50": 0.1},
                          {"concurrency": 15, "ttft_p50": 0.9}]}
    assert au.level_for(summary, 15)["ttft_p50"] == 0.9
    assert au.level_for(summary, 999) is None
    assert au.level_for(None, 1) is None


def test_render_table_aligns_and_survives_short_rows():
    out = au.render_table(["a", "bb"], [["1", "2"], ["333"]])
    lines = out.splitlines()
    assert len(lines) == 4  # header, rule, 2 rows
    assert lines[0].startswith("a")


def test_fmt_handles_none_and_units():
    assert au.fmt(None) == "—"
    assert au.fmt(1.234, " ms", 1) == "1.2 ms"
    assert au.fmt(True) == "yes"


# --------------------------------------------------------------------------
# GIL ablation internals
# --------------------------------------------------------------------------


def test_gil_cost_sign_and_magnitude():
    import gil_attribution as gil

    # threadpool slower than processpool -> positive attributable GIL cost
    assert gil._gil_cost({"wall_time_s": 10.0}, {"wall_time_s": 5.0}) == pytest.approx(50.0)
    # process overhead dominating is a real low-concurrency outcome, not a bug
    assert gil._gil_cost({"wall_time_s": 5.0}, {"wall_time_s": 10.0}) == pytest.approx(-100.0)
    assert gil._gil_cost({"wall_time_s": 0.0}, {"wall_time_s": 1.0}) is None


def test_spin_holds_the_gil_and_scales():
    """The busy loop must actually cost time — a no-op would make the whole
    ablation report zero contention."""
    import time

    import gil_attribution as gil

    t0 = time.perf_counter()
    gil._spin(200_000)
    small = time.perf_counter() - t0

    t0 = time.perf_counter()
    gil._spin(800_000)
    large = time.perf_counter() - t0

    assert large > small
    assert small > 0


def test_calibrate_spin_returns_positive_rate():
    import gil_attribution as gil

    rate = gil.calibrate_spin(target_s=0.01)
    assert rate > 0


def test_synthetic_request_streams_every_token():
    """Mirrors arm B's producer-thread + queue structure, and must not drop
    chunks — dropping them is precisely the v1 ITL bug."""
    import gil_attribution as gil

    out = gil.synthetic_request(("t0", 5, 0, 0.001, 0.1))
    assert out["tokens"] == 5
    assert out["ttft"] is not None
    assert out["e2e"] > 0


def test_pyspy_absent_degrades_gracefully(monkeypatch):
    import gil_attribution as gil

    monkeypatch.setattr(gil.shutil, "which", lambda name: None)
    res = gil.start_pyspy("/tmp/out.svg", duration=1)
    assert res["available"] is False
    assert "py-spy" in res["reason"]


# --------------------------------------------------------------------------
# ablation CLIs
# --------------------------------------------------------------------------


def test_all_ablations_expose_dry_run():
    import continuous_batching as cb
    import gil_attribution as gil
    import prefix_cache as pc

    for mod in (pc, cb, gil):
        assert "--dry-run" in _option_strings(mod.build_parser())


def test_prefix_cache_runs_four_conditions(monkeypatch, tmp_path):
    """Two fixtures x caching on/off, each a separate subprocess."""
    import prefix_cache as pc

    seen = []

    def fake_run(script, run_id, extra_args, out_dir, dry_run=False, echo=True):
        seen.append({"run_id": run_id, "args": list(extra_args)})
        return {"run_id": run_id, "ok": True, "returncode": 0,
                "summary_path": str(tmp_path / f"{run_id}_summary.json"),
                "manifest_path": "", "command": [], "elapsed_s": 0.0,
                "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(pc, "run_condition", fake_run)
    rc = pc.main(["--dry-run", "--out-dir", str(tmp_path), "--concurrency", "15"])

    assert rc == 0
    assert len(seen) == 4
    fixtures = {a for s in seen for a in s["args"] if a in ("shared_prefix", "low_overlap")}
    assert fixtures == {"shared_prefix", "low_overlap"}
    on = sum(1 for s in seen if "--enable-prefix-caching" in s["args"])
    off = sum(1 for s in seen if "--no-prefix-caching" in s["args"])
    assert on == 2 and off == 2


def test_continuous_batching_holds_caching_off_in_both_conditions(monkeypatch, tmp_path):
    """Caching must be constant, or the measured gap is batching+caching."""
    import continuous_batching as cb

    seen = []

    def fake_run(script, run_id, extra_args, out_dir, dry_run=False, echo=True):
        seen.append(list(extra_args))
        return {"run_id": run_id, "ok": True, "returncode": 0,
                "summary_path": str(tmp_path / "s.json"), "manifest_path": "",
                "command": [], "elapsed_s": 0.0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(cb, "run_condition", fake_run)
    rc = cb.main(["--dry-run", "--out-dir", str(tmp_path), "--concurrency", "1", "8"])

    assert rc == 0
    assert len(seen) == 2
    for args in seen:
        assert "--no-prefix-caching" in args
        assert "--enable-prefix-caching" not in args
    # sequential condition pins the scheduler to one sequence
    seq = [a for a in seen if "--max-num-seqs" in a]
    assert len(seq) == 1
    assert seq[0][seq[0].index("--max-num-seqs") + 1] == "1"


def test_continuous_batching_defaults_to_low_overlap_fixture():
    """With caching off, a high-overlap fixture could still leak block reuse
    into the batching comparison."""
    import continuous_batching as cb

    assert cb.build_parser().parse_args([]).fixture == "low_overlap"
