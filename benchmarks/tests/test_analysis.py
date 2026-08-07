"""Tests for the analysis layer, against small synthetic artifacts.

No GPU, no real run data. The fixtures deliberately mirror the *shape* of the
measured runs — arm B throughput falling with load, arm C rising; arm B VRAM
growing linearly, arm C flat — so the pipeline is exercised on data that
behaves like the real thing without hardcoding any measured value.

The single most important test here is
``test_refuses_to_compare_across_gpus``: a silent cross-card comparison is the
failure the whole study exists to prevent, and the guard must be a refusal
rather than a warning.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import compare_arms, make_results, raw_stats  # noqa: E402
from analysis.loader import (  # noqa: E402
    GUARDED_FIELDS,
    ProvenanceMismatch,
    load_run,
    require_comparable,
    shared_concurrencies,
)
from common import trace_analysis as ta  # noqa: E402

GPU_A = "GPU-3bda9160-0000-0000-0000-000000000000"
GPU_B = "GPU-afb936de-0000-0000-0000-000000000000"
FIXTURE_SHA = "a69c2be7" + "0" * 56


def _manifest(**over):
    m = {
        "run_id": "r", "arm": "B", "concurrency": 1,
        "gpu_uuid": GPU_A, "gpu_name": "NVIDIA A100-SXM4-80GB",
        "fixture_sha256": FIXTURE_SHA, "config_sha256": "cfg" + "0" * 61,
        "max_new_tokens": 256, "pip_freeze_sha256": "pip" + "0" * 61,
        "driver_version": "580.82.07", "cuda_version": "13.0",
        "model": "Qwen/Qwen2.5-7B-Instruct", "context_tokens": 2620,
        "fixture_name": "shared_prefix", "profiling_layer": 1,
        "manifest_complete": True, "collection_errors": [],
        "utc_timestamp": "2026-08-05T00:00:00Z", "extra": {},
    }
    m.update(over)
    return m


def _level(c, ttft, out_tps, vram, wall=60.0, warmup=5.0, steady=True, notes=None):
    return {
        "concurrency": c, "requested": 8, "completed": 8, "errored": 0,
        "wall_time": wall,
        "ttft_p50": ttft, "ttft_p95": ttft * 1.3, "ttft_p99": ttft * 1.5,
        "e2e_p50": ttft * 40, "e2e_p95": ttft * 52, "e2e_p99": ttft * 60,
        "itl_mean": 0.037, "itl_p95": 0.05,
        "input_tokens_total": 2620 * 8, "output_tokens_total": 256 * 8,
        "total_tokens": (2620 + 256) * 8,
        "input_tok_per_s": 1000.0, "output_tok_per_s": out_tps,
        "total_tok_per_s": 1000.0 + out_tps,
        "completed_requests_per_s": 0.1,
        "resources": {
            "vram_peak_mb": vram,
            "vram_steady_state_mb": (vram - 300) if steady else None,
            "util_gpu_mean_pct": 24.0, "util_gpu_max_pct": 60.0,
            "cpu_mean_pct_across_cores": 30.0,
            "cpu_peak_pct_across_cores": 55.0, "cpu_core_count": 12,
            "warmup_s": warmup,
        },
        "notes": notes or {"itl_sample_count": 100},
    }


def _write_run(tmp_path, run_id, arm, levels, manifest_over=None, records=None):
    summary = {"run_id": run_id, "arm": arm, "levels": levels}
    if arm == "B":
        summary["vram_slope"] = {"available": True, "slope_mb_per_request": 314.9,
                                 "r_squared": 0.999}
    spath = tmp_path / f"{run_id}_summary.json"
    spath.write_text(json.dumps(summary))
    m = _manifest(run_id=run_id, arm=arm, **(manifest_over or {}))
    (tmp_path / f"{run_id}_manifest.json").write_text(json.dumps(m))
    if records is not None:
        (tmp_path / f"{run_id}_raw.json").write_text(
            json.dumps({"run_id": run_id, "records": records})
        )
    return str(spath)


def _records(n=8, tokens=6, itl=0.037, ttft=0.2, degrade=0.0, stagger=0.0):
    out = []
    for i in range(n):
        t_submit = i * stagger
        t_first = t_submit + ttft + i * stagger * 0.5
        ts = [t_first]
        for k in range(1, tokens):
            ts.append(ts[-1] + itl * (1.0 + degrade * k))
        out.append({
            "request_id": f"req-{i:03d}", "prompt_tokens": 2620,
            "output_tokens": tokens, "t_submit": t_submit,
            "t_first_token": t_first, "per_token_timestamps": ts,
            "t_done": ts[-1], "error": None,
        })
    return out


@pytest.fixture
def two_arms(tmp_path):
    b = _write_run(tmp_path, "armB_shared", "B", [
        _level(1, 0.197, 28.0, 15648.0),
        _level(8, 0.9, 13.8, 18000.0),
        _level(32, 3.2, 13.7, 25502.0),
    ], records=_records())
    c = _write_run(tmp_path, "armC_shared", "C", [
        _level(1, 0.05, 96.6, 74933.0),
        _level(8, 0.06, 744.0, 74933.0),
        _level(32, 0.07, 2376.0, 74933.0, wall=3.45, steady=False,
               notes={"kv_cache": {"stats_source": None,
                                   "usage_perc_peak": None}}),
    ], manifest_over={"extra": {"gpu_memory_utilization": 0.9}},
        records=_records(itl=0.0123))
    return b, c


# --------------------------------------------------------------------------
# the provenance guard
# --------------------------------------------------------------------------


def test_matched_runs_compare_fine(two_arms):
    runs = [load_run(p) for p in two_arms]
    require_comparable(runs)  # must not raise
    assert shared_concurrencies(runs) == [1, 8, 32]


@pytest.mark.parametrize("field,bad", [
    ("gpu_uuid", GPU_B),
    ("fixture_sha256", "deadbeef" + "0" * 56),
    ("max_new_tokens", 5),
    ("pip_freeze_sha256", "moved" + "0" * 59),
])
def test_refuses_to_compare_on_each_always_guarded_field(tmp_path, field, bad):
    """Every always-guarded field must be a hard refusal, naming the field."""
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={field: bad})
    runs = [load_run(a), load_run(b)]
    with pytest.raises(ProvenanceMismatch) as exc:
        require_comparable(runs)
    assert field in str(exc.value)
    assert "REFUSING TO COMPARE" in str(exc.value)


def test_config_sha_mismatch_refused_WITHIN_an_arm(tmp_path):
    """Two runs of the same arm must share a resolved config."""
    a = _write_run(tmp_path, "a1", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "a2", "B", [_level(1, 0.2, 28.0, 15648.0)],
                   manifest_over={"config_sha256": "other" + "0" * 59})
    with pytest.raises(ProvenanceMismatch, match="config_sha256"):
        require_comparable([load_run(a), load_run(b)])


def test_config_sha_may_differ_ACROSS_arms(tmp_path):
    """Arm C's resolved config carries the vLLM serving knobs, which ARE the
    independent variable — so the hashes differ by construction. Enforcing
    equality here would make the study's headline comparison impossible."""
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={"config_sha256": "vllmcfg" + "0" * 57})
    require_comparable([load_run(a), load_run(b)])  # must not raise


@pytest.mark.parametrize("field,bad", [
    ("model", "meta-llama/Llama-3-8B"),
    ("context_tokens", 1024),
    ("fixture_name", "low_overlap"),
])
def test_cross_arm_still_checks_shared_controls(tmp_path, field, bad):
    """Relaxing config_sha256 across arms must not relax what it stood for."""
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={field: bad})
    with pytest.raises(ProvenanceMismatch, match=field):
        require_comparable([load_run(a), load_run(b)])


def test_cross_arm_config_check_is_reported(tmp_path):
    """The substitution must be visible to the reader, not silent."""
    from analysis.loader import config_check_mode

    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)])
    cc = config_check_mode([load_run(a), load_run(b)])
    assert cc["cross_arm"] is True
    assert "independent variable" in cc["note"]
    md = compare_arms.render_markdown([load_run(a), load_run(b)], [1])
    assert "Config check:" in md


def test_refuses_to_compare_across_gpus(tmp_path):
    """The failure this study exists to prevent. Refusal, never a warning."""
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={"gpu_uuid": GPU_B})
    with pytest.raises(ProvenanceMismatch) as exc:
        require_comparable([load_run(a), load_run(b)])
    msg = str(exc.value)
    assert GPU_A in msg and GPU_B in msg
    assert "hard refusal" in msg


def test_cli_exits_nonzero_on_mismatch(tmp_path, capsys):
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={"gpu_uuid": GPU_B})
    rc = compare_arms.main([a, b, "--out-dir", str(tmp_path / "o")])
    assert rc == 3
    assert not os.path.exists(tmp_path / "o" / "compare_arms.md")


def test_mismatch_can_be_waived_but_is_recorded(tmp_path):
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={"gpu_uuid": GPU_B})
    runs = [load_run(a), load_run(b)]
    require_comparable(runs, allow=["gpu_uuid"])  # no raise
    md = compare_arms.render_markdown(runs, [1], waived=["gpu_uuid"])
    assert "PROVENANCE GUARD WAIVED" in md
    assert "gpu_uuid" in md


def test_missing_manifest_is_refused(tmp_path):
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    spath = tmp_path / "b_summary.json"
    spath.write_text(json.dumps({"run_id": "b", "arm": "C",
                                 "levels": [_level(1, 0.05, 96.6, 74933.0)]}))
    with pytest.raises(ProvenanceMismatch, match="no manifest"):
        require_comparable([load_run(a), load_run(str(spath))])


def test_single_run_needs_no_guard(tmp_path):
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    require_comparable([load_run(a)])


# --------------------------------------------------------------------------
# VRAM separation
# --------------------------------------------------------------------------


def test_comparison_table_contains_no_vram_column(two_arms):
    runs = [load_run(p) for p in two_arms]
    rows = compare_arms.build_table(runs, 8)
    metrics = {r["metric"].lower() for r in rows}
    assert not any("vram" in m or "memory" in m for m in metrics)


def test_vram_appears_only_in_its_own_section(two_arms):
    runs = [load_run(p) for p in two_arms]
    md = compare_arms.render_markdown(runs, [1, 8, 32])
    assert "NOT comparable across arms" in md
    # The explanation must travel with the numbers.
    assert "pre-allocates" in md
    assert "gpu_memory_utilization" in md
    # And the VRAM section must come after the comparison tables.
    assert md.index("## Concurrency 1") < md.index("## Memory")


def test_csv_tags_vram_rows_as_not_comparable(two_arms):
    runs = [load_run(p) for p in two_arms]
    text = compare_arms.render_csv(runs, [1])
    assert "[NOT COMPARABLE ACROSS ARMS]" in text


def test_missing_steady_state_is_explained(two_arms):
    runs = [load_run(p) for p in two_arms]
    md = compare_arms.render_markdown(runs, [1, 8, 32])
    assert "before the monitor's warm-up window" in md


def test_null_stats_source_is_called_out(two_arms):
    runs = [load_run(p) for p in two_arms]
    md = compare_arms.render_markdown(runs, [32])
    assert "Absent is not zero" in md


# --------------------------------------------------------------------------
# raw_stats
# --------------------------------------------------------------------------


def test_distribution_reports_more_than_three_percentiles():
    d = raw_stats.distribution([0.1, 0.2, 0.3, 0.4, 0.5])
    assert d["n"] == 5
    assert set(d["percentiles"]) >= {"p0", "p25", "p50", "p75", "p99", "p100"}
    assert d["spread_ratio"] == pytest.approx(5.0)


def test_itl_trajectory_detects_flat():
    """Steady serialisation: every token pays the same cost."""
    recs = _records(n=4, tokens=40, itl=2.3, degrade=0.0)
    t = raw_stats.itl_trajectory(recs, bins=10)
    assert t["available"]
    assert abs(t["drift_pct"]) < 10.0
    assert "flat-but-slow" in t["verdict"]


def test_itl_trajectory_detects_degradation():
    """Progressive build-up: later tokens cost more than earlier ones."""
    recs = _records(n=4, tokens=40, itl=0.05, degrade=0.05)
    t = raw_stats.itl_trajectory(recs, bins=10)
    assert t["available"]
    assert t["drift_pct"] > 10.0
    assert "degrading" in t["verdict"]


def test_itl_trajectory_unavailable_when_too_few_tokens():
    t = raw_stats.itl_trajectory(_records(n=2, tokens=3), bins=10)
    assert t["available"] is False


def test_ttft_vs_order_detects_starvation():
    recs = _records(n=12, tokens=6, stagger=0.5)
    o = raw_stats.ttft_vs_order(recs)
    assert o["available"]
    assert o["spearman_submit_vs_ttft"] > 0.9
    assert "FIFO" in o["verdict"]


def test_ttft_vs_order_detects_no_penalty():
    recs = _records(n=12, tokens=6, stagger=0.0)
    o = raw_stats.ttft_vs_order(recs)
    assert o["available"]
    assert "no strong submission-order penalty" in o["verdict"]


def test_verify_output_lengths_passes():
    v = raw_stats.verify_output_lengths(_records(n=5, tokens=256), 256)
    assert v["all_equal_expected"] is True
    assert v["mismatch_count"] == 0


def test_verify_output_lengths_flags_early_stop():
    """A short request means output throughput is not purely a stack property."""
    recs = _records(n=5, tokens=256)
    recs[2]["output_tokens"] = 91
    v = raw_stats.verify_output_lengths(recs, 256)
    assert v["all_equal_expected"] is False
    assert v["mismatch_count"] == 1
    assert v["mismatches"][0]["request_id"] == "req-002"
    assert "NOT purely a property" in v["note"]


def test_analyse_end_to_end(two_arms):
    run = load_run(two_arms[0])
    res = raw_stats.analyse(run, concurrency=None)
    assert res["completed"] == 8
    report = raw_stats.format_report(res)
    assert "PER-REQUEST ANALYSIS" in report
    assert "ignore_eos check" in report


# --------------------------------------------------------------------------
# prefill/decode split (deliverable 4)
# --------------------------------------------------------------------------


def _trace(kernels):
    return {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": n, "ts": ts, "dur": d,
         "args": {"stream": 7, "grid": [64, 1, 1]}}
        for n, ts, d in kernels
    ]}


def test_prefill_boundary_is_last_first_token():
    recs = _records(n=3, tokens=5, ttft=0.2, stagger=0.1)
    b = ta.prefill_boundary_from_records(recs)
    assert b["available"]
    expected = max(r["t_first_token"] for r in recs) - min(r["t_submit"] for r in recs)
    assert b["prefill_end_rel_s"] == pytest.approx(expected)


def test_measured_boundary_beats_the_name_heuristic():
    """The heuristic leaves work unattributed; the measured boundary does not."""
    trace = _trace([
        ("ampere_s16816gemm_f16", 0.0, 100.0),      # prefill-shaped
        ("elementwise_kernel", 120.0, 50.0),        # unattributable by name
        ("gemv2T_kernel", 200.0, 30.0),             # decode-shaped
    ])
    heuristic = ta.analyze_trace(trace)["prefill_decode"]
    assert heuristic["method"] == "heuristic_name_shape"
    assert heuristic["unattributed_kernel_time_us"] == 50.0
    assert heuristic["method_confidence"].startswith("inferred")

    recs = _records(n=2, tokens=5, ttft=0.2, itl=0.1)
    derived = ta.derive_decode_start_us(trace, recs)
    assert derived["available"]
    measured = ta.analyze_trace(
        trace, decode_start_us=derived["decode_start_us"], boundary_meta=derived
    )["prefill_decode"]
    assert measured["method_confidence"] == "measured"
    # Every microsecond is attributed once the boundary is known.
    total = measured["prefill_kernel_time_us"] + measured["decode_kernel_time_us"]
    assert total == pytest.approx(180.0)


def test_decode_only_window_is_identified_not_mapped():
    """A bounded Layer 3 window opens after prefill; do not fabricate a split."""
    trace = _trace([("gemv2T_kernel", 1000.0, 40.0)])
    recs = _records(n=2, tokens=5, ttft=0.2, itl=0.1)
    boundary = ta.prefill_boundary_from_records(recs)["prefill_end_rel_s"]
    derived = ta.derive_decode_start_us(
        trace, recs, window_start_rel_s=boundary + 1.0
    )
    assert derived["method"] == "measured_boundary_window_after_prefill"
    assert "DECODE ONLY" in derived["note"]
    split = ta.analyze_trace(trace, decode_start_us=derived["decode_start_us"],
                             boundary_meta=derived)["prefill_decode"]
    assert split["prefill_kernel_time_us"] == 0.0


def test_boundary_unavailable_without_first_tokens():
    recs = [{"request_id": "x", "t_submit": 0.0, "t_done": 1.0,
             "t_first_token": None, "per_token_timestamps": [], "error": None}]
    assert ta.prefill_boundary_from_records(recs)["available"] is False


# --------------------------------------------------------------------------
# RESULTS.md generation
# --------------------------------------------------------------------------


def test_results_document_is_generated_from_the_data(two_arms, tmp_path):
    runs = [load_run(p) for p in two_arms]
    doc = make_results.build_results(
        runs,
        excluded=[("smoke_c1_summary.json",
                   "superseded: different stack AND different GPU")],
        command="python -m analysis.make_results ...",
    )
    assert "Generated file. Do not edit." in doc
    assert GPU_A in doc                      # provenance
    assert "armB_shared" in doc and "armC_shared" in doc
    assert "smoke_c1_summary.json" in doc    # exclusion recorded, not dropped
    assert "superseded" in doc
    assert "## 5. Measured vs not yet measured" in doc
    assert "## 6. Caveats" in doc


def test_results_detects_caveats_from_the_artifacts(two_arms):
    runs = [load_run(p) for p in two_arms]
    caveats = {c["id"] for c in make_results.detect_caveats(runs)}
    assert "vram-not-comparable" in caveats
    assert "kv-stats-unresolved" in caveats
    assert any(c.startswith("steady-state-missing") for c in caveats)


def test_measured_status_marks_layer4_as_not_measured(two_arms):
    runs = [load_run(p) for p in two_arms]
    rows = {r["metric"]: r["status"] for r in make_results.measured_status(runs, None)}
    assert rows["TTFT p50/p95/p99"] == "measured"
    assert rows["Achieved occupancy"] == "not yet measured"
    assert rows["Prefix cache hit rate"] == "not yet measured"


def test_measured_status_requires_a_measured_split(two_arms):
    runs = [load_run(p) for p in two_arms]
    heuristic = {"prefill_decode": {"method_confidence": "inferred — nope"}}
    rows = {r["metric"]: r["status"]
            for r in make_results.measured_status(runs, heuristic)}
    assert rows["Prefill vs decode kernel time split"] == "not yet measured"

    measured = {"prefill_decode": {"method_confidence": "measured"}}
    rows = {r["metric"]: r["status"]
            for r in make_results.measured_status(runs, measured)}
    assert rows["Prefill vs decode kernel time split"] == "measured"


def test_make_results_cli_writes_the_file(two_arms, tmp_path):
    out = tmp_path / "RESULTS.md"
    rc = make_results.main([
        "--arm-b", two_arms[0], "--arm-c", two_arms[1], "--out", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert "# GPU serving benchmark — results" in text
    assert "Concurrency 32" in text


def test_make_results_cli_refuses_mismatch(tmp_path):
    a = _write_run(tmp_path, "a", "B", [_level(1, 0.2, 28.0, 15648.0)])
    b = _write_run(tmp_path, "b", "C", [_level(1, 0.05, 96.6, 74933.0)],
                   manifest_over={"gpu_uuid": GPU_B})
    out = tmp_path / "RESULTS.md"
    rc = make_results.main(["--arm-b", a, "--arm-c", b, "--out", str(out)])
    assert rc == 3
    assert not out.exists()


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def test_footnote_carries_provenance(two_arms):
    from analysis import plots

    runs = [load_run(p) for p in two_arms]
    text = plots.footnote_text(runs, "C=32")
    assert GPU_A in text
    assert FIXTURE_SHA[:12] in text
    assert "max_new_tokens=256" in text
    assert "armB_shared" in text and "armC_shared" in text


def test_empty_figure_is_skipped_not_written(two_arms, tmp_path):
    """A blank PNG in the output directory looks like a rendered figure."""
    from analysis import plots

    runs = [load_run(p) for p in two_arms]
    out = str(tmp_path / "figs")
    # Level 999 exists nowhere, so there is nothing to plot.
    with pytest.raises(plots.NoDataForFigure, match="NOT written"):
        plots.fig2_latency_cdf(runs, out, concurrency=999)
    assert not os.path.exists(os.path.join(out, "fig2_latency_cdf.png"))


def test_render_all_records_skips_with_reasons(two_arms, tmp_path):
    from analysis import plots

    arm_b, arm_c = (load_run(p) for p in two_arms)
    result = plots.render_all(arm_b, arm_c, str(tmp_path / "figs"),
                              cdf_concurrency=1, trace_summary=None)
    # fig6 has no trace input; its absence must be explained.
    assert "fig6_launch_overhead" in result["skipped"]
    assert "Layer 3" in result["skipped"]["fig6_launch_overhead"]
    assert "fig1_throughput_vs_concurrency" in result["figures"]


def test_figures_emit_png_and_svg(two_arms, tmp_path):
    from analysis import plots

    runs = [load_run(p) for p in two_arms]
    paths = plots.fig1_throughput_vs_concurrency(runs, str(tmp_path / "figs"))
    assert len(paths) == 2
    assert any(p.endswith(".png") for p in paths)
    assert any(p.endswith(".svg") for p in paths)
    for p in paths:
        assert os.path.getsize(p) > 0


def test_vram_figure_is_arm_b_only(two_arms, tmp_path):
    """Arm C must not be plottable here — its VRAM is a config value."""
    from analysis import plots

    arm_b = load_run(two_arms[0])
    paths = plots.fig4_vram_slope(arm_b, str(tmp_path / "figs"))
    assert len(paths) == 2
    svg = open([p for p in paths if p.endswith(".svg")][0]).read()
    assert "Arm C is deliberately absent" in svg or "deliberately absent" in svg


def test_itl_trajectory_flat_after_initial_transient():
    """A fast first bin must not be reported as progressive degradation.

    Observed on the real arm C data: bins 10.3, 12.4, 12.4 ... 12.7. Endpoint
    drift is +22.8% purely because the first decile is faster (the batch is
    still filling); everything after it is flat to within 2.4%.
    """
    recs = _records(n=4, tokens=40, itl=0.0124)
    # Make the first tenth of each request faster, leave the rest steady.
    for r in recs:
        ts = r["per_token_timestamps"]
        shift = 0.0
        for i in range(1, len(ts)):
            gap = 0.0103 if i <= len(ts) // 10 else 0.0124
            shift += gap
            ts[i] = ts[0] + shift
        r["t_done"] = ts[-1]
    t = raw_stats.itl_trajectory(recs, bins=10)
    assert t["available"]
    assert t["drift_pct"] > 10.0                     # endpoint view says "rising"
    assert abs(t["drift_pct_excl_first_bin"]) < 10.0  # but the trend is flat
    assert "initial transient" in t["verdict"]
    assert "NOT progressive build-up" in t["verdict"]


def test_ttft_vs_order_names_a_reversed_correlation():
    """rho ~ -1 means later requests were served FASTER, not 'no effect'."""
    recs = _records(n=12, tokens=6)
    for i, r in enumerate(recs):
        # TTFT falls monotonically with submission order.
        r["t_submit"] = i * 0.01
        r["t_first_token"] = r["t_submit"] + (0.4 - i * 0.02)
        span = r["per_token_timestamps"][-1] - r["per_token_timestamps"][0]
        r["per_token_timestamps"] = [
            r["t_first_token"] + (span * k / 5) for k in range(6)
        ]
        r["t_done"] = r["per_token_timestamps"][-1]
    o = raw_stats.ttft_vs_order(recs)
    assert o["spearman_submit_vs_ttft"] < -0.9
    assert "FASTER" in o["verdict"]
    assert "already-warm batch" in o["verdict"]
