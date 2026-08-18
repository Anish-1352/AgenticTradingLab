"""Heterogeneous multi-model serving: plan arithmetic and guards.

No GPU and no vllm here, so engine construction is mocked. What IS tested is
everything that decides whether a GPU session succeeds or is wasted:

* the memory budget is DIVIDED across engines — passing the full fraction to
  each is the mistake that makes multi-model look impossible when it is only
  misconfigured;
* a configuration whose weights cannot fit is refused before anything loads;
* absent KV stats read as absent, never as zero.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import cadence_model as cm            # noqa: E402
from runners import bench_vllm_multimodel as mm     # noqa: E402

ROSTER = os.path.join(os.path.dirname(__file__), "..", "configs", "model_roster.yaml")


@pytest.fixture(scope="module")
def roster():
    return mm.load_roster(ROSTER)


# ------------------------------------------------------------- the divisor --

def test_memory_fraction_is_divided_across_engines():
    """The whole point. Each engine claims a fraction of the WHOLE card."""
    assert mm.per_engine_memory_fraction(0.9, 1) == pytest.approx(0.9)
    assert mm.per_engine_memory_fraction(0.9, 4) == pytest.approx(0.225)
    assert mm.per_engine_memory_fraction(0.9, 7) == pytest.approx(0.12857, rel=1e-4)


def test_per_engine_fractions_sum_to_the_card_budget(roster):
    models = mm.select_models(roster, n_models=7)
    plan = mm.build_engine_configs(
        models, total_concurrency=32, quantization="awq",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic")
    assert sum(e.gpu_memory_utilization for e in plan.engines) == pytest.approx(0.9)


def test_zero_models_is_rejected():
    with pytest.raises(ValueError):
        mm.per_engine_memory_fraction(0.9, 0)


# ------------------------------------------------------- load fragmentation --

def test_total_concurrency_is_split_not_multiplied():
    """Total is held FIXED as N rises — that is the independent variable."""
    assert mm.split_concurrency(32, 1) == [32]
    assert mm.split_concurrency(32, 2) == [16, 16]
    assert mm.split_concurrency(32, 4) == [8, 8, 8, 8]
    assert sum(mm.split_concurrency(32, 7)) == 32


def test_remainder_goes_to_the_earliest_engines():
    split = mm.split_concurrency(32, 7)
    assert split == [5, 5, 5, 5, 4, 4, 4]
    assert max(split) - min(split) <= 1


def test_more_models_than_requests_is_warned_not_silently_idle(roster):
    models = mm.select_models(roster, n_models=7)
    plan = mm.build_engine_configs(
        models, total_concurrency=4, quantization="awq",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic")
    assert any("idle" in w for w in plan.warnings)


# -------------------------------------------------------------- VRAM guard --

def test_seven_models_unquantized_do_not_fit_an_a100(roster):
    """The hypothesis that makes quantization an enabling condition."""
    models = mm.select_models(roster, n_models=7)
    plan = mm.build_engine_configs(
        models, total_concurrency=32, quantization="none",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic", card_vram_gb=80.0)
    assert plan.vram["weights_gb"] > 72.0
    assert any(w.startswith("WEIGHTS ALONE") for w in plan.warnings)


def test_seven_models_quantized_do_fit_an_a100(roster):
    models = mm.select_models(roster, n_models=7)
    plan = mm.build_engine_configs(
        models, total_concurrency=32, quantization="awq",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic", card_vram_gb=80.0)
    assert not any(w.startswith("WEIGHTS ALONE") for w in plan.warnings)


def test_seven_models_quantized_do_NOT_fit_an_l4(roster):
    """Decides the card recommendation — L4 cannot hold the full roster."""
    models = mm.select_models(roster, n_models=7)
    plan = mm.build_engine_configs(
        models, total_concurrency=32, quantization="awq",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic", card_vram_gb=24.0)
    assert any(w.startswith("WEIGHTS ALONE") for w in plan.warnings)


def test_four_models_quantized_fit_an_l4(roster):
    models = mm.select_models(roster, n_models=4)
    plan = mm.build_engine_configs(
        models, total_concurrency=32, quantization="awq",
        gpu_memory_utilization=0.9, max_new_tokens=860,
        fixture_name="atl_realistic", card_vram_gb=24.0)
    assert not any(w.startswith("WEIGHTS ALONE") for w in plan.warnings)


def test_vram_estimate_excludes_kv_and_says_so(roster):
    models = mm.select_models(roster, n_models=4)
    est = mm.estimate_vram(models, "awq")
    assert "KV" in est["excludes"]
    assert est["tier"] == "DERIVED"


def test_quantization_reduces_the_estimate(roster):
    models = mm.select_models(roster, n_models=7)
    assert (mm.estimate_vram(models, "awq")["weights_gb"]
            < mm.estimate_vram(models, "none")["weights_gb"] / 2)


# ---------------------------------------------------------------- roster ----

def test_roster_has_seven_models(roster):
    assert len(roster["models"]) == 7


def test_roster_includes_the_benchmarked_anchor(roster):
    anchors = [m for m in roster["models"] if m.get("anchor")]
    assert len(anchors) == 1
    assert anchors[0]["id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_roster_names_the_models_that_cannot_be_self_hosted(roster):
    """The limitation is in the config, not only in the report."""
    blocked = {m["id"] for m in roster["not_self_hostable"]}
    assert "openai/gpt-5.5" in blocked
    assert "google/gemini-3.1-pro" in blocked
    assert all(m["reason"] == "weights not public"
               for m in roster["not_self_hostable"])


def test_unknown_model_id_is_carried_not_dropped(roster):
    chosen = mm.select_models(roster, models=["some/unlisted-model"])
    assert chosen[0]["unknown_to_roster"] is True
    assert mm.estimate_vram(chosen, "awq")["models_without_estimate"] == \
        ["some/unlisted-model"]


def test_asking_for_more_models_than_the_roster_has_fails_loudly(roster):
    with pytest.raises(SystemExit):
        mm.select_models(roster, n_models=99)


# ------------------------------------------------------------- KV absence ---

def test_kv_stats_absent_reads_as_absent_not_zero():
    """vLLM 0.26 reported stats_source null; that is not a zero hit rate."""
    engine = types.SimpleNamespace()
    stats = mm.read_kv_stats(engine)
    assert stats["available"] is False
    assert stats["source"] is None
    assert "ABSENCE" in stats["note"]
    assert all("absent" in a for a in stats["attempts"])


def test_kv_stats_present_is_reported():
    engine = types.SimpleNamespace(get_metrics=lambda: {"kv": 0.5})
    stats = mm.read_kv_stats(engine)
    assert stats["available"] is True and stats["source"] == "get_metrics"


def test_kv_probe_survives_a_raising_engine():
    def boom():
        raise RuntimeError("nope")
    stats = mm.read_kv_stats(types.SimpleNamespace(get_metrics=boom))
    assert stats["available"] is False


# ------------------------------------------------------- engine, mocked -----

def test_engine_construction_passes_quantization_and_divided_memory(monkeypatch):
    """vLLM is mocked — what matters is the kwargs the runner would send."""
    captured = {}

    class FakeArgs:
        def __init__(self, **kw):
            captured.update(kw)

    class FakeEngine:
        @staticmethod
        def from_engine_args(args):
            return "ENGINE"

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeArgs
    fake_vllm.AsyncLLMEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    engine, kind = mm.build_engine(
        "Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.225,
        quantization="awq")
    assert engine == "ENGINE" and kind == "vllm.AsyncLLMEngine"
    assert captured["model"] == "Qwen/Qwen2.5-7B-Instruct"   # singular!
    assert captured["gpu_memory_utilization"] == 0.225
    assert captured["quantization"] == "awq"


def test_no_quantization_omits_the_kwarg(monkeypatch):
    captured = {}

    class FakeArgs:
        def __init__(self, **kw):
            captured.update(kw)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeArgs
    fake_vllm.AsyncLLMEngine = types.SimpleNamespace(
        from_engine_args=lambda a: "E")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    mm.build_engine("m", gpu_memory_utilization=0.9, quantization=None)
    assert "quantization" not in captured


# ------------------------------------------------------------------- CLI ----

def test_dry_run_refuses_a_configuration_that_cannot_load():
    code = mm.main(["--dry-run", "--n-models", "7", "--quantization", "none"])
    assert code == 1


def test_dry_run_accepts_a_configuration_that_fits():
    code = mm.main(["--dry-run", "--n-models", "4", "--quantization", "awq"])
    assert code == 0


def test_process_isolation_is_declared_unimplemented_not_faked():
    code = mm.main(["--dry-run", "--n-models", "2", "--isolation", "process"])
    assert code == 2


# ------------------------------------------------------------ cost model ----

def test_hosted_baseline_flags_the_unhostable_majority():
    h = cm.hosted_baseline_monthly(calls_per_decision=3)
    assert h["not_self_hostable_share"] > 0.9
    top = h["rows"][0]
    assert top["self_hostable"] is False


def test_cards_required_refuses_without_a_measured_models_per_card():
    """models_per_card is the sweep's output; inventing it would fabricate."""
    out = cm.self_hosted_cards(n_models=7, models_per_card=None)
    assert out["available"] is False
    assert "not measured" in out["reason"]


def test_cards_required_computes_when_measured():
    out = cm.self_hosted_cards(n_models=7, models_per_card=4, card="A100-80GB")
    assert out["cards_required"] == 2
    assert out["monthly_usd"] == pytest.approx(2880.0)
    assert "engineering" in out["excludes"]


def test_every_card_price_states_a_source():
    for name, spec in cm.GPU_MONTHLY_USD.items():
        assert spec["source"] and "hr" in spec["source"]


def test_unknown_card_raises():
    with pytest.raises(KeyError):
        cm.self_hosted_cards(n_models=4, models_per_card=4, card="RTX-9090")
