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


# =========================================================================
# the execution path
#
# The runner used to stop before constructing anything, so none of this was
# ever exercised. vLLM is mocked: what these assert is the SHAPE of the run —
# that N engines are built, that load is split rather than multiplied, that a
# timestamp is recorded per output token, and that the aggregate divides by the
# run's wall clock rather than by a sum of overlapping per-model spans.
#
# A mock cannot tell us whether two engines coexist on a real card. It can tell
# us the runner asks for the right thing, which is the part that was untested.
# =========================================================================

import asyncio                                          # noqa: E402
import time                                             # noqa: E402


class _FakeCompletion:
    def __init__(self, n):
        self.token_ids = list(range(n))


class _FakeOut:
    def __init__(self, n):
        self.outputs = [_FakeCompletion(n)]


class FakeEngineRuntime:
    """Streams `n_tokens` outputs, one at a time, recording peak concurrency."""

    def __init__(self, n_tokens=4, delay=0.001):
        self.n_tokens = n_tokens
        self.delay = delay
        self.in_flight = 0
        self.peak_in_flight = 0
        self.seen_request_ids = []
        self.shutdown_called = False

    async def generate(self, prompt, sampling_params, request_id):
        self.seen_request_ids.append(request_id)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            for i in range(1, self.n_tokens + 1):
                await asyncio.sleep(self.delay)
                yield _FakeOut(i)
        finally:
            self.in_flight -= 1

    def shutdown(self):
        self.shutdown_called = True


def _cfg(model_id="m/one", concurrency=2):
    return mm.EngineConfig(model_id=model_id, concurrency=concurrency,
                           gpu_memory_utilization=0.45, quantization=None)


def test_one_request_records_a_timestamp_per_token():
    eng = FakeEngineRuntime(n_tokens=5)
    sem = asyncio.Semaphore(1)
    rec = asyncio.run(mm.run_one(eng, [1, 2, 3], "r0", None, sem))
    assert rec.ok
    assert rec.output_tokens == 5
    assert len(rec.per_token_timestamps) == 5
    assert len(rec.itl) == 4          # N timestamps -> N-1 gaps
    assert rec.ttft is not None and rec.ttft > 0
    assert rec.e2e >= rec.ttft


def test_ttft_clock_starts_after_the_semaphore():
    """Same convention as arm C: client-side queue wait is excluded.

    A request that waits behind a full semaphore must not have that wait
    charged to TTFT, or the arms stop being comparable.
    """
    eng = FakeEngineRuntime(n_tokens=2, delay=0.01)
    sem = asyncio.Semaphore(1)

    async def go():
        return await asyncio.gather(*[
            mm.run_one(eng, [1], f"r{i}", None, sem) for i in range(3)])

    recs = asyncio.run(go())
    # The last to run waited ~2 slots, but its TTFT is one token's delay.
    assert all(r.ok for r in recs)
    assert max(r.ttft for r in recs) < 0.05


def test_a_failing_request_is_recorded_not_raised():
    class Boom:
        async def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")
            yield  # pragma: no cover

    rec = asyncio.run(mm.run_one(Boom(), [1], "r", None, asyncio.Semaphore(1)))
    assert rec.ok is False
    assert rec.error == "CUDA_OOM"


def test_per_model_concurrency_caps_in_flight_requests():
    """The semaphore IS the load split. If it does not bind, adding models
    raises offered load and the throughput comparison measures nothing."""
    eng = FakeEngineRuntime(n_tokens=2, delay=0.005)
    out = asyncio.run(mm.run_model(eng, _cfg(concurrency=3),
                                   [[1]] * 12, None))
    assert len(out["records"]) == 12
    assert eng.peak_in_flight <= 3


def test_all_models_run_concurrently_not_in_sequence():
    """Sequential execution would measure N independent single-model runs."""
    engines = [(FakeEngineRuntime(n_tokens=3, delay=0.01), _cfg(f"m/{i}", 4))
               for i in range(3)]
    t0 = time.perf_counter()
    run = asyncio.run(mm.run_all(engines, [[1]] * 4, max_new_tokens=3, seed=1, sampling_params=object()))
    wall = time.perf_counter() - t0
    per_model_sum = sum(e["wall_seconds"] for e in run["per_model"])
    # Overlapped: the whole run is far shorter than the models added up.
    assert wall < per_model_sum
    assert len(run["per_model"]) == 3


def test_request_ids_are_namespaced_by_model():
    """Three models generating id 'r0' would collide in the record table."""
    engines = [(FakeEngineRuntime(n_tokens=1), _cfg(f"org/model{i}", 2))
               for i in range(2)]
    run = asyncio.run(mm.run_all(engines, [[1]] * 2, max_new_tokens=1, seed=1, sampling_params=object()))
    ids = [r.request_id for e in run["per_model"] for r in e["records"]]
    assert len(set(ids)) == len(ids)
    assert any(i.startswith("model0-") for i in ids)


def test_aggregate_emits_per_model_and_a_total():
    engines = [(FakeEngineRuntime(n_tokens=2), _cfg(f"org/m{i}", 2))
               for i in range(3)]
    run = asyncio.run(mm.run_all(engines, [[1]] * 4, max_new_tokens=2, seed=1, sampling_params=object()))
    agg = mm.aggregate(run["per_model"], run["wall_seconds"], "RID")
    assert len(agg["per_model"]) == 3
    assert len(agg["all_records"]) == 12
    assert agg["aggregate_rates"]["completed"] == 12
    assert agg["aggregate_rates"]["total_output_tokens"] == 24


def test_aggregate_rate_uses_wall_clock_not_summed_spans():
    """Per-model spans OVERLAP. Summing them would divide by ~N times too
    much and understate card throughput by that factor."""
    engines = [(FakeEngineRuntime(n_tokens=2, delay=0.01), _cfg(f"o/m{i}", 2))
               for i in range(3)]
    run = asyncio.run(mm.run_all(engines, [[1]] * 4, max_new_tokens=2, seed=1, sampling_params=object()))
    agg = mm.aggregate(run["per_model"], run["wall_seconds"], "RID")
    summed = sum(e["wall_seconds"] for e in run["per_model"])
    rate = agg["aggregate_rates"]["completed_requests_per_s"]
    assert rate == pytest.approx(12 / run["wall_seconds"])
    assert rate > 12 / summed


def test_aggregate_concurrency_is_the_total_offered_load():
    engines = [(FakeEngineRuntime(n_tokens=1), _cfg(f"o/m{i}", 8))
               for i in range(4)]
    run = asyncio.run(mm.run_all(engines, [[1]] * 2, max_new_tokens=1, seed=1, sampling_params=object()))
    agg = mm.aggregate(run["per_model"], run["wall_seconds"], "RID")
    assert agg["aggregate"].concurrency == 32


def test_failed_requests_are_excluded_from_the_rate_not_counted_as_fast():
    class Half:
        def __init__(self):
            self.n = 0

        async def generate(self, prompt, sp, rid):
            self.n += 1
            if self.n % 2 == 0:
                raise RuntimeError("boom")
            yield _FakeOut(1)

    run = asyncio.run(mm.run_all([(Half(), _cfg("o/m", 1))], [[1]] * 4,
                                 max_new_tokens=1, seed=1, sampling_params=object()))
    agg = mm.aggregate(run["per_model"], run["wall_seconds"], "RID")
    assert agg["aggregate_rates"]["failed"] == 2
    assert agg["aggregate_rates"]["completed"] == 2


def test_shutdown_is_attempted_and_survives_an_engine_without_one():
    eng = FakeEngineRuntime()
    mm.shutdown_engine(eng)
    assert eng.shutdown_called
    mm.shutdown_engine(object())        # must not raise


def test_short_name_is_the_bare_model_name():
    assert _cfg("Qwen/Qwen2.5-7B-Instruct").short_name == "Qwen2.5-7B-Instruct"


# ------------------------------------------------- the capability probe -----

def test_probe_reports_absence_as_absence_with_the_check_it_ran():
    """The predecessor printed 'no GPU, no vllm' unconditionally — on a
    healthy A100 too. Every field here must come from an executed check."""
    caps = mm.probe_capabilities()
    for key in ("torch", "vllm", "engine_class"):
        assert "available" in caps[key]
        assert caps[key]["available"] or caps[key].get("error")
    assert caps["can_run"] is (
        caps["vllm"]["available"] and caps["engine_class"]["available"]
        and caps["torch"]["cuda_available"])


def test_probe_says_can_run_when_torch_cuda_and_vllm_are_all_present(monkeypatch):
    """The failing case that motivated this: a working environment must not
    be told its hardware is missing."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.13.0+cu130"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda i: "NVIDIA A100-SXM4-80GB")
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.27.1"

    mod_v1 = types.ModuleType("vllm.v1.engine.async_llm")
    mod_v1.AsyncLLM = type("AsyncLLM", (), {
        "from_engine_args": staticmethod(lambda *a, **k: None),
        "generate": lambda self: None})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", mod_v1)

    caps = mm.probe_capabilities()
    assert caps["can_run"] is True
    assert caps["torch"]["cuda_available"] is True
    assert caps["vllm"]["version"] == "0.27.1"
    assert caps["engine_class"]["has_from_engine_args"] is True
    text = mm.format_capabilities(caps)
    assert "A100" in text and "0.27.1" in text
    assert "MISSING" not in text


def test_probe_distinguishes_no_vllm_from_no_cuda(monkeypatch):
    """Different failures need different fixes; one message for both cost a
    debugging session."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.13.0"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0,
        get_device_name=lambda i: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    caps = mm.probe_capabilities()
    assert caps["torch"]["available"] is True
    assert caps["torch"]["cuda_available"] is False
    assert caps["can_run"] is False
    text = mm.format_capabilities(caps)
    assert "torch.cuda.is_available() is False" in text


def test_engine_class_absence_is_marked_not_probed_when_vllm_is_missing(monkeypatch):
    """It must not claim a probe it could not perform."""
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(mm.sys, "path", mm.sys.path)
    caps = mm.probe_capabilities()
    if not caps["vllm"]["available"]:
        assert caps["engine_class"]["probe"] is None
        assert "not probed" in caps["engine_class"]["error"]


def test_runner_refuses_without_capability_and_says_which_check_failed(capsys):
    """No GPU here, so this is the live path: exit 3, and the reason names the
    probe rather than asserting a general absence."""
    code = mm.main(["--n-models", "2"])
    err = capsys.readouterr().err
    assert code == 3
    assert "Cannot run here:" in err
    assert "vllm does not import" in err or "torch" in err


def test_engine_construction_prefers_the_v1_class(monkeypatch):
    """vLLM 0.27.1 removed the V0 engine: AsyncLLMEngine is an alias of
    AsyncLLM. The v1 path is tried first so the run records the real class."""
    captured = {}

    class FakeArgs:
        def __init__(self, **kw):
            captured.update(kw)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeArgs
    mod_v1 = types.ModuleType("vllm.v1.engine.async_llm")
    mod_v1.AsyncLLM = types.SimpleNamespace(
        from_engine_args=lambda a: "V1ENGINE")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", mod_v1)

    engine, kind = mm.build_engine("m", gpu_memory_utilization=0.45,
                                   quantization=None)
    assert engine == "V1ENGINE"
    assert kind == "vllm.v1.engine.async_llm.AsyncLLM"


def test_max_model_len_and_max_num_seqs_are_only_sent_when_set(monkeypatch):
    """Both are `| None` in 0.27.1's AsyncEngineArgs, so None is the default
    rather than a value — passing it explicitly would be a no-op at best."""
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
    assert "max_model_len" not in captured and "max_num_seqs" not in captured

    captured.clear()
    mm.build_engine("m", gpu_memory_utilization=0.9, quantization=None,
                    max_model_len=4096, max_num_seqs=64)
    assert captured["max_model_len"] == 4096 and captured["max_num_seqs"] == 64


# ------------------------------------------------- main(), end to end -------

@pytest.fixture
def fake_vllm_stack(monkeypatch):
    """A complete enough vLLM to drive main() to completion.

    This is the test that would have caught the manifest and results wiring
    being wrong — none of it had ever executed. It cannot say whether two real
    engines fit on a real card.
    """
    built = []

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.13.0+cu130"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda i: "NVIDIA A100-SXM4-80GB")

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.27.1"

    class FakeArgs:
        def __init__(self, **kw):
            self.kw = kw

    class FakeSampling:
        def __init__(self, **kw):
            self.kw = kw

    fake_vllm.AsyncEngineArgs = FakeArgs
    fake_vllm.SamplingParams = FakeSampling

    def from_engine_args(args):
        eng = FakeEngineRuntime(n_tokens=3, delay=0.0)
        built.append((eng, args.kw))
        return eng

    mod_v1 = types.ModuleType("vllm.v1.engine.async_llm")
    mod_v1.AsyncLLM = types.SimpleNamespace(from_engine_args=from_engine_args)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", mod_v1)
    return built


def test_main_runs_end_to_end_and_writes_results(fake_vllm_stack, tmp_path, capsys):
    code = mm.main(["--n-models", "2", "--n-requests", "4",
                    "--max-new-tokens", "3", "--total-concurrency", "4",
                    "--out-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Per model:" in out
    assert "Aggregate (all models, one card):" in out
    assert "stats_source:" in out
    assert list(tmp_path.iterdir()), "no results written"


def test_main_builds_one_engine_per_model_at_the_divided_fraction(fake_vllm_stack,
                                                                  tmp_path):
    """The division is the whole point: the full 0.9 to each engine would OOM
    every engine after the first."""
    mm.main(["--n-models", "3", "--n-requests", "2", "--max-new-tokens", "2",
             "--total-concurrency", "6", "--out-dir", str(tmp_path)])
    assert len(fake_vllm_stack) == 3
    fracs = [kw["gpu_memory_utilization"] for _e, kw in fake_vllm_stack]
    assert all(f == pytest.approx(0.9 / 3) for f in fracs)
    assert sum(fracs) == pytest.approx(0.9)
    models = [kw["model"] for _e, kw in fake_vllm_stack]
    assert len(set(models)) == 3        # `model` is singular in 0.27.1


def test_main_splits_the_offered_load_across_engines(fake_vllm_stack, tmp_path):
    """Total concurrency is held fixed as N rises — that is the experiment."""
    mm.main(["--n-models", "4", "--n-requests", "8", "--max-new-tokens", "2",
             "--total-concurrency", "8", "--out-dir", str(tmp_path)])
    engines = [e for e, _kw in fake_vllm_stack]
    assert len(engines) == 4
    assert sum(e.peak_in_flight for e in engines) <= 8
    for e in engines:
        assert e.peak_in_flight <= 2    # 8 total / 4 models


def test_main_shuts_every_engine_down(fake_vllm_stack, tmp_path):
    mm.main(["--n-models", "2", "--n-requests", "2", "--max-new-tokens", "2",
             "--out-dir", str(tmp_path)])
    assert all(e.shutdown_called for e, _kw in fake_vllm_stack)


def test_main_shuts_engines_down_even_when_the_run_fails(fake_vllm_stack,
                                                         tmp_path, monkeypatch):
    """A leaked engine holds the card and the next run OOMs for the wrong
    reason."""
    async def boom(*a, **k):
        raise RuntimeError("scheduler died")

    monkeypatch.setattr(mm, "run_all", boom)
    with pytest.raises(RuntimeError):
        mm.main(["--n-models", "2", "--n-requests", "2", "--out-dir", str(tmp_path)])
    assert fake_vllm_stack and all(e.shutdown_called for e, _kw in fake_vllm_stack)


def test_manifest_records_stats_source_and_the_engine_api(fake_vllm_stack,
                                                          tmp_path):
    """stats_source was null on 0.26 and may resolve on 0.27.1. Either way it
    is recorded, so absence stays distinguishable from a reported zero."""
    import json as _json
    mm.main(["--n-models", "2", "--n-requests", "2", "--max-new-tokens", "2",
             "--out-dir", str(tmp_path)])
    blobs = [_json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    found = [b for b in blobs if "capabilities" in b or "manifest" in b]
    assert found, "no manifest-bearing results file"
    blob = found[0]
    assert "stats_source" in _json.dumps(blob)
    assert "vllm.v1.engine.async_llm.AsyncLLM" in _json.dumps(blob)


def test_manifest_leaves_model_unset_because_there_are_n_of_them(fake_vllm_stack,
                                                                 tmp_path):
    """Naming one of N would misattribute the run; the list is in the config."""
    import json as _json
    mm.main(["--n-models", "3", "--n-requests", "2", "--max-new-tokens", "2",
             "--out-dir", str(tmp_path)])
    blobs = [_json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    man = None
    for b in blobs:
        man = b.get("manifest") or (b if "arm" in b else None)
        if man:
            break
    assert man is not None
    assert man.get("model") in (None, "")
    assert man["arm"] == mm.ARM


def test_manifest_file_is_actually_written_where_the_runner_says(fake_vllm_stack,
                                                                 tmp_path, capsys):
    """The runner prints a manifest path; a path that names nothing is worse
    than no path at all."""
    mm.main(["--n-models", "2", "--n-requests", "2", "--max-new-tokens", "2",
             "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.strip().startswith("manifest:")]
    assert line, "no manifest path printed"
    path = line[0].split("manifest:", 1)[1].strip()
    assert os.path.exists(path), f"printed {path} but nothing is there"


def test_fixture_size_mismatch_is_stated_not_silently_reconciled(fake_vllm_stack,
                                                                 tmp_path, capsys):
    """--n-requests does not resize a stored fixture. Slicing it would break
    the sha256 provenance, so the count actually sent is announced."""
    mm.main(["--n-models", "2", "--n-requests", "4", "--max-new-tokens", "2",
             "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "does not resize it" in out
