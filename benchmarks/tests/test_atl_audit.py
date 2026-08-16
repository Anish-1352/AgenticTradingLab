"""Tests for the ATL pipeline audit, token extract, fixture and cost levers.

Synthetic code samples and synthetic token records — nothing reads the real
backend or a real database.
"""
import json, os, sqlite3, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis import arm_a_vs_bc_model as model  # noqa: E402
from analysis import atl_pipeline_audit as audit  # noqa: E402
from analysis import atl_token_extract as extract  # noqa: E402
from common.fixtures import StubTokenizer, build_fixture  # noqa: E402

LOOP_SRC = '''
def run_pipeline(client, steps):
    for i, step in enumerate(steps):
        r = client.messages.create(model="m", max_tokens=4096, system="S",
                                   messages=[{"role": "user", "content": "x"}])
'''
FIXED_SRC = '''
def one_shot(client):
    return client.messages.create(model="m", max_tokens=900, messages=[])
'''


def test_finds_call_in_loop_and_marks_count_variable():
    calls = audit.find_llm_calls(LOOP_SRC, "f.py")
    assert len(calls) == 1
    assert calls[0]["in_loop"] is True
    assert calls[0]["function"] == "run_pipeline"
    assert calls[0]["max_tokens"] == "4096"
    assert not audit.summarise_calls(calls)["calls_per_decision_is_fixed"]


def test_fixed_call_is_not_flagged_as_variable():
    calls = audit.find_llm_calls(FIXED_SRC, "f.py")
    assert calls[0]["in_loop"] is False
    assert audit.summarise_calls(calls)["calls_per_decision_is_fixed"]


def test_retry_surface_requires_loop_and_try():
    src = '''
def retrying(client):
    for attempt in range(3):
        try:
            return client.messages.create(model="m", messages=[])
        except Exception:
            pass
'''
    s = audit.summarise_calls(audit.find_llm_calls(src, "f.py"))
    assert len(s["retry_surfaces"]) == 1


def test_prompt_segments_separate_static_from_dynamic():
    src = '''
import json
def build(step, snapshot):
    parts = ["=== HEADER ===", (step.get("prompt") or "").strip()]
    if snapshot:
        parts.append(json.dumps(snapshot))
    return "\\n".join(parts)
'''
    c = audit.classify_prompt_segments(src, "build")
    assert c["available"]
    assert c["static_segment_count"] >= 1
    exprs = [d["source_expression"] for d in c["dynamic_segments"]]
    assert any("snapshot" in e for e in exprs)
    # The snapshot is inside an `if`, so it must be flagged conditional.
    assert any(d["conditional"] for d in c["dynamic_segments"] if "snapshot" in d["source_expression"])


def test_missing_function_is_reported_not_guessed():
    c = audit.classify_prompt_segments("x = 1", "nope")
    assert c["available"] is False and "not found" in c["reason"]


def test_syntax_error_is_reported():
    assert "parse_error" in audit.find_llm_calls("def (:", "bad.py")[0]


# ---- token extract ----

def _db(tmp_path, runs):
    p = tmp_path / "t.db"
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE agent_runs (run_id TEXT, model TEXT, llm_calls INT,"
              " input_tokens INT, output_tokens INT, metadata TEXT)")
    c.execute("CREATE TABLE backtest_decisions (run_id TEXT, step_index INT)")
    for r in runs:
        c.execute("INSERT INTO agent_runs VALUES (?,?,?,?,?,?)", r)
    c.commit(); c.close()
    return str(p)


def test_extract_computes_means_only(tmp_path):
    meta = json.dumps({"initial_pipeline": [
        {"presetKey": "a"}, {"presetKey": "b"},
        {"presetKey": "post_trade_analysis"}], "llm_max_output_tokens": 4096})
    db = _db(tmp_path, [("r1", "m", 10, 47900, 26520, meta)])
    e = extract.extract_runs(db)
    assert e["available"] and e["has_per_call_table"] is False
    run = e["runs"][0]
    assert run["mean_input_tokens_per_call"] == pytest.approx(4790.0)
    assert run["pipeline"]["calls_per_decision_from_pipeline"] == 2
    assert run["pipeline"]["post_trade_calls_per_day"] == 1


def test_per_call_distribution_is_reported_unavailable(tmp_path):
    """The headline finding — it must never be silently synthesised."""
    db = _db(tmp_path, [("r1", "m", 10, 47900, 26520, "{}")])
    s = extract.summarise(extract.extract_runs(db))
    d = s["per_call_distribution"]
    assert d["available"] is False
    assert "summation" in d["reason"]
    assert "llm_call_usage" in d["instrumentation_gap"]["minimal_fix"]


def test_summary_reports_run_level_means_not_percentiles(tmp_path):
    db = _db(tmp_path, [("r1", "m", 10, 47900, 2560, "{}"),
                        ("r2", "m", 20, 60000, 5120, "{}")])
    s = extract.summarise(extract.extract_runs(db))
    agg = s["input_tokens_per_call"]
    assert agg["n_runs"] == 2
    assert "not over calls" in agg["note"]
    assert "p95" not in agg


def test_rule_based_runs_are_reported_as_no_data(tmp_path):
    db = _db(tmp_path, [("r1", "rule", 0, 0, 0, "{}")])
    s = extract.summarise(extract.extract_runs(db))
    assert s["available"] is False and "llm_calls > 0" in s["reason"]


def test_missing_db_is_an_error_not_a_guess():
    with pytest.raises(FileNotFoundError):
        extract.extract_runs("/nonexistent.db")


# ---- fixture ----

def test_atl_realistic_sits_between_the_bounds():
    fx = build_fixture("atl_realistic", StubTokenizer(), 8, context_tokens=4790)
    m = fx.to_meta()
    assert m["token_counts_all_equal"]
    # Between low_overlap (0.3%) and shared_prefix (99.4%).
    assert 0.03 < m["common_prefix_fraction"] < 0.95


def test_atl_realistic_prefix_matches_its_static_segment():
    fx = build_fixture("atl_realistic", StubTokenizer(), 6, context_tokens=1000,
                       seed=7)
    m = fx.to_meta()
    assert m["common_prefix_tokens"] >= m["expected_cross_agent_prefix_tokens"]


def test_atl_realistic_warns_when_not_derived_from_measurement():
    fx = build_fixture("atl_realistic", StubTokenizer(), 4, context_tokens=500)
    assert "warning" in fx.to_meta()["derived_from"]


def test_atl_realistic_records_provenance_when_given():
    from common.fixtures import build_atl_realistic
    fx = build_atl_realistic(StubTokenizer(), 4, context_tokens=500,
                             derived_from={"audit": "atl_audit.json"})
    assert fx.to_meta()["derived_from"]["audit"] == "atl_audit.json"


def test_atl_realistic_rejects_impossible_split():
    from common.fixtures import build_atl_realistic
    with pytest.raises(ValueError, match="must be"):
        build_atl_realistic(StubTokenizer(), 2, context_tokens=100,
                            static_tokens=80, per_agent_tokens=40)


def test_existing_bounds_are_untouched():
    for name, lo, hi in (("shared_prefix", 0.9, 1.0), ("low_overlap", 0.0, 0.05)):
        m = build_fixture(name, StubTokenizer(), 6, context_tokens=1000).to_meta()
        assert lo <= m["common_prefix_fraction"] <= hi


# ---- cost model ----

def test_calls_per_decision_scales_cost_linearly():
    one = model.cost_per_decision(1, 4790, 2652, 0.05, 0.20)
    three = model.cost_per_decision(3, 4790, 2652, 0.05, 0.20)
    assert three == pytest.approx(3 * one)


def test_prefix_cache_discounts_input_only():
    """Output tokens are generated fresh every time; a cache cannot help them."""
    no_cache = model.cost_per_decision(1, 10000, 0, 0.05, 0.20)
    full_cache = model.cost_per_decision(1, 10000, 0, 0.05, 0.20,
                                         prefix_cache_hit_rate=1.0)
    assert full_cache == pytest.approx(no_cache * 0.1)
    out_only = model.cost_per_decision(1, 0, 1000, 0.05, 0.20,
                                       prefix_cache_hit_rate=1.0)
    assert out_only == pytest.approx(model.cost_per_decision(1, 0, 1000, 0.05, 0.20))


def test_model_choice_dominates_the_lever_table():
    lv = model.lever_sensitivity(baseline_calls=3, baseline_input=4790,
                                 baseline_output=2652, price_in=0.05,
                                 price_out=0.20)
    assert lv["levers"][0]["lever"] == "model choice"
    assert lv["levers"][0]["span_factor"] > 50
    # And it must outrank the cache by a wide margin.
    cache = [l for l in lv["levers"] if l["lever"] == "prefix cache hit rate"][0]
    assert lv["levers"][0]["span_factor"] > cache["span_factor"] * 10


def test_measured_inputs_move_the_crossover_down():
    """More calls and bigger outputs make the API pricier, so self-hosting
    wins sooner."""
    assumed = model.find_crossover(api_cost_per_request=0.0001822,
                                   decisions_per_agent_per_day=390,
                                   gpu_usd_per_hour=2.0, rps=9.28)
    measured_cost = model.cost_per_decision(3, 4790, 2652, 0.05, 0.20)
    measured = model.find_crossover(api_cost_per_request=measured_cost,
                                    decisions_per_agent_per_day=390,
                                    gpu_usd_per_hour=2.0, rps=9.28)
    assert measured["crossover_agents"] < assumed["crossover_agents"]
