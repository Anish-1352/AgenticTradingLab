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


# ---- D2: calls per decision, derived two ways ----

def _db2(tmp_path, runs, *, bars=None, decisions=None, name="t2.db"):
    """A DB shaped like the real one: `llm_model`, plus the two tables a
    decision count can come from."""
    p = tmp_path / name
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE agent_runs (run_id TEXT, llm_model TEXT,"
              " llm_calls INT, input_tokens INT, output_tokens INT,"
              " metadata TEXT)")
    c.execute("CREATE TABLE backtest_decisions (run_id TEXT, step_index INT)")
    c.execute("CREATE TABLE equity_timeseries (run_id TEXT, equity REAL)")
    for r in runs:
        c.execute("INSERT INTO agent_runs VALUES (?,?,?,?,?,?)", r)
    for run_id, n in (bars or {}).items():
        c.executemany("INSERT INTO equity_timeseries VALUES (?,?)",
                      [(run_id, 1.0)] * n)
    for run_id, n in (decisions or {}).items():
        c.executemany("INSERT INTO backtest_decisions VALUES (?,?)",
                      [(run_id, 0)] * n)
    c.commit(); c.close()
    return str(p)


def _pipeline(n_steps):
    return json.dumps({"initial_pipeline": [{"presetKey": f"s{i}"}
                                            for i in range(n_steps)]})


def test_llm_model_column_is_read():
    """Regression: only `model` was selected, so every row lost its model —
    and output tokens must be tagged with the model that produced them."""
    assert "llm_model" in open(os.path.join(
        os.path.dirname(__file__), "..", "analysis", "atl_token_extract.py")
    ).read()


def test_model_is_recovered_from_llm_model_column(tmp_path):
    db = _db2(tmp_path, [("r1", "nemotron_3_nano_30b", 10, 47900, 8600, "{}")],
              bars={"r1": 10})
    run = extract.extract_runs(db)["runs"][0]
    assert run["model"] == "nemotron_3_nano_30b"
    assert run["output_tokens_model_tag"] == "nemotron_3_nano_30b"


def test_decision_count_falls_back_to_equity_bars_and_says_so(tmp_path):
    """backtest_decisions is empty for the pipeline runtime BY CONSTRUCTION —
    engine.py only writes it under the ai_hedge_fund runtime."""
    db = _db2(tmp_path, [("r1", "m", 161, 47900, 8600, "{}")], bars={"r1": 161})
    run = extract.extract_runs(db)["runs"][0]
    dc = run["decision_count"]
    assert dc["decisions"] == 161
    assert dc["source"] == "equity_timeseries"
    assert dc["is_proxy"] is True
    assert run["observed_calls_per_decision"] == pytest.approx(1.0)


def test_backtest_decisions_wins_when_populated(tmp_path):
    db = _db2(tmp_path, [("r1", "m", 20, 47900, 8600, "{}")],
              bars={"r1": 99}, decisions={"r1": 10})
    dc = extract.extract_runs(db)["runs"][0]["decision_count"]
    assert dc["source"] == "backtest_decisions" and dc["decisions"] == 10
    assert dc["is_proxy"] is False


def test_both_derivations_agree_when_no_retries(tmp_path):
    db = _db2(tmp_path, [("r1", "m", 30, 47900, 8600, _pipeline(3))],
              bars={"r1": 10})
    rec = extract.extract_runs(db)["runs"][0]["calls_per_decision_reconciliation"]
    assert rec["agree"] is True
    assert rec["observed_llm_calls_over_decisions"] == pytest.approx(3.0)
    assert rec["configured_pipeline_steps"] == 3


def test_disagreement_above_configured_is_reported_as_retries(tmp_path):
    """36 calls over 10 decisions against 3 configured steps: 0.6 extra calls
    per decision that no step asked for."""
    db = _db2(tmp_path, [("r1", "m", 36, 47900, 8600, _pipeline(3))],
              bars={"r1": 10})
    rec = extract.extract_runs(db)["runs"][0]["calls_per_decision_reconciliation"]
    assert rec["agree"] is False
    assert "EXCEEDS" in rec["verdict"]
    assert rec["excess_calls_per_decision"] == pytest.approx(0.6)
    assert "RETRY INFLATION" in rec["interpretation"]


def test_disagreement_below_configured_is_not_called_retries(tmp_path):
    db = _db2(tmp_path, [("r1", "m", 25, 47900, 8600, _pipeline(3))],
              bars={"r1": 10})
    rec = extract.extract_runs(db)["runs"][0]["calls_per_decision_reconciliation"]
    assert rec["agree"] is False and "BELOW" in rec["verdict"]
    assert rec["missing_calls_per_decision"] == pytest.approx(0.5)
    assert "RETRY" not in rec["interpretation"].upper()


def test_single_call_path_reports_configured_count_as_unavailable(tmp_path):
    """No initial_pipeline is the single-call path — the multi-step path stays
    unmeasured, and that must be said rather than defaulted to 1."""
    db = _db2(tmp_path, [("r1", "m", 161, 47900, 8600, "{}")], bars={"r1": 161})
    rec = extract.extract_runs(db)["runs"][0]["calls_per_decision_reconciliation"]
    assert rec["agree"] is None
    assert rec["configured_pipeline_steps"] is None
    assert "SINGLE-CALL" in rec["interpretation"]


def test_disagreement_is_counted_in_the_summary(tmp_path):
    db = _db2(tmp_path, [("r1", "m", 36, 47900, 8600, _pipeline(3)),
                         ("r2", "m", 30, 47900, 8600, _pipeline(3))],
              bars={"r1": 10, "r2": 10})
    der = extract.summarise(extract.extract_runs(db))["calls_per_decision_derivations"]
    assert der["runs_reconciled"] == 2 and der["runs_in_disagreement"] == 1
    assert der["decision_denominator_sources_used"] == ["equity_timeseries"]


# ---- D2: output tokens are model-tagged and not substitutable ----

def test_output_tokens_are_keyed_by_model(tmp_path):
    db = _db2(tmp_path, [("r1", "nemotron", 10, 47900, 8600, "{}"),
                         ("r2", "gemini", 10, 47900, 50000, "{}")],
              bars={"r1": 10, "r2": 10})
    bym = extract.summarise(extract.extract_runs(db))["output_tokens_by_model"]
    assert bym["models"]["nemotron"]["mean_output_tokens_per_call"] == pytest.approx(860)
    assert bym["models"]["gemini"]["mean_output_tokens_per_call"] == pytest.approx(5000)
    assert bym["spread_factor"] == pytest.approx(5000 / 860)
    assert all(e["substitutable_across_models"] is False
               for e in bym["models"].values())


def test_output_lookup_refuses_an_unmeasured_model(tmp_path):
    db = _db2(tmp_path, [("r1", "nemotron", 10, 47900, 8600, "{}")],
              bars={"r1": 10})
    bym = extract.summarise(extract.extract_runs(db))["output_tokens_by_model"]
    assert extract.output_tokens_for_model(bym, "nemotron") == pytest.approx(860)
    with pytest.raises(extract.CrossModelSubstitution):
        extract.output_tokens_for_model(bym, "gemini")


def test_cross_model_output_average_is_flagged_as_unusable(tmp_path):
    db = _db2(tmp_path, [("r1", "a", 10, 47900, 8600, "{}"),
                         ("r2", "b", 10, 47900, 50000, "{}")],
              bars={"r1": 10, "r2": 10})
    s = extract.summarise(extract.extract_runs(db))
    assert "model-specific" in s["output_tokens_per_call_warning"]


# ---- D2: three-way comparison ----

def test_three_way_keeps_the_original_assumption_intact(tmp_path):
    db = _db2(tmp_path, [("r1", "m", 161, 771190, 138460, "{}")],
              bars={"r1": 161})
    cmp_ = extract.three_way_comparison(extract.summarise(extract.extract_runs(db)))
    cols = cmp_["columns"]
    # The assumption is not overwritten by the measurement that corrected it.
    assert cols["original_assumption"]["input_tokens_per_call"] == 2620.0
    assert cols["original_assumption"]["output_tokens_per_call"] == 256.0
    assert cols["seed_db"]["input_tokens_per_call"] == pytest.approx(4790, rel=1e-3)
    assert cmp_["deltas"]["seed_vs_assumption_input_tokens_per_call"] > 1.8


def test_three_way_reports_a_missing_local_run_rather_than_inventing_one():
    cmp_ = extract.three_way_comparison(None, None)
    local = cmp_["columns"]["local_pipeline_run"]
    assert local["available"] is False
    assert "not supplied" in local["reason"]
    assert extract.format_comparison(cmp_).count("NOT AVAILABLE") >= 1


def test_local_db_column_is_separate_from_seed(tmp_path):
    seed = _db2(tmp_path, [("s1", "m", 161, 771190, 138460, "{}")],
                bars={"s1": 161}, name="seed.db")
    local = _db2(tmp_path, [("l1", "m", 30, 143700, 25800, _pipeline(3))],
                 bars={"l1": 10}, name="local.db")
    cmp_ = extract.three_way_comparison(
        extract.summarise(extract.extract_runs(seed)),
        extract.summarise(extract.extract_runs(local)))
    assert cmp_["columns"]["seed_db"]["calls_per_decision_observed"] == pytest.approx(1.0)
    assert cmp_["columns"]["local_pipeline_run"]["calls_per_decision_observed"] == pytest.approx(3.0)
    assert cmp_["deltas"]["local_vs_seed_calls_per_decision"] == pytest.approx(3.0)


def test_local_db_pointing_at_seed_is_rejected():
    seed = os.path.join(os.path.dirname(__file__), "..", "..",
                        "dashboard", "storage", "data", "backtest.db")
    assert extract.main(["--db", seed, "--local-db", seed]) == 2


# ---- D3: per-model cost table ----

def test_per_model_uses_each_models_own_measured_output():
    t = model.per_model_cost_table(calls_per_decision=1, calls_are_measured=False)
    rows = {r["model"]: r for r in t["rows"]}
    assert rows["nvidia/nemotron-3-nano-30b-a3b"]["output_tokens_per_call"] == pytest.approx(860.2)
    assert rows["google/gemini-3.1-pro"]["output_tokens_per_call"] == pytest.approx(5004.5)
    assert all("measured" in r["output_tokens_source"] for r in t["rows"])


def test_per_model_spread_exceeds_price_spread_alone():
    """Price and verbosity multiply — the whole point of the table."""
    t = model.per_model_cost_table(calls_per_decision=1, calls_are_measured=False)
    assert t["cost_span_factor"] > t["output_price_span_factor"]
    assert t["verbosity_span_factor"] == pytest.approx(5004.5 / 860.2)


def test_shared_output_length_understates_the_spread():
    t = model.per_model_cost_table(calls_per_decision=1, calls_are_measured=False,
                                   shared_output_tokens=860.2)
    rows = {r["model"]: r for r in t["rows"]}
    # Nemotron IS the shared figure, so it is unchanged; verbose models are not.
    assert rows["nvidia/nemotron-3-nano-30b-a3b"]["own_over_shared_factor"] == pytest.approx(1.0)
    assert rows["google/gemini-3.1-pro"]["own_over_shared_factor"] > 3.0


def test_shared_output_length_errs_in_both_directions():
    """A shared length OVERstates the terse models and UNDERstates the verbose
    ones at the same time — so the factor is not named for one direction."""
    t = model.per_model_cost_table(calls_per_decision=1, calls_are_measured=False,
                                   shared_output_tokens=2652.0)
    rows = {r["model"]: r for r in t["rows"]}
    assert rows["nvidia/nemotron-3-nano-30b-a3b"]["own_over_shared_factor"] < 1.0
    assert rows["google/gemini-3.1-pro"]["own_over_shared_factor"] > 1.0
    # And the shared-length view reports a narrower spread than the truth.
    own = [r["cost_per_decision"] for r in t["rows"]]
    sh = [r["cost_per_decision_shared_output"] for r in t["rows"]]
    assert (max(own) / min(own)) > (max(sh) / min(sh))


def test_production_default_is_cheapest_on_both_axes():
    t = model.per_model_cost_table(calls_per_decision=1, calls_are_measured=False)
    assert t["production_default"]["rank"] == 1
    rows = {r["model"]: r for r in t["rows"]}
    nem = rows[model.PRODUCTION_DEFAULT_MODEL]
    assert nem["price_out_per_m"] == min(r["price_out_per_m"] for r in t["rows"])
    assert nem["output_tokens_per_call"] == min(
        r["output_tokens_per_call"] for r in t["rows"])


def test_illustrative_calls_per_decision_is_labelled_as_such():
    t = model.per_model_cost_table(
        calls_per_decision=model.ILLUSTRATIVE_CALLS_PER_DECISION,
        calls_are_measured=False)
    assert "ILLUSTRATIVE" in t["calls_per_decision_basis"]
    m = model.per_model_cost_table(calls_per_decision=1.0, calls_are_measured=True)
    assert m["calls_per_decision_basis"] == "MEASURED"


def test_measured_flag_is_required_for_the_measured_label():
    """An assumed call count must never print as a measured one."""
    argv = ["--api-cost-per-request", "0.0002", "--calls-per-decision", "3"]
    import io as _io, contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert model.main(argv) == 0
    assert "ILLUSTRATIVE — not measured" in buf.getvalue()
    buf2 = _io.StringIO()
    with contextlib.redirect_stdout(buf2):
        assert model.main(argv + ["--calls-per-decision-measured"]) == 0
    assert "[MEASURED]" in buf2.getvalue()


def test_per_model_prices_match_the_backend_pricing_table():
    """The DB's underscored model names do not substring-match token_cost.py's
    slugs, so the mapping is asserted here rather than inferred at runtime."""
    for slug, m in model.MEASURED_OUTPUT_TOKENS.items():
        assert slug in model.MODEL_PRICES, slug
        assert m["db_model"] and m["run_id"].startswith("lb_")


_SEED_DB = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "dashboard", "storage", "data", "backtest.db"))


@pytest.mark.skipif(not os.path.exists(_SEED_DB), reason="seed DB not present")
def test_measured_output_lengths_match_the_seed_db():
    """The table is a transcription of real rows; drift must fail loudly."""
    conn = sqlite3.connect(f"file:{_SEED_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = {r["llm_model"]: r for r in conn.execute(
        "SELECT llm_model, llm_calls, input_tokens, output_tokens, est_cost_usd"
        " FROM agent_runs WHERE llm_calls > 0")}
    conn.close()
    assert len(rows) == 7
    for slug, m in model.MEASURED_OUTPUT_TOKENS.items():
        row = rows[m["db_model"]]
        calls = row["llm_calls"]
        assert m["output"] == pytest.approx(row["output_tokens"] / calls, rel=1e-3)
        assert m["input"] == pytest.approx(row["input_tokens"] / calls, rel=1e-3)
        # And the price pairing: recomputing the run's cost from its own token
        # totals must reproduce the cost the run stored at the time.
        pin, pout = model.MODEL_PRICES[slug]
        recomputed = (row["input_tokens"] / 1e6) * pin + (
            row["output_tokens"] / 1e6) * pout
        assert recomputed == pytest.approx(row["est_cost_usd"], abs=1e-5), slug
