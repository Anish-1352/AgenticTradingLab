"""The reliability harness's classifier, and the harness loop against a fake.

``classify_response`` is the measurement instrument. If it miscounts, every
routing decision downstream is made on a wrong number, so the categories are
pinned individually — including the awkward ones where the production parser's
tolerance means a visibly bad response still passes.
"""

import importlib.util
import json
import os
import sys

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "scripts")
_HARNESS = os.path.join(_SCRIPTS, "measure_step_reliability.py")
# The script imports ``_bootstrap`` as a sibling, which works when it is run
# directly (Python puts the script's own directory on sys.path) but not when it
# is loaded from here. Doing this in the SCRIPT would trip the architecture
# guard that allows sys.path mutation only in _bootstrap itself.
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_spec = importlib.util.spec_from_file_location("_msr", _HARNESS)
msr = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["_msr"] = msr
_spec.loader.exec_module(msr)

from dashboard.backend.infrastructure.llm.backtest_harness import (  # noqa: E402
    parse_llm_response,
)


def _classify(text, *, is_last=False):
    parsed = parse_llm_response(text) if text.strip() else None
    return msr.classify_response(text, parsed, is_last=is_last)


# ------------------------------------------------------------- categories ---

def test_clean_json_passes_clean():
    ok, cat, _ = _classify('{"facts": [], "confidence": 0.8}')
    assert ok is True and cat == msr.OK_CLEAN


def test_empty_response_is_empty_not_malformed():
    ok, cat, _ = _classify("   ")
    assert ok is False and cat == msr.FAIL_EMPTY


def test_pure_prose_is_no_json():
    ok, cat, detail = _classify("I cannot help with financial advice.")
    assert ok is False and cat == msr.FAIL_NO_JSON
    assert "cannot help" in detail


def test_truncated_json_is_truncated_not_malformed():
    """An opened-and-never-closed object means raise max_tokens, not swap model."""
    ok, cat, _ = _classify('{"facts": [{"source": "reuters", "summary": "aaa')
    assert ok is False and cat == msr.FAIL_TRUNCATED


def test_prose_wrapped_json_PASSES_because_production_tolerates_it():
    """The parser slices first-brace to last-brace, so this survives.

    Counting it as a failure would understate the model; counting it as clean
    would hide that the model is one long preamble away from breaking.
    """
    ok, cat, _ = _classify('Sure! Here is the result:\n{"facts": []}\nHope that helps.')
    assert ok is True
    assert cat == msr.OK_WRAPPED


def test_code_fenced_json_passes_but_is_flagged_separately():
    ok, cat, _ = _classify('```json\n{"facts": []}\n```')
    assert ok is True and cat == msr.OK_FENCED


def test_malformed_json_with_both_braces_is_malformed():
    ok, cat, _ = _classify('{"a": 1, "b": [unquoted, garbage}')
    assert ok is False and cat == msr.FAIL_MALFORMED


# -------------------------------------------------- the final step's gate ---

def test_final_step_needs_actions_not_merely_valid_json():
    """The second gate. Valid JSON is necessary and not sufficient."""
    ok_mid, cat_mid, _ = _classify('{"signal": "bullish"}', is_last=False)
    ok_last, cat_last, detail = _classify('{"signal": "bullish"}', is_last=True)
    assert ok_mid is True and cat_mid == msr.OK_CLEAN
    assert ok_last is False and cat_last == msr.FAIL_SCHEMA
    assert "signal" in detail


def test_empty_orders_is_a_valid_no_trade_not_a_schema_failure():
    """`{"orders": []}` is a well-formed "make no trades this hour" — the
    correct answer most hours — and production aborts the decision on it.

    Verified against pipeline_output_to_decision directly, with no model
    involved. Categorising it as wrong_schema would blame the model for a
    pipeline limitation."""
    ok, cat, detail = _classify('{"orders": []}', is_last=True)
    assert ok is False
    assert cat == msr.FAIL_EMPTY_ACTIONS
    assert detail == "orders"

    ok2, cat2, _ = _classify('{"actions": []}', is_last=True)
    assert cat2 == msr.FAIL_EMPTY_ACTIONS


def test_genuinely_wrong_schema_stays_wrong_schema():
    ok, cat, _ = _classify('{"summary": "looks bullish"}', is_last=True)
    assert ok is False and cat == msr.FAIL_SCHEMA


def test_final_step_with_orders_passes_via_normalisation():
    ok, cat, _ = _classify(
        '{"orders": [{"symbol": "AAPL", "side": "buy", "qty": 3}]}', is_last=True)
    assert ok is True and cat == msr.OK_CLEAN


def test_final_step_with_actions_passes():
    ok, _cat, _ = _classify(
        '{"actions": [{"action": "hold", "symbol": "AAPL"}]}', is_last=True)
    assert ok is True


# ------------------------------------------------------ pipeline loading ----

def test_default_pipeline_is_a_real_shipped_template():
    steps, source = msr.load_pipeline(None)
    assert source == "pipeline-analyst"
    assert len(steps) == 3
    assert [s["label"] for s in steps] == [
        "Information Gathering", "Information to Signal", "Signal to Execution"]
    for s in steps:
        assert s.get("prompt") and s.get("outputFormat")


def test_template_without_any_pipeline_is_refused_with_the_alternatives():
    with pytest.raises(SystemExit, match="no pipeline"):
        msr.load_pipeline("ai-hedge-fund")


def test_single_step_template_loads_even_though_routing_cannot_help_it():
    """A 1-step pipeline has nothing to route, but its reliability is still
    measurable and the harness should not refuse to look."""
    steps, _ = msr.load_pipeline("balanced-starter")
    assert len(steps) == 1


def test_unknown_pipeline_lists_what_exists():
    with pytest.raises(SystemExit, match="no template"):
        msr.load_pipeline("does-not-exist")


def test_pipeline_can_be_loaded_from_a_file(tmp_path):
    p = tmp_path / "pipe.json"
    p.write_text(json.dumps([{"id": "a", "label": "A", "prompt": "x",
                              "outputFormat": '{"a":1}'}]))
    steps, source = msr.load_pipeline(str(p))
    assert len(steps) == 1 and source == "pipe.json"


# ---------------------------------------------------------------- prompts ---

def test_prompts_come_from_the_production_builder():
    """Real step prompts, not synthetic ones."""
    steps, _ = msr.load_pipeline(None)
    ref = msr.build_reference_context(steps)
    first = msr._prompt_for(steps, 0, [])
    last = msr._prompt_for(steps, 2, ref[:2])

    assert "=== SUB-AGENT: Information Gathering ===" in first
    assert "=== MARKET SNAPSHOT ===" in first          # step 1 only
    assert "=== MARKET SNAPSHOT ===" not in last
    assert "=== UPSTREAM PIPELINE OUTPUTS ===" in last
    assert "=== EXECUTION RULES ===" in last           # last step only
    assert steps[0]["prompt"] in first


def test_reference_context_is_identical_for_every_model():
    """Every model must see the same step-3 input or the comparison is invalid."""
    steps, _ = msr.load_pipeline(None)
    a = msr._prompt_for(steps, 2, msr.build_reference_context(steps)[:2])
    b = msr._prompt_for(steps, 2, msr.build_reference_context(steps)[:2])
    assert a == b


def test_market_snapshot_has_the_production_shape():
    snap = msr.REFERENCE_SNAPSHOT
    assert set(snap) >= {"timestamp", "portfolio", "current_holdings",
                         "recent_trades", "top_signals"}
    assert set(snap["portfolio"]) == {
        "cash", "positions_value", "total_equity", "num_positions"}
    for sig in snap["top_signals"].values():
        assert "price" in sig and "rsi" in sig


# ------------------------------------------------------------- cost ceiling -

def test_cost_ceiling_assumes_the_max_token_cap_not_the_answer():
    steps, _ = msr.load_pipeline(None)
    est = msr.estimate_run_cost(steps, ["anthropic/claude-sonnet-4-6"], 10, "isolated")
    assert est["total_calls"] == 30
    assert est["total_max_cost_usd"] > 0
    assert "max_tokens cap" in est["assumption"]


def test_self_hosted_model_is_not_priced_at_claude_rates():
    """token_cost defaults ANY unknown model to (1.0, 5.0) with no signal.
    Left unhandled, a local Qwen would look as expensive as Claude Haiku and
    routing to it would appear to save nothing."""
    steps, _ = msr.load_pipeline(None)
    est = msr.estimate_run_cost(steps, ["Qwen/Qwen2.5-1.5B-Instruct"], 10, "isolated")
    row = est["rows"][0]
    assert row["priced"] is False
    assert row["price_provenance"] == "local"
    assert row["max_cost_usd"] == 0.0


def test_cheap_model_ceiling_is_far_below_the_expensive_one():
    steps, _ = msr.load_pipeline(None)
    est = msr.estimate_run_cost(
        steps, ["nvidia/nemotron-3-nano-30b-a3b", "openai/gpt-5.5"], 10, "isolated")
    cheap, dear = est["rows"][0]["max_cost_usd"], est["rows"][1]["max_cost_usd"]
    assert dear > cheap * 50


# -------------------------------------------------------- the harness loop --

class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Response:
    def __init__(self, text, out_tok):
        self.content = [_Block(text)]
        self.usage = _Usage(500, out_tok)


class _FakeMessages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kw):
        self._owner.calls.append(kw)
        content = kw["messages"][0]["content"]
        text = '{"ok": 1}'
        for needle, reply in self._owner.by_step.items():
            if f"=== SUB-AGENT: {needle} ===" in content:
                text = reply
                break
        return _Response(text, self._owner.out_tok)


class FakeClient:
    """Replies chosen by step label, so a model can be good at one step and
    bad at another — the case routing has to be able to detect."""

    def __init__(self, by_step, out_tok=40):
        self.by_step = by_step
        self.out_tok = out_tok
        self.calls = []
        self.messages = _FakeMessages(self)


def test_isolated_mode_measures_every_step_for_every_attempt():
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": '{"facts": []}',
                         "Information to Signal": '{"signal": "up"}',
                         "Signal to Execution":
                             '{"actions": [{"action": "hold", "symbol": "AAPL"}]}'})
    results = msr.run_isolated(client, steps, "m", attempts=4, verbose=False)
    assert len(results) == 3
    assert all(r.n == 4 for r in results)
    assert all(r.parse_rate == 1.0 for r in results)
    assert len(client.calls) == 12


def test_a_step_specific_failure_is_attributed_to_that_step():
    """A model fine at extraction and bad at the decision step must show that."""
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": '{"facts": []}',
                         "Information to Signal": '{"signal": "up"}',
                         "Signal to Execution": 'I recommend holding everything.'})
    results = msr.run_isolated(client, steps, "m", attempts=5, verbose=False)
    assert results[0].parse_rate == 1.0
    assert results[1].parse_rate == 1.0
    assert results[2].parse_rate == 0.0
    assert results[2].categories() == {msr.FAIL_NO_JSON: 5}


def test_isolated_mode_keeps_measuring_after_a_failure():
    """Unlike production, isolation must not stop at the first bad step —
    otherwise a weak step 1 hides everything downstream."""
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": 'nope',
                         "Information to Signal": '{"signal": "up"}',
                         "Signal to Execution":
                             '{"actions": [{"action": "hold"}]}'})
    results = msr.run_isolated(client, steps, "m", attempts=3, verbose=False)
    assert results[0].parse_rate == 0.0
    assert results[1].parse_rate == 1.0      # still measured
    assert results[2].parse_rate == 1.0


def test_output_tokens_are_recorded_since_they_are_the_cost_term():
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": '{"facts": []}'}, out_tok=123)
    results = msr.run_isolated(client, steps, "m", attempts=2, verbose=False)
    assert results[0].mean_output_tokens == 123


def test_api_errors_are_recorded_not_raised():
    class Boom:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("429 rate limited")

    steps, _ = msr.load_pipeline(None)
    results = msr.run_isolated(Boom(), steps, "m", attempts=2, verbose=False)
    assert all(r.parse_rate == 0.0 for r in results)
    assert results[0].categories() == {"api_error": 2}


def test_endtoend_mode_stops_at_the_first_failure_like_production():
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": '{"facts": []}',
                         "Information to Signal": 'garbage',
                         "Signal to Execution": '{"actions": [{"action": "hold"}]}'})
    out = msr.run_endtoend(client, steps, "m", attempts=4, verbose=False)
    assert out["completed"] == 0
    assert out["completion_rate"] == 0.0
    assert list(out["aborted_at"]) == ["step2:no_json"]
    # step 3 was never reached, so it has no token record at all.
    assert 2 not in out["mean_output_tokens_by_step"]


def test_endtoend_completion_counts_full_pipelines():
    steps, _ = msr.load_pipeline(None)
    client = FakeClient({"Information Gathering": '{"facts": []}',
                         "Information to Signal": '{"signal": "up"}',
                         "Signal to Execution":
                             '{"actions": [{"action": "hold", "symbol": "AAPL"}]}'})
    out = msr.run_endtoend(client, steps, "m", attempts=3, verbose=False)
    assert out["completed"] == 3 and out["completion_rate"] == 1.0
    assert out["aborted_at"] == {}


# ------------------------------------------------------------------- CLI ----

def test_dry_run_spends_nothing_and_returns_zero(capsys):
    code = msr.main(["--dry-run", "--models", "openai/gpt-5.5", "--attempts", "10"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no API calls made, nothing spent" in out
    assert "COST CEILING" in out


def test_absurd_attempt_count_is_refused():
    code = msr.main(["--models", "x", "--attempts", "5000", "--dry-run"])
    assert code == 2


def test_no_models_measures_nothing():
    assert msr.main(["--dry-run"]) == 2
