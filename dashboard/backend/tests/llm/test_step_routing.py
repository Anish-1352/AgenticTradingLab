"""Per-step model routing: resolution, fallback, and the untouched default.

The property that matters most here is the boring one — that a default or
absent config produces byte-identical behaviour to the pipeline runner before
routing existed. Routing is a cost lever aimed at a live leaderboard; if
switching it off is not exactly a no-op, nothing measured with it can be
trusted against anything measured without it.

The second property is that turning routing ON does not change the PROMPTS.
Only the model answering them changes. Otherwise a routed-vs-unrouted cost
comparison would be measuring two different workloads.
"""

import json

import pytest

from dashboard.backend.infrastructure.llm import pipeline_runner as pr
from dashboard.backend.infrastructure.llm.step_routing import (
    MATCH_PRECEDENCE,
    RoutingConfig,
    StepRoute,
    load_routing_config,
)

CHEAP = "nvidia/nemotron-nano-9b-v2"
EXPENSIVE = "claude-haiku-4-5-20251001"

STEPS = [
    {"id": "bench-3-1", "label": "Technical Read", "prompt": "p1",
     "outputFormat": '{"a": 1}'},
    {"id": "bench-3-2", "label": "Risk Assessment", "prompt": "p2",
     "outputFormat": '{"b": 2}'},
    {"id": "bench-3-3", "label": "Order Construction", "prompt": "p3",
     "outputFormat": '{"actions": []}'},
]


# ------------------------------------------------------- the default path ---

def test_disabled_config_returns_the_agents_model():
    cfg = RoutingConfig.disabled()
    res = cfg.resolve(STEPS[0], 0, EXPENSIVE)
    assert res.model == EXPENSIVE
    assert res.is_default is True
    assert res.routed is False
    assert res.matched_by is None


def test_absent_config_is_disabled():
    for source in (None, {}, ""):
        assert load_routing_config(source).enabled is False


def test_config_present_but_not_enabled_still_routes_nothing():
    """A table can be committed and reviewed while switched off."""
    cfg = RoutingConfig.from_dict(
        {"enabled": False, "steps": {"Technical Read": CHEAP}})
    assert cfg.by_label                     # the rule was parsed
    assert cfg.resolve(STEPS[0], 0, EXPENSIVE).model == EXPENSIVE


def test_unmapped_step_uses_the_agents_configured_model():
    """The brief's explicit requirement: no rule, no change."""
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"Technical Read": CHEAP}})
    res = cfg.resolve(STEPS[1], 1, EXPENSIVE)   # Risk Assessment: unmapped
    assert res.model == EXPENSIVE
    assert res.is_default is True
    assert res.routed is False


def test_every_step_unmapped_leaves_the_whole_pipeline_on_one_model():
    cfg = RoutingConfig.from_dict({"enabled": True, "steps": {"Nothing": CHEAP}})
    plan = cfg.plan(STEPS, EXPENSIVE)
    assert [r.model for r in plan] == [EXPENSIVE] * 3
    assert all(r.is_default for r in plan)


# ------------------------------------------------------------- resolution ---

def test_label_routing_matches_case_and_whitespace_insensitively():
    """Labels are free text someone typed into a UI field."""
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"  technical READ ": CHEAP}})
    assert cfg.resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP


def test_label_routing_generalises_across_pipelines():
    """The 3-step and 5-step configs share step labels; that is the point."""
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"Technical Read": CHEAP}})
    five_step_first = {"id": "bench-5-1", "label": "Technical Read"}
    assert cfg.resolve(five_step_first, 0, EXPENSIVE).model == CHEAP


def test_index_routing_is_one_based():
    cfg = RoutingConfig.from_dict({"enabled": True, "steps": {"1": CHEAP}})
    assert cfg.resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP
    assert cfg.resolve(STEPS[1], 1, EXPENSIVE).model == EXPENSIVE


def test_id_routing_targets_one_step_in_one_pipeline():
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"id:bench-3-2": CHEAP}})
    assert cfg.resolve(STEPS[1], 1, EXPENSIVE).model == CHEAP
    assert cfg.resolve({"id": "bench-5-2", "label": "x"}, 1, EXPENSIVE).model == EXPENSIVE


def test_preset_key_routing():
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"preset:risk_check": CHEAP}})
    step = {"presetKey": "risk_check", "label": "Whatever"}
    res = cfg.resolve(step, 0, EXPENSIVE)
    assert res.model == CHEAP and res.matched_by == "presetKey"


def test_precedence_is_id_then_preset_then_label_then_index():
    assert MATCH_PRECEDENCE == ("id", "presetKey", "label", "index")


def test_id_beats_label_so_one_step_can_be_excepted():
    """'Every Risk Assessment on the cheap model, except this one.'"""
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Risk Assessment": CHEAP, "id:bench-3-2": EXPENSIVE},
    })
    assert cfg.resolve(STEPS[1], 1, "other").model == EXPENSIVE
    other_risk_step = {"id": "bench-5-3", "label": "Risk Assessment"}
    assert cfg.resolve(other_risk_step, 2, "other").model == CHEAP


def test_label_beats_index():
    cfg = RoutingConfig.from_dict({
        "enabled": True, "steps": {"Technical Read": CHEAP, "1": EXPENSIVE}})
    res = cfg.resolve(STEPS[0], 0, "other")
    assert res.model == CHEAP and res.matched_by == "label"


def test_a_step_labelled_with_a_number_is_addressable_unambiguously():
    """A pipeline may label a step '3'; the config must be able to say which."""
    step = {"id": "x", "label": "3"}
    by_label = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"label:3": CHEAP}})
    assert by_label.resolve(step, 0, EXPENSIVE).model == CHEAP     # index is 1
    by_index = RoutingConfig.from_dict({"enabled": True, "steps": {"3": CHEAP}})
    assert by_index.resolve(step, 0, EXPENSIVE).model == EXPENSIVE  # label ignored
    assert by_index.resolve(step, 2, EXPENSIVE).model == CHEAP      # index 3


def test_resolution_records_what_matched():
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"Technical Read": CHEAP}})
    res = cfg.resolve(STEPS[0], 0, EXPENSIVE)
    assert res.matched_by == "label"
    assert res.matched_key == "Technical Read"
    assert res.to_dict()["routed"] is True


# --------------------------------------------------------------- fallback ---

def test_fallback_catches_unmapped_steps():
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Order Construction": EXPENSIVE},
        "fallback": CHEAP,
    })
    plan = cfg.plan(STEPS, "agent-model")
    assert [r.model for r in plan] == [CHEAP, CHEAP, EXPENSIVE]
    assert plan[0].matched_by == "fallback"
    assert plan[2].matched_by == "label"


def test_fallback_means_the_agents_model_is_never_used():
    """Worth stating: a fallback silently removes the agent's own model."""
    cfg = RoutingConfig.from_dict({"enabled": True, "fallback": CHEAP})
    plan = cfg.plan(STEPS, EXPENSIVE)
    assert all(r.model == CHEAP for r in plan)
    assert not any(r.is_default for r in plan)


def test_default_is_accepted_as_a_synonym_for_fallback():
    cfg = RoutingConfig.from_dict({"enabled": True, "default": CHEAP})
    assert cfg.fallback == StepRoute(model=CHEAP)


# ---------------------------------------------------------- route targets ---

def test_route_can_name_an_integration_for_cross_provider_routing():
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Technical Read": {"model": CHEAP, "integration": "openrouter"}},
    })
    res = cfg.resolve(STEPS[0], 0, EXPENSIVE, default_integration="commonstack")
    assert res.model == CHEAP
    assert res.integration == "openrouter"


def test_route_without_an_integration_inherits_the_agents():
    cfg = RoutingConfig.from_dict(
        {"enabled": True, "steps": {"Technical Read": CHEAP}})
    res = cfg.resolve(STEPS[0], 0, EXPENSIVE, default_integration="commonstack")
    assert res.integration == "commonstack"


def test_local_models_need_no_new_provider_type():
    """A self-hosted engine is an OpenRouter-compatible base_url, not a new kind."""
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Technical Read": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                                     "integration": "openrouter"}},
    })
    res = cfg.resolve(STEPS[0], 0, EXPENSIVE)
    assert res.model == "Qwen/Qwen2.5-1.5B-Instruct"


# ------------------------------------------------------- malformed config ---

def test_unknown_top_level_key_is_rejected():
    """A typo must not read as 'route nothing' and show up as a cost mystery."""
    with pytest.raises(ValueError, match="unknown step-routing keys"):
        RoutingConfig.from_dict({"enabled": True, "stpes": {"a": CHEAP}})


def test_underscore_keys_are_comments_not_typos():
    """JSON has no comments; the example config uses this convention."""
    cfg = RoutingConfig.from_dict({
        "enabled": True, "_why": ["a note"], "steps": {"Technical Read": CHEAP}})
    assert cfg.resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP


def test_the_shipped_example_config_parses_and_is_switched_off():
    """A committed example that does not load is worse than none."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    path = os.path.join(root, "dashboard", "config", "step_routing.example.json")
    cfg = load_routing_config(path)
    assert cfg.enabled is False
    assert cfg.by_label                     # rules parsed, just not active
    assert cfg.fallback is None             # a fallback would bypass the agent
    plan = cfg.plan(STEPS, "AGENT-MODEL")
    assert [r.model for r in plan] == ["AGENT-MODEL"] * 3


def test_route_without_a_model_is_rejected():
    with pytest.raises(ValueError, match="missing a model"):
        RoutingConfig.from_dict({"enabled": True, "steps": {"a": {"integration": "x"}}})


def test_empty_model_id_is_rejected():
    with pytest.raises(ValueError):
        RoutingConfig.from_dict({"enabled": True, "steps": {"a": "  "}})


def test_non_numeric_step_index_is_rejected():
    with pytest.raises(ValueError, match="not a step number"):
        RoutingConfig.from_dict({"enabled": True, "steps": {"step:two": CHEAP}})


# ------------------------------------------------------------------ load ----

def test_load_from_inline_json():
    cfg = load_routing_config(json.dumps(
        {"enabled": True, "steps": {"Technical Read": CHEAP}}))
    assert cfg.resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP


def test_load_from_file_accepts_a_nested_key(tmp_path):
    p = tmp_path / "routing.json"
    p.write_text(json.dumps(
        {"step_routing": {"enabled": True, "steps": {"Technical Read": CHEAP}}}))
    assert load_routing_config(str(p)).resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP


def test_env_var_is_only_read_when_no_source_is_given(monkeypatch):
    monkeypatch.setenv("ATL_STEP_ROUTING", json.dumps(
        {"enabled": True, "fallback": CHEAP}))
    assert load_routing_config().resolve(STEPS[0], 0, EXPENSIVE).model == CHEAP
    # An explicit source wins, so a stray env var cannot override a caller.
    explicit = load_routing_config({"enabled": False})
    assert explicit.resolve(STEPS[0], 0, EXPENSIVE).model == EXPENSIVE


def test_describe_is_serialisable_for_manifests():
    cfg = RoutingConfig.from_dict({
        "enabled": True, "steps": {"Technical Read": CHEAP}, "fallback": EXPENSIVE})
    d = cfg.describe()
    assert json.loads(json.dumps(d))["n_rules"] == 1
    assert d["fallback"] == EXPENSIVE


# ================================================================= runner ===
# Routing wired into run_pipeline_decision. A fake client records every call so
# the tests can assert on the model each step actually received.


class _Block:
    """Anthropic-shaped text block — attributes, not dict keys."""

    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Response:
    def __init__(self, text, in_tok=10, out_tok=5):
        self.content = [_Block(text)]
        self.usage = _Usage(in_tok, out_tok)


class FakeMessages:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return _Response(reply)


class FakeClient:
    def __init__(self, replies):
        self.messages = FakeMessages(replies)


def _replies():
    return ['{"read": "ok"}', '{"risk": "ok"}',
            '{"actions": [{"action": "hold", "symbol": "AAPL"}]}']


def _run(routing=None, **kw):
    client = FakeClient(_replies())
    decision, tokens, calls, outputs = pr.run_pipeline_decision(
        client,
        pipeline=[dict(s) for s in STEPS],
        market_snapshot={"top_signals": {}},
        model=EXPENSIVE,
        routing=routing,
        **kw,
    )
    return client, decision, calls


def test_runner_without_routing_uses_the_agent_model_for_every_step():
    client, decision, calls = _run(routing=None)
    assert decision is not None and calls == 3
    assert [c["model"] for c in client.messages.calls] == [EXPENSIVE] * 3


def test_runner_with_disabled_routing_is_identical_to_no_routing():
    a, _, _ = _run(routing=None)
    b, _, _ = _run(routing=RoutingConfig.disabled())
    assert [c["model"] for c in a.messages.calls] == \
           [c["model"] for c in b.messages.calls]
    assert [c["messages"] for c in a.messages.calls] == \
           [c["messages"] for c in b.messages.calls]


def test_runner_routes_only_the_mapped_steps():
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Technical Read": CHEAP, "Risk Assessment": CHEAP},
    })
    client, decision, _ = _run(routing=cfg)
    assert decision is not None
    assert [c["model"] for c in client.messages.calls] == [CHEAP, CHEAP, EXPENSIVE]


def test_routing_does_not_change_the_prompts():
    """Only the model answering changes. Otherwise a routed-vs-unrouted cost
    comparison measures two different workloads."""
    plain, _, _ = _run(routing=None)
    cfg = RoutingConfig.from_dict({"enabled": True, "fallback": CHEAP})
    routed, _, _ = _run(routing=cfg)
    assert [c["messages"] for c in plain.messages.calls] == \
           [c["messages"] for c in routed.messages.calls]
    assert [c["system"] for c in plain.messages.calls] == \
           [c["system"] for c in routed.messages.calls]


def test_a_routed_step_returning_unparseable_json_aborts_the_whole_decision():
    """The risk that decides viability: no retry, the decision never completes."""
    client = FakeClient(['not json at all', '{"risk": "ok"}',
                         '{"actions": [{"action": "hold"}]}'])
    cfg = RoutingConfig.from_dict({"enabled": True, "fallback": CHEAP})
    decision, tokens, calls, outputs = pr.run_pipeline_decision(
        client, pipeline=[dict(s) for s in STEPS],
        market_snapshot={}, model=EXPENSIVE, routing=cfg)
    assert decision is None
    assert calls == 1                   # aborted at step 1; steps 2-3 never ran
    assert outputs == []


def test_final_step_parsing_but_yielding_no_actions_also_aborts():
    """The last step has a second, stricter gate than JSON validity."""
    client = FakeClient(['{"read": "ok"}', '{"risk": "ok"}', '{"actions": []}'])
    decision, _tokens, calls, _outputs = pr.run_pipeline_decision(
        client, pipeline=[dict(s) for s in STEPS],
        market_snapshot={}, model=EXPENSIVE)
    assert decision is None
    assert calls == 3                   # all three ran; the conversion failed


def test_integration_route_swaps_the_client():
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Technical Read": {"model": CHEAP, "integration": "openrouter"}},
    })
    cheap_client = FakeClient(_replies())
    seen = []

    def client_for(integration):
        seen.append(integration)
        return cheap_client

    main_client, decision, _ = _run(routing=cfg, client_for_integration=client_for)
    assert seen == ["openrouter"]
    assert [c["model"] for c in cheap_client.messages.calls] == [CHEAP]
    assert [c["model"] for c in main_client.messages.calls] == [EXPENSIVE, EXPENSIVE]


def test_missing_integration_client_falls_back_to_the_callers_client():
    """A provider that will not construct must not take the decision down."""
    cfg = RoutingConfig.from_dict({
        "enabled": True,
        "steps": {"Technical Read": {"model": CHEAP, "integration": "openrouter"}},
    })
    client, decision, _ = _run(routing=cfg, client_for_integration=lambda i: None)
    assert decision is not None
    assert client.messages.calls[0]["model"] == CHEAP
