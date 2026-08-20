"""Cost projection from measured reliability.

The behaviour under test is mostly refusal. A projection that reports a saving
without a measured completion rate is the exact failure the brief names — "a
40% cost reduction that drops 10% of decisions is not a saving" — so the
effective saving is withheld rather than defaulted to the nominal one.
"""

import pytest

from dashboard.backend.infrastructure.llm.routing_cost import (
    PRICE_DEFAULTED,
    PRICE_KNOWN,
    PRICE_LOCAL,
    format_projection,
    price_provenance,
    project_routing,
)

DEAR = "openai/gpt-5.5"                        # 5.0 / 30.0 per Mtok
CHEAP = "nvidia/nemotron-3-nano-30b-a3b"       # 0.05 / 0.20 per Mtok
LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"

STEPS = [
    {"label": "Information Gathering"},
    {"label": "Information to Signal"},
    {"label": "Signal to Execution"},
]


def _tokens(model_a, model_b=None, out=(300.0, 200.0, 150.0)):
    """Measured output tokens keyed by (model, step)."""
    d = {(model_a, i): out[i] for i in range(3)}
    if model_b:
        d.update({(model_b, i): out[i] for i in range(3)})
    return d


def _rates(model, value, n=3):
    return {(model, i): value for i in range(n)}


# ------------------------------------------------------- price provenance ---

def test_known_model_prices_are_marked_known():
    prov, (i, o) = price_provenance(DEAR)
    assert prov == PRICE_KNOWN and (i, o) == (5.0, 30.0)


def test_self_hosted_model_has_no_per_token_price():
    prov, price = price_provenance(LOCAL)
    assert prov == PRICE_LOCAL and price == (0.0, 0.0)


def test_unknown_hosted_model_is_marked_defaulted_not_known():
    """token_cost silently returns (1.0, 5.0) for anything it does not know."""
    prov, price = price_provenance("some-vendor/brand-new-model-v9")
    assert prov == PRICE_DEFAULTED
    assert price == (1.0, 5.0)


def test_free_markers_are_local():
    assert price_provenance("local-model")[0] == PRICE_LOCAL
    assert price_provenance("rule-based")[0] == PRICE_LOCAL


# ---------------------------------------------------------------- refusal ---

def test_saving_is_withheld_without_measured_parse_rates():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, CHEAP, DEAR],
        measured_output_tokens=_tokens(DEAR, CHEAP))
    assert proj.nominal_saving_fraction is not None      # arithmetic is fine
    assert proj.effective_saving_fraction is None        # the claim is not
    assert proj.routed_completion_rate is None
    assert proj.reliable is False
    assert any("not a result" in w for w in proj.warnings)


def test_defaulted_price_makes_the_projection_unreliable():
    proj = project_routing(
        STEPS, baseline_model=DEAR,
        routed_models=["mystery/model", "mystery/model", DEAR],
        measured_output_tokens=_tokens(DEAR, "mystery/model"),
        measured_parse_rates={**_rates("mystery/model", 1.0), (DEAR, 2): 1.0})
    assert proj.reliable is False
    assert any("not a price" in w for w in proj.warnings)


def test_missing_token_measurement_is_warned_not_invented():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, CHEAP, DEAR],
        measured_output_tokens={(DEAR, i): 300.0 for i in range(3)})
    assert any("stand-in" in w for w in proj.warnings)


# ------------------------------------------------------------- arithmetic ---

def test_routing_two_of_three_steps_saves_money():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, CHEAP, DEAR],
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates={**_rates(CHEAP, 1.0), (DEAR, 2): 1.0},
        baseline_completion_rate=1.0,
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.routed_cost < proj.baseline_cost
    assert 0 < proj.nominal_saving_fraction < 1
    assert proj.effective_saving_fraction == pytest.approx(
        proj.nominal_saving_fraction)          # nothing lost at 100% parse


def test_baseline_completion_is_not_assumed_to_be_perfect():
    """Assuming the expensive model never fails would flatter routing."""
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, CHEAP, DEAR],
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates={**_rates(CHEAP, 1.0), (DEAR, 2): 1.0})
    assert proj.routed_completion_rate is not None
    assert proj.baseline_completion_rate is None
    assert proj.effective_saving_fraction is None
    assert any("flatter routing" in w for w in proj.warnings)


def test_routing_every_step_to_local_zeroes_the_api_cost():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[LOCAL] * 3,
        measured_output_tokens=_tokens(DEAR, LOCAL),
        measured_parse_rates=_rates(LOCAL, 1.0),
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.routed_cost == 0.0
    assert proj.nominal_saving_fraction == pytest.approx(1.0)
    assert any("GPU time" in w for w in proj.warnings)


def test_completion_is_the_product_of_every_step_rate():
    """No retry: the pipeline completes only if EVERY step parses."""
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP] * 3,
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates={(CHEAP, 0): 0.9, (CHEAP, 1): 0.9, (CHEAP, 2): 0.9})
    assert proj.routed_completion_rate == pytest.approx(0.729)


def test_a_big_nominal_saving_with_losses_shows_both_numbers():
    """The brief's case: 60% on paper, 5% of decisions aborted."""
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, CHEAP, DEAR],
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates={(CHEAP, 0): 0.98, (CHEAP, 1): 0.97, (DEAR, 2): 1.0},
        baseline_completion_rate=1.0,
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.nominal_saving_fraction > proj.effective_saving_fraction
    assert proj.decisions_lost_per_100 == pytest.approx(4.94, abs=0.01)
    text = format_projection(proj)
    assert "nominal saving" in text and "decisions lost" in text


def test_cost_per_completed_exceeds_raw_cost_when_decisions_abort():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP] * 3,
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates=_rates(CHEAP, 0.8),
        baseline_completion_rate=1.0,
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.routed_cost_per_completed > proj.routed_cost
    assert any("UPPER bound" in w for w in proj.warnings)


def test_enough_failures_can_erase_the_saving_entirely():
    """A cheap model that aborts most decisions is not cheap."""
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, DEAR, DEAR],
        measured_output_tokens=_tokens(DEAR, CHEAP),
        measured_parse_rates={(CHEAP, 0): 0.10, (DEAR, 1): 1.0, (DEAR, 2): 1.0},
        baseline_completion_rate=1.0,
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.nominal_saving_fraction > 0
    assert proj.effective_saving_fraction < 0       # routing now COSTS more
    assert proj.decisions_lost_per_100 == pytest.approx(90.0)


# ------------------------------------------------------------ compounding ---

def test_a_terser_step_one_shrinks_later_steps_input():
    """_build_step_prompt embeds every prior output into each later prompt."""
    verbose = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[DEAR] * 3,
        measured_output_tokens={(DEAR, 0): 800.0, (DEAR, 1): 200.0, (DEAR, 2): 150.0},
        measured_parse_rates=_rates(DEAR, 1.0),
        base_input_tokens=[2000.0, 400.0, 500.0])
    terse = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[CHEAP, DEAR, DEAR],
        measured_output_tokens={(DEAR, 0): 800.0, (DEAR, 1): 200.0, (DEAR, 2): 150.0,
                                (CHEAP, 0): 100.0},
        measured_parse_rates={(CHEAP, 0): 1.0, (DEAR, 1): 1.0, (DEAR, 2): 1.0},
        base_input_tokens=[2000.0, 400.0, 500.0])

    # Step 2 and 3 are on the SAME model in both, yet their input differs.
    assert terse.routed_steps[1].input_tokens < verbose.routed_steps[1].input_tokens
    assert terse.routed_steps[2].input_tokens < verbose.routed_steps[2].input_tokens
    assert terse.compounding_applied is True


def test_step_one_input_carries_no_upstream():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[DEAR] * 3,
        measured_output_tokens=_tokens(DEAR),
        measured_parse_rates=_rates(DEAR, 1.0),
        base_input_tokens=[2000.0, 400.0, 500.0])
    assert proj.routed_steps[0].input_tokens == 2000.0


def test_compounding_is_cumulative_not_just_the_previous_step():
    proj = project_routing(
        STEPS, baseline_model=DEAR, routed_models=[DEAR] * 3,
        measured_output_tokens={(DEAR, 0): 100.0, (DEAR, 1): 100.0, (DEAR, 2): 0.0},
        measured_parse_rates=_rates(DEAR, 1.0),
        base_input_tokens=[0.0, 0.0, 0.0])
    # step 3 inherits BOTH prior outputs, step 2 only the first.
    assert proj.routed_steps[2].input_tokens > proj.routed_steps[1].input_tokens
    assert proj.routed_steps[2].input_tokens == pytest.approx(200.0 + 25.0 * 2)


# ------------------------------------------------------------- guardrails ---

def test_model_list_must_match_step_count():
    with pytest.raises(ValueError, match="for 3 steps"):
        project_routing(STEPS, baseline_model=DEAR, routed_models=[CHEAP])


def test_report_renders_without_measurements():
    proj = project_routing(STEPS, baseline_model=DEAR, routed_models=[CHEAP] * 3)
    text = format_projection(proj)
    assert "WITHHELD" in text
    assert "NOT MEASURED" in text
