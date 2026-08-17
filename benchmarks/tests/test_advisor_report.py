"""Tests for the advisor package: tier discipline, and the notebook's refusal.

Two properties are load-bearing and are enforced here rather than trusted:

1. **Every numeric claim in ADVISOR_REPORT.md carries an evidence tier.** The
   report's whole value is that a reader can tell a measurement from arithmetic
   from a guess. One untagged number undermines that for all of them.
2. **The cost model refuses to compute with blank unmeasured inputs.** A silent
   default is how an assumption becomes a finding.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import cost_model_lib as lib          # noqa: E402
from analysis import make_advisor_report as report  # noqa: E402
from analysis import make_notebook as nbgen         # noqa: E402
from analysis import tier_check                     # noqa: E402

_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
_REPORT_MD = os.path.join(_ANALYSIS, "ADVISOR_REPORT.md")
_NOTEBOOK = os.path.join(_ANALYSIS, "cost_model.ipynb")


@pytest.fixture(scope="module")
def measured():
    return lib.load_measured()


@pytest.fixture(scope="module")
def report_text(measured):
    return report.build_report(measured)


# ---- tier discipline -------------------------------------------------------

def test_generated_report_has_no_untagged_numbers(report_text):
    result = tier_check.check_report(report_text)
    assert result["ok"], (
        f"{result['n_violations']} numeric line(s) lack an evidence tier:\n"
        + tier_check.format_violations(result["violations"]))


def test_committed_report_matches_the_generator(report_text):
    """The committed file must be the generator's output, not hand-edited."""
    assert os.path.exists(_REPORT_MD), "run: python -m analysis.make_advisor_report"
    with open(_REPORT_MD) as fh:
        committed = fh.read()
    assert committed == report_text, (
        "ADVISOR_REPORT.md is stale or hand-edited; regenerate it with "
        "`python -m analysis.make_advisor_report`")


def test_committed_report_also_passes_the_tier_check():
    with open(_REPORT_MD) as fh:
        assert tier_check.check_report(fh.read())["ok"]


def test_tier_checker_catches_an_untagged_number():
    """The checker must fail on bad input, or it proves nothing."""
    bad = "# Title\n\nThe platform made 4,182 calls last month.\n"
    result = tier_check.check_report(bad)
    assert not result["ok"] and result["n_violations"] == 1


def test_tier_checker_accepts_a_tagged_number():
    good = "# Title\n\nThe platform made 4,182 calls last month. [MEASURED]\n"
    assert tier_check.check_report(good)["ok"]


def test_tier_checker_accepts_a_tagged_table_row():
    good = ("# T\n\n| Model | Calls | Tier |\n|---|---:|---|\n"
            "| `x` | 1,234 | MEASURED |\n")
    assert tier_check.check_report(good)["ok"]


def test_tier_checker_rejects_an_untagged_table_row():
    bad = "# T\n\n| Model | Calls | Note |\n|---|---:|---|\n| `x` | 1,234 | hi |\n"
    assert not tier_check.check_report(bad)["ok"]


def test_tier_checker_ignores_fenced_code():
    """Fenced blocks are quoted source; the claim is the prose around them."""
    text = "# T\n\n```python\nx = 42\n```\n"
    assert tier_check.check_report(text)["ok"]


def test_report_names_all_three_tiers(report_text):
    for tier in (lib.MEASURED, lib.DERIVED, lib.NOT_MEASURED):
        assert f"[{tier}]" in report_text or f"| {tier} |" in report_text


# ---- the report's content contract -----------------------------------------

def test_report_still_excludes_modelled_crossover(report_text):
    """The crossover remains model output, not measurement."""
    assert "crossover_agents" not in report_text.lower()
    for line in report_text.splitlines():
        if "crossover" in line.lower():
            assert any(m in line.lower() for m in (
                "never", "not measured", "excluded", "lower bound",
                "dirty", "omits")), (
                f"crossover appears as a claim rather than an exclusion: {line}")


def test_arm_a_is_now_measured_not_excluded(report_text):
    """Arm A moved tiers once it was actually run — the report must say so."""
    assert "Arm A — hosted API, measured for the first time" in report_text
    # Its cost/latency rows carry MEASURED, and it is gone from the
    # never-executed list.
    assert "no hosted-API run has ever been executed" not in report_text
    assert "Every Arm A number" not in report_text


def test_arm_a_throughput_is_still_flagged_incomparable(report_text):
    """min_tokens was ignored, so tok/s does not cross arms."""
    assert "min_tokens" in report_text
    assert "NOT comparable to arms B/C" in report_text


def test_prefix_cache_benefit_is_still_unmeasured(report_text):
    """A measured ZERO on the hosted side is not a measured benefit."""
    assert "cached_tokens" in report_text
    for line in report_text.splitlines():
        if "prefix-cache benefit" in line.lower():
            assert any(m in line.lower() for m in
                       ("never", "not measured", "ablation"))


def test_calls_per_decision_is_measured_at_both_depths(report_text):
    assert "| 3 | 3.000 | yes |" in report_text
    assert "| 5 | 5.000 | yes |" in report_text
    assert "no retry inflation was observed" in report_text


def test_thresholds_shown_at_one_three_and_five_calls(report_text):
    assert "| Model | 1 call/decision | 3 calls | 5 calls | Tier |" in report_text


def test_rate_limit_ceiling_carries_the_account_tier_qualifier(report_text):
    assert "property of THIS ACCOUNT TIER" in report_text
    assert "not of the api" in report_text.lower()


def test_report_states_the_answer_as_a_conditional(report_text):
    assert "conditional" in report_text.lower()
    assert "only if" in report_text.lower()


def test_report_flags_the_dirty_sha_caveat(report_text):
    assert "-dirty" in report_text
    assert "provisional" in report_text.lower()
    assert "RUN_MANIFEST_SCHEMA" in report_text


def test_report_keeps_dashboard_and_orchestration_distinct(report_text):
    assert "orchestration/FinAgents" in report_text
    assert "Neither imports the other" in report_text


def test_report_lists_every_unmeasured_input(report_text):
    for key in lib.UNMEASURED_INPUTS:
        assert key in report_text, f"unmeasured input {key} not listed"


# ---- measured values are read, not transcribed -----------------------------

def test_pricing_is_verified_against_stored_cost(measured):
    pv = measured["price_verification"]
    assert pv["checked"] == 7
    assert pv["max_delta_usd"] < pv["tolerance"]


def test_loader_raises_when_pricing_disagrees(monkeypatch):
    """A drifted price must stop the build, not quietly change every figure."""
    bogus = {k: {**v, "in": v["in"] * 3} for k, v in lib.PRICE_BY_DB_MODEL.items()}
    monkeypatch.setattr(lib, "PRICE_BY_DB_MODEL", bogus)
    with pytest.raises(ValueError, match="pricing disagrees"):
        lib.load_measured()


def test_every_code_finding_citation_is_live():
    findings = lib.verify_code_findings()
    assert findings, "no code findings declared"
    bad = [f"{f['file']}:{f['line']}" for f in findings if not f["verified"]]
    assert not bad, f"citations no longer match the source: {bad}"


def test_seed_db_facts(measured):
    assert measured["total_runs"] == 17
    assert measured["runs_with_llm"] == 7
    # The finding, not an accident: insert_decisions never fires for this path.
    assert measured["backtest_decisions_rows"] == 0


def test_output_token_spread_is_larger_than_input_spread(measured):
    outs = [m["output_per_call"] for m in measured["models"].values()]
    ins = [m["input_per_call"] for m in measured["models"].values()]
    assert (max(outs) / min(outs)) > (max(ins) / min(ins))


# ---- the notebook refuses to compute ---------------------------------------

def test_require_inputs_names_every_blank():
    with pytest.raises(lib.MissingInput) as exc:
        lib.require_inputs({}, ["n_users", "model_mix"])
    msg = str(exc.value)
    assert "n_users" in msg and "model_mix" in msg
    assert "Render" in msg          # names what would resolve it


def test_monthly_breakdown_refuses_blank_params(measured):
    with pytest.raises(lib.MissingInput):
        lib.monthly_breakdown(measured, {k: None for k in lib.UNMEASURED_INPUTS})


def test_monthly_breakdown_has_no_silent_defaults(measured):
    """Every unmeasured input must be individually required."""
    full = dict(n_users=10, n_agents=10, decisions_per_agent_per_day=7,
                backtests_per_user_per_day=1, calls_per_decision=1,
                model_mix={"nemotron_3_nano_30b": 1.0},
                trading_days_per_month=21, calendar_days_per_month=30,
                infrastructure_usd_per_month=0.0)
    lib.monthly_breakdown(measured, full)          # baseline works
    for key in full:
        blanked = dict(full)
        blanked[key] = None
        with pytest.raises(lib.MissingInput, match=key):
            lib.monthly_breakdown(measured, blanked)


def test_model_mix_refuses_an_unmeasured_model(measured):
    params = dict(n_users=1, n_agents=1, decisions_per_agent_per_day=1,
                  backtests_per_user_per_day=1, calls_per_decision=1,
                  model_mix={"some_model_we_never_ran": 1.0},
                  trading_days_per_month=21, calendar_days_per_month=30,
                  infrastructure_usd_per_month=0.0)
    with pytest.raises(lib.MissingInput, match="no measured token counts"):
        lib.monthly_breakdown(measured, params)


def test_backtest_and_live_costs_are_reported_separately(measured):
    params = dict(n_users=100, n_agents=1000, decisions_per_agent_per_day=7,
                  backtests_per_user_per_day=2, calls_per_decision=1,
                  model_mix={"gemini_3_1_pro_preview": 1.0},
                  trading_days_per_month=21, calendar_days_per_month=30,
                  infrastructure_usd_per_month=2000.0)
    b = lib.monthly_breakdown(measured, params)
    assert b["live_trading"]["usd"] > 0 and b["backtesting"]["usd"] > 0
    assert b["live_trading"]["usd"] != b["backtesting"]["usd"]
    total = (b["live_trading"]["usd"] + b["backtesting"]["usd"]
             + b["infrastructure"]["usd"])
    assert b["total_usd_per_month"] == pytest.approx(total)


def test_sensitivity_is_ordered_by_effect_size(measured):
    params = dict(n_users=100, n_agents=1000, decisions_per_agent_per_day=7,
                  backtests_per_user_per_day=2, calls_per_decision=1,
                  model_mix={"gemini_3_1_pro_preview": 1.0},
                  trading_days_per_month=21, calendar_days_per_month=30,
                  infrastructure_usd_per_month=2000.0)
    rows = lib.sensitivity(measured, params)
    swings = [r["swing_usd"] for r in rows]
    assert swings == sorted(swings, reverse=True)
    assert any("model_mix" in r["lever"] for r in rows)


# ---- the notebook itself ---------------------------------------------------

def _notebook_code_cells(path=_NOTEBOOK):
    with open(path) as fh:
        nb = json.load(fh)
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebook_matches_the_generator():
    assert os.path.exists(_NOTEBOOK), "run: python -m analysis.make_notebook"
    with open(_NOTEBOOK) as fh:
        committed = json.load(fh)
    assert committed == nbgen.build_notebook(), (
        "cost_model.ipynb is stale; regenerate with "
        "`python -m analysis.make_notebook`")


def test_notebook_unmeasured_params_are_all_blank():
    src = "\n".join(_notebook_code_cells())
    for param in nbgen.UNMEASURED_PARAMS:
        assert f"{param} = None" in src, (
            f"{param} must ship blank — a pre-filled value would be read as a "
            f"measurement")


def test_notebook_runs_end_to_end_with_blanks_and_says_what_it_needs(capsys):
    """The whole contract: completes, computes nothing, names every input."""
    cwd = os.getcwd()
    os.chdir(os.path.dirname(_ANALYSIS))
    try:
        ns: dict = {}
        for src in _notebook_code_cells():
            exec(compile(src, "<nb>", "exec"), ns)      # noqa: S102
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "CANNOT COMPUTE" in out
    for param in ("n_users", "backtests_per_user_per_day", "model_mix"):
        assert param in out
    # It must not have invented a monthly total.
    assert "TOTAL" not in out


def test_notebook_computes_once_params_are_supplied(capsys):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(_ANALYSIS))
    try:
        cells = _notebook_code_cells()
        ns: dict = {}
        exec(compile(cells[0], "<nb>", "exec"), ns)     # noqa: S102
        ns["PARAMS"] = dict(
            n_users=100, n_agents=1000, decisions_per_agent_per_day=7,
            backtests_per_user_per_day=2, calls_per_decision=1,
            model_mix={"nemotron_3_nano_30b": 1.0},
            trading_days_per_month=21, calendar_days_per_month=30,
            infrastructure_usd_per_month=2000.0)
        for src in cells[2:]:
            exec(compile(src, "<nb>", "exec"), ns)      # noqa: S102
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "CANNOT COMPUTE" not in out
    assert "TOTAL" in out and "Dominant lever" in out


# ---- companion documents ---------------------------------------------------

@pytest.mark.parametrize("name", ["QUESTIONS_FOR_ADVISOR.md",
                                  "INSTRUMENTATION_PATCH.md"])
def test_companion_docs_exist(name):
    assert os.path.exists(os.path.join(_ANALYSIS, name))


def test_instrumentation_patch_cites_the_three_live_call_sites():
    with open(os.path.join(_ANALYSIS, "INSTRUMENTATION_PATCH.md")) as fh:
        text = fh.read()
    repo = os.path.abspath(os.path.join(_ANALYSIS, "..", ".."))
    for path, line, needle in (
        ("dashboard/backend/infrastructure/llm/pipeline_runner.py", 473,
         "extract_token_usage"),
        ("dashboard/backend/infrastructure/llm/pipeline_runner.py", 388,
         "extract_token_usage"),
        ("dashboard/backend/domain/backtesting/portfolio_manager.py", 398,
         "_extract_token_usage"),
    ):
        assert f"{os.path.basename(path)}:{line}" in text, f"{path}:{line} not cited"
        with open(os.path.join(repo, path)) as fh:
            assert needle in fh.readlines()[line - 1], (
                f"{path}:{line} no longer calls {needle}")


def test_questions_doc_covers_the_required_topics():
    with open(os.path.join(_ANALYSIS, "QUESTIONS_FOR_ADVISOR.md")) as fh:
        text = fh.read().lower()
    for topic in ("backtest", "model mix", "pipeline depth", "paper trading",
                  "orchestration/finagents"):
        assert topic in text, f"missing topic: {topic}"
