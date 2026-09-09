"""Phase 23: the schema probe, the captured pairs, and the report.

The claim these guard is that the schema was established by CALLING the
shipped code, not by reading it. A specification written from a reading is
exactly the artifact this phase was asked not to produce.
"""

import io
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, ".."))
_ANALYSIS = os.path.join(_BENCH, "analysis")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from analysis import pair_analysis as pa  # noqa: E402
from analysis import schema_probe as sp  # noqa: E402
from analysis.tier_check import check_report  # noqa: E402

REPORT = os.path.join(_ANALYSIS, "DECISION_SCHEMA.md")
RESULTS = os.path.join(_BENCH, "results")


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not committed")
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def probe():
    return _load("schema_probe.json")


@pytest.fixture(scope="module")
def pairs():
    return _load("pair_analysis.json")


# ------------------------------------------------------------- provenance --

def test_the_probe_records_which_tree_it_ran_against(probe):
    """The first run used a checkout 533 commits behind and reported parser
    behaviour that had since been fixed. The revision is the guard."""
    assert probe["tree"]["head"]
    assert probe["tree"]["is_origin_main"], (
        "probe was run against a tree that is not origin/main; regenerate")


def test_the_seed_db_was_untouched(probe):
    assert probe["seed_db_sha256_before"] == sp.SEED_DB_SHA256
    assert probe["seed_db_sha256_after"] == sp.SEED_DB_SHA256


def test_guard_refuses_the_seed_db(monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", sp.SEED_DB)
    with pytest.raises(SystemExit):
        sp.guard_seed_db()


def test_guard_refuses_an_unset_path(monkeypatch):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    with pytest.raises(SystemExit):
        sp.guard_seed_db()


# ------------------------------------------------------------ the schema ---

def test_every_probe_case_has_a_recorded_outcome(probe):
    for r in probe["pipeline_parser"]:
        assert r["accepted"] in (True, False)
        assert r["group"] and r["case"]


def test_an_empty_envelope_is_a_decision_not_a_failure(probe):
    """Phase 16's finding, re-checked: it has been fixed upstream."""
    by = {r["case"]: r for r in probe["pipeline_parser"]}
    for case in ("orders, empty", "actions, empty", "risk_actions, empty"):
        assert by[case]["accepted"], case
        assert by[case]["parsed"] == {"actions": []}, case


def test_unknown_side_is_coerced_rather_than_rejected(probe):
    by = {r["case"]: r for r in probe["pipeline_parser"]}
    r = by["side=short (unknown)"]
    assert r["accepted"]
    assert r["parsed"]["actions"][0]["action"] == "hold"


def test_a_non_numeric_confidence_is_the_one_hard_failure(probe):
    by = {r["case"]: r for r in probe["pipeline_parser"]}
    r = by["confidence='high'"]
    assert not r["accepted"]
    assert "ValueError" in (r["error"] or "")


def test_one_parser_emits_more_than_one_action_shape(probe):
    """The finding that blocks a single training target."""
    assert probe["envelope_output_shapes"]["distinct_shapes"] > 1


def test_the_two_parsers_disagree(probe):
    ag = probe["parser_agreement"]
    assert ag["compared"] > 0
    assert ag["agreement_rate"] < 0.5
    assert ag["accepted_by_pipeline_only"] > 0


def test_templates_agree_with_each_other(probe):
    t = probe["templates"]
    assert t["all_templates_agree_on_final"]
    assert t["distinct_final_envelopes"] == ["orders"]


def test_templates_ask_for_fields_no_parser_reads(probe):
    fields = probe["template_fields_read_by_pipeline_parser"]
    assert fields["order_type"] is False
    assert fields["limit_price"] is False
    assert fields["symbol"] is True


# ------------------------------------------------------------- the pairs ---

def test_pairs_came_from_more_than_one_path_and_regime(pairs):
    cr = pairs["crossed"]
    assert cr["fully_crossed"], (
        "regime and path are confounded; the regime column cannot be read")
    assert len(cr["regimes"]) >= 3


def test_enough_pairs_to_answer_the_question(pairs):
    assert pairs["n_pairs"] >= 50


def test_most_billed_output_is_never_shown(pairs):
    ts = pairs["token_split"]
    assert ts["billed_output_tokens"] > ts["visible_text_tokens_est"]
    assert ts["hidden_share_est"] > 0.5


def test_the_no_text_failure_was_reproduced(pairs):
    assert pairs["no_text_responses"] > 0
    assert any("thinking" in k and "text" not in k
               for k in pairs["content_type_shapes"])


def test_the_pipeline_path_outperformed_the_single_prompt_path(pairs):
    bp = pairs["by_path"]
    assert bp["pipeline"]["parse_rate"] > bp["single_prompt"]["parse_rate"]
    assert bp["pipeline"]["ceiling_rate"] < bp["single_prompt"]["ceiling_rate"]


def test_intermediate_steps_are_not_counted_as_conversion_failures(pairs):
    """facts and signals are correctly declined; only the last step converts."""
    c = pairs["conversion"]
    assert c["pipeline_final_conversion_rate"] is not None
    assert c["pipeline_final_steps"] < pairs["by_path"]["pipeline"]["n"]


def test_token_split_is_arithmetically_consistent(pairs):
    ts = pairs["token_split"]
    assert (ts["visible_text_tokens_est"] + ts["not_shown_in_the_response_est"]
            == ts["billed_output_tokens"])


def test_field_usage_separates_read_from_ignored(pairs):
    fu = pairs["field_usage"]
    assert set(fu["keys_asked_for_but_ignored"]) <= pa.FIELDS_ASKED_BUT_IGNORED
    assert set(fu["keys_read_by_parser"]) <= pa.FIELDS_READ


# ------------------------------------------------------------- the report --

def test_report_is_not_stale():
    from analysis import make_schema_report as mk
    with io.open(REPORT, encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "DECISION_SCHEMA.md is stale; regenerate with "
            "`python benchmarks/analysis/make_schema_report.py`")


def test_report_is_fully_tagged():
    with io.open(REPORT, encoding="utf-8") as fh:
        assert check_report(fh.read())["ok"]


def test_report_says_there_is_not_one_schema():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "more than one schema" in t
    assert "do not agree" in t


def test_report_does_not_claim_schema_validity_implies_good_trading():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "must not be substituted for it" in t
    assert "not a claim about decision quality" in t


def test_report_gives_the_hold_recommendation():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "not an empty array" in t


def test_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_schema_report.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
