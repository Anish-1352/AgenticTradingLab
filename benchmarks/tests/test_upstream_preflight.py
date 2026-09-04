"""The pre-flight, and the report built from it.

Findings in this project have gone stale twice by being written against a tree
that moved. These tests guard the checker that catches it, not the findings.
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

from analysis import upstream_preflight as pf  # noqa: E402
from analysis.tier_check import check_report  # noqa: E402

REPORT = os.path.join(_ANALYSIS, "UPSTREAM_AND_OUTPUT_CAP.md")
PREFLIGHT = os.path.join(_BENCH, "results", "preflight.json")


@pytest.fixture(scope="module")
def pre():
    if not os.path.exists(PREFLIGHT):
        pytest.skip("no preflight committed")
    with io.open(PREFLIGHT, encoding="utf-8") as fh:
        return json.load(fh)


def test_every_probe_states_which_way_it_holds():
    for p in pf.PROBES:
        assert p["holds_when"] in ("present", "absent"), p["id"]
        assert p["claim"] and p["paths"] and p["pattern"]


def test_probe_ids_are_unique():
    ids = [p["id"] for p in pf.PROBES]
    assert len(ids) == len(set(ids))


def test_a_present_probe_holds_when_found_and_an_absent_one_does_not():
    """The polarity is the whole point; a flipped sign would silently invert
    every conclusion in the report."""
    found = {"id": "x", "claim": "c", "holds_when": "present",
             "found_upstream": True}
    assert (found["holds_when"] == "present") == found["found_upstream"]
    for p in pf.PROBES:
        r = pf.run_probe(p)
        expected = (r["found_upstream"] if p["holds_when"] == "present"
                    else not r["found_upstream"])
        assert r["still_holds"] == expected, p["id"]


def test_test_matches_do_not_count_as_production_code():
    """An assertion in the upstream suite proves the behaviour is tested, not
    that production grew the feature."""
    r = pf.run_probe({
        "id": "t", "claim": "c", "holds_when": "absent",
        "paths": ["dashboard/backend/tests/"],
        "pattern": r"OPENROUTER_REASONING_EFFORT",
    })
    assert r["n_matches"] == 0


def test_the_two_superseded_findings_are_recorded(pre):
    by_id = {p["id"]: p for p in pre["probes"]}
    assert by_id["reasoning_off_rescue"]["still_holds"] is False
    assert by_id["attempt_level_column"]["still_holds"] is False


def test_the_findings_this_work_relies_on_still_hold(pre):
    by_id = {p["id"]: p for p in pre["probes"]}
    for k in ("retry_loop_unchanged", "retry_trigger_unchanged",
              "per_call_usage_table", "structured_outputs_unused",
              "backtest_client_has_no_timeout"):
        assert by_id[k]["still_holds"], f"{k} was superseded; revisit the report"


def test_every_cited_anchor_was_located_upstream(pre):
    for a in pre["citation_anchors"]:
        assert a["upstream_line"], f"{a['describes']} no longer found upstream"


def test_preflight_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "upstream_preflight.py")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------ report --

def test_report_is_not_stale():
    from analysis import make_phase22_report as mk
    with io.open(REPORT, encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "UPSTREAM_AND_OUTPUT_CAP.md is stale; regenerate with "
            "`python benchmarks/analysis/make_phase22_report.py`")


def test_report_is_fully_tagged():
    with io.open(REPORT, encoding="utf-8") as fh:
        assert check_report(fh.read())["ok"]


def test_report_says_the_clamp_does_not_work():
    """The negative result is the finding; it must not get quietly softened."""
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "does not work" in t
    assert "advisory" in t


def test_report_refuses_structured_outputs():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "Do not\nimplement it" in t or "Do not implement it" in t


def test_report_marks_the_quality_delta_underpowered():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "underpowered" in t
    assert "one window" in t.lower()


def test_report_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_phase22_report.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
