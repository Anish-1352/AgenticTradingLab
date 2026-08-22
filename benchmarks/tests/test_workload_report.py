"""The generated report: staleness, tier tagging, and scope.

Staleness is checked byte-for-byte against a fresh build, so the document
cannot drift from the data it claims to report. The scope check exists because
this report has an explicit exclusion list — prefix caching, bar-skipping,
self-hosting and fallback rates are all things a cost report naturally reaches
for, and all four are things this one must not claim.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import make_workload_report as mk        # noqa: E402
from analysis.tier_check import check_report           # noqa: E402

_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
REPORT = os.path.join(_ANALYSIS, "WORKLOAD_AND_API_CEILING.md")


@pytest.fixture(scope="module")
def built():
    return mk.build()


def test_report_exists():
    assert os.path.exists(REPORT), (
        "WORKLOAD_AND_API_CEILING.md missing; generate with "
        "`python benchmarks/analysis/make_workload_report.py`")


def test_report_is_not_stale(built):
    with open(REPORT, "r", encoding="utf-8") as fh:
        on_disk = fh.read()
    assert on_disk == built, (
        "WORKLOAD_AND_API_CEILING.md is stale or hand-edited; regenerate with "
        "`python benchmarks/analysis/make_workload_report.py`")


def test_every_numeric_line_is_tagged(built):
    r = check_report(built)
    assert r["ok"], r["violations"][:5]


def test_check_mode_writes_nothing(tmp_path):
    out = tmp_path / "x.md"
    assert mk.main(["--check", "--out", str(out)]) == 0
    assert not out.exists()


# ------------------------------------------------------------ the content ---

def test_both_workloads_are_present_and_separate(built):
    assert "EXPERIMENT" in built and "USER PLATFORM" in built
    assert "LAB-CONTROLLED" in built and "USER-DETERMINED" in built


def test_the_independent_variable_is_marked_off_limits(built):
    assert "the independent variable" in built
    assert "makes it a different experiment" in built


def test_both_cadences_are_reported_side_by_side(built):
    """The brief keeps the hourly figures; the ratio is the finding."""
    assert "hourly (3600s)" in built and "Nof1 (150s)" in built
    assert "24x" in built or "22x" in built


def test_the_power_question_is_asked_not_answered(built):
    assert "not answered here" in built
    assert "effect size" in built


def test_fewer_models_is_shown_as_a_range(built):
    assert "range, not a saving" in built
    assert "raises the blended rate" in built


def test_the_leaderboard_recurring_cost_is_zero_by_pause_not_absence(built):
    """Upstream now ships the scheduler with its cron commented out, so the
    $0 stands but its reason changed — and the counterfactual is priceable."""
    assert "Recurring cost today is `$0`" in built
    assert "deliberate pause" in built
    assert "daily-leaderboard.yml" in built
    assert "per month if re-enabled" in built


def test_the_daily_window_is_not_assumed_to_be_the_contest_window(built):
    assert "rolling **one-day** window" in built


def test_the_compound_effect_is_reported_as_running_backwards(built):
    assert "compound effect runs backwards" in built
    assert "inflates" in built.lower()


def test_the_parse_rate_bound_is_stated_not_rounded_to_perfect(built):
    assert "rule of three" in built
    assert "not\nat zero" in built or "not at zero" in built


def test_no_end_to_end_routing_saving_is_claimed(built):
    assert "No end-to-end saving is stated" in built


def test_caching_saving_is_refused(built):
    assert "cannot be sized" in built
    assert "no telemetry" in built


def test_the_dropdown_is_named_as_a_product_decision(built):
    assert "not an engineering change" in built.lower()
    assert "advisor's call" in built


def test_the_residual_is_reported_beside_the_saving(built):
    """60-70% off $600k is still ~$200k — the point of the whole report."""
    assert "order of magnitude" in built
    assert "193x" in built


# ------------------------------------------------------------------ scope ---

# Phrases that would only appear if a saving were being CLAIMED for an
# excluded lever. Bare mentions are fine and in fact required — the report has
# to say why each is excluded, and the workload table has to say that skipping
# bars is available to the experiment and not to the platform.
EXCLUDED_CLAIMS = (
    "prefix cache saves",
    "prefix caching saves",
    "saving from prefix",
    "by staggering",
    "by skipping bars",
    "self-hosting saves",
    "self-hosted saving",
    "fallback rate of",
    "estimated fallback",
)


def test_the_report_claims_no_saving_for_an_excluded_lever(built):
    low = built.lower()
    for phrase in EXCLUDED_CLAIMS:
        assert phrase not in low, f"out-of-scope claim: {phrase!r}"


def test_skipping_bars_is_mentioned_only_as_unavailable_to_the_platform(built):
    """It must appear — as something the platform cannot do — but never as a
    modelled saving."""
    assert "skip bars to save money | yes | no" in built
    assert "Experiment-only" in built


def test_the_exclusions_are_stated_rather_than_silent(built):
    assert "deliberately not modelled" in built
    for topic in ("Prefix caching", "Self-hosting", "fallback-rate"):
        assert topic in built


def test_prefix_caching_is_named_as_zero_not_omitted(built):
    assert "zero cached tokens" in built


def test_the_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_workload_report.py"),
         "--check"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "all tagged" in r.stdout
