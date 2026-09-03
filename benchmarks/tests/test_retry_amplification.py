"""Phase 21: the retry probe, its transport counter, and the report.

The claim these guard is narrow and specific: a retry recorded here is a retry
because the condition that causes retries fired, and a request counted here is
a request that left the process. Phase 20 inferred both from call counts and
got a number it could not defend.
"""

import io
import json
import os
import subprocess
import sys

import httpx
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.abspath(os.path.join(_HERE, ".."))
_ANALYSIS = os.path.join(_BENCH, "analysis")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from analysis import retry_amplification as ra  # noqa: E402
from analysis.tier_check import check_report  # noqa: E402

REPORT = os.path.join(_ANALYSIS, "RETRY_AMPLIFICATION.md")
RESULTS = os.path.join(_BENCH, "results")


def _arms():
    out = {}
    for label, fname, _ in ra_arms():
        p = os.path.join(RESULTS, fname)
        if os.path.exists(p):
            with io.open(p, encoding="utf-8") as fh:
                out[label] = json.load(fh)
    return out


def ra_arms():
    from analysis import make_retry_report as mk
    return mk.ARMS


@pytest.fixture(scope="module")
def arms():
    a = _arms()
    if not a:
        pytest.skip("no arm results committed")
    return a


# --------------------------------------------------------- the seed-DB guard --

def test_guard_refuses_when_database_path_is_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    with pytest.raises(SystemExit):
        ra.guard_seed_db()


def test_guard_refuses_when_pointed_at_the_seed_db(monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", ra.SEED_DB)
    with pytest.raises(SystemExit):
        ra.guard_seed_db()


def test_the_seed_db_is_currently_intact():
    assert ra.seed_db_sha256() == ra.SEED_DB_SHA256


def test_every_arm_left_the_seed_db_untouched(arms):
    for label, d in arms.items():
        assert d["seed_db_sha256_before"] == ra.SEED_DB_SHA256, label
        assert d["seed_db_sha256_after"] == ra.SEED_DB_SHA256, label


# ------------------------------------------------------- the transport layer --

def _mock_client(statuses):
    """An Anthropic client whose transport returns `statuses` in order."""
    seq = list(statuses)
    calls = {"n": 0}

    def handler(request):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        code = seq[i]
        if code != 200:
            return httpx.Response(code, json={"error": "x"})
        return httpx.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "x",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2}})

    import anthropic
    return anthropic.Anthropic(
        api_key="k",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_transport_counter_sees_sdk_retries_that_llm_calls_cannot():
    """The load-bearing test. ATL counts one attempt; the wire saw three."""
    rec = ra.RetryRecorder()
    tr = ra.TransportRecorder(key_fn=rec.current_key)
    ra.install_transport_counter(tr)
    ra.install_client_timeout(30.0)
    try:
        c = _mock_client([529, 529, 200])
        rec.new_decision()
        rec.attempt_started()
        c.messages.create(model="x", max_tokens=8,
                          messages=[{"role": "user", "content": "hi"}])
        rec.attempt_finished(0.1)
    finally:
        ra.restore()

    s = ra.summarise(rec, "x", tr)
    assert s["total_attempts"] == 1, "ATL's loop ran once"
    assert s["transport_requests"] == 3, "three requests reached the provider"
    assert s["sdk_level_retries"] == 2
    assert s["transport_requests_per_atl_attempt"] == 3.0
    assert s["status_counts"] == {"529": 2, "200": 1}


def test_transport_rows_are_attributed_to_the_attempt_that_issued_them():
    rec = ra.RetryRecorder()
    tr = ra.TransportRecorder(key_fn=rec.current_key)
    ra.install_transport_counter(tr)
    ra.install_client_timeout(30.0)
    try:
        c = _mock_client([200])
        rec.new_decision()
        rec.new_decision()          # decision_index == 1
        rec.attempt_started()
        c.messages.create(model="x", max_tokens=8,
                          messages=[{"role": "user", "content": "hi"}])
        rec.attempt_finished(0.1)
    finally:
        ra.restore()
    assert [(r["decision_index"], r["attempt_index"]) for r in tr.requests] \
        == [(1, 0)]


def test_sdk_retry_index_is_read_from_the_header_not_inferred():
    rec = ra.RetryRecorder()
    tr = ra.TransportRecorder(key_fn=rec.current_key)
    ra.install_transport_counter(tr)
    ra.install_client_timeout(30.0)
    try:
        c = _mock_client([529, 200])
        rec.new_decision()
        rec.attempt_started()
        c.messages.create(model="x", max_tokens=8,
                          messages=[{"role": "user", "content": "hi"}])
        rec.attempt_finished(0.1)
    finally:
        ra.restore()
    assert [r["sdk_retry_count"] for r in tr.requests] == [0, 1]
    assert [r["is_sdk_retry"] for r in tr.requests] == [False, True]


def test_transport_errors_are_recorded_and_reraised():
    def boom(request):
        raise httpx.ConnectError("refused")

    rec = ra.RetryRecorder()
    tr = ra.TransportRecorder(key_fn=rec.current_key)
    ra.install_transport_counter(tr)
    ra.install_client_timeout(30.0)
    try:
        import anthropic
        c = anthropic.Anthropic(
            api_key="k",
            http_client=httpx.Client(transport=httpx.MockTransport(boom)))
        with pytest.raises(Exception):
            c.messages.create(model="x", max_tokens=8,
                              messages=[{"role": "user", "content": "hi"}])
    finally:
        ra.restore()
    assert tr.summary()["transport_errors"].get("ConnectError")
    assert all(r["status"] is None for r in tr.requests)


# ----------------------------------------------------------- the timeout fix --

def test_backend_clients_carry_no_timeout_of_their_own():
    """The reason the fix is needed. If this ever fails, the backend grew a
    timeout and the probe should stop installing one."""
    import inspect
    from dashboard.backend.infrastructure.llm.providers import (
        openrouter, commonstack, anthropic_native)
    for mod in (openrouter, commonstack, anthropic_native):
        src = inspect.getsource(mod)
        assert "timeout" not in src.lower(), (
            f"{mod.__name__} now sets a timeout; revisit the probe")


def test_installed_timeout_replaces_the_sdk_default():
    cfg = ra.install_client_timeout(45.0)
    try:
        import anthropic
        c = anthropic.Anthropic(api_key="k")
    finally:
        ra.restore()
    assert c.timeout.read == 45.0
    assert cfg["sdk_default_max_retries"] == 2
    assert "600" in cfg["sdk_default_timeout"]
    assert cfg["clients_patched"] == 1


def test_restore_puts_httpx_and_anthropic_back():
    import anthropic
    send_before = httpx.Client.send
    init_before = anthropic.Anthropic.__init__
    tr = ra.TransportRecorder()
    ra.install_transport_counter(tr)
    ra.install_client_timeout(10.0)
    assert httpx.Client.send is not send_before
    ra.restore()
    assert httpx.Client.send is send_before
    assert anthropic.Anthropic.__init__ is init_before


# ---------------------------------------------------------- the retry loop ---

def test_only_a_no_text_attributeerror_counts_as_the_trigger():
    """The loop advances on exactly one condition; the recorder must agree."""
    rec = ra.RetryRecorder()
    rec.new_decision()
    rec.attempt_started()
    rec.attempt_finished(1.0)
    rec.record_outcome("no_text_content")
    rec.attempt_started()
    rec.attempt_finished(1.0)
    rec.record_outcome("text_returned")
    s = ra.summarise(rec, "m")
    assert s["attempts_per_decision"] == 2.0
    assert s["retries"] == 1
    assert s["triggers"] == {"no_text_content": 1}


def test_the_fifth_attempt_is_flagged_as_the_rescue():
    rec = ra.RetryRecorder()
    rec.new_decision()
    for _ in range(ra.MAX_ATTEMPTS_PER_DECISION):
        rec.attempt_started()
        rec.attempt_finished(1.0)
    assert ra.summarise(rec, "m")["rescue_calls"] == 1


def test_bars_without_a_request_do_not_dilute_the_rate():
    rec = ra.RetryRecorder()
    rec.new_decision()
    rec.attempt_started()
    rec.attempt_finished(1.0)
    rec.new_decision()          # a bar that issued nothing
    s = ra.summarise(rec, "m")
    assert s["decisions_with_requests"] == 1
    assert s["attempts_per_decision"] == 1.0


# ------------------------------------------------------------ arm invariants --

def test_every_recorded_retry_was_triggered_not_inferred(arms):
    for label, d in arms.items():
        s = d["summary"]
        if s["retries"]:
            assert s["triggers"], f"{label} counted retries with no trigger"


def test_no_arm_exceeds_the_loops_own_attempt_cap(arms):
    for label, d in arms.items():
        worst = max((int(k) for k in d["summary"]["attempts_histogram"]),
                    default=0)
        assert worst <= ra.MAX_ATTEMPTS_PER_DECISION, label


def test_every_arm_installed_a_client_timeout(arms):
    for label, d in arms.items():
        cfg = d.get("client_timeout")
        assert cfg, f"{label} ran without the timeout fix"
        assert cfg["read_timeout_seconds"] < ra_sdk_default_read(), label


def ra_sdk_default_read():
    from anthropic import _base_client
    return _base_client.DEFAULT_TIMEOUT.read


def test_trigger_count_equals_retries_plus_exhausted_decisions(arms):
    """The accounting identity. A failure either causes a retry or exhausts
    the loop; there is no third outcome, so these must balance exactly."""
    from analysis.make_retry_report import _exhausted
    for label, d in arms.items():
        s = d["summary"]
        failures = sum(s["triggers"].values())
        assert failures == s["retries"] + _exhausted(d), label


def test_attempts_and_decisions_reconcile_with_the_histogram(arms):
    for label, d in arms.items():
        s = d["summary"]
        hist = {int(k): v for k, v in s["attempts_histogram"].items()}
        assert sum(hist.values()) == s["decisions_with_requests"], label
        assert sum(k * v for k, v in hist.items()) == s["total_attempts"], label


def test_log_and_instrumentation_agree_on_the_trigger_count(arms):
    """The cross-check Phase 20 could not make. The backend's own print and
    the instrumented exception must count the same failures."""
    p = os.path.join(RESULTS, "retry_content_types.json")
    if not os.path.exists(p):
        pytest.skip("no content-type extract committed")
    with io.open(p, encoding="utf-8") as fh:
        ct = json.load(fh)
    from_log = sum(ct["content_types_total"].values())
    from_probe = sum(sum(d["summary"]["triggers"].values())
                     for d in arms.values())
    assert from_log == from_probe


def test_only_one_response_shape_ever_triggered_a_retry(arms):
    p = os.path.join(RESULTS, "retry_content_types.json")
    if not os.path.exists(p):
        pytest.skip("no content-type extract committed")
    with io.open(p, encoding="utf-8") as fh:
        ct = json.load(fh)
    assert ct["distinct_shapes"] == 1
    assert "thinking" in next(iter(ct["content_types_total"]))


def test_transport_count_is_at_least_the_atl_attempt_count(arms):
    """Every ATL attempt issues at least one request; SDK retries add more."""
    for label, d in arms.items():
        s = d["summary"]
        if s.get("transport_requests") is None:
            continue
        assert s["transport_requests"] >= s["total_attempts"], label


# ----------------------------------------------------------------- report ----

def test_report_is_not_stale():
    from analysis import make_retry_report as mk
    with io.open(REPORT, encoding="utf-8") as fh:
        assert fh.read() == mk.build(), (
            "RETRY_AMPLIFICATION.md is stale; regenerate with "
            "`python benchmarks/analysis/make_retry_report.py`")


def test_report_is_fully_tagged():
    with io.open(REPORT, encoding="utf-8") as fh:
        assert check_report(fh.read())["ok"]


def test_report_names_both_retry_layers():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "two retry layers" in t
    assert "undercounts" in t


def test_report_states_the_trigger_is_the_only_one():
    with io.open(REPORT, encoding="utf-8") as fh:
        t = fh.read()
    assert "No text content" in t
    assert "cannot" in t and "abort" in t


def test_generator_runs_as_a_script():
    r = subprocess.run(
        [sys.executable, os.path.join(_ANALYSIS, "make_retry_report.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
