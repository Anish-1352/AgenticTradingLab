"""Unit tests for fixture construction.

Uses the stub tokenizer, so no transformers install and no model download.
The properties under test are tokenizer-independent: exact length,
determinism, and the overlap gap between the two sets.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.fixtures import (  # noqa: E402
    StubTokenizer,
    build_fixture,
    build_low_overlap,
    build_shared_prefix,
    load_fixture,
    save_fixture,
)

CTX = 512  # smaller than the 2620 default to keep tests fast


def test_shared_prefix_is_exactly_context_tokens():
    """Context length is a controlled variable; 'about 2620' is not good enough."""
    fx = build_shared_prefix(StubTokenizer(), 6, context_tokens=CTX, seed=1)
    counts = fx.token_counts()
    assert counts == [CTX] * 6


def test_low_overlap_is_exactly_context_tokens():
    fx = build_low_overlap(StubTokenizer(), 6, context_tokens=CTX, seed=1)
    assert fx.token_counts() == [CTX] * 6


def test_shared_prefix_actually_shares_a_long_prefix():
    fx = build_shared_prefix(StubTokenizer(), 8, context_tokens=CTX, seed=1,
                             tail_tokens=24)
    common = fx.common_prefix_tokens()
    # At least the intended shared region; possibly a little more, because the
    # per-ticker tails begin with identical boilerplate. Measured, not assumed.
    assert common >= CTX - 24
    assert common < CTX  # the requests are not identical


def test_low_overlap_shares_almost_nothing():
    fx = build_low_overlap(StubTokenizer(), 8, context_tokens=CTX, seed=1)
    assert fx.common_prefix_tokens() < 20


def test_the_gap_between_sets_is_the_measurement():
    """shared_prefix and low_overlap must differ sharply in shared prefix.

    If they did not, running both would measure the same thing twice and the
    prefix-caching result would be an artifact of the fixture.
    """
    shared = build_shared_prefix(StubTokenizer(), 8, context_tokens=CTX, seed=1)
    low = build_low_overlap(StubTokenizer(), 8, context_tokens=CTX, seed=1)
    assert shared.common_prefix_tokens() > 10 * max(low.common_prefix_tokens(), 1)


def test_requests_are_distinct_within_each_set():
    """v1 used one identical prompt for every request, which is a ~100% cache
    hit rate by construction."""
    for builder in (build_shared_prefix, build_low_overlap):
        fx = builder(StubTokenizer(), 6, context_tokens=CTX, seed=1)
        ids = [tuple(r["prompt_token_ids"]) for r in fx.requests]
        assert len(set(ids)) == len(ids)


def test_determinism_same_seed_same_hash():
    a = build_shared_prefix(StubTokenizer(), 5, context_tokens=CTX, seed=99)
    b = build_shared_prefix(StubTokenizer(), 5, context_tokens=CTX, seed=99)
    assert a.sha256() == b.sha256()


def test_different_seed_changes_hash():
    a = build_shared_prefix(StubTokenizer(), 5, context_tokens=CTX, seed=99)
    c = build_shared_prefix(StubTokenizer(), 5, context_tokens=CTX, seed=100)
    assert a.sha256() != c.sha256()


def test_hash_covers_context_length():
    a = build_low_overlap(StubTokenizer(), 4, context_tokens=CTX, seed=7)
    b = build_low_overlap(StubTokenizer(), 4, context_tokens=CTX + 8, seed=7)
    assert a.sha256() != b.sha256()


def test_meta_records_stub_tokenizer():
    """The runner refuses stub-built fixtures; the flag has to survive to disk."""
    meta = build_low_overlap(StubTokenizer(), 3, context_tokens=CTX, seed=1).to_meta()
    assert meta["tokenizer_is_stub"] is True
    assert meta["token_counts_all_equal"] is True
    assert meta["common_prefix_fraction"] == pytest.approx(
        meta["common_prefix_tokens"] / CTX
    )


def test_save_and_load_roundtrip(tmp_path):
    fx = build_shared_prefix(StubTokenizer(), 4, context_tokens=CTX, seed=3)
    paths = save_fixture(fx, str(tmp_path))
    loaded = load_fixture(paths["full"])
    assert loaded.sha256() == fx.sha256()
    assert loaded.token_counts() == fx.token_counts()
    with open(paths["meta"]) as fh:
        meta = json.load(fh)
    assert meta["sha256"] == fx.sha256()


def test_load_rejects_tampered_fixture(tmp_path):
    """The sha256 in the file is an integrity check, not decoration."""
    fx = build_low_overlap(StubTokenizer(), 3, context_tokens=CTX, seed=3)
    paths = save_fixture(fx, str(tmp_path))
    with open(paths["full"]) as fh:
        data = json.load(fh)
    data["requests"][0]["prompt_token_ids"][0] += 1
    with open(paths["full"], "w") as fh:
        json.dump(data, fh)
    with pytest.raises(ValueError, match="integrity check"):
        load_fixture(paths["full"])


def test_tail_must_be_smaller_than_context():
    with pytest.raises(ValueError):
        build_shared_prefix(StubTokenizer(), 2, context_tokens=32, tail_tokens=32)


def test_build_fixture_rejects_unknown_name():
    with pytest.raises(KeyError):
        build_fixture("does_not_exist", StubTokenizer(), 2, context_tokens=CTX)
