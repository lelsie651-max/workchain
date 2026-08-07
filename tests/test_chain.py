import hashlib

import pytest

from evidence_core.chain import (
    ZERO_HASH,
    compute_chain_hash,
    compute_content_hash,
    compute_record_digest,
    is_valid_hash,
)


def _base_record() -> dict:
    return {
        "captured_at": 1723000000,
        "content_hash": "a" * 64,
        "evidence_id": "ev-1",
        "media_type": "text",
        "occurred_at": 1722990000,
        "seq": 1,
        "source_hint": "feishu",
    }


def test_compute_content_hash_matches_known_empty_sha256():
    assert (
        compute_content_hash(b"")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_compute_content_hash_treats_string_and_bytes_equally():
    assert compute_content_hash("abc") == compute_content_hash(b"abc")


def test_compute_content_hash_uses_utf8_for_chinese_text():
    assert compute_content_hash("中文") == compute_content_hash("中文".encode("utf-8"))


def test_compute_content_hash_rejects_unsupported_types():
    with pytest.raises(TypeError):
        compute_content_hash(123)


def test_compute_record_digest_ignores_non_digest_fields_per_d2():
    base = _base_record()
    enriched = {
        **base,
        "slot_requester": "act-1",
        "slot_due": 1723990000,
        "plain_summary": "do it",
        "caveats": ["ambiguous"],
        "kind": "request",
        "thread_id": "thr-1",
    }

    assert compute_record_digest(base) == compute_record_digest(enriched)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("captured_at", 1723000001),
        ("content_hash", "b" * 64),
        ("evidence_id", "ev-2"),
        ("media_type", "image"),
        ("occurred_at", 1722990001),
        ("seq", 2),
        ("source_hint", "wechat"),
    ],
)
def test_compute_record_digest_changes_when_any_digest_field_changes(
    field: str, changed_value
):
    base = _base_record()
    changed = {**base, field: changed_value}

    assert compute_record_digest(base) != compute_record_digest(changed)


@pytest.mark.parametrize(
    ("prev_hash", "record_digest"),
    [
        ("abc", "a" * 64),
        ("A" * 64, "a" * 64),
        ("g" * 64, "a" * 64),
        ("a" * 64, "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789"),
        ("a" * 64, "z" * 64),
    ],
)
def test_compute_chain_hash_rejects_invalid_hash_inputs(prev_hash: str, record_digest: str):
    with pytest.raises(ValueError):
        compute_chain_hash(prev_hash, record_digest)


def test_zero_hash_is_valid():
    assert is_valid_hash(ZERO_HASH) is True


def test_same_digest_with_different_prev_hashes_produce_different_chain_hashes():
    record_digest = compute_record_digest(_base_record())

    left = compute_chain_hash(ZERO_HASH, record_digest)
    right = compute_chain_hash("1" * 64, record_digest)

    assert left != right


def test_hash_functions_are_deterministic_across_repeated_calls():
    record = _base_record()
    content_results = {compute_content_hash("abc") for _ in range(100)}
    digest_results = {compute_record_digest(record) for _ in range(100)}
    chain_results = {
        compute_chain_hash(ZERO_HASH, compute_record_digest(record)) for _ in range(100)
    }

    assert len(content_results) == 1
    assert len(digest_results) == 1
    assert len(chain_results) == 1


def test_compute_chain_hash_matches_direct_sha256_composition():
    record_digest = compute_record_digest(_base_record())
    expected = hashlib.sha256((ZERO_HASH + record_digest).encode("ascii")).hexdigest()

    assert compute_chain_hash(ZERO_HASH, record_digest) == expected
