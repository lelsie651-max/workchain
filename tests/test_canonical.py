import pytest

from evidence_core.canonical import DIGEST_FIELDS, build_digest_payload, canonical_json


def test_canonical_json_is_identical_for_equivalent_dicts():
    left = {
        "source_hint": "chat",
        "captured_at": 1723000000,
        "nested": {"b": 2, "a": 1},
    }
    right = {
        "nested": {"a": 1, "b": 2},
        "captured_at": 1723000000,
        "source_hint": "chat",
    }

    assert canonical_json(left) == canonical_json(right)


def test_canonical_json_keeps_chinese_unescaped():
    payload = {"captured_at": 1723000000, "text": "中文内容"}

    result = canonical_json(payload)

    assert "中文内容".encode("utf-8") in result
    assert b"\\u4e2d" not in result


def test_canonical_json_keeps_none_as_null_without_dropping_keys():
    payload = {"captured_at": 1723000000, "source_hint": None}

    result = canonical_json(payload)

    assert b'"source_hint":null' in result


def test_canonical_json_rejects_float_timestamp():
    payload = {"captured_at": 1723000000.5}

    with pytest.raises(TypeError, match="captured_at"):
        canonical_json(payload)


def test_build_digest_payload_drops_extra_keys_and_fills_missing_keys():
    record = {
        "captured_at": 1723000000,
        "content_hash": "abc",
        "evidence_id": "ev-1",
        "extra": "discard me",
        "seq": 3,
    }

    result = build_digest_payload(record)

    assert list(result.keys()) == DIGEST_FIELDS
    assert result == {
        "captured_at": 1723000000,
        "content_hash": "abc",
        "evidence_id": "ev-1",
        "media_type": None,
        "occurred_at": None,
        "seq": 3,
        "source_hint": None,
    }


def test_canonical_json_sorts_nested_dict_keys():
    payload = {
        "captured_at": 1723000000,
        "nested": {"z": 1, "a": 2, "mid": {"y": 3, "b": 4}},
    }

    result = canonical_json(payload)

    assert result == (
        b'{"captured_at":1723000000,"nested":{"a":2,"mid":{"b":4,"y":3},"z":1}}'
    )


def test_canonical_json_rejects_float_for_int_field():
    assert canonical_json({"seq": 1}) == b'{"seq":1}'

    with pytest.raises(TypeError, match="seq"):
        canonical_json({"seq": 1.0})


def test_canonical_json_rejects_bool_for_int_field():
    with pytest.raises(TypeError, match="seq"):
        canonical_json({"seq": True})


def test_canonical_json_rejects_nan_values():
    with pytest.raises(TypeError, match="score"):
        canonical_json({"score": float("nan")})


def test_canonical_json_rejects_non_string_keys():
    with pytest.raises(TypeError, match="dict key must be str"):
        canonical_json({1: "a"})


def test_canonical_json_normalizes_unicode_to_nfc():
    decomposed = {"captured_at": 1723000000, "text": "e\u0301"}
    precomposed = {"captured_at": 1723000000, "text": "\u00e9"}

    assert canonical_json(decomposed) == canonical_json(precomposed)


def test_canonical_json_validates_and_normalizes_nested_list_dicts():
    payload = {
        "captured_at": 1723000000,
        "items": [
            {"text": "e\u0301", "seq": 1},
            {"seq": 2, "text": "\u00e9"},
        ],
    }

    result = canonical_json(payload)

    assert result == (
        b'{"captured_at":1723000000,"items":[{"seq":1,"text":"\xc3\xa9"},{"seq":2,"text":"\xc3\xa9"}]}'
    )

    with pytest.raises(TypeError, match="seq"):
        canonical_json({"items": [{"seq": 1.5}]})


def test_canonical_json_does_not_mutate_input():
    payload = {
        "captured_at": 1723000000,
        "text": "e\u0301",
        "items": [{"text": "e\u0301"}],
    }
    original = {
        "captured_at": 1723000000,
        "text": "e\u0301",
        "items": [{"text": "e\u0301"}],
    }

    canonical_json(payload)

    assert payload == original
