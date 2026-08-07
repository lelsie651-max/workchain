from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    try:
        canonical_json(payload)
        assert False, "expected TypeError"
    except TypeError:
        assert True


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
