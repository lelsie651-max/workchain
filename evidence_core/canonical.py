from __future__ import annotations

import json
import unicodedata
from typing import Any


DIGEST_FIELDS = [
    "captured_at",
    "content_hash",
    "evidence_id",
    "media_type",
    "occurred_at",
    "seq",
    "source_hint",
]

INT_FIELDS = frozenset(
    {
        "seq",
        "occurred_at",
        "captured_at",
        "slot_due",
        "current_due",
        "created_at",
        "last_activity_at",
        "first_seen_at",
        "at_seq",
        "version",
    }
)


def _normalize_for_canonical(value: Any, current_key: str | None = None) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"dict key must be str, got {type(key).__name__} at key {key!r}")

            normalized_key = unicodedata.normalize("NFC", key)
            normalized[normalized_key] = _normalize_for_canonical(item, normalized_key)
        return normalized

    if isinstance(value, list):
        return [_normalize_for_canonical(item, current_key) for item in value]

    if isinstance(value, tuple):
        return [_normalize_for_canonical(item, current_key) for item in value]

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)

    if isinstance(value, bool):
        if current_key in INT_FIELDS:
            raise TypeError(f"field '{current_key}' must be int or None")
        return value

    if isinstance(value, float):
        field_name = current_key or "<root>"
        raise TypeError(f"float is not allowed for field '{field_name}'")

    if current_key in INT_FIELDS and value is not None and not isinstance(value, int):
        raise TypeError(f"field '{current_key}' must be int or None")

    return value


def canonical_json(obj: dict) -> bytes:
    if not isinstance(obj, dict):
        raise TypeError("obj must be a dict")

    normalized = _normalize_for_canonical(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build_digest_payload(record: dict) -> dict:
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")

    return {field: record.get(field, None) for field in DIGEST_FIELDS}
