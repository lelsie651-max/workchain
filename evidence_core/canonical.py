from __future__ import annotations

import json
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


def _validate_timestamp_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.endswith("_at"):
                if item is not None and (isinstance(item, bool) or not isinstance(item, int)):
                    raise TypeError(f"timestamp field '{key}' must be int or None")
            _validate_timestamp_fields(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_timestamp_fields(item)


def canonical_json(obj: dict) -> bytes:
    if not isinstance(obj, dict):
        raise TypeError("obj must be a dict")

    _validate_timestamp_fields(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_digest_payload(record: dict) -> dict:
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")

    return {field: record.get(field, None) for field in DIGEST_FIELDS}
