from __future__ import annotations

import hashlib

from evidence_core import canonical


ZERO_HASH = "0" * 64


def is_valid_hash(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False

    return all(char in "0123456789abcdef" for char in value)


def compute_content_hash(payload: bytes | str) -> str:
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        raise TypeError("payload must be bytes or str")

    return hashlib.sha256(data).hexdigest()


def compute_record_digest(record: dict) -> str:
    digest_payload = canonical.build_digest_payload(record)
    digest_bytes = canonical.canonical_json(digest_payload)
    return hashlib.sha256(digest_bytes).hexdigest()


def compute_chain_hash(prev_hash: str, record_digest: str) -> str:
    if not is_valid_hash(prev_hash):
        raise ValueError("prev_hash must be a 64-character lowercase hex string")
    if not is_valid_hash(record_digest):
        raise ValueError("record_digest must be a 64-character lowercase hex string")

    return hashlib.sha256((prev_hash + record_digest).encode("ascii")).hexdigest()
