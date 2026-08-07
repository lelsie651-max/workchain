from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any


ZERO_HASH = "0" * 64
DIGEST_FIELDS = (
    "captured_at",
    "content_hash",
    "evidence_id",
    "media_type",
    "occurred_at",
    "seq",
    "source_hint",
)
INT_FIELDS = frozenset({"seq", "occurred_at", "captured_at"})


def build_digest_payload(record: dict) -> dict:
    return {field: record.get(field, None) for field in DIGEST_FIELDS}


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


def compute_content_hash(payload: bytes | str) -> str:
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        raise TypeError("payload must be bytes or str")
    return hashlib.sha256(data).hexdigest()


def compute_record_digest(record: dict) -> str:
    return hashlib.sha256(canonical_json(build_digest_payload(record))).hexdigest()


def compute_chain_hash(prev_hash: str, record_digest: str) -> str:
    return hashlib.sha256((prev_hash + record_digest).encode("ascii")).hexdigest()


def _blob_relative_path(content_hash: str) -> Path:
    return Path(content_hash[:2]) / f"{content_hash}.bin"


def verify_export_dir(directory: Path) -> tuple[bool, int | None, str | None]:
    manifest_path = directory / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    checkpoints = data.get("checkpoints", [])

    expected_seq = 1
    previous_chain_hash = ZERO_HASH
    chain_hashes_by_seq: dict[int, str] = {}

    for record in records:
        seq = record["seq"]
        if seq != expected_seq:
            return False, expected_seq, "seq gap"

        digest = compute_record_digest(record)
        if digest != record["record_digest"]:
            return False, seq, "record_digest mismatch"

        if record["prev_hash"] != previous_chain_hash:
            return False, seq, "prev_hash mismatch"

        chain_hash = compute_chain_hash(previous_chain_hash, digest)
        if chain_hash != record["chain_hash"]:
            return False, seq, "chain_hash mismatch"

        blob_path = directory / "blobs" / _blob_relative_path(record["content_hash"])
        if not blob_path.exists():
            return False, seq, "blob missing"
        content_hash = compute_content_hash(blob_path.read_bytes())
        if content_hash != record["content_hash"]:
            return False, seq, "content_hash mismatch"

        previous_chain_hash = record["chain_hash"]
        chain_hashes_by_seq[seq] = record["chain_hash"]
        expected_seq += 1

    for checkpoint in checkpoints:
        at_seq = checkpoint["at_seq"]
        if at_seq not in chain_hashes_by_seq:
            return False, at_seq, "chain truncated"
        if chain_hashes_by_seq[at_seq] != checkpoint["chain_hash"]:
            return False, at_seq, "checkpoint mismatch"

    return True, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        print("manifest missing")
        return 2

    try:
        ok, seq, reason = verify_export_dir(directory)
    except json.JSONDecodeError as exc:
        print(f"invalid manifest json: {exc.msg}")
        return 2
    except (KeyError, TypeError, ValueError) as exc:
        print(f"verification failed: {exc}")
        return 1

    if ok:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"OK {len(manifest.get('records', []))}")
        return 0

    print(f"FAIL seq={seq} reason={reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
