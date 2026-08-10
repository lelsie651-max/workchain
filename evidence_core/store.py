from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from evidence_core import chain


ALLOWED_MEDIA_TYPES = {"image", "text", "file"}
ALLOWED_KINDS = {"request", "confirm", "change", "deliver", "dispute", "reference"}
UPDATABLE_SLOT_FIELDS = {
    "slot_requester",
    "slot_owner",
    "slot_deliverable",
    "slot_due",
    "slot_due_raw",
    "slot_direction",
    "plain_summary",
    "caveats",
}
COUNTED_SLOT_FIELDS = (
    "slot_requester",
    "slot_owner",
    "slot_deliverable",
    "slot_due",
)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        raise ValueError("row not found")
    return dict(row)


def _normalize_payload(media_type: str, payload: bytes | str) -> tuple[bytes, str | None]:
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise ValueError("media_type must be one of image, text, file")

    if media_type == "text":
        if not isinstance(payload, str):
            raise ValueError("text media_type requires str payload")
        return payload.encode("utf-8"), payload

    if not isinstance(payload, bytes):
        raise ValueError(f"{media_type} media_type requires bytes payload")
    return payload, None


def _blob_relative_path(content_hash: str) -> Path:
    return Path(content_hash[:2]) / f"{content_hash}.bin"


def _write_blob(blobs_root: Path, content_hash: str, blob_bytes: bytes) -> str:
    relative_path = _blob_relative_path(content_hash)
    full_path = blobs_root / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    if not full_path.exists():
        full_path.write_bytes(blob_bytes)
    return relative_path.as_posix()


def _compute_slots_filled(values: dict[str, Any]) -> int:
    return sum(1 for field in COUNTED_SLOT_FIELDS if values.get(field) is not None)


def append_evidence(
    conn,
    *,
    blobs_root: Path,
    media_type: str,
    payload: bytes | str,
    captured_at: int,
    occurred_at: int | None = None,
    source_hint: str | None = None,
    kind: str = "reference",
    evidence_id: str | None = None,
) -> dict:
    if captured_at is None:
        raise ValueError("captured_at must not be None")
    if media_type is None:
        raise ValueError("media_type must not be None")
    if kind not in ALLOWED_KINDS:
        raise ValueError("kind must be one of request, confirm, change, deliver, dispute, reference")

    blob_bytes, raw_text = _normalize_payload(media_type, payload)
    blobs_root = Path(blobs_root)
    generated_evidence_id = evidence_id or f"ev_{uuid.uuid4().hex[:12]}"

    started_transaction = not conn.in_transaction
    savepoint_name = f"sp_{uuid.uuid4().hex[:12]}"

    try:
        if started_transaction:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute(f"SAVEPOINT {savepoint_name}")
        content_hash = chain.compute_content_hash(payload)
        blob_path = _write_blob(blobs_root, content_hash, blob_bytes)

        last_row = conn.execute(
            "SELECT seq, chain_hash FROM evidence ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if last_row is None:
            seq = 1
            prev_hash = chain.ZERO_HASH
        else:
            seq = last_row["seq"] + 1
            prev_hash = last_row["chain_hash"]

        if generated_evidence_id is None:
            raise ValueError("evidence_id must not be None")
        if seq is None:
            raise ValueError("seq must not be None")
        if content_hash is None:
            raise ValueError("content_hash must not be None")

        record = {
            "evidence_id": generated_evidence_id,
            "seq": seq,
            "thread_id": None,
            "kind": kind,
            "media_type": media_type,
            "blob_path": blob_path,
            "raw_text": raw_text,
            "source_hint": source_hint,
            "slot_requester": None,
            "slot_owner": None,
            "slot_deliverable": None,
            "slot_due": None,
            "slot_due_raw": None,
            "slot_direction": None,
            "slots_filled": 0,
            "plain_summary": None,
            "caveats": None,
            "occurred_at": occurred_at,
            "captured_at": captured_at,
            "content_hash": content_hash,
        }
        record_digest = chain.compute_record_digest(record)
        chain_hash = chain.compute_chain_hash(prev_hash, record_digest)

        conn.execute(
            """
            INSERT INTO evidence (
                evidence_id, seq, thread_id, kind, media_type, blob_path,
                raw_text, source_hint, slot_requester, slot_owner, slot_deliverable,
                slot_due, slot_due_raw, slot_direction, slots_filled, plain_summary,
                caveats, occurred_at, captured_at, content_hash, prev_hash, chain_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generated_evidence_id,
                seq,
                None,
                kind,
                media_type,
                blob_path,
                raw_text,
                source_hint,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                occurred_at,
                captured_at,
                content_hash,
                prev_hash,
                chain_hash,
            ),
        )
        if seq % 100 == 0:
            _insert_checkpoint(conn)
        row = conn.execute(
            "SELECT * FROM evidence WHERE evidence_id = ?",
            (generated_evidence_id,),
        ).fetchone()
        if started_transaction:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        return _row_to_dict(row)
    except Exception:
        if started_transaction:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        raise


def update_slots(conn, evidence_id: str, **slots) -> dict:
    unknown_fields = set(slots) - UPDATABLE_SLOT_FIELDS
    if unknown_fields:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown_fields))}")

    current = conn.execute(
        """
        SELECT slot_requester, slot_owner, slot_deliverable, slot_due, slot_due_raw,
               slot_direction, plain_summary, caveats
        FROM evidence
        WHERE evidence_id = ?
        """,
        (evidence_id,),
    ).fetchone()
    if current is None:
        raise ValueError("evidence_id not found")

    updated = dict(current)
    updated.update(slots)
    if isinstance(updated.get("caveats"), list):
        updated["caveats"] = json.dumps(
            updated["caveats"],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    slots_filled = _compute_slots_filled(updated)
    conn.execute(
        """
        UPDATE evidence
        SET slot_requester = ?, slot_owner = ?, slot_deliverable = ?, slot_due = ?,
            slot_due_raw = ?, slot_direction = ?, plain_summary = ?, caveats = ?,
            slots_filled = ?
        WHERE evidence_id = ?
        """,
        (
            updated.get("slot_requester"),
            updated.get("slot_owner"),
            updated.get("slot_deliverable"),
            updated.get("slot_due"),
            updated.get("slot_due_raw"),
            updated.get("slot_direction"),
            updated.get("plain_summary"),
            updated.get("caveats"),
            slots_filled,
            evidence_id,
        ),
    )
    row = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
    conn.commit()
    return _row_to_dict(row)


def _insert_checkpoint(conn) -> dict:
    latest = conn.execute(
        "SELECT seq, chain_hash FROM evidence ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    if latest is None:
        raise ValueError("cannot checkpoint empty evidence table")

    checkpoint_id = f"ck_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO checkpoints (
            checkpoint_id, at_seq, chain_hash, created_at, tsa_token
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (checkpoint_id, latest["seq"], latest["chain_hash"], created_at, None),
    )
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
        (checkpoint_id,),
    ).fetchone()
    return _row_to_dict(row)


def make_checkpoint(conn) -> dict:
    row = _insert_checkpoint(conn)
    conn.commit()
    return row


def verify_chain(conn, blobs_root: Path | None = None) -> tuple[bool, int | None, str | None]:
    rows = conn.execute("SELECT * FROM evidence ORDER BY seq ASC").fetchall()
    expected_seq = 1
    previous_chain_hash = chain.ZERO_HASH
    blobs_root = None if blobs_root is None else Path(blobs_root)
    chain_hashes_by_seq: dict[int, str] = {}

    for row in rows:
        record = dict(row)
        seq = record["seq"]

        if seq != expected_seq:
            return False, expected_seq, "seq gap"

        if record["prev_hash"] != previous_chain_hash:
            return False, seq, "prev_hash mismatch"

        expected_digest = chain.compute_record_digest(record)
        expected_chain_hash = chain.compute_chain_hash(previous_chain_hash, expected_digest)

        if record["chain_hash"] != expected_chain_hash:
            return False, seq, "chain_hash mismatch"

        if blobs_root is not None:
            blob_path = record["blob_path"]
            if blob_path is None:
                return False, seq, "missing blob_path"
            blob_bytes_path = blobs_root / blob_path
            if not blob_bytes_path.exists():
                return False, seq, "blob missing"
            blob_hash = chain.compute_content_hash(blob_bytes_path.read_bytes())
            if blob_hash != record["content_hash"]:
                return False, seq, "content_hash mismatch"

        previous_chain_hash = record["chain_hash"]
        chain_hashes_by_seq[seq] = record["chain_hash"]
        expected_seq += 1

    checkpoints = conn.execute(
        "SELECT at_seq, chain_hash FROM checkpoints ORDER BY at_seq ASC"
    ).fetchall()
    max_seq = rows[-1]["seq"] if rows else 0
    for checkpoint in checkpoints:
        at_seq = checkpoint["at_seq"]
        if at_seq > max_seq:
            return False, at_seq, "chain truncated"
        if chain_hashes_by_seq.get(at_seq) != checkpoint["chain_hash"]:
            return False, at_seq, "checkpoint mismatch"

    return True, None, None
