from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from evidence_core.extraction_contract import normalize_observations


class ExtractionStoreError(ValueError):
    """Base error for controlled extraction writes."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return f"ext_{uuid.uuid4().hex[:12]}"


def _begin(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExtractionStoreError("text fields must be strings or None")
    value = value.strip()
    return value or None


def _ensure_evidence_exists(conn: sqlite3.Connection, evidence_id: str) -> None:
    row = conn.execute(
        "SELECT evidence_id FROM evidence WHERE evidence_id = ?",
        (evidence_id,),
    ).fetchone()
    if row is None:
        raise ExtractionStoreError(f"evidence not found: {evidence_id}")


def _get_extraction_row(conn: sqlite3.Connection, extraction_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM evidence_extractions WHERE extraction_id = ?",
        (extraction_id,),
    ).fetchone()
    if row is None:
        raise ExtractionStoreError(f"extraction not found: {extraction_id}")
    return row


def _decode_observations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return normalize_observations(parsed)


def _row_to_extraction(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["observations"] = _decode_observations(row["observations"])
    return result


def create_extraction(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    origin: str,
    provider: str,
    model: str | None = None,
    transcript: str | None = None,
    observations: Any = None,
    created_at: int | None = None,
    extraction_id: str | None = None,
    supersedes_extraction_id: str | None = None,
) -> dict[str, Any]:
    if origin not in {"machine", "user"}:
        raise ExtractionStoreError("origin must be 'machine' or 'user'")

    provider = _coerce_text(provider)
    if provider is None:
        raise ExtractionStoreError("provider must not be blank")

    transcript = _coerce_text(transcript)
    model = _coerce_text(model)
    normalized_observations = normalize_observations(observations if observations is not None else [])
    if transcript is None and not normalized_observations:
        raise ExtractionStoreError("extraction requires transcript or observations")

    created_at = _now_ms() if created_at is None else created_at
    extraction_id = extraction_id or _new_id()

    started_transaction = not conn.in_transaction

    try:
        if started_transaction:
            _begin(conn)
        _ensure_evidence_exists(conn, evidence_id)
        if supersedes_extraction_id is not None:
            superseded = _get_extraction_row(conn, supersedes_extraction_id)
            if superseded["evidence_id"] != evidence_id:
                raise ExtractionStoreError(
                    "supersedes_extraction_id must belong to the same evidence"
                )

        conn.execute(
            """
            INSERT INTO evidence_extractions (
                extraction_id, evidence_id, origin, provider, model,
                transcript, observations, created_at, supersedes_extraction_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                extraction_id,
                evidence_id,
                origin,
                provider,
                model,
                transcript,
                json.dumps(normalized_observations, ensure_ascii=False, separators=(",", ":")),
                created_at,
                supersedes_extraction_id,
            ),
        )
        row = _get_extraction_row(conn, extraction_id)
        if started_transaction:
            conn.commit()
        return _row_to_extraction(row)
    except Exception:
        if started_transaction:
            conn.rollback()
        raise


def get_latest_extraction(conn: sqlite3.Connection, evidence_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM evidence_extractions
        WHERE evidence_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """,
        (evidence_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_extraction(row)


def list_extractions(conn: sqlite3.Connection, evidence_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM evidence_extractions
        WHERE evidence_id = ?
        ORDER BY created_at ASC, rowid ASC
        """,
        (evidence_id,),
    ).fetchall()
    return [_row_to_extraction(row) for row in rows]
