from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


_UNSET = object()
_FACT_MUTABLE_FIELDS = (
    "fact_type",
    "content",
    "occurred_at",
    "due_at",
    "due_raw",
    "due_anchor_at",
)


class SemanticStoreError(ValueError):
    """Base error for controlled semantic writes."""


class ProtectedFactError(SemanticStoreError):
    """Raised when a protected fact cannot be overwritten."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        raise SemanticStoreError("row not found")
    return dict(row)


def _begin(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticStoreError("text fields must be strings or None")
    value = value.strip()
    return value or None


def _coerce_required_text(name: str, value: Any) -> str:
    normalized = _coerce_optional_text(value)
    if normalized is None:
        raise SemanticStoreError(f"{name} must be a non-empty string")
    return normalized


def _normalize_non_empty_ids(name: str, values: Sequence[str]) -> list[str]:
    normalized = list(values)
    if not normalized:
        raise SemanticStoreError(f"{name} must contain at least one item")
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise SemanticStoreError(f"{name} must contain non-empty string ids")
    if len(set(normalized)) != len(normalized):
        raise SemanticStoreError(f"{name} must not contain duplicate ids")
    return normalized


def _normalize_actor_roles(
    actor_roles: Sequence[tuple[str, str] | Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    if actor_roles is None:
        return []

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in actor_roles:
        if isinstance(item, Mapping):
            actor_id = item.get("actor_id")
            role = item.get("role")
        else:
            actor_id, role = item
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise SemanticStoreError("actor_roles actor_id must be a non-empty string")
        if not isinstance(role, str) or not role.strip():
            raise SemanticStoreError("actor_roles role must be a non-empty string")
        key = (actor_id, role.strip())
        if key in seen:
            raise SemanticStoreError("actor_roles must not contain duplicate (actor_id, role)")
        seen.add(key)
        normalized.append({"actor_id": actor_id, "role": role.strip()})
    return normalized


def _fetch_existing_ids(
    conn: sqlite3.Connection, *, table: str, id_column: str, ids: Sequence[str]
) -> set[str]:
    if not ids:
        return set()
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {id_column} FROM {table} WHERE {id_column} IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return {row[id_column] for row in rows}


def _ensure_evidence_exists(conn: sqlite3.Connection, evidence_ids: Sequence[str]) -> None:
    existing = _fetch_existing_ids(
        conn, table="evidence", id_column="evidence_id", ids=evidence_ids
    )
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in existing]
    if missing:
        raise SemanticStoreError(
            f"evidence not found: {', '.join(missing)}"
        )


def _ensure_actors_exist(conn: sqlite3.Connection, actor_ids: Sequence[str]) -> None:
    existing = _fetch_existing_ids(conn, table="actors", id_column="actor_id", ids=actor_ids)
    missing = [actor_id for actor_id in actor_ids if actor_id not in existing]
    if missing:
        raise SemanticStoreError(f"actor not found: {', '.join(missing)}")


def _ensure_event_exists(conn: sqlite3.Connection, event_id: str) -> None:
    row = conn.execute(
        "SELECT event_id FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        raise SemanticStoreError(f"event not found: {event_id}")


def _get_extraction_row(conn: sqlite3.Connection, extraction_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM evidence_extractions WHERE extraction_id = ?",
        (extraction_id,),
    ).fetchone()
    if row is None:
        raise SemanticStoreError(f"extraction not found: {extraction_id}")
    return row


def _get_fact_row(conn: sqlite3.Connection, fact_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
    if row is None:
        raise SemanticStoreError(f"fact not found: {fact_id}")
    return row


def _get_semantic_run_row(conn: sqlite3.Connection, semantic_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM semantic_runs WHERE semantic_run_id = ?",
        (semantic_run_id,),
    ).fetchone()
    if row is None:
        raise SemanticStoreError(f"semantic run not found: {semantic_run_id}")
    return row


def _normalize_semantic_run_inputs(
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not inputs:
        raise SemanticStoreError("inputs must contain at least one item")

    normalized: list[dict[str, Any]] = []
    seen_evidence_ids: set[str] = set()
    seen_positions: set[int] = set()
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise SemanticStoreError("each semantic run input must be an object")
        evidence_id = _coerce_required_text("evidence_id", item.get("evidence_id"))
        extraction_id = _coerce_optional_text(item.get("extraction_id"))
        raw_position = item.get("position", index)
        if not isinstance(raw_position, int) or isinstance(raw_position, bool) or raw_position < 0:
            raise SemanticStoreError("input position must be an integer >= 0")
        if evidence_id in seen_evidence_ids:
            raise SemanticStoreError("inputs must not contain duplicate evidence_id")
        if raw_position in seen_positions:
            raise SemanticStoreError("inputs must not contain duplicate position")
        seen_evidence_ids.add(evidence_id)
        seen_positions.add(raw_position)
        normalized.append(
            {
                "evidence_id": evidence_id,
                "extraction_id": extraction_id,
                "position": raw_position,
            }
        )
    return sorted(normalized, key=lambda item: item["position"])


def _validate_semantic_run_inputs(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str | None = None,
    inputs: Sequence[dict[str, Any]],
) -> None:
    _ensure_evidence_exists(conn, [item["evidence_id"] for item in inputs])
    for item in inputs:
        extraction_id = item.get("extraction_id")
        if extraction_id is None:
            continue
        extraction = _get_extraction_row(conn, extraction_id)
        if extraction["evidence_id"] != item["evidence_id"]:
            raise SemanticStoreError(
                "extraction_id must belong to the same evidence_id"
            )

    if semantic_run_id is not None:
        _get_semantic_run_row(conn, semantic_run_id)


def _semantic_run_input_evidence_ids(conn: sqlite3.Connection, semantic_run_id: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT evidence_id
        FROM semantic_run_inputs
        WHERE semantic_run_id = ?
        """,
        (semantic_run_id,),
    ).fetchall()
    return {row["evidence_id"] for row in rows}


def _ensure_semantic_run_covers_evidence_ids(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str,
    evidence_ids: Sequence[str],
) -> None:
    _get_semantic_run_row(conn, semantic_run_id)
    allowed = _semantic_run_input_evidence_ids(conn, semantic_run_id)
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in allowed]
    if missing:
        raise SemanticStoreError(
            "semantic run does not include evidence: " + ", ".join(missing)
        )


def _load_submission(conn: sqlite3.Connection, submission_id: str) -> dict[str, Any]:
    submission = conn.execute(
        "SELECT * FROM submissions WHERE submission_id = ?",
        (submission_id,),
    ).fetchone()
    if submission is None:
        raise SemanticStoreError(f"submission not found: {submission_id}")
    evidence_rows = conn.execute(
        """
        SELECT e.*
        FROM submission_evidence se
        JOIN evidence e ON e.evidence_id = se.evidence_id
        WHERE se.submission_id = ?
        ORDER BY se.position ASC
        """,
        (submission_id,),
    ).fetchall()
    result = dict(submission)
    result["evidence"] = [dict(row) for row in evidence_rows]
    return result


def _load_semantic_run(conn: sqlite3.Connection, semantic_run_id: str) -> dict[str, Any]:
    run = _get_semantic_run_row(conn, semantic_run_id)
    input_rows = conn.execute(
        """
        SELECT semantic_run_id, evidence_id, extraction_id, position
        FROM semantic_run_inputs
        WHERE semantic_run_id = ?
        ORDER BY position ASC, evidence_id ASC
        """,
        (semantic_run_id,),
    ).fetchall()
    result = dict(run)
    result["inputs"] = [dict(row) for row in input_rows]
    return result


def _load_fact(conn: sqlite3.Connection, fact_id: str) -> dict[str, Any]:
    fact = _get_fact_row(conn, fact_id)
    evidence_rows = conn.execute(
        """
        SELECT e.evidence_id, e.seq
        FROM fact_evidence fe
        JOIN evidence e ON e.evidence_id = fe.evidence_id
        WHERE fe.fact_id = ?
        ORDER BY e.seq ASC, e.evidence_id ASC
        """,
        (fact_id,),
    ).fetchall()
    actor_rows = conn.execute(
        """
        SELECT actor_id, role
        FROM fact_actors
        WHERE fact_id = ?
        ORDER BY actor_id ASC, role ASC
        """,
        (fact_id,),
    ).fetchall()
    result = dict(fact)
    result["evidence_ids"] = [row["evidence_id"] for row in evidence_rows]
    result["actors"] = [dict(row) for row in actor_rows]
    return result


def _load_interpretation(conn: sqlite3.Connection, interpretation_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM interpretations WHERE interpretation_id = ?",
        (interpretation_id,),
    ).fetchone()
    if row is None:
        raise SemanticStoreError(f"interpretation not found: {interpretation_id}")
    return dict(row)


def _replace_fact_actors(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    actor_roles: Sequence[dict[str, str]],
) -> None:
    actor_ids = [item["actor_id"] for item in actor_roles]
    _ensure_actors_exist(conn, actor_ids)
    conn.execute("DELETE FROM fact_actors WHERE fact_id = ?", (fact_id,))
    for item in actor_roles:
        conn.execute(
            "INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)",
            (fact_id, item["actor_id"], item["role"]),
        )


def _validate_assignment_fields(
    *,
    event_id: str | None,
    event_assignment: str,
    event_assignment_confidence: float | None,
) -> None:
    if event_assignment == "unassigned":
        if event_id is not None:
            raise SemanticStoreError("event_assignment 'unassigned' requires event_id=None")
        if event_assignment_confidence is not None:
            raise SemanticStoreError(
                "event_assignment_confidence requires an assigned event"
            )
        return

    if event_assignment not in {"auto", "suggested", "confirmed"}:
        raise SemanticStoreError(f"unsupported event_assignment: {event_assignment}")
    if event_id is None:
        raise SemanticStoreError(
            f"event_assignment '{event_assignment}' requires a non-null event_id"
        )


def create_submission(
    conn: sqlite3.Connection,
    *,
    evidence_ids: Sequence[str],
    submission_id: str | None = None,
    created_at: int | None = None,
    source_hint: str | None = None,
) -> dict[str, Any]:
    normalized_evidence_ids = _normalize_non_empty_ids("evidence_ids", evidence_ids)
    created_at = _now_ms() if created_at is None else created_at
    submission_id = submission_id or _new_id("sub")

    try:
        _begin(conn)
        _ensure_evidence_exists(conn, normalized_evidence_ids)

        occupied = _fetch_existing_ids(
            conn,
            table="submission_evidence",
            id_column="evidence_id",
            ids=normalized_evidence_ids,
        )
        if occupied:
            raise SemanticStoreError(
                "evidence already linked to another submission: "
                + ", ".join(sorted(occupied))
            )

        conn.execute(
            "INSERT INTO submissions (submission_id, created_at, source_hint) VALUES (?, ?, ?)",
            (submission_id, created_at, source_hint),
        )
        for position, evidence_id in enumerate(normalized_evidence_ids):
            conn.execute(
                """
                INSERT INTO submission_evidence (submission_id, evidence_id, position)
                VALUES (?, ?, ?)
                """,
                (submission_id, evidence_id, position),
            )
        result = _load_submission(conn, submission_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def create_event(
    conn: sqlite3.Connection,
    *,
    title: str,
    status: str = "active",
    summary: str | None = None,
    event_id: str | None = None,
    created_at: int | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    normalized_title = title.strip()
    if not normalized_title:
        raise SemanticStoreError("title must not be blank")
    if status not in {"active", "resolved", "archived"}:
        raise SemanticStoreError(f"unsupported event status: {status}")

    created_at = _now_ms() if created_at is None else created_at
    updated_at = created_at if updated_at is None else updated_at
    event_id = event_id or _new_id("evt")

    try:
        _begin(conn)
        conn.execute(
            """
            INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, normalized_title, status, summary, created_at, updated_at),
        )
        row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        conn.commit()
        return _row_to_dict(row)
    except Exception:
        conn.rollback()
        raise


def create_semantic_run(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    parser_version: str,
    inputs: Sequence[Mapping[str, Any]],
    semantic_run_id: str | None = None,
    anchor_date: str | None = None,
    created_at: int | None = None,
    supersedes_run_id: str | None = None,
) -> dict[str, Any]:
    provider = _coerce_required_text("provider", provider)
    model = _coerce_required_text("model", model)
    parser_version = _coerce_required_text("parser_version", parser_version)
    anchor_date = _coerce_optional_text(anchor_date)
    supersedes_run_id = _coerce_optional_text(supersedes_run_id)
    created_at = _now_ms() if created_at is None else created_at
    semantic_run_id = semantic_run_id or _new_id("srun")
    normalized_inputs = _normalize_semantic_run_inputs(inputs)

    try:
        _begin(conn)
        if supersedes_run_id is not None:
            _get_semantic_run_row(conn, supersedes_run_id)
        _validate_semantic_run_inputs(conn, inputs=normalized_inputs)

        conn.execute(
            """
            INSERT INTO semantic_runs (
                semantic_run_id, provider, model, parser_version, status,
                anchor_date, created_at, completed_at, failure_type, supersedes_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                semantic_run_id,
                provider,
                model,
                parser_version,
                "running",
                anchor_date,
                created_at,
                None,
                None,
                supersedes_run_id,
            ),
        )
        for item in normalized_inputs:
            conn.execute(
                """
                INSERT INTO semantic_run_inputs (
                    semantic_run_id, evidence_id, extraction_id, position
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    semantic_run_id,
                    item["evidence_id"],
                    item["extraction_id"],
                    item["position"],
                ),
            )
        result = _load_semantic_run(conn, semantic_run_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def mark_semantic_run_succeeded(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str,
    completed_at: int | None = None,
) -> dict[str, Any]:
    completed_at = _now_ms() if completed_at is None else completed_at

    try:
        _begin(conn)
        current = _get_semantic_run_row(conn, semantic_run_id)
        if current["status"] != "running":
            raise SemanticStoreError("semantic run is not running")
        conn.execute(
            """
            UPDATE semantic_runs
            SET status = ?, completed_at = ?, failure_type = ?
            WHERE semantic_run_id = ?
            """,
            ("succeeded", completed_at, None, semantic_run_id),
        )
        result = _load_semantic_run(conn, semantic_run_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def mark_semantic_run_failed(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str,
    failure_type: str | None = None,
    completed_at: int | None = None,
) -> dict[str, Any]:
    completed_at = _now_ms() if completed_at is None else completed_at
    failure_type = _coerce_optional_text(failure_type)

    try:
        _begin(conn)
        current = _get_semantic_run_row(conn, semantic_run_id)
        if current["status"] != "running":
            raise SemanticStoreError("semantic run is not running")
        conn.execute(
            """
            UPDATE semantic_runs
            SET status = ?, completed_at = ?, failure_type = ?
            WHERE semantic_run_id = ?
            """,
            ("failed", completed_at, failure_type, semantic_run_id),
        )
        result = _load_semantic_run(conn, semantic_run_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def get_semantic_run(conn: sqlite3.Connection, semantic_run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT semantic_run_id FROM semantic_runs WHERE semantic_run_id = ?",
        (semantic_run_id,),
    ).fetchone()
    if row is None:
        return None
    return _load_semantic_run(conn, semantic_run_id)


def create_fact(
    conn: sqlite3.Connection,
    *,
    fact_type: str,
    content: str,
    evidence_ids: Sequence[str],
    semantic_run_id: str | None = None,
    event_id: str | None = None,
    occurred_at: int | None = None,
    due_at: int | None = None,
    due_raw: str | None = None,
    due_anchor_at: int | None = None,
    confidence: float | None = None,
    event_assignment: str = "unassigned",
    event_assignment_confidence: float | None = None,
    origin: str = "ai",
    review_status: str = "unreviewed",
    actor_roles: Sequence[tuple[str, str] | Mapping[str, str]] | None = None,
    fact_id: str | None = None,
    created_at: int | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    normalized_evidence_ids = _normalize_non_empty_ids("evidence_ids", evidence_ids)
    normalized_actor_roles = _normalize_actor_roles(actor_roles)
    _validate_assignment_fields(
        event_id=event_id,
        event_assignment=event_assignment,
        event_assignment_confidence=event_assignment_confidence,
    )

    created_at = _now_ms() if created_at is None else created_at
    updated_at = created_at if updated_at is None else updated_at
    fact_id = fact_id or _new_id("fact")

    try:
        _begin(conn)
        _ensure_evidence_exists(conn, normalized_evidence_ids)
        if event_id is not None:
            _ensure_event_exists(conn, event_id)
        if semantic_run_id is not None:
            _ensure_semantic_run_covers_evidence_ids(
                conn,
                semantic_run_id=semantic_run_id,
                evidence_ids=normalized_evidence_ids,
            )
        if normalized_actor_roles:
            _ensure_actors_exist(
                conn, [item["actor_id"] for item in normalized_actor_roles]
            )

        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
                confidence, event_assignment, created_at, updated_at,
                due_anchor_at, event_assignment_confidence, origin, review_status, semantic_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_id,
                event_id,
                fact_type,
                content,
                occurred_at,
                due_at,
                due_raw,
                confidence,
                event_assignment,
                created_at,
                updated_at,
                due_anchor_at,
                event_assignment_confidence,
                origin,
                review_status,
                semantic_run_id,
            ),
        )
        for evidence_id in normalized_evidence_ids:
            conn.execute(
                "INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)",
                (fact_id, evidence_id),
            )
        for item in normalized_actor_roles:
            conn.execute(
                "INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)",
                (fact_id, item["actor_id"], item["role"]),
            )
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def create_interpretation(
    conn: sqlite3.Connection,
    *,
    kind: str,
    content: str,
    fact_id: str | None = None,
    evidence_id: str | None = None,
    semantic_run_id: str | None = None,
    confidence: float | None = None,
    interpretation_id: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    if fact_id is None and evidence_id is None:
        raise SemanticStoreError("interpretation requires fact_id or evidence_id")

    interpretation_id = interpretation_id or _new_id("itp")
    created_at = _now_ms() if created_at is None else created_at

    try:
        _begin(conn)
        fact_row = None
        if fact_id is not None:
            fact_row = _get_fact_row(conn, fact_id)
        if evidence_id is not None:
            _ensure_evidence_exists(conn, [evidence_id])
        if semantic_run_id is not None:
            _get_semantic_run_row(conn, semantic_run_id)
            if evidence_id is not None:
                _ensure_semantic_run_covers_evidence_ids(
                    conn,
                    semantic_run_id=semantic_run_id,
                    evidence_ids=[evidence_id],
                )
            if fact_row is not None:
                fact_semantic_run_id = fact_row["semantic_run_id"]
                if fact_semantic_run_id is None:
                    raise SemanticStoreError(
                        "interpretation semantic_run_id requires fact semantic_run_id to be set"
                    )
                if fact_semantic_run_id != semantic_run_id:
                    raise SemanticStoreError(
                        "interpretation semantic_run_id must match fact semantic_run_id"
                    )
        conn.execute(
            """
            INSERT INTO interpretations (
                interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at, semantic_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interpretation_id,
                fact_id,
                evidence_id,
                kind,
                content,
                confidence,
                created_at,
                semantic_run_id,
            ),
        )
        result = _load_interpretation(conn, interpretation_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def set_event_assignment_by_ai(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    event_id: str,
    assignment: str,
    event_assignment_confidence: float | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    if assignment not in {"auto", "suggested"}:
        raise SemanticStoreError("AI assignment must be 'auto' or 'suggested'")
    updated_at = _now_ms() if updated_at is None else updated_at

    try:
        _begin(conn)
        current = _get_fact_row(conn, fact_id)
        if current["event_assignment"] == "confirmed":
            raise ProtectedFactError(
                f"fact {fact_id} has confirmed event assignment and cannot be overwritten by AI"
            )
        _ensure_event_exists(conn, event_id)
        conn.execute(
            """
            UPDATE facts
            SET event_id = ?, event_assignment = ?, event_assignment_confidence = ?, updated_at = ?
            WHERE fact_id = ?
            """,
            (event_id, assignment, event_assignment_confidence, updated_at, fact_id),
        )
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def set_event_assignment_by_user(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    event_id: str | None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    updated_at = _now_ms() if updated_at is None else updated_at
    assignment = "unassigned" if event_id is None else "confirmed"

    try:
        _begin(conn)
        _get_fact_row(conn, fact_id)
        if event_id is not None:
            _ensure_event_exists(conn, event_id)
        conn.execute(
            """
            UPDATE facts
            SET event_id = ?, event_assignment = ?, event_assignment_confidence = ?, updated_at = ?
            WHERE fact_id = ?
            """,
            (event_id, assignment, None, updated_at, fact_id),
        )
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def confirm_fact(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    updated_at: int | None = None,
) -> dict[str, Any]:
    updated_at = _now_ms() if updated_at is None else updated_at

    try:
        _begin(conn)
        _get_fact_row(conn, fact_id)
        conn.execute(
            "UPDATE facts SET review_status = ?, updated_at = ? WHERE fact_id = ?",
            ("confirmed", updated_at, fact_id),
        )
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def correct_fact_by_user(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    fact_type: str | object = _UNSET,
    content: str | object = _UNSET,
    occurred_at: int | None | object = _UNSET,
    due_at: int | None | object = _UNSET,
    due_raw: str | None | object = _UNSET,
    due_anchor_at: int | None | object = _UNSET,
    actor_roles: Sequence[tuple[str, str] | Mapping[str, str]] | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    updates = {
        "fact_type": fact_type,
        "content": content,
        "occurred_at": occurred_at,
        "due_at": due_at,
        "due_raw": due_raw,
        "due_anchor_at": due_anchor_at,
    }
    normalized_updates = {key: value for key, value in updates.items() if value is not _UNSET}
    normalized_actor_roles = None if actor_roles is None else _normalize_actor_roles(actor_roles)
    if not normalized_updates and normalized_actor_roles is None:
        raise SemanticStoreError("correct_fact_by_user requires at least one semantic change")

    updated_at = _now_ms() if updated_at is None else updated_at

    try:
        _begin(conn)
        current = _get_fact_row(conn, fact_id)
        assignments = {
            "fact_type": current["fact_type"],
            "content": current["content"],
            "occurred_at": current["occurred_at"],
            "due_at": current["due_at"],
            "due_raw": current["due_raw"],
            "due_anchor_at": current["due_anchor_at"],
        }
        assignments.update(normalized_updates)
        assignments["origin"] = "user"
        assignments["review_status"] = "corrected"
        assignments["updated_at"] = updated_at

        conn.execute(
            """
            UPDATE facts
            SET fact_type = ?, content = ?, occurred_at = ?, due_at = ?, due_raw = ?,
                due_anchor_at = ?, origin = ?, review_status = ?, updated_at = ?
            WHERE fact_id = ?
            """,
            (
                assignments["fact_type"],
                assignments["content"],
                assignments["occurred_at"],
                assignments["due_at"],
                assignments["due_raw"],
                assignments["due_anchor_at"],
                assignments["origin"],
                assignments["review_status"],
                assignments["updated_at"],
                fact_id,
            ),
        )
        if normalized_actor_roles is not None:
            _replace_fact_actors(conn, fact_id=fact_id, actor_roles=normalized_actor_roles)
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def update_fact_by_ai(
    conn: sqlite3.Connection,
    *,
    fact_id: str,
    fact_type: str | object = _UNSET,
    content: str | object = _UNSET,
    occurred_at: int | None | object = _UNSET,
    due_at: int | None | object = _UNSET,
    due_raw: str | None | object = _UNSET,
    due_anchor_at: int | None | object = _UNSET,
    confidence: float | None | object = _UNSET,
    actor_roles: Sequence[tuple[str, str] | Mapping[str, str]] | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    updates = {
        "fact_type": fact_type,
        "content": content,
        "occurred_at": occurred_at,
        "due_at": due_at,
        "due_raw": due_raw,
        "due_anchor_at": due_anchor_at,
        "confidence": confidence,
    }
    normalized_updates = {key: value for key, value in updates.items() if value is not _UNSET}
    normalized_actor_roles = None if actor_roles is None else _normalize_actor_roles(actor_roles)
    if not normalized_updates and normalized_actor_roles is None:
        raise SemanticStoreError("update_fact_by_ai requires at least one semantic change")

    updated_at = _now_ms() if updated_at is None else updated_at

    try:
        _begin(conn)
        current = _get_fact_row(conn, fact_id)
        if current["review_status"] in {"confirmed", "corrected"}:
            raise ProtectedFactError(
                f"fact {fact_id} has review_status={current['review_status']} and cannot be overwritten by AI"
            )

        assignments = {
            "fact_type": current["fact_type"],
            "content": current["content"],
            "occurred_at": current["occurred_at"],
            "due_at": current["due_at"],
            "due_raw": current["due_raw"],
            "due_anchor_at": current["due_anchor_at"],
            "confidence": current["confidence"],
        }
        assignments.update(normalized_updates)
        conn.execute(
            """
            UPDATE facts
            SET fact_type = ?, content = ?, occurred_at = ?, due_at = ?, due_raw = ?,
                due_anchor_at = ?, confidence = ?, origin = ?, updated_at = ?
            WHERE fact_id = ?
            """,
            (
                assignments["fact_type"],
                assignments["content"],
                assignments["occurred_at"],
                assignments["due_at"],
                assignments["due_raw"],
                assignments["due_anchor_at"],
                assignments["confidence"],
                "ai",
                updated_at,
                fact_id,
            ),
        )
        if normalized_actor_roles is not None:
            _replace_fact_actors(conn, fact_id=fact_id, actor_roles=normalized_actor_roles)
        result = _load_fact(conn, fact_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
