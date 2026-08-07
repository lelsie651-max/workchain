import sqlite3

import pytest

from evidence_core.db import get_schema_version, init_db


def _insert_actor(conn: sqlite3.Connection, actor_id: str, is_self: int) -> None:
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (actor_id, actor_id, '[]', None, None, is_self, None, 1723000000),
    )


def _insert_thread(
    conn: sqlite3.Connection,
    thread_id: str = "thr-1",
    status: str = "open",
    owner_actor_id: str | None = None,
    requester_actor_id: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO threads (
            thread_id, title, status, owner_actor_id, requester_actor_id,
            current_deliverable, current_due, version, risk_flags,
            first_seen_at, last_activity_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            "thread",
            status,
            owner_actor_id,
            requester_actor_id,
            None,
            None,
            1,
            "[]",
            1723000000,
            1723000000,
        ),
    )


def _insert_evidence(
    conn: sqlite3.Connection,
    evidence_id: str = "ev-1",
    seq: int = 1,
    thread_id: str | None = "thr-1",
    kind: str = "request",
    media_type: str = "text",
    slots_filled: int = 0,
) -> None:
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
            evidence_id,
            seq,
            thread_id,
            kind,
            media_type,
            None,
            "raw",
            "hint",
            None,
            None,
            None,
            None,
            None,
            None,
            slots_filled,
            None,
            "[]",
            None,
            1723000000,
            "content-hash",
            "prev-hash",
            "chain-hash",
        ),
    )


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "workchain.db"


def test_init_db_creates_all_tables(db_file):
    conn = init_db(db_file)

    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name IN ('actors', 'threads', 'evidence', 'checkpoints', 'meta')
        ORDER BY name
        """
    ).fetchall()

    assert [row["name"] for row in rows] == [
        "actors",
        "checkpoints",
        "evidence",
        "meta",
        "threads",
    ]


def test_init_db_is_idempotent_and_keeps_schema_version(db_file):
    conn1 = init_db(db_file)
    conn1.close()

    conn2 = init_db(db_file)

    assert get_schema_version(conn2) == 1


def test_only_one_actor_can_have_is_self_true(db_file):
    conn = init_db(db_file)

    _insert_actor(conn, "act-1", 1)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_actor(conn, "act-2", 1)


def test_multiple_actors_can_have_is_self_false(db_file):
    conn = init_db(db_file)

    _insert_actor(conn, "act-1", 0)
    _insert_actor(conn, "act-2", 0)
    _insert_actor(conn, "act-3", 0)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS count FROM actors WHERE is_self = 0").fetchone()

    assert count["count"] == 3


def test_evidence_kind_rejects_invalid_value(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, kind="gossip")


def test_threads_status_rejects_invalid_value(db_file):
    conn = init_db(db_file)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_thread(conn, status="pending")


def test_evidence_seq_must_be_unique(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)
    _insert_evidence(conn, evidence_id="ev-1", seq=1)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, evidence_id="ev-2", seq=1)


def test_foreign_keys_are_enabled_and_missing_thread_is_rejected(db_file):
    conn = init_db(db_file)

    pragma = conn.execute("PRAGMA foreign_keys").fetchone()

    assert pragma[0] == 1

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, thread_id="thr-missing")


def test_init_db_supports_memory_database():
    conn = init_db(":memory:")

    assert get_schema_version(conn) == 1
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'actors'").fetchone() is not None


def test_evidence_slots_filled_rejects_out_of_range_value(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, slots_filled=5)
