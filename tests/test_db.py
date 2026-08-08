import sqlite3

import pytest

from evidence_core import db as db_module
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
    seq: int | None = 1,
    thread_id: str | None = "thr-1",
    kind: str = "request",
    media_type: str = "text",
    slot_direction: str | None = None,
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
            slot_direction,
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


def _make_v1_db(path):
    conn = db_module._connect(path)
    try:
        db_module._ensure_meta_table(conn)
        db_module._create_v1_schema(conn)
        db_module._set_schema_version(conn, 1)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "workchain.db"


def test_init_db_creates_all_tables(db_file):
    conn = init_db(db_file)

    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name IN (
            'actors', 'threads', 'evidence', 'checkpoints', 'meta',
            'submissions', 'submission_evidence', 'events', 'facts',
            'fact_evidence', 'fact_actors', 'interpretations'
        )
        ORDER BY name
        """
    ).fetchall()

    assert [row["name"] for row in rows] == [
        "actors",
        "checkpoints",
        "events",
        "evidence",
        "fact_actors",
        "fact_evidence",
        "facts",
        "interpretations",
        "meta",
        "submission_evidence",
        "submissions",
        "threads",
    ]
    assert get_schema_version(conn) == 2


def test_init_db_is_idempotent_and_keeps_schema_version(db_file):
    conn1 = init_db(db_file)
    conn1.close()

    conn2 = init_db(db_file)

    assert get_schema_version(conn2) == 2


def test_v1_database_is_migrated_to_v2_without_losing_existing_rows(db_file):
    _make_v1_db(db_file)
    conn = init_db(db_file)
    try:
        _insert_actor(conn, "act-1", 0)
        _insert_thread(conn, thread_id="thr-1")
        _insert_evidence(conn, evidence_id="ev-1", seq=1, thread_id="thr-1")
        conn.commit()
        conn.close()

        reopened = init_db(db_file)
        actor = reopened.execute("SELECT actor_id FROM actors WHERE actor_id = 'act-1'").fetchone()
        thread = reopened.execute("SELECT thread_id FROM threads WHERE thread_id = 'thr-1'").fetchone()
        evidence = reopened.execute("SELECT evidence_id FROM evidence WHERE evidence_id = 'ev-1'").fetchone()
        facts_table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts'"
        ).fetchone()

        assert get_schema_version(reopened) == 2
        assert actor["actor_id"] == "act-1"
        assert thread["thread_id"] == "thr-1"
        assert evidence["evidence_id"] == "ev-1"
        assert facts_table["name"] == "facts"
    finally:
        reopened.close()


def test_v1_to_v2_migration_is_idempotent(db_file):
    _make_v1_db(db_file)

    first = init_db(db_file)
    first.close()
    second = init_db(db_file)
    try:
        assert get_schema_version(second) == 2
        tables = second.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN ('submissions', 'events', 'facts', 'interpretations')
            ORDER BY name
            """
        ).fetchall()
        assert [row["name"] for row in tables] == ["events", "facts", "interpretations", "submissions"]
    finally:
        second.close()


def test_init_db_rejects_future_schema_version(db_file):
    conn = db_module._connect(db_file)
    try:
        db_module._ensure_meta_table(conn)
        db_module._set_schema_version(conn, 999)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="newer than supported version"):
        init_db(db_file)


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

    assert get_schema_version(conn) == 2
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'actors'").fetchone() is not None


def test_evidence_slots_filled_rejects_out_of_range_value(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, slots_filled=5)


@pytest.mark.parametrize("slot_direction", ["i_owe", "owed_to_me", "none"])
def test_evidence_slot_direction_accepts_allowed_values(db_file, slot_direction: str):
    conn = init_db(db_file)
    _insert_thread(conn)

    _insert_evidence(conn, slot_direction=slot_direction)
    conn.commit()

    row = conn.execute(
        "SELECT slot_direction FROM evidence WHERE evidence_id = ?",
        ("ev-1",),
    ).fetchone()

    assert row["slot_direction"] == slot_direction


def test_evidence_slot_direction_accepts_null(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    _insert_evidence(conn, slot_direction=None)
    conn.commit()

    row = conn.execute(
        "SELECT slot_direction FROM evidence WHERE evidence_id = ?",
        ("ev-1",),
    ).fetchone()

    assert row["slot_direction"] is None


def test_evidence_slot_direction_rejects_invalid_value(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, slot_direction="garbage")


def test_evidence_seq_rejects_null(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, seq=None)


def test_evidence_seq_still_rejects_duplicate_values(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)
    _insert_evidence(conn, evidence_id="ev-1", seq=1)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_evidence(conn, evidence_id="ev-2", seq=1)


def test_file_database_uses_wal_journal_mode(db_file):
    conn = init_db(db_file)

    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()

    assert journal_mode[0].lower() == "wal"


def test_submission_evidence_enforces_order_uniqueness_and_fk_constraints(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)
    _insert_evidence(conn, evidence_id="ev-1", seq=1)
    _insert_evidence(conn, evidence_id="ev-2", seq=2)
    conn.execute(
        "INSERT INTO submissions (submission_id, created_at, source_hint) VALUES (?, ?, ?)",
        ("sub-1", 1723000000, "hint"),
    )
    conn.execute(
        "INSERT INTO submission_evidence (submission_id, evidence_id, position) VALUES (?, ?, ?)",
        ("sub-1", "ev-1", 0),
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO submission_evidence (submission_id, evidence_id, position) VALUES (?, ?, ?)",
            ("sub-1", "ev-2", 0),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO submission_evidence (submission_id, evidence_id, position) VALUES (?, ?, ?)",
            ("sub-missing", "ev-2", 1),
        )


def test_facts_constraints_cover_enum_confidence_and_assignment(db_file):
    conn = init_db(db_file)
    conn.execute(
        """
        INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("evt-1", "渠道复盘", "active", None, 1723000000, 1723000000),
    )
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "fact-1",
            None,
            "statement",
            "先记一下",
            None,
            None,
            None,
            0.5,
            "unassigned",
            1723000000,
            1723000000,
        ),
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-confidence", None, "statement", "x", 1.5, "unassigned", 1, 1),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-type", None, "gossip", "x", None, "unassigned", 1, 1),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-assignment", "evt-1", "statement", "x", None, "unassigned", 1, 1),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-missing-event", None, "statement", "x", None, "auto", 1, 1),
        )


def test_fact_evidence_and_fact_actors_allow_many_to_many_links(db_file):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1", 0)
    _insert_actor(conn, "act-2", 0)
    _insert_thread(conn)
    _insert_evidence(conn, evidence_id="ev-1", seq=1)
    _insert_evidence(conn, evidence_id="ev-2", seq=2)
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("fact-1", None, "statement", "原始事实", None, None, None, None, "unassigned", 1, 1),
    )
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("fact-2", None, "reference", "补充事实", None, None, None, None, "unassigned", 2, 2),
    )

    conn.execute("INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)", ("fact-1", "ev-1"))
    conn.execute("INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)", ("fact-1", "ev-2"))
    conn.execute("INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)", ("fact-2", "ev-1"))
    conn.execute("INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)", ("fact-1", "act-1", "speaker"))
    conn.execute("INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)", ("fact-1", "act-2", "target"))
    conn.commit()

    fact_evidence_count = conn.execute(
        "SELECT COUNT(*) AS count FROM fact_evidence WHERE evidence_id = ?",
        ("ev-1",),
    ).fetchone()
    fact_actor_count = conn.execute(
        "SELECT COUNT(*) AS count FROM fact_actors WHERE fact_id = ?",
        ("fact-1",),
    ).fetchone()

    assert fact_evidence_count["count"] == 2
    assert fact_actor_count["count"] == 2


def test_interpretations_require_parent_and_validate_constraints(db_file):
    conn = init_db(db_file)
    _insert_thread(conn)
    _insert_evidence(conn, evidence_id="ev-1", seq=1)
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("fact-1", None, "statement", "原始事实", None, None, None, None, "unassigned", 1, 1),
    )
    conn.execute(
        """
        INSERT INTO interpretations (
            interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("itp-1", "fact-1", None, "explanation", "这是一段解释", 0.8, 1),
    )
    conn.execute(
        """
        INSERT INTO interpretations (
            interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("itp-2", None, "ev-1", "term", "术语解释", None, 2),
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO interpretations (
                interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("itp-bad-parent", None, None, "explanation", "无父节点", None, 3),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO interpretations (
                interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("itp-bad-confidence", "fact-1", None, "uncertainty", "x", -0.1, 4),
        )
