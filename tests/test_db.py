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


def _make_v2_db(path):
    conn = db_module._connect(path)
    try:
        db_module._ensure_meta_table(conn)
        db_module._create_v1_schema(conn)
        db_module._create_v2_schema(conn)
        db_module._set_schema_version(conn, 2)
        conn.commit()
    finally:
        conn.close()


def _make_v3_db(path):
    conn = db_module._connect(path)
    try:
        db_module._ensure_meta_table(conn)
        db_module._create_v1_schema(conn)
        db_module._create_v2_schema(conn)
        db_module._create_v3_schema(conn)
        db_module._set_schema_version(conn, 3)
        conn.commit()
    finally:
        conn.close()


def _make_v5_db(path):
    conn = db_module._connect(path)
    try:
        db_module._ensure_meta_table(conn)
        db_module._create_v1_schema(conn)
        db_module._create_v2_schema(conn)
        db_module._create_v3_schema(conn)
        db_module._create_v4_schema(conn)
        db_module._create_v5_schema(conn)
        db_module._set_schema_version(conn, 5)
        conn.commit()
    finally:
        conn.close()


def _make_v6_db(path):
    conn = db_module._connect(path)
    try:
        db_module._ensure_meta_table(conn)
        db_module._create_v1_schema(conn)
        db_module._create_v2_schema(conn)
        db_module._create_v3_schema(conn)
        db_module._create_v4_schema(conn)
        db_module._create_v5_schema(conn)
        db_module._create_v6_schema(conn)
        db_module._set_schema_version(conn, 6)
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
            'fact_evidence', 'fact_actors', 'interpretations',
            'evidence_extractions', 'semantic_runs', 'semantic_run_inputs',
            'event_match_runs', 'event_change_runs', 'event_changes'
        )
        ORDER BY name
        """
    ).fetchall()

    assert [row["name"] for row in rows] == [
        "actors",
        "checkpoints",
        "event_change_runs",
        "event_changes",
        "event_match_runs",
        "events",
        "evidence",
        "evidence_extractions",
        "fact_actors",
        "fact_evidence",
        "facts",
        "interpretations",
        "meta",
        "semantic_run_inputs",
        "semantic_runs",
        "submission_evidence",
        "submissions",
        "threads",
    ]
    fact_columns = conn.execute("PRAGMA table_info(facts)").fetchall()
    interpretation_columns = conn.execute("PRAGMA table_info(interpretations)").fetchall()

    assert get_schema_version(conn) == 9
    assert {column["name"] for column in fact_columns} >= {
        "due_anchor_at",
        "event_assignment_confidence",
        "origin",
        "review_status",
        "semantic_run_id",
    }
    assert {column["name"] for column in interpretation_columns} >= {"semantic_run_id"}


def test_init_db_is_idempotent_and_keeps_schema_version(db_file):
    conn1 = init_db(db_file)
    conn1.close()

    conn2 = init_db(db_file)

    assert get_schema_version(conn2) == 9


def test_v1_database_is_migrated_to_v8_without_losing_existing_rows(db_file):
    _make_v1_db(db_file)
    reopened = None
    try:
        conn = db_module._connect(db_file)
        _insert_actor(conn, "act-1", 0)
        _insert_thread(conn, thread_id="thr-1")
        _insert_evidence(conn, evidence_id="ev-1", seq=1, thread_id="thr-1")
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            ("custom:v1-note", "kept"),
        )
        conn.commit()
        conn.close()

        reopened = init_db(db_file)
        actor = reopened.execute("SELECT actor_id FROM actors WHERE actor_id = 'act-1'").fetchone()
        thread = reopened.execute("SELECT thread_id FROM threads WHERE thread_id = 'thr-1'").fetchone()
        evidence = reopened.execute("SELECT evidence_id FROM evidence WHERE evidence_id = 'ev-1'").fetchone()
        meta_row = reopened.execute(
            "SELECT value FROM meta WHERE key = ?",
            ("custom:v1-note",),
        ).fetchone()
        facts_table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts'"
        ).fetchone()

        extraction_table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'evidence_extractions'"
        ).fetchone()

        semantic_runs_table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'semantic_runs'"
        ).fetchone()

        event_match_runs_table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'event_match_runs'"
        ).fetchone()

        assert get_schema_version(reopened) == 9
        assert actor["actor_id"] == "act-1"
        assert thread["thread_id"] == "thr-1"
        assert evidence["evidence_id"] == "ev-1"
        assert meta_row["value"] == "kept"
        assert facts_table["name"] == "facts"
        assert extraction_table["name"] == "evidence_extractions"
        assert semantic_runs_table["name"] == "semantic_runs"
        assert event_match_runs_table["name"] == "event_match_runs"
    finally:
        if reopened is not None:
            reopened.close()


def test_v1_to_v8_migration_is_idempotent(db_file):
    _make_v1_db(db_file)

    first = init_db(db_file)
    first.close()
    second = init_db(db_file)
    try:
        assert get_schema_version(second) == 9
        tables = second.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN (
                'submissions', 'events', 'facts', 'interpretations',
                'evidence_extractions', 'semantic_runs', 'semantic_run_inputs',
                'event_match_runs'
            )
            ORDER BY name
            """
        ).fetchall()
        assert [row["name"] for row in tables] == [
            "event_match_runs",
            "events",
            "evidence_extractions",
            "facts",
            "interpretations",
            "semantic_run_inputs",
            "semantic_runs",
            "submissions",
        ]
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


def test_init_db_rejects_unsupported_low_schema_version(db_file):
    conn = db_module._connect(db_file)
    try:
        db_module._ensure_meta_table(conn)
        db_module._set_schema_version(conn, 0)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="unsupported"):
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

    assert get_schema_version(conn) == 9
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


def test_v2_database_with_existing_events_and_facts_migrates_to_v8_without_data_loss(db_file):
    _make_v2_db(db_file)

    conn = db_module._connect(db_file)
    try:
        conn.execute(
            """
            INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("evt-1", "渠道复盘", "active", "既有事件", 1723000000, 1723000001),
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
                "evt-1",
                "deadline_change",
                "原计划下周五交付",
                1723000000,
                1723600000,
                "下周五",
                0.61,
                "suggested",
                1723000002,
                1723000003,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = init_db(db_file)
    try:
        row = reopened.execute(
            """
            SELECT
                event_id, fact_type, content, due_at, due_raw, confidence,
                event_assignment, due_anchor_at, event_assignment_confidence,
                origin, review_status
            FROM facts
            WHERE fact_id = ?
            """,
            ("fact-1",),
        ).fetchone()

        assert get_schema_version(reopened) == 9
        assert row["event_id"] == "evt-1"
        assert row["fact_type"] == "deadline_change"
        assert row["content"] == "原计划下周五交付"
        assert row["due_at"] == 1723600000
        assert row["due_raw"] == "下周五"
        assert row["confidence"] == 0.61
        assert row["event_assignment"] == "suggested"
        assert row["due_anchor_at"] is None
        assert row["event_assignment_confidence"] is None
        assert row["origin"] == "ai"
        assert row["review_status"] == "unreviewed"
    finally:
        reopened.close()


def test_facts_v3_defaults_and_checks_are_enforced(db_file):
    conn = init_db(db_file)

    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("fact-defaults", None, "statement", "先记一条", None, None, "下下周五", 0.4, "unassigned", 1, 1),
    )

    row = conn.execute(
        """
        SELECT due_anchor_at, event_assignment_confidence, origin, review_status
        FROM facts WHERE fact_id = ?
        """,
        ("fact-defaults",),
    ).fetchone()

    assert row["due_anchor_at"] is None
    assert row["event_assignment_confidence"] is None
    assert row["origin"] == "ai"
    assert row["review_status"] == "unreviewed"

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                event_assignment_confidence, event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-assignment-confidence", None, "statement", "x", 0.5, 1.1, "unassigned", 1, 1),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                origin, event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-origin", None, "statement", "x", 0.5, "human", "unassigned", 1, 1),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, confidence,
                review_status, event_assignment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("fact-bad-review", None, "statement", "x", 0.5, "approved", "unassigned", 1, 1),
        )


def test_v3_database_migrates_to_v8_without_losing_existing_rows(db_file):
    _make_v3_db(db_file)
    conn = db_module._connect(db_file)
    try:
        _insert_thread(conn)
        _insert_evidence(conn, evidence_id="ev-1", seq=1)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            ("custom:v3-note", "still-here"),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = init_db(db_file)
    try:
        extraction_columns = reopened.execute("PRAGMA table_info(evidence_extractions)").fetchall()
        evidence = reopened.execute(
            "SELECT evidence_id FROM evidence WHERE evidence_id = ?",
            ("ev-1",),
        ).fetchone()
        meta_row = reopened.execute(
            "SELECT value FROM meta WHERE key = ?",
            ("custom:v3-note",),
        ).fetchone()

        assert get_schema_version(reopened) == 9
        assert evidence["evidence_id"] == "ev-1"
        assert meta_row["value"] == "still-here"
        assert {column["name"] for column in extraction_columns} >= {
            "extraction_id",
            "evidence_id",
            "origin",
            "provider",
            "model",
            "transcript",
            "observations",
            "warnings",
            "created_at",
            "supersedes_extraction_id",
        }
    finally:
        reopened.close()


def test_v5_database_migrates_to_v8_without_losing_semantic_rows(db_file):
    _make_v5_db(db_file)

    conn = db_module._connect(db_file)
    try:
        _insert_thread(conn)
        _insert_evidence(conn, evidence_id="ev-1", seq=1)
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
                confidence, event_assignment, created_at, updated_at,
                due_anchor_at, event_assignment_confidence, origin, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fact-1",
                None,
                "statement",
                "原有事实",
                1,
                None,
                None,
                0.4,
                "unassigned",
                2,
                3,
                None,
                None,
                "ai",
                "unreviewed",
            ),
        )
        conn.execute(
            """
            INSERT INTO interpretations (
                interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("itp-1", "fact-1", None, "explanation", "原有解释", 0.5, 4),
        )
        conn.execute(
            """
            INSERT INTO evidence_extractions (
                extraction_id, evidence_id, origin, provider, model,
                transcript, observations, warnings, created_at, supersedes_extraction_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ext-1", "ev-1", "machine", "dashscope", "ocr-v1", "原有转录", "[]", "[]", 5, None),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = init_db(db_file)
    try:
        fact_row = reopened.execute(
            """
            SELECT fact_id, semantic_run_id, origin, review_status
            FROM facts
            WHERE fact_id = ?
            """,
            ("fact-1",),
        ).fetchone()
        interpretation_row = reopened.execute(
            """
            SELECT interpretation_id, semantic_run_id, fact_id, content
            FROM interpretations
            WHERE interpretation_id = ?
            """,
            ("itp-1",),
        ).fetchone()
        extraction_row = reopened.execute(
            "SELECT extraction_id, transcript, warnings FROM evidence_extractions WHERE extraction_id = ?",
            ("ext-1",),
        ).fetchone()
        run_tables = reopened.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN ('semantic_runs', 'semantic_run_inputs')
            ORDER BY name
            """
        ).fetchall()

        assert get_schema_version(reopened) == 9
        assert fact_row["fact_id"] == "fact-1"
        assert fact_row["semantic_run_id"] is None
        assert interpretation_row["interpretation_id"] == "itp-1"
        assert interpretation_row["semantic_run_id"] is None
        assert extraction_row["extraction_id"] == "ext-1"
        assert extraction_row["transcript"] == "原有转录"
        assert [row["name"] for row in run_tables] == ["semantic_run_inputs", "semantic_runs"]
    finally:
        reopened.close()


def test_v6_database_migrates_to_v8_without_losing_existing_rows(db_file):
    _make_v6_db(db_file)

    conn = db_module._connect(db_file)
    try:
        _insert_thread(conn)
        _insert_evidence(conn, evidence_id="ev-1", seq=1)
        conn.execute(
            """
            INSERT INTO semantic_runs (
                semantic_run_id, provider, model, parser_version, status,
                anchor_date, created_at, completed_at, failure_type, supersedes_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("srun-1", "deepseek", "deepseek-v4-flash", "2.2", "succeeded", None, 1, 2, None, None),
        )
        conn.execute(
            """
            INSERT INTO semantic_run_inputs (
                semantic_run_id, evidence_id, extraction_id, position
            ) VALUES (?, ?, ?, ?)
            """,
            ("srun-1", "ev-1", None, 0),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = init_db(db_file)
    try:
        event_match_columns = reopened.execute("PRAGMA table_info(event_match_runs)").fetchall()
        semantic_run = reopened.execute(
            "SELECT semantic_run_id, status FROM semantic_runs WHERE semantic_run_id = ?",
            ("srun-1",),
        ).fetchone()

        assert get_schema_version(reopened) == 9
        assert semantic_run["semantic_run_id"] == "srun-1"
        assert semantic_run["status"] == "succeeded"
        assert {column["name"] for column in event_match_columns} >= {
            "event_match_run_id",
            "semantic_run_id",
            "provider",
            "model",
            "matcher_version",
            "status",
            "routing_mode",
            "result_json",
            "failure_type",
            "created_at",
            "completed_at",
            "supersedes_run_id",
            "review_status",
            "reviewed_at",
        }
    finally:
        reopened.close()


def test_fact_confidence_and_event_assignment_confidence_are_independent(db_file):
    conn = init_db(db_file)
    conn.execute(
        """
        INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("evt-1", "渠道复盘", "active", None, 1, 1),
    )
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, confidence,
            event_assignment, event_assignment_confidence, created_at, updated_at,
            due_raw, due_anchor_at, origin, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "fact-1",
            "evt-1",
            "request",
            "请补一份渠道复盘数据",
            0.27,
            "confirmed",
            0.91,
            1,
            2,
            "下下周五",
            1723000000,
            "user",
            "corrected",
        ),
    )
    conn.commit()

    row = conn.execute(
        """
        SELECT confidence, event_assignment_confidence, due_raw, due_anchor_at, origin, review_status
        FROM facts WHERE fact_id = ?
        """,
        ("fact-1",),
    ).fetchone()

    assert row["confidence"] == 0.27
    assert row["event_assignment_confidence"] == 0.91
    assert row["due_raw"] == "下下周五"
    assert row["due_anchor_at"] == 1723000000
    assert row["origin"] == "user"
    assert row["review_status"] == "corrected"
