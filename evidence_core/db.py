from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 11


def _connect(path: str | Path) -> sqlite3.Connection:
    database = str(path)
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _create_v1_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS actors (
            actor_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            aliases TEXT,
            org TEXT,
            role_hint TEXT,
            is_self INTEGER NOT NULL CHECK (is_self IN (0, 1)),
            confidence REAL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('open', 'delivered', 'closed', 'disputed', 'abandoned')
            ),
            owner_actor_id TEXT,
            requester_actor_id TEXT,
            current_deliverable TEXT,
            current_due INTEGER,
            version INTEGER NOT NULL,
            risk_flags TEXT,
            first_seen_at INTEGER NOT NULL,
            last_activity_at INTEGER NOT NULL,
            FOREIGN KEY (owner_actor_id) REFERENCES actors(actor_id) ON DELETE RESTRICT,
            FOREIGN KEY (requester_actor_id) REFERENCES actors(actor_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            seq INTEGER NOT NULL UNIQUE,
            thread_id TEXT,
            kind TEXT NOT NULL CHECK (
                kind IN ('request', 'confirm', 'change', 'deliver', 'dispute', 'reference')
            ),
            media_type TEXT NOT NULL CHECK (media_type IN ('image', 'text', 'file')),
            blob_path TEXT,
            raw_text TEXT,
            source_hint TEXT,
            slot_requester TEXT,
            slot_owner TEXT,
            slot_deliverable TEXT,
            slot_due INTEGER,
            slot_due_raw TEXT,
            slot_direction TEXT CHECK (slot_direction IN ('i_owe', 'owed_to_me', 'none')),
            slots_filled INTEGER NOT NULL CHECK (slots_filled BETWEEN 0 AND 4),
            plain_summary TEXT,
            caveats TEXT,
            occurred_at INTEGER,
            captured_at INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            chain_hash TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE RESTRICT,
            FOREIGN KEY (slot_requester) REFERENCES actors(actor_id) ON DELETE RESTRICT,
            FOREIGN KEY (slot_owner) REFERENCES actors(actor_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            at_seq INTEGER NOT NULL,
            chain_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            tsa_token BLOB
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_actors_is_self_true
            ON actors(is_self) WHERE is_self = 1;

        CREATE INDEX IF NOT EXISTS idx_evidence_thread_id
            ON evidence(thread_id);

        CREATE INDEX IF NOT EXISTS idx_evidence_captured_at
            ON evidence(captured_at);

        CREATE INDEX IF NOT EXISTS idx_threads_status
            ON threads(status);

        CREATE INDEX IF NOT EXISTS idx_checkpoints_at_seq
            ON checkpoints(at_seq);
        """
    )


def _create_v2_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            source_hint TEXT
        );

        CREATE TABLE IF NOT EXISTS submission_evidence (
            submission_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL UNIQUE,
            position INTEGER NOT NULL CHECK(position >= 0),
            PRIMARY KEY (submission_id, evidence_id),
            UNIQUE (submission_id, position),
            FOREIGN KEY (submission_id) REFERENCES submissions(submission_id) ON DELETE RESTRICT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('active', 'resolved', 'archived')
            ),
            summary TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            event_id TEXT,
            fact_type TEXT NOT NULL CHECK (
                fact_type IN (
                    'request', 'commitment', 'confirmation', 'scope_change',
                    'responsibility_change', 'deadline_change', 'delivery',
                    'cancellation', 'denial', 'statement', 'reference'
                )
            ),
            content TEXT NOT NULL,
            occurred_at INTEGER,
            due_at INTEGER,
            due_raw TEXT,
            confidence REAL CHECK (
                confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
            ),
            event_assignment TEXT NOT NULL DEFAULT 'unassigned' CHECK (
                event_assignment IN ('unassigned', 'auto', 'suggested', 'confirmed')
            ),
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE RESTRICT,
            CHECK (
                (event_assignment = 'unassigned' AND event_id IS NULL)
                OR
                (event_assignment IN ('auto', 'suggested', 'confirmed') AND event_id IS NOT NULL)
            )
        );

        CREATE TABLE IF NOT EXISTS fact_evidence (
            fact_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            PRIMARY KEY (fact_id, evidence_id),
            FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE RESTRICT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS fact_actors (
            fact_id TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY (fact_id, actor_id, role),
            FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE RESTRICT,
            FOREIGN KEY (actor_id) REFERENCES actors(actor_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS interpretations (
            interpretation_id TEXT PRIMARY KEY,
            fact_id TEXT,
            evidence_id TEXT,
            kind TEXT NOT NULL CHECK (
                kind IN ('explanation', 'term', 'action_hint', 'uncertainty')
            ),
            content TEXT NOT NULL,
            confidence REAL CHECK (
                confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
            ),
            created_at INTEGER NOT NULL,
            FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE RESTRICT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT,
            CHECK (fact_id IS NOT NULL OR evidence_id IS NOT NULL)
        );

        CREATE INDEX IF NOT EXISTS idx_submission_evidence_submission_id
            ON submission_evidence(submission_id);

        CREATE INDEX IF NOT EXISTS idx_facts_event_id
            ON facts(event_id);

        CREATE INDEX IF NOT EXISTS idx_fact_evidence_evidence_id
            ON fact_evidence(evidence_id);

        CREATE INDEX IF NOT EXISTS idx_fact_actors_actor_id
            ON fact_actors(actor_id);

        CREATE INDEX IF NOT EXISTS idx_interpretations_fact_id
            ON interpretations(fact_id);

        CREATE INDEX IF NOT EXISTS idx_interpretations_evidence_id
            ON interpretations(evidence_id);
        """
    )


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def _create_v3_schema(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "facts", "due_anchor_at"):
        conn.execute("ALTER TABLE facts ADD COLUMN due_anchor_at INTEGER")
    if not _column_exists(conn, "facts", "event_assignment_confidence"):
        conn.execute(
            """
            ALTER TABLE facts
            ADD COLUMN event_assignment_confidence REAL CHECK (
                event_assignment_confidence IS NULL
                OR (
                    event_assignment_confidence >= 0
                    AND event_assignment_confidence <= 1
                )
            )
            """
        )
    if not _column_exists(conn, "facts", "origin"):
        conn.execute(
            """
            ALTER TABLE facts
            ADD COLUMN origin TEXT NOT NULL DEFAULT 'ai' CHECK (
                origin IN ('ai', 'user')
            )
            """
        )
    if not _column_exists(conn, "facts", "review_status"):
        conn.execute(
            """
            ALTER TABLE facts
            ADD COLUMN review_status TEXT NOT NULL DEFAULT 'unreviewed' CHECK (
                review_status IN ('unreviewed', 'confirmed', 'corrected')
            )
            """
        )


def _create_v4_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS evidence_extractions (
            extraction_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            origin TEXT NOT NULL CHECK (origin IN ('machine', 'user')),
            provider TEXT NOT NULL,
            model TEXT,
            transcript TEXT,
            observations TEXT NOT NULL DEFAULT '[]',
            warnings TEXT NOT NULL DEFAULT '[]',
            created_at INTEGER NOT NULL,
            supersedes_extraction_id TEXT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT,
            FOREIGN KEY (supersedes_extraction_id) REFERENCES evidence_extractions(extraction_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_evidence_extractions_evidence_id_created_at
            ON evidence_extractions(evidence_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_evidence_extractions_supersedes
            ON evidence_extractions(supersedes_extraction_id);
        """
    )


def _create_v5_schema(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "evidence_extractions", "warnings"):
        conn.execute(
            """
            ALTER TABLE evidence_extractions
            ADD COLUMN warnings TEXT NOT NULL DEFAULT '[]'
            """
        )


def _create_v6_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS semantic_runs (
            semantic_run_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
            anchor_date TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            failure_type TEXT,
            supersedes_run_id TEXT,
            FOREIGN KEY (supersedes_run_id) REFERENCES semantic_runs(semantic_run_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS semantic_run_inputs (
            semantic_run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            extraction_id TEXT,
            position INTEGER NOT NULL CHECK (position >= 0),
            PRIMARY KEY (semantic_run_id, evidence_id),
            UNIQUE (semantic_run_id, position),
            FOREIGN KEY (semantic_run_id) REFERENCES semantic_runs(semantic_run_id) ON DELETE RESTRICT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT,
            FOREIGN KEY (extraction_id) REFERENCES evidence_extractions(extraction_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_semantic_runs_supersedes
            ON semantic_runs(supersedes_run_id);

        CREATE INDEX IF NOT EXISTS idx_semantic_run_inputs_evidence_id
            ON semantic_run_inputs(evidence_id);
        """
    )

    if not _column_exists(conn, "facts", "semantic_run_id"):
        conn.execute(
            """
            ALTER TABLE facts
            ADD COLUMN semantic_run_id TEXT REFERENCES semantic_runs(semantic_run_id) ON DELETE RESTRICT
            """
        )
    if not _column_exists(conn, "interpretations", "semantic_run_id"):
        conn.execute(
            """
            ALTER TABLE interpretations
            ADD COLUMN semantic_run_id TEXT REFERENCES semantic_runs(semantic_run_id) ON DELETE RESTRICT
            """
        )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_facts_semantic_run_id
            ON facts(semantic_run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_interpretations_semantic_run_id
            ON interpretations(semantic_run_id)
        """
    )


def _create_v7_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_match_runs (
            event_match_run_id TEXT PRIMARY KEY,
            semantic_run_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            matcher_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
            routing_mode TEXT CHECK (routing_mode IN ('auto', 'confirm', 'needs_context')),
            result_json TEXT,
            failure_type TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            supersedes_run_id TEXT,
            FOREIGN KEY (semantic_run_id) REFERENCES semantic_runs(semantic_run_id) ON DELETE RESTRICT,
            FOREIGN KEY (supersedes_run_id) REFERENCES event_match_runs(event_match_run_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_event_match_runs_semantic_run_id
            ON event_match_runs(semantic_run_id);

        CREATE INDEX IF NOT EXISTS idx_event_match_runs_supersedes
            ON event_match_runs(supersedes_run_id);
        """
    )


def _create_v8_schema(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "event_match_runs", "review_status"):
        conn.execute(
            """
            ALTER TABLE event_match_runs
            ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending' CHECK (
                review_status IN ('pending', 'completed')
            )
            """
        )
    if not _column_exists(conn, "event_match_runs", "reviewed_at"):
        conn.execute(
            """
            ALTER TABLE event_match_runs
            ADD COLUMN reviewed_at INTEGER
            """
        )


def _create_v9_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_change_runs (
            change_run_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            detector_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            failure_type TEXT,
            FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS event_changes (
            change_id TEXT PRIMARY KEY,
            change_run_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            change_type TEXT NOT NULL CHECK (
                change_type IN (
                    'requirement_change',
                    'deadline_change',
                    'responsibility_change',
                    'contradiction'
                )
            ),
            earlier_fact_id TEXT NOT NULL,
            later_fact_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            created_at INTEGER NOT NULL,
            FOREIGN KEY (change_run_id) REFERENCES event_change_runs(change_run_id) ON DELETE RESTRICT,
            FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE RESTRICT,
            FOREIGN KEY (earlier_fact_id) REFERENCES facts(fact_id) ON DELETE RESTRICT,
            FOREIGN KEY (later_fact_id) REFERENCES facts(fact_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_event_change_runs_event_id
            ON event_change_runs(event_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_event_changes_change_run_id
            ON event_changes(change_run_id);

        CREATE INDEX IF NOT EXISTS idx_event_changes_event_id
            ON event_changes(event_id);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_event_changes_run_tuple
            ON event_changes(change_run_id, change_type, earlier_fact_id, later_fact_id);
        """
    )


def _create_v10_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_reviews (
            review_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            extraction_id TEXT NOT NULL,
            original_source_hint TEXT NOT NULL,
            observed_platform TEXT,
            resolved_source_hint TEXT NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('confirmed_declared', 'corrected')),
            created_at INTEGER NOT NULL,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT,
            FOREIGN KEY (extraction_id) REFERENCES evidence_extractions(extraction_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_source_reviews_evidence_id_created_at
            ON source_reviews(evidence_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_source_reviews_extraction_id_created_at
            ON source_reviews(extraction_id, created_at DESC);
        """
    )


def _create_v11_schema(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "evidence_extractions", "structured_payload"):
        conn.execute(
            """
            ALTER TABLE evidence_extractions
            ADD COLUMN structured_payload TEXT
            """
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS extraction_speaker_reviews (
            review_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            extraction_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('provided', 'skipped')),
            labels_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id) ON DELETE RESTRICT,
            FOREIGN KEY (extraction_id) REFERENCES evidence_extractions(extraction_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_extraction_speaker_reviews_evidence_id_created_at
            ON extraction_speaker_reviews(evidence_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_extraction_speaker_reviews_extraction_id_created_at
            ON extraction_speaker_reviews(extraction_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS context_assembly_runs (
            assembly_run_id TEXT PRIMARY KEY,
            submission_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            assembler_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
            result_json TEXT,
            failure_type TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            FOREIGN KEY (submission_id) REFERENCES submissions(submission_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_context_assembly_runs_submission_id_created_at
            ON context_assembly_runs(submission_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS context_group_reviews (
            review_id TEXT PRIMARY KEY,
            assembly_run_id TEXT NOT NULL,
            group_key TEXT NOT NULL,
            review_status TEXT NOT NULL CHECK (review_status IN ('accepted', 'needs_user_review')),
            decision_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (assembly_run_id) REFERENCES context_assembly_runs(assembly_run_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_context_group_reviews_assembly_run_id_created_at
            ON context_group_reviews(assembly_run_id, created_at DESC);
        """
    )

    if not _column_exists(conn, "semantic_runs", "context_group_key"):
        conn.execute(
            """
            ALTER TABLE semantic_runs
            ADD COLUMN context_group_key TEXT
            """
        )
    if not _column_exists(conn, "semantic_runs", "context_assembly_run_id"):
        conn.execute(
            """
            ALTER TABLE semantic_runs
            ADD COLUMN context_assembly_run_id TEXT REFERENCES context_assembly_runs(assembly_run_id) ON DELETE RESTRICT
            """
        )


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        ("schema_version", str(version)),
    )


def _read_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        ("schema_version",),
    ).fetchone()
    if row is None:
        return None
    return int(row["value"])


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    _create_v2_schema(conn)
    _set_schema_version(conn, 2)


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    _create_v3_schema(conn)
    _set_schema_version(conn, 3)


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    _create_v4_schema(conn)
    _set_schema_version(conn, 4)


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    _create_v5_schema(conn)
    _set_schema_version(conn, 5)


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    _create_v6_schema(conn)
    _set_schema_version(conn, 6)


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    _create_v7_schema(conn)
    _set_schema_version(conn, 7)


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    _create_v8_schema(conn)
    conn.execute(
        """
        UPDATE event_match_runs
        SET review_status = 'completed',
            reviewed_at = completed_at
        WHERE status = 'succeeded' AND routing_mode = 'auto'
        """
    )
    conn.execute(
        """
        UPDATE event_match_runs
        SET review_status = 'pending',
            reviewed_at = NULL
        WHERE NOT (status = 'succeeded' AND routing_mode = 'auto')
        """
    )
    _set_schema_version(conn, 8)


def _migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    _create_v9_schema(conn)
    _set_schema_version(conn, 9)


def _migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    _create_v10_schema(conn)
    _set_schema_version(conn, 10)


def _migrate_v10_to_v11(conn: sqlite3.Connection) -> None:
    _create_v11_schema(conn)
    _set_schema_version(conn, 11)


def _is_fresh_database(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name <> 'meta'
        """
    ).fetchall()
    return len(rows) == 0


def init_db(path: str | Path) -> sqlite3.Connection:
    conn = _connect(path)
    _ensure_meta_table(conn)
    schema_version = _read_schema_version(conn)

    if schema_version is None:
        if not _is_fresh_database(conn):
            conn.close()
            raise ValueError("database schema_version is missing or unsupported")
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _create_v8_schema(conn)
        _create_v9_schema(conn)
        _create_v10_schema(conn)
        _create_v11_schema(conn)
        _set_schema_version(conn, SCHEMA_VERSION)
    elif schema_version > SCHEMA_VERSION:
        conn.close()
        raise ValueError(
            f"database schema_version {schema_version} is newer than supported version {SCHEMA_VERSION}"
        )
    elif schema_version < 1:
        conn.close()
        raise ValueError(
            f"database schema_version {schema_version} is unsupported; supported versions are 1..{SCHEMA_VERSION}"
        )
    elif schema_version == 1:
        _create_v1_schema(conn)
        _migrate_v1_to_v2(conn)
        _migrate_v2_to_v3(conn)
        _migrate_v3_to_v4(conn)
        _migrate_v4_to_v5(conn)
        _migrate_v5_to_v6(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 2:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _migrate_v2_to_v3(conn)
        _migrate_v3_to_v4(conn)
        _migrate_v4_to_v5(conn)
        _migrate_v5_to_v6(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 3:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _migrate_v3_to_v4(conn)
        _migrate_v4_to_v5(conn)
        _migrate_v5_to_v6(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 4:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _migrate_v4_to_v5(conn)
        _migrate_v5_to_v6(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 5:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _migrate_v5_to_v6(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 6:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _migrate_v6_to_v7(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 7:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _migrate_v7_to_v8(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 8:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _create_v8_schema(conn)
        _migrate_v8_to_v9(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 9:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _create_v8_schema(conn)
        _create_v9_schema(conn)
        _migrate_v9_to_v10(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 10:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _create_v8_schema(conn)
        _create_v9_schema(conn)
        _create_v10_schema(conn)
        _migrate_v10_to_v11(conn)
    elif schema_version == 11:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
        _create_v6_schema(conn)
        _create_v7_schema(conn)
        _create_v8_schema(conn)
        _create_v9_schema(conn)
        _create_v10_schema(conn)
        _create_v11_schema(conn)
    else:
        conn.close()
        raise ValueError(
            f"database schema_version {schema_version} is unsupported; supported versions are 1..{SCHEMA_VERSION}"
        )

    conn.commit()
    return conn


def get_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?",
        ("schema_version",),
    ).fetchone()
    if row is None:
        raise ValueError("schema_version not found")
    return int(row["value"])
