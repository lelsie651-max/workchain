from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 5


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
    elif schema_version == 2:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _migrate_v2_to_v3(conn)
        _migrate_v3_to_v4(conn)
        _migrate_v4_to_v5(conn)
    elif schema_version == 3:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _migrate_v3_to_v4(conn)
        _migrate_v4_to_v5(conn)
    elif schema_version == 4:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _migrate_v4_to_v5(conn)
    elif schema_version == 5:
        _create_v1_schema(conn)
        _create_v2_schema(conn)
        _create_v3_schema(conn)
        _create_v4_schema(conn)
        _create_v5_schema(conn)
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
