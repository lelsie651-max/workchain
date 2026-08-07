from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


def _connect(path: str | Path) -> sqlite3.Connection:
    database = str(path)
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
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
            slot_direction TEXT CHECK (
                slot_direction IS NULL
                OR slot_direction IN ('i_owe', 'owed_to_me', 'none')
            ),
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

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
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


def _ensure_meta(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def init_db(path: str | Path) -> sqlite3.Connection:
    conn = _connect(path)
    _create_schema(conn)
    _ensure_meta(conn)
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
