from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from evidence_core.db import init_db
from evidence_core.export import export_evidence_package
from evidence_core.store import verify_chain


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_seed(out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.seed_demo", "--out", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _evidence_snapshot(conn: sqlite3.Connection) -> list[tuple]:
    rows = conn.execute(
        """
        SELECT
            evidence_id, seq, thread_id, kind, source_hint, occurred_at, captured_at,
            slots_filled, plain_summary, caveats, content_hash, prev_hash, chain_hash
        FROM evidence
        ORDER BY seq
        """
    ).fetchall()
    return [tuple(row) for row in rows]


def _thread_snapshot(conn: sqlite3.Connection) -> list[tuple]:
    rows = conn.execute(
        """
        SELECT
            thread_id, title, status, owner_actor_id, requester_actor_id,
            current_deliverable, current_due, version, risk_flags,
            first_seen_at, last_activity_at
        FROM threads
        ORDER BY thread_id
        """
    ).fetchall()
    return [tuple(row) for row in rows]


def test_seed_demo_creates_18_evidence_with_continuous_seq(tmp_path):
    out_dir = tmp_path / "demo_data"
    result = _run_seed(out_dir)

    assert result.returncode == 0, result.stdout + result.stderr

    conn = _connect(out_dir / "workchain.db")
    try:
        rows = conn.execute("SELECT seq FROM evidence ORDER BY seq").fetchall()
        assert len(rows) == 18
        assert [row["seq"] for row in rows] == list(range(1, 19))
    finally:
        conn.close()


def test_seed_demo_verify_chain_returns_true(tmp_path):
    out_dir = tmp_path / "demo_data"
    result = _run_seed(out_dir)

    assert result.returncode == 0, result.stdout + result.stderr

    conn = init_db(out_dir / "workchain.db")
    try:
        assert verify_chain(conn, blobs_root=out_dir / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_reference_records_keep_zero_slots(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        rows = conn.execute(
            "SELECT slots_filled FROM evidence WHERE kind = 'reference' ORDER BY seq"
        ).fetchall()
        assert len(rows) == 4
        assert all(row["slots_filled"] == 0 for row in rows)
    finally:
        conn.close()


def test_non_reference_records_have_rich_slot_fill(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        rows = conn.execute(
            "SELECT slots_filled FROM evidence WHERE kind != 'reference' ORDER BY seq"
        ).fetchall()
        assert len(rows) == 14
        assert all(row["slots_filled"] >= 3 for row in rows)
    finally:
        conn.close()


def test_seed_demo_creates_three_threads_and_channel_version_four(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        count = conn.execute("SELECT COUNT(*) AS count FROM threads").fetchone()["count"]
        version = conn.execute(
            "SELECT version FROM threads WHERE thread_id = 'thr_channel'"
        ).fetchone()["version"]
        assert count == 3
        assert version == 4
    finally:
        conn.close()


def test_reference_records_have_null_thread_id(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        rows = conn.execute(
            "SELECT thread_id FROM evidence WHERE kind = 'reference' ORDER BY seq"
        ).fetchall()
        assert len(rows) == 4
        assert all(row["thread_id"] is None for row in rows)
    finally:
        conn.close()


def test_non_reference_records_have_non_null_thread_id(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        rows = conn.execute(
            "SELECT thread_id FROM evidence WHERE kind != 'reference' ORDER BY seq"
        ).fetchall()
        assert len(rows) == 14
        assert all(row["thread_id"] is not None for row in rows)
    finally:
        conn.close()


def test_slots_filled_distribution_matches_story_design(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        count_three = conn.execute(
            "SELECT COUNT(*) AS count FROM evidence WHERE slots_filled = 3"
        ).fetchone()["count"]
        count_four = conn.execute(
            "SELECT COUNT(*) AS count FROM evidence WHERE slots_filled = 4"
        ).fetchone()["count"]
        assert count_three == 2
        assert count_four == 12
    finally:
        conn.close()


def test_thr_apidoc_status_is_open(tmp_path):
    out_dir = tmp_path / "demo_data"
    _run_seed(out_dir)

    conn = _connect(out_dir / "workchain.db")
    try:
        status = conn.execute(
            "SELECT status FROM threads WHERE thread_id = 'thr_apidoc'"
        ).fetchone()["status"]
        assert status == "open"
    finally:
        conn.close()


def test_seed_demo_is_repeatable_across_two_runs(tmp_path):
    out_dir = tmp_path / "demo_data"

    first = _run_seed(out_dir)
    assert first.returncode == 0, first.stdout + first.stderr
    conn = init_db(out_dir / "workchain.db")
    try:
        first_verify = verify_chain(conn, blobs_root=out_dir / "blobs")
    finally:
        conn.close()
    conn = _connect(out_dir / "workchain.db")
    try:
        first_evidence = _evidence_snapshot(conn)
        first_threads = _thread_snapshot(conn)
        first_blobs = sorted(
            str(path.relative_to(out_dir))
            for path in out_dir.rglob("*")
            if path.is_file() and path.name != "workchain.db"
        )
    finally:
        conn.close()

    second = _run_seed(out_dir)
    assert second.returncode == 0, second.stdout + second.stderr
    conn = init_db(out_dir / "workchain.db")
    try:
        second_verify = verify_chain(conn, blobs_root=out_dir / "blobs")
    finally:
        conn.close()
    conn = _connect(out_dir / "workchain.db")
    try:
        second_evidence = _evidence_snapshot(conn)
        second_threads = _thread_snapshot(conn)
        second_blobs = sorted(
            str(path.relative_to(out_dir))
            for path in out_dir.rglob("*")
            if path.is_file() and path.name != "workchain.db"
        )
    finally:
        conn.close()

    assert first_verify == (True, None, None)
    assert second_verify == (True, None, None)
    assert first_evidence == second_evidence
    assert first_threads == second_threads
    assert first_blobs == second_blobs


def test_seed_demo_database_exports_and_verify_py_passes(tmp_path):
    out_dir = tmp_path / "demo_data"
    result = _run_seed(out_dir)

    assert result.returncode == 0, result.stdout + result.stderr

    export_dir = tmp_path / "export"
    conn = init_db(out_dir / "workchain.db")
    try:
        export_evidence_package(conn, blobs_root=out_dir / "blobs", out_dir=export_dir)
    finally:
        conn.close()

    verify_result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert verify_result.returncode == 0
    assert "OK 18" in verify_result.stdout
