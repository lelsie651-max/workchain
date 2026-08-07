from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from evidence_core.db import init_db
from evidence_core.store import (
    append_evidence,
    make_checkpoint,
    update_slots,
    verify_chain,
)


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "workchain.db"


@pytest.fixture
def blobs_root(tmp_path):
    path = tmp_path / "blobs"
    path.mkdir()
    return path


def _insert_actor(conn, actor_id: str) -> None:
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (actor_id, actor_id, "[]", None, None, 0, None, 1723000000),
    )
    conn.commit()


def _all_seqs(conn) -> list[int]:
    rows = conn.execute("SELECT seq FROM evidence ORDER BY seq").fetchall()
    return [row["seq"] for row in rows]


def _append_text(conn, blobs_root: Path, *, text: str, captured_at: int, **kwargs):
    return append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="text",
        payload=text,
        captured_at=captured_at,
        **kwargs,
    )


def test_append_evidence_100_times_verifies_cleanly(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    assert verify_chain(conn) == (True, None, None)


def test_seq_starts_at_one_and_is_strictly_continuous(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(5):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    assert _all_seqs(conn) == [1, 2, 3, 4, 5]


def test_same_payload_reuses_single_blob_file_but_creates_two_evidence_rows(db_file, blobs_root):
    conn = init_db(db_file)

    first = _append_text(conn, blobs_root, text="same payload", captured_at=1723000000)
    second = _append_text(conn, blobs_root, text="same payload", captured_at=1723000001)

    files = [path for path in blobs_root.rglob("*") if path.is_file()]

    assert len(files) == 1
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["blob_path"] == second["blob_path"]


def test_concurrent_appends_keep_seq_unique_and_continuous(db_file, blobs_root):
    init_db(db_file).close()
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker(prefix: str) -> None:
        conn = init_db(db_file)
        try:
            barrier.wait()
            for index in range(50):
                _append_text(
                    conn,
                    blobs_root,
                    text=f"{prefix}-{index}",
                    captured_at=1723000000 + index,
                    source_hint=prefix,
                )
        except Exception as exc:  # pragma: no cover - exercised only on failure
            errors.append(exc)
        finally:
            conn.close()

    left = threading.Thread(target=worker, args=("left",))
    right = threading.Thread(target=worker, args=("right",))
    left.start()
    right.start()
    left.join()
    right.join()

    assert errors == []

    conn = init_db(db_file)
    seqs = _all_seqs(conn)

    assert len(seqs) == 100
    assert seqs == list(range(1, 101))
    assert verify_chain(conn, blobs_root=blobs_root) == (True, None, None)


def test_verify_chain_detects_tampered_blob_and_reports_correct_seq(db_file, blobs_root):
    conn = init_db(db_file)
    first = _append_text(conn, blobs_root, text="safe", captured_at=1723000000)
    second = _append_text(conn, blobs_root, text="target", captured_at=1723000001)

    blob_path = blobs_root / second["blob_path"]
    blob_path.write_bytes(b"tampered")

    assert first["seq"] == 1
    assert verify_chain(conn, blobs_root=blobs_root) == (False, 2, "content_hash mismatch")


def test_verify_chain_detects_database_tampering_on_occurred_at(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="one", captured_at=1723000000, occurred_at=1722990000)
    row = _append_text(conn, blobs_root, text="two", captured_at=1723000001, occurred_at=1722990001)

    conn.execute(
        "UPDATE evidence SET occurred_at = ? WHERE evidence_id = ?",
        (1722999999, row["evidence_id"]),
    )
    conn.commit()

    assert verify_chain(conn) == (False, 2, "chain_hash mismatch")


def test_verify_chain_detects_seq_gap_after_delete(db_file, blobs_root):
    conn = init_db(db_file)
    rows = [
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)
        for index in range(5)
    ]

    conn.execute("DELETE FROM evidence WHERE evidence_id = ?", (rows[2]["evidence_id"],))
    conn.commit()

    assert verify_chain(conn) == (False, 3, "seq gap")


def test_update_slots_does_not_break_chain_verification(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    row = _append_text(conn, blobs_root, text="payload", captured_at=1723000000, kind="request")

    updated = update_slots(
        conn,
        row["evidence_id"],
        slot_requester="act-1",
        slot_owner="act-2",
        slot_deliverable="slides",
        slot_due=1724000000,
        slot_due_raw="下周五",
        slot_direction="i_owe",
        plain_summary="整理并提交",
        caveats=["未指定具体时间"],
    )

    assert json.loads(updated["caveats"]) == ["未指定具体时间"]
    assert verify_chain(conn) == (True, None, None)


@pytest.mark.parametrize("field_name", ["chain_hash", "seq", "slots_filled"])
def test_update_slots_rejects_protected_fields(db_file, blobs_root, field_name: str):
    conn = init_db(db_file)
    row = _append_text(conn, blobs_root, text="payload", captured_at=1723000000)

    with pytest.raises(ValueError):
        update_slots(conn, row["evidence_id"], **{field_name: "x"})


@pytest.mark.parametrize(
    ("slot_values", "expected_count"),
    [
        ({}, 0),
        ({"slot_requester": "act-1", "slot_owner": "act-2"}, 2),
        (
            {
                "slot_requester": "act-1",
                "slot_owner": "act-2",
                "slot_deliverable": "doc",
                "slot_due": 1724000000,
            },
            4,
        ),
    ],
)
def test_update_slots_recomputes_slots_filled(db_file, blobs_root, slot_values, expected_count: int):
    conn = init_db(db_file)
    if "slot_requester" in slot_values:
        _insert_actor(conn, "act-1")
    if "slot_owner" in slot_values:
        _insert_actor(conn, "act-2")
    row = _append_text(conn, blobs_root, text="payload", captured_at=1723000000)

    updated = update_slots(conn, row["evidence_id"], **slot_values)

    assert updated["slots_filled"] == expected_count


def test_append_evidence_validates_media_type_payload_match_and_required_captured_at(
    db_file, blobs_root
):
    conn = init_db(db_file)

    with pytest.raises(ValueError):
        append_evidence(
            conn,
            blobs_root=blobs_root,
            media_type="text",
            payload=b"bytes",
            captured_at=1723000000,
        )

    with pytest.raises(ValueError):
        append_evidence(
            conn,
            blobs_root=blobs_root,
            media_type="image",
            payload="not-bytes",
            captured_at=1723000000,
        )

    with pytest.raises(ValueError):
        append_evidence(
            conn,
            blobs_root=blobs_root,
            media_type="text",
            payload="ok",
            captured_at=None,
        )


def test_make_checkpoint_records_latest_chain_hash(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="one", captured_at=1723000000)
    latest = _append_text(conn, blobs_root, text="two", captured_at=1723000001)

    checkpoint = make_checkpoint(conn)

    assert checkpoint["at_seq"] == latest["seq"]
    assert checkpoint["chain_hash"] == latest["chain_hash"]
    assert checkpoint["tsa_token"] is None


def test_make_checkpoint_rejects_empty_database(db_file):
    conn = init_db(db_file)

    with pytest.raises(ValueError):
        make_checkpoint(conn)
