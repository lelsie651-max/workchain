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


def _checkpoint_rows(conn):
    return conn.execute(
        "SELECT at_seq, chain_hash FROM checkpoints ORDER BY at_seq"
    ).fetchall()


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


def test_append_evidence_auto_creates_checkpoint_at_100(db_file, blobs_root):
    conn = init_db(db_file)

    latest = None
    for index in range(100):
        latest = _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    checkpoints = _checkpoint_rows(conn)

    assert latest is not None
    assert len(checkpoints) == 1
    assert checkpoints[0]["at_seq"] == 100
    assert checkpoints[0]["chain_hash"] == latest["chain_hash"]


def test_append_evidence_auto_creates_checkpoints_every_100_records(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(250):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    checkpoints = _checkpoint_rows(conn)

    assert [row["at_seq"] for row in checkpoints] == [100, 200]


def test_verify_chain_detects_truncation_past_checkpoint(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    conn.execute("DELETE FROM evidence WHERE seq BETWEEN 95 AND 100")
    conn.commit()

    assert verify_chain(conn) == (False, 100, "chain truncated")


def test_verify_chain_reports_chain_hash_mismatch_before_checkpoint_check(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    conn.execute("UPDATE evidence SET chain_hash = ? WHERE seq = 100", ("f" * 64,))
    conn.commit()

    assert verify_chain(conn) == (False, 100, "chain_hash mismatch")


def test_verify_chain_detects_manual_checkpoint_truncation(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(50):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)
    make_checkpoint(conn)

    conn.execute("DELETE FROM evidence WHERE seq = 50")
    conn.commit()

    assert verify_chain(conn) == (False, 50, "chain truncated")


def test_verify_chain_detects_empty_evidence_with_remaining_checkpoint(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    conn.execute("DELETE FROM evidence")
    conn.commit()

    assert verify_chain(conn) == (False, 100, "chain truncated")


def test_append_evidence_rejects_invalid_kind_before_db_insert(db_file, blobs_root):
    conn = init_db(db_file)

    with pytest.raises(ValueError):
        _append_text(
            conn,
            blobs_root,
            text="payload",
            captured_at=1723000000,
            kind="gossip",
        )


def test_verify_chain_still_passes_after_100_writes_with_auto_checkpoint(db_file, blobs_root):
    conn = init_db(db_file)

    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    assert verify_chain(conn) == (True, None, None)


def test_semantic_v4_tables_do_not_affect_verify_chain(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    ev1 = _append_text(conn, blobs_root, text="第一条原始记录", captured_at=1723000000)
    ev2 = _append_text(conn, blobs_root, text="第二条原始记录", captured_at=1723000001)

    before = verify_chain(conn, blobs_root=blobs_root)

    conn.execute(
        "INSERT INTO submissions (submission_id, created_at, source_hint) VALUES (?, ?, ?)",
        ("sub-1", 1723000100, "飞书-项目A"),
    )
    conn.execute(
        "INSERT INTO submission_evidence (submission_id, evidence_id, position) VALUES (?, ?, ?)",
        ("sub-1", ev1["evidence_id"], 0),
    )
    conn.execute(
        "INSERT INTO submission_evidence (submission_id, evidence_id, position) VALUES (?, ?, ?)",
        ("sub-1", ev2["evidence_id"], 1),
    )
    conn.execute(
        """
        INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("evt-1", "渠道复盘", "active", "事件摘要", 1723000200, 1723000200),
    )
    conn.execute(
        """
        INSERT INTO facts (
            fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw,
            confidence, event_assignment, event_assignment_confidence,
            due_anchor_at, origin, review_status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "fact-1",
            "evt-1",
            "request",
            "张总要求补一份渠道复盘",
            1723000000,
            1723600000,
            "下周五",
            0.92,
            "confirmed",
            0.74,
            1723000000,
            "ai",
            "unreviewed",
            1723000200,
            1723000200,
        ),
    )
    conn.execute(
        "INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)",
        ("fact-1", ev1["evidence_id"]),
    )
    conn.execute(
        "INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)",
        ("fact-1", ev2["evidence_id"]),
    )
    conn.execute(
        "INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)",
        ("fact-1", "act-1", "requester"),
    )
    conn.execute(
        "INSERT INTO fact_actors (fact_id, actor_id, role) VALUES (?, ?, ?)",
        ("fact-1", "act-2", "owner"),
    )
    conn.execute(
        """
        INSERT INTO interpretations (
            interpretation_id, fact_id, evidence_id, kind, content, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("itp-1", "fact-1", None, "explanation", "这里是在明确交付物范围", 0.88, 1723000300),
    )
    conn.execute(
        """
        UPDATE facts
        SET due_anchor_at = ?, event_assignment_confidence = ?, origin = ?, review_status = ?
        WHERE fact_id = ?
        """,
        (1723086400, 0.81, "user", "corrected", "fact-1"),
    )
    conn.execute(
        """
        INSERT INTO evidence_extractions (
            extraction_id, evidence_id, origin, provider, model,
            transcript, observations, created_at, supersedes_extraction_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ext-1",
            ev1["evidence_id"],
            "machine",
            "dashscope",
            "vanchin/deepseek-ocr",
            "原始识别文字",
            "[]",
            1723000400,
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO evidence_extractions (
            extraction_id, evidence_id, origin, provider, model,
            transcript, observations, created_at, supersedes_extraction_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ext-2",
            ev1["evidence_id"],
            "user",
            "manual",
            None,
            "人工修正后的识别文字",
            "[]",
            1723000500,
            "ext-1",
        ),
    )
    conn.commit()

    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)
