from __future__ import annotations

import pytest

from evidence_core.db import get_schema_version, init_db
from evidence_core.extraction_store import (
    ExtractionStoreError,
    create_extraction,
    get_latest_extraction,
    list_extractions,
)
from evidence_core.store import append_evidence, verify_chain


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "workchain.db"


@pytest.fixture
def blobs_root(tmp_path):
    path = tmp_path / "blobs"
    path.mkdir()
    return path


def _append_text(conn, blobs_root, *, text: str, captured_at: int):
    return append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="text",
        payload=text,
        captured_at=captured_at,
        occurred_at=captured_at,
        source_hint="飞书-项目A",
        kind="reference",
    )


def test_fresh_db_supports_evidence_extractions(db_file):
    conn = init_db(db_file)

    assert get_schema_version(conn) == 4
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'evidence_extractions'"
    ).fetchone()

    assert row["name"] == "evidence_extractions"


def test_machine_extraction_can_be_saved_with_transcript_only(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    created = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="machine",
        provider="dashscope",
        model="vanchin/deepseek-ocr",
        transcript="审批通过,周五前交付渠道复盘数据",
        observations=[],
        created_at=1723000100,
        extraction_id="ext-machine",
    )

    assert created["extraction_id"] == "ext-machine"
    assert created["transcript"] == "审批通过,周五前交付渠道复盘数据"
    assert created["observations"] == []
    assert created["provider"] == "dashscope"
    assert created["model"] == "vanchin/deepseek-ocr"


def test_user_correction_can_supersede_machine_extraction_without_deleting_history(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    machine = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="machine",
        provider="dashscope",
        model="vanchin/deepseek-ocr",
        transcript="原始识别文字",
        observations=[],
        created_at=1723000100,
        extraction_id="ext-machine",
    )
    user = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="user",
        provider="manual",
        model=None,
        transcript="人工修正后的识别文字",
        observations=[],
        created_at=1723000200,
        extraction_id="ext-user",
        supersedes_extraction_id=machine["extraction_id"],
    )

    history = list_extractions(conn, evidence["evidence_id"])
    latest = get_latest_extraction(conn, evidence["evidence_id"])

    assert [item["extraction_id"] for item in history] == ["ext-machine", "ext-user"]
    assert history[0]["transcript"] == "原始识别文字"
    assert history[1]["supersedes_extraction_id"] == "ext-machine"
    assert latest["extraction_id"] == "ext-user"
    assert latest["origin"] == "user"


def test_create_extraction_rejects_cross_evidence_supersede(db_file, blobs_root):
    conn = init_db(db_file)
    first = _append_text(conn, blobs_root, text="第一条", captured_at=1723000000)
    second = _append_text(conn, blobs_root, text="第二条", captured_at=1723000001)

    machine = create_extraction(
        conn,
        evidence_id=first["evidence_id"],
        origin="machine",
        provider="dashscope",
        model="vanchin/deepseek-ocr",
        transcript="原始识别文字",
        observations=[],
        created_at=1723000100,
    )

    with pytest.raises(ExtractionStoreError, match="same evidence"):
        create_extraction(
            conn,
            evidence_id=second["evidence_id"],
            origin="user",
            provider="manual",
            model=None,
            transcript="试图跨证据覆盖",
            observations=[],
            created_at=1723000200,
            supersedes_extraction_id=machine["extraction_id"],
        )

    assert list_extractions(conn, second["evidence_id"]) == []


def test_create_extraction_allows_observations_without_transcript(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    created = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="machine",
        provider="future-visual-provider",
        model="visual-v1",
        transcript=None,
        observations=[
            {"kind": "reaction", "content": "小王账号对该消息显示👍反应", "confidence": 0.81}
        ],
        created_at=1723000100,
    )

    assert created["transcript"] is None
    assert created["observations"] == [
        {"kind": "reaction", "content": "小王账号对该消息显示👍反应", "confidence": 0.81}
    ]


def test_create_extraction_normalizes_observations_to_json_array(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    created = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="machine",
        provider="future-visual-provider",
        model="visual-v1",
        transcript="有文字",
        observations={"bad": "not-a-list"},
        created_at=1723000100,
    )

    assert created["observations"] == []


def test_create_extraction_rejects_when_transcript_and_observations_are_both_empty(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    with pytest.raises(ExtractionStoreError, match="requires transcript or observations"):
        create_extraction(
            conn,
            evidence_id=evidence["evidence_id"],
            origin="machine",
            provider="dashscope",
            model="vanchin/deepseek-ocr",
            transcript=None,
            observations=[],
        )


def test_extraction_writes_do_not_affect_verify_chain(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = _append_text(conn, blobs_root, text="原始截图", captured_at=1723000000)

    before = verify_chain(conn, blobs_root=blobs_root)
    create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="machine",
        provider="dashscope",
        model="vanchin/deepseek-ocr",
        transcript="原始识别文字",
        observations=[],
        created_at=1723000100,
        extraction_id="ext-machine",
    )
    create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        origin="user",
        provider="manual",
        model=None,
        transcript="人工修正后的识别文字",
        observations=[],
        created_at=1723000200,
        supersedes_extraction_id="ext-machine",
        extraction_id="ext-user",
    )
    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)
