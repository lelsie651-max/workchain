from __future__ import annotations

from pathlib import Path

import pytest

from evidence_core import db as db_module
from app.llm import due_date_to_millis
from app.semantic_llm import SEMANTIC_PARSER_VERSION
from evidence_core.db import get_schema_version, init_db
from evidence_core.extraction_store import create_extraction
from evidence_core.semantic_store import (
    ProtectedFactError,
    SemanticStoreError,
    confirm_fact,
    create_event,
    create_event_change_run,
    create_event_match_run,
    create_source_review,
    create_fact,
    create_interpretation,
    create_semantic_run,
    create_submission,
    correct_relative_due_dates_by_user,
    correct_fact_by_user,
    create_context_assembly_run,
    create_context_group_review,
    create_extraction_speaker_review,
    get_effective_source_hint,
    get_latest_context_assembly_run,
    get_latest_context_group_review,
    get_latest_event_change_run_for_event,
    get_latest_event_match_for_evidence,
    get_latest_extraction_speaker_review,
    get_semantic_run,
    get_latest_semantic_run_for_evidence,
    get_latest_source_review,
    list_event_candidates,
    list_facts_for_semantic_run,
    list_interpretations_for_semantic_run,
    mark_event_change_run_failed,
    mark_event_match_run_failed,
    mark_context_assembly_run_failed,
    mark_semantic_run_failed,
    mark_semantic_run_succeeded,
    persist_context_assembly_run_result,
    persist_event_change_run_result,
    persist_event_match_run_result,
    persist_semantic_run_result,
    review_event_match_run_by_user,
    set_event_assignment_by_ai,
    set_event_assignment_by_user,
    update_fact_by_ai,
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


def _append_text(conn, blobs_root: Path, *, evidence_id: str, text: str, captured_at: int):
    return append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="text",
        payload=text,
        captured_at=captured_at,
        evidence_id=evidence_id,
    )


def _count(conn, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return row["count"]


def _create_machine_extraction(
    conn,
    *,
    evidence_id: str,
    extraction_id: str,
    transcript: str = "原始识别文字",
    created_at: int = 1723000000,
):
    return create_extraction(
        conn,
        evidence_id=evidence_id,
        extraction_id=extraction_id,
        origin="machine",
        provider="dashscope",
        model="vanchin/deepseek-ocr",
        transcript=transcript,
        observations=[],
        warnings=[],
        created_at=created_at,
    )


def _prepare_reviewable_match_run(
    conn,
    blobs_root: Path,
    *,
    routing_mode: str = "confirm",
    normalized_match: dict | None = None,
):
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run = create_semantic_run(
        conn,
        semantic_run_id=f"srun-{routing_mode}",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-a",
                "fact_type": "request",
                "content": "请补一版渠道复盘",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 11,
                "updated_at": 11,
            },
            {
                "fact_id": "fact-b",
                "fact_type": "statement",
                "content": "客户回访也要整理",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 12,
                "updated_at": 12,
            },
        ],
        interpretations=[],
        completed_at=13,
    )
    match_run = create_event_match_run(
        conn,
        event_match_run_id=f"mrun-{routing_mode}",
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=14,
    )
    result = persist_event_match_run_result(
        conn,
        event_match_run_id=match_run["event_match_run_id"],
        semantic_run_id=run["semantic_run_id"],
        routing_mode=routing_mode,
        normalized_match=normalized_match
        or {
            "groups": [
                {
                    "fact_indexes": [0],
                    "target": "existing",
                    "event_id": "evt-1",
                    "proposed_title": None,
                    "confidence": 0.74,
                    "reason": "像是在延续旧事项",
                },
                {
                    "fact_indexes": [1],
                    "target": "new",
                    "event_id": None,
                    "proposed_title": "整理客户回访",
                    "confidence": 0.71,
                    "reason": "也可能是另一件事",
                },
            ],
            "ambiguities": [],
        },
        facts=persisted["facts"],
        completed_at=15,
    )
    return ev1, run, persisted, result


def test_create_submission_keeps_evidence_order(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)

    created = create_submission(
        conn,
        submission_id="sub-1",
        created_at=10,
        source_hint="飞书",
        evidence_ids=[ev2["evidence_id"], ev1["evidence_id"]],
    )

    assert created["submission_id"] == "sub-1"
    assert created["created_at"] == 10
    assert [row["evidence_id"] for row in created["evidence"]] == ["ev-2", "ev-1"]


def test_source_review_updates_effective_source_hint_without_touching_original_evidence(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="image",
        payload=b"fake-image",
        captured_at=1,
        occurred_at=1,
        source_hint="飞书-项目A",
        kind="reference",
        evidence_id="ev-1",
    )
    extraction_v1 = _create_machine_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id="ext-1",
        transcript="第一次提取",
        created_at=10,
    )

    assert get_effective_source_hint(conn, evidence["evidence_id"]) == "飞书-项目A"

    corrected_review = create_source_review(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id=extraction_v1["extraction_id"],
        original_source_hint="飞书-项目A",
        observed_platform="微信",
        resolved_source_hint="微信-项目A",
        decision="corrected",
        review_id="srev-1",
        created_at=11,
    )
    extraction_v2 = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id="ext-2",
        origin="machine",
        provider="doubao-ark",
        model="seed-2.1",
        transcript="第二次提取",
        observations=[],
        warnings=[],
        created_at=12,
        supersedes_extraction_id=extraction_v1["extraction_id"],
    )
    confirmed_review = create_source_review(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id=extraction_v2["extraction_id"],
        original_source_hint="微信-项目A",
        observed_platform="微信",
        resolved_source_hint="微信-项目A",
        decision="confirmed_declared",
        review_id="srev-2",
        created_at=13,
    )

    evidence_row = conn.execute(
        "SELECT source_hint FROM evidence WHERE evidence_id = ?",
        (evidence["evidence_id"],),
    ).fetchone()

    assert evidence_row["source_hint"] == "飞书-项目A"
    assert corrected_review["decision"] == "corrected"
    assert confirmed_review["decision"] == "confirmed_declared"
    assert get_effective_source_hint(conn, evidence["evidence_id"]) == "微信-项目A"
    assert get_latest_source_review(conn, evidence["evidence_id"])["review_id"] == "srev-2"
    assert get_latest_source_review(conn, evidence["evidence_id"], extraction_id="ext-1")["review_id"] == "srev-1"
    assert get_latest_source_review(conn, evidence["evidence_id"], extraction_id="ext-2")["review_id"] == "srev-2"


def test_create_semantic_run_supports_single_and_multiple_inputs(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    ext1 = _create_machine_extraction(conn, evidence_id=ev1["evidence_id"], extraction_id="ext-1")

    created = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        anchor_date="2026-08-08",
        inputs=[
            {"evidence_id": ev2["evidence_id"], "position": 1},
            {
                "evidence_id": ev1["evidence_id"],
                "extraction_id": ext1["extraction_id"],
                "position": 0,
            },
        ],
        created_at=10,
    )

    loaded = get_semantic_run(conn, "srun-1")

    assert created["status"] == "running"
    assert created["provider"] == "deepseek"
    assert created["parser_version"] == SEMANTIC_PARSER_VERSION
    assert created["inputs"] == [
        {
            "semantic_run_id": "srun-1",
            "evidence_id": "ev-1",
            "extraction_id": "ext-1",
            "position": 0,
        },
        {
            "semantic_run_id": "srun-1",
            "evidence_id": "ev-2",
            "extraction_id": None,
            "position": 1,
        },
    ]
    assert loaded == created


def test_v11_structured_payload_and_speaker_review_round_trip(db_file, blobs_root):
    conn = init_db(db_file)
    evidence = append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="image",
        payload=b"fake-image",
        captured_at=1,
        occurred_at=1,
        source_hint="微信-单聊",
        kind="reference",
        evidence_id="ev-1",
    )

    extraction = create_extraction(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id="ext-1",
        origin="machine",
        provider="doubao-ark",
        model="seed-vision",
        transcript="[message 1][left_account] 今天先这样",
        observations=[{"kind": "timestamp", "content": "2026-08-09 19:21", "confidence": 0.9}],
        warnings=[],
        structured_payload={
            "payload_version": "1.0",
            "conversation_type": "direct_chat",
            "participants": [
                {"speaker_ref": "left_account", "side": "left", "display_name": None},
                {"speaker_ref": "right_account", "side": "right", "display_name": None},
            ],
            "messages": [
                {
                    "index": 1,
                    "speaker_ref": "left_account",
                    "side": "left",
                    "text": "今天先这样",
                    "quote": None,
                    "reply": None,
                    "reactions": [],
                }
            ],
            "system_events": [{"type": "message_recalled", "visible_text": "撤回了一条消息", "actor_display_name": "张三"}],
        },
        created_at=10,
    )
    review = create_extraction_speaker_review(
        conn,
        evidence_id=evidence["evidence_id"],
        extraction_id=extraction["extraction_id"],
        status="provided",
        labels={"left_account": "甲方", "right_account": "乙方"},
        review_id="sprev-1",
        created_at=11,
    )

    latest_extraction = conn.execute(
        "SELECT structured_payload FROM evidence_extractions WHERE extraction_id = ?",
        (extraction["extraction_id"],),
    ).fetchone()

    assert latest_extraction["structured_payload"] is not None
    assert extraction["structured_payload"]["messages"][0]["speaker_ref"] == "left_account"
    assert get_latest_extraction_speaker_review(conn, evidence["evidence_id"])["review_id"] == "sprev-1"
    assert review["labels"] == {"left_account": "甲方", "right_account": "乙方"}


def test_v11_context_assembly_and_semantic_run_provenance(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    submission = create_submission(
        conn,
        submission_id="sub-1",
        created_at=10,
        source_hint="飞书-项目群",
        evidence_ids=[ev1["evidence_id"], ev2["evidence_id"]],
    )

    assembly_run = create_context_assembly_run(
        conn,
        submission_id=submission["submission_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        assembler_version="1.0",
        assembly_run_id="arun-1",
        created_at=11,
    )
    persisted_assembly = persist_context_assembly_run_result(
        conn,
        assembly_run_id=assembly_run["assembly_run_id"],
        result={
            "groups": [
                {
                    "group_key": "group-1",
                    "evidence_ids": ["ev-1", "ev-2"],
                    "analysis_order": ["ev-2", "ev-1"],
                    "relation": "continuation",
                    "confidence": 0.91,
                }
            ],
            "ambiguities": [],
        },
        completed_at=12,
    )
    review = create_context_group_review(
        conn,
        assembly_run_id=assembly_run["assembly_run_id"],
        group_key="group-1",
        review_status="accepted",
        decision={"analysis_order": ["ev-2", "ev-1"]},
        review_id="cgr-1",
        created_at=13,
    )
    semantic_run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        context_group_key="group-1",
        context_assembly_run_id=assembly_run["assembly_run_id"],
        inputs=[
            {"evidence_id": ev2["evidence_id"], "position": 0},
            {"evidence_id": ev1["evidence_id"], "position": 1},
        ],
        created_at=14,
    )

    loaded_run = get_semantic_run(conn, "srun-1")
    assert persisted_assembly["status"] == "succeeded"
    assert get_latest_context_assembly_run(conn, submission["submission_id"])["assembly_run_id"] == "arun-1"
    assert get_latest_context_group_review(conn, assembly_run["assembly_run_id"])["review_id"] == "cgr-1"
    assert review["review_status"] == "accepted"
    assert loaded_run["context_group_key"] == "group-1"
    assert loaded_run["context_assembly_run_id"] == "arun-1"


def test_v10_to_v11_migration_adds_structured_payload_and_context_tables(tmp_path, monkeypatch):
    db_file = tmp_path / "workchain.db"

    monkeypatch.setattr(db_module, "SCHEMA_VERSION", 10)
    conn = init_db(db_file)
    conn.close()

    monkeypatch.setattr(db_module, "SCHEMA_VERSION", 11)
    migrated = init_db(db_file)
    try:
        version = get_schema_version(migrated)
        extraction_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(evidence_extractions)").fetchall()
        }
        semantic_run_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(semantic_runs)").fetchall()
        }
        table_names = {
            row["name"]
            for row in migrated.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    finally:
        migrated.close()

    assert version == 11
    assert "structured_payload" in extraction_columns
    assert "context_group_key" in semantic_run_columns
    assert "context_assembly_run_id" in semantic_run_columns
    assert {"extraction_speaker_reviews", "context_assembly_runs", "context_group_reviews"} <= table_names


def test_create_semantic_run_rejects_cross_evidence_extraction_and_rolls_back(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    ext1 = _create_machine_extraction(conn, evidence_id=ev1["evidence_id"], extraction_id="ext-1")

    with pytest.raises(SemanticStoreError, match="same evidence_id"):
        create_semantic_run(
            conn,
            semantic_run_id="srun-1",
            provider="deepseek",
            model="deepseek-v4-flash",
            parser_version=SEMANTIC_PARSER_VERSION,
            inputs=[
                {
                    "evidence_id": ev2["evidence_id"],
                    "extraction_id": ext1["extraction_id"],
                    "position": 0,
                }
            ],
            created_at=10,
        )

    assert _count(conn, "semantic_runs") == 0
    assert _count(conn, "semantic_run_inputs") == 0


@pytest.mark.parametrize(
    ("evidence_ids", "expected_message"),
    [
        (["ev-1", "ev-1"], "must not contain duplicate ids"),
        (["ev-1", "ev-missing"], "evidence not found"),
    ],
)
def test_create_submission_rejects_bad_evidence_ids_and_rolls_back(
    db_file, blobs_root, evidence_ids, expected_message: str
):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)

    with pytest.raises(SemanticStoreError, match=expected_message):
        create_submission(conn, submission_id="sub-1", created_at=10, evidence_ids=evidence_ids)

    assert _count(conn, "submissions") == 0
    assert _count(conn, "submission_evidence") == 0


def test_create_submission_rolls_back_when_evidence_already_belongs_to_other_submission(
    db_file, blobs_root
):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    create_submission(conn, submission_id="sub-1", created_at=10, evidence_ids=["ev-1"])

    with pytest.raises(SemanticStoreError, match="already linked to another submission"):
        create_submission(conn, submission_id="sub-2", created_at=11, evidence_ids=["ev-1", "ev-2"])

    assert _count(conn, "submissions") == 1
    assert _count(conn, "submission_evidence") == 1


def test_create_submission_supports_nested_transaction(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)

    conn.execute("BEGIN IMMEDIATE")
    create_submission(conn, submission_id="sub-1", created_at=10, evidence_ids=["ev-1", "ev-2"])
    conn.rollback()

    assert _count(conn, "submissions") == 0
    assert _count(conn, "submission_evidence") == 0


def test_create_event_trims_title_and_defaults_active(db_file):
    conn = init_db(db_file)

    event = create_event(conn, event_id="evt-1", title="  渠道复盘  ", created_at=10)

    assert event["event_id"] == "evt-1"
    assert event["title"] == "渠道复盘"
    assert event["status"] == "active"
    assert event["created_at"] == 10
    assert event["updated_at"] == 10


def test_create_fact_supports_many_to_many_and_multiple_actor_roles(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    event = create_event(conn, event_id="evt-1", title="渠道复盘", created_at=10)

    fact1 = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="request",
        content="请补渠道复盘",
        evidence_ids=[ev1["evidence_id"], ev2["evidence_id"]],
        event_id=event["event_id"],
        event_assignment="suggested",
        event_assignment_confidence=0.62,
        due_raw="下下周五",
        due_at=1723600000,
        due_anchor_at=1723000000,
        actor_roles=[("act-1", "requester"), ("act-2", "owner"), ("act-2", "reviewer")],
        created_at=11,
        updated_at=12,
    )
    fact2 = create_fact(
        conn,
        fact_id="fact-2",
        fact_type="reference",
        content="同一证据还能支撑补充事实",
        evidence_ids=[ev1["evidence_id"]],
        created_at=13,
        updated_at=14,
    )

    assert fact1["origin"] == "ai"
    assert fact1["review_status"] == "unreviewed"
    assert fact1["evidence_ids"] == ["ev-1", "ev-2"]
    assert fact1["event_assignment_confidence"] == 0.62
    assert fact1["due_anchor_at"] == 1723000000
    assert fact1["actors"] == [
        {"actor_id": "act-1", "role": "requester"},
        {"actor_id": "act-2", "role": "owner"},
        {"actor_id": "act-2", "role": "reviewer"},
    ]
    assert fact2["evidence_ids"] == ["ev-1"]

    linked_facts = conn.execute(
        "SELECT COUNT(*) AS count FROM fact_evidence WHERE evidence_id = ?",
        ("ev-1",),
    ).fetchone()
    assert linked_facts["count"] == 2


def test_create_fact_can_bind_to_semantic_run(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )

    fact = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=[ev1["evidence_id"]],
        semantic_run_id=run["semantic_run_id"],
        created_at=11,
        updated_at=12,
    )

    assert fact["semantic_run_id"] == "srun-1"


def test_create_fact_rejects_evidence_outside_semantic_run_and_rolls_back(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )

    with pytest.raises(SemanticStoreError, match="semantic run does not include evidence"):
        create_fact(
            conn,
            fact_id="fact-1",
            fact_type="statement",
            content="原始事实",
            evidence_ids=[ev2["evidence_id"]],
            semantic_run_id="srun-1",
            created_at=11,
            updated_at=12,
        )

    assert _count(conn, "facts") == 0
    assert _count(conn, "fact_evidence") == 0


def test_create_fact_without_semantic_run_id_stays_backward_compatible(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)

    fact = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="旧调用仍可创建事实",
        evidence_ids=[ev1["evidence_id"]],
        created_at=10,
        updated_at=11,
    )

    assert fact["semantic_run_id"] is None


def test_create_fact_rolls_back_when_any_association_is_invalid(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)

    with pytest.raises(SemanticStoreError, match="actor not found: act-missing"):
        create_fact(
            conn,
            fact_id="fact-1",
            fact_type="statement",
            content="先记下来",
            evidence_ids=["ev-1"],
            actor_roles=[("act-1", "speaker"), ("act-missing", "target")],
            created_at=10,
            updated_at=11,
        )

    assert _count(conn, "facts") == 0
    assert _count(conn, "fact_evidence") == 0
    assert _count(conn, "fact_actors") == 0


@pytest.mark.parametrize("assignment", ["auto", "suggested"])
def test_set_event_assignment_by_ai_accepts_auto_and_suggested(db_file, blobs_root, assignment: str):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="事件一", created_at=10)
    fact = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=11,
        updated_at=11,
    )

    updated = set_event_assignment_by_ai(
        conn,
        fact_id=fact["fact_id"],
        event_id="evt-1",
        assignment=assignment,
        event_assignment_confidence=0.73,
        updated_at=12,
    )

    assert updated["event_id"] == "evt-1"
    assert updated["event_assignment"] == assignment
    assert updated["event_assignment_confidence"] == 0.73
    assert updated["updated_at"] == 12


def test_confirmed_event_assignment_cannot_be_overwritten_by_ai(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="事件一", created_at=10)
    create_event(conn, event_id="evt-2", title="事件二", created_at=11)
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=12,
        updated_at=12,
    )
    set_event_assignment_by_user(conn, fact_id="fact-1", event_id="evt-1", updated_at=13)

    with pytest.raises(ProtectedFactError, match="confirmed event assignment"):
        set_event_assignment_by_ai(
            conn,
            fact_id="fact-1",
            event_id="evt-2",
            assignment="suggested",
            event_assignment_confidence=0.4,
            updated_at=14,
        )

    row = conn.execute(
        "SELECT event_id, event_assignment FROM facts WHERE fact_id = ?",
        ("fact-1",),
    ).fetchone()
    assert row["event_id"] == "evt-1"
    assert row["event_assignment"] == "confirmed"


def test_user_can_change_or_cancel_event_assignment(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="事件一", created_at=10)
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=11,
        updated_at=11,
    )

    confirmed = set_event_assignment_by_user(conn, fact_id="fact-1", event_id="evt-1", updated_at=12)
    cleared = set_event_assignment_by_user(conn, fact_id="fact-1", event_id=None, updated_at=13)

    assert confirmed["event_id"] == "evt-1"
    assert confirmed["event_assignment"] == "confirmed"
    assert confirmed["event_assignment_confidence"] is None
    assert cleared["event_id"] is None
    assert cleared["event_assignment"] == "unassigned"
    assert cleared["event_assignment_confidence"] is None


def test_confirmed_or_corrected_fact_cannot_be_overwritten_by_ai(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_fact(
        conn,
        fact_id="fact-confirmed",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=10,
        updated_at=10,
    )
    create_fact(
        conn,
        fact_id="fact-corrected",
        fact_type="statement",
        content="另一个原始事实",
        evidence_ids=["ev-1"],
        created_at=11,
        updated_at=11,
    )
    confirm_fact(conn, fact_id="fact-confirmed", updated_at=12)
    correct_fact_by_user(
        conn,
        fact_id="fact-corrected",
        content="用户修正后的事实",
        actor_roles=[("act-1", "speaker")],
        updated_at=13,
    )

    with pytest.raises(ProtectedFactError, match="review_status=confirmed"):
        update_fact_by_ai(
            conn,
            fact_id="fact-confirmed",
            content="AI 想覆盖它",
            updated_at=14,
        )
    with pytest.raises(ProtectedFactError, match="review_status=corrected"):
        update_fact_by_ai(
            conn,
            fact_id="fact-corrected",
            content="AI 也不能覆盖它",
            updated_at=15,
        )


def test_correct_fact_by_user_sets_origin_and_review_status_and_keeps_evidence_links(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    fact = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        actor_roles=[("act-1", "speaker")],
        created_at=10,
        updated_at=10,
    )

    corrected = correct_fact_by_user(
        conn,
        fact_id=fact["fact_id"],
        fact_type="deadline_change",
        content="用户修正后的事实",
        occurred_at=20,
        due_at=30,
        due_raw="下周五",
        due_anchor_at=18,
        actor_roles=[("act-2", "owner")],
        updated_at=21,
    )

    assert corrected["fact_type"] == "deadline_change"
    assert corrected["content"] == "用户修正后的事实"
    assert corrected["origin"] == "user"
    assert corrected["review_status"] == "corrected"
    assert corrected["evidence_ids"] == ["ev-1"]
    assert corrected["actors"] == [{"actor_id": "act-2", "role": "owner"}]


def test_correct_fact_by_user_respects_outer_transaction(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="补材料", created_at=10, updated_at=10)
    create_fact(
        conn,
        fact_id="fact-1",
        event_id="evt-1",
        event_assignment="confirmed",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=10,
        updated_at=10,
    )

    conn.execute("BEGIN IMMEDIATE")
    correct_fact_by_user(
        conn,
        fact_id="fact-1",
        content="事务内修正后的事实",
        updated_at=20,
    )
    conn.execute("UPDATE events SET updated_at = ? WHERE event_id = ?", (21, "evt-1"))
    conn.rollback()

    fact_row = conn.execute(
        "SELECT content, origin, review_status, updated_at FROM facts WHERE fact_id = 'fact-1'"
    ).fetchone()
    event_row = conn.execute("SELECT updated_at FROM events WHERE event_id = 'evt-1'").fetchone()

    assert fact_row["content"] == "原始事实"
    assert fact_row["origin"] == "ai"
    assert fact_row["review_status"] == "unreviewed"
    assert fact_row["updated_at"] == 10
    assert event_row["updated_at"] == 10


def test_correct_relative_due_dates_by_user_updates_due_fields_and_event_timestamp(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="周五前补材料", captured_at=1)
    create_event(conn, event_id="evt-1", title="补材料", created_at=10, updated_at=10)
    create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="deadline_change",
        content="周五前补材料",
        evidence_ids=[ev1["evidence_id"]],
        semantic_run_id="srun-1",
        event_id="evt-1",
        event_assignment="confirmed",
        due_raw="周五前",
        created_at=12,
        updated_at=12,
    )

    result = correct_relative_due_dates_by_user(
        conn,
        evidence_id=ev1["evidence_id"],
        semantic_run_id="srun-1",
        due_updates=[
            {
                "fact_id": "fact-1",
                "due_at": due_date_to_millis("2026-08-14"),
                "due_anchor_at": due_date_to_millis("2026-08-09"),
            }
        ],
        updated_at=20,
    )

    corrected = result["facts"][0]
    event_row = conn.execute("SELECT updated_at FROM events WHERE event_id = 'evt-1'").fetchone()

    assert corrected["due_at"] == due_date_to_millis("2026-08-14")
    assert corrected["due_anchor_at"] == due_date_to_millis("2026-08-09")
    assert corrected["origin"] == "user"
    assert corrected["review_status"] == "corrected"
    assert event_row["updated_at"] == 20


def test_correct_relative_due_dates_by_user_rolls_back_entire_batch(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="周五前补材料", captured_at=1)
    create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="deadline_change",
        content="周五前补材料",
        evidence_ids=[ev1["evidence_id"]],
        semantic_run_id="srun-1",
        due_raw="周五前",
        created_at=12,
        updated_at=12,
    )

    with pytest.raises(SemanticStoreError, match="fact not found"):
        correct_relative_due_dates_by_user(
            conn,
            evidence_id=ev1["evidence_id"],
            semantic_run_id="srun-1",
            due_updates=[
                {
                    "fact_id": "fact-1",
                    "due_at": due_date_to_millis("2026-08-14"),
                    "due_anchor_at": due_date_to_millis("2026-08-09"),
                },
                {
                    "fact_id": "fact-missing",
                    "due_at": due_date_to_millis("2026-08-15"),
                    "due_anchor_at": due_date_to_millis("2026-08-09"),
                },
            ],
            updated_at=20,
        )

    fact_row = conn.execute(
        "SELECT due_at, due_anchor_at, origin, review_status, updated_at FROM facts WHERE fact_id = 'fact-1'"
    ).fetchone()
    assert fact_row["due_at"] is None
    assert fact_row["due_anchor_at"] is None
    assert fact_row["origin"] == "ai"
    assert fact_row["review_status"] == "unreviewed"
    assert fact_row["updated_at"] == 12


def test_create_interpretation_validates_parent_records(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=["ev-1"],
        created_at=10,
        updated_at=10,
    )

    interpretation = create_interpretation(
        conn,
        interpretation_id="itp-1",
        fact_id="fact-1",
        kind="explanation",
        content="这是一段解释",
        created_at=11,
    )

    assert interpretation["interpretation_id"] == "itp-1"
    assert interpretation["fact_id"] == "fact-1"
    assert interpretation["semantic_run_id"] is None

    with pytest.raises(SemanticStoreError, match="requires fact_id or evidence_id"):
        create_interpretation(conn, kind="term", content="缺父节点")
    with pytest.raises(SemanticStoreError, match="fact not found"):
        create_interpretation(
            conn,
            interpretation_id="itp-2",
            fact_id="fact-missing",
            kind="explanation",
            content="无效父节点",
            created_at=12,
        )


def test_create_interpretation_enforces_semantic_run_consistency(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ev2 = _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )
    fact = create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=[ev1["evidence_id"]],
        semantic_run_id=run["semantic_run_id"],
        created_at=11,
        updated_at=11,
    )

    interpretation = create_interpretation(
        conn,
        interpretation_id="itp-1",
        fact_id=fact["fact_id"],
        evidence_id=ev1["evidence_id"],
        semantic_run_id=run["semantic_run_id"],
        kind="uncertainty",
        content="这条事实仍需补上下文",
        created_at=12,
    )

    assert interpretation["semantic_run_id"] == "srun-1"

    other_run = create_semantic_run(
        conn,
        semantic_run_id="srun-2",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev2["evidence_id"], "position": 0}],
        created_at=14,
    )
    with pytest.raises(SemanticStoreError, match="must match fact semantic_run_id"):
        create_interpretation(
            conn,
            interpretation_id="itp-3",
            fact_id=fact["fact_id"],
            semantic_run_id=other_run["semantic_run_id"],
            kind="explanation",
            content="run 不一致",
            created_at=15,
        )
    with pytest.raises(SemanticStoreError, match="does not include evidence"):
        create_interpretation(
            conn,
            interpretation_id="itp-4",
            evidence_id=ev2["evidence_id"],
            semantic_run_id=run["semantic_run_id"],
            kind="explanation",
            content="run 未消费该 evidence",
            created_at=16,
        )


def test_create_interpretation_rejects_run_provenance_when_fact_has_no_run(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="statement",
        content="原始事实",
        evidence_ids=[ev1["evidence_id"]],
        created_at=10,
        updated_at=10,
    )
    create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )

    with pytest.raises(SemanticStoreError, match="requires fact semantic_run_id to be set"):
        create_interpretation(
            conn,
            interpretation_id="itp-1",
            fact_id="fact-1",
            semantic_run_id="srun-1",
            kind="explanation",
            content="不能伪造 provenance",
            created_at=12,
        )


def test_semantic_run_lifecycle_and_supersedes_history(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)

    first = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )
    succeeded = mark_semantic_run_succeeded(conn, semantic_run_id=first["semantic_run_id"], completed_at=11)

    second = create_semantic_run(
        conn,
        semantic_run_id="srun-2",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=12,
        supersedes_run_id=first["semantic_run_id"],
    )
    failed = mark_semantic_run_failed(
        conn,
        semantic_run_id=second["semantic_run_id"],
        failure_type="provider_timeout",
        completed_at=13,
    )

    assert succeeded["status"] == "succeeded"
    assert succeeded["completed_at"] == 11
    assert failed["status"] == "failed"
    assert failed["failure_type"] == "provider_timeout"
    assert failed["supersedes_run_id"] == "srun-1"
    assert get_semantic_run(conn, "srun-1")["status"] == "succeeded"
    assert get_semantic_run(conn, "srun-2")["status"] == "failed"

    with pytest.raises(SemanticStoreError, match="not running"):
        mark_semantic_run_succeeded(conn, semantic_run_id="srun-2", completed_at=14)


def test_persist_semantic_run_result_creates_atomic_provenance_bundle(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-zhang")
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    ext1 = _create_machine_extraction(conn, evidence_id=ev1["evidence_id"], extraction_id="ext-1")
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[
            {
                "evidence_id": ev1["evidence_id"],
                "extraction_id": ext1["extraction_id"],
                "position": 0,
            }
        ],
        created_at=10,
    )

    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_type": "request",
                "content": "张伟要求补复盘",
                "evidence_ids": [ev1["evidence_id"]],
                "actor_roles": [("act-zhang", "requester")],
                "occurred_at": 1723000000000,
                "due_at": 1723600000000,
                "due_raw": "下周五",
                "due_anchor_at": 1723000000000,
                "event_assignment": "unassigned",
                "origin": "ai",
                "review_status": "unreviewed",
                "created_at": 11,
                "updated_at": 11,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "explanation",
                "content": "这里的复盘指渠道复盘",
                "created_at": 12,
            },
            {
                "evidence_id": ev1["evidence_id"],
                "kind": "uncertainty",
                "content": "尚未看到明确交付格式",
                "created_at": 13,
            },
        ],
        completed_at=14,
    )

    latest = get_latest_semantic_run_for_evidence(conn, ev1["evidence_id"], status="succeeded")
    facts = list_facts_for_semantic_run(conn, run["semantic_run_id"], evidence_id=ev1["evidence_id"])
    interpretations = list_interpretations_for_semantic_run(
        conn,
        run["semantic_run_id"],
        evidence_id=ev1["evidence_id"],
    )

    assert persisted["semantic_run"]["status"] == "succeeded"
    assert persisted["semantic_run"]["completed_at"] == 14
    assert latest["semantic_run_id"] == run["semantic_run_id"]
    assert len(facts) == 1
    assert facts[0]["semantic_run_id"] == run["semantic_run_id"]
    assert facts[0]["origin"] == "ai"
    assert facts[0]["review_status"] == "unreviewed"
    assert facts[0]["actors"] == [{"actor_id": "act-zhang", "role": "requester"}]
    assert [item["kind"] for item in interpretations] == ["explanation", "uncertainty"]
    assert interpretations[0]["fact_id"] == facts[0]["fact_id"]
    assert interpretations[0]["semantic_run_id"] == run["semantic_run_id"]
    assert interpretations[1]["evidence_id"] == ev1["evidence_id"]
    assert interpretations[1]["fact_id"] is None


def test_persist_semantic_run_result_rolls_back_on_late_failure(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )

    with pytest.raises(SemanticStoreError, match="out of range"):
        persist_semantic_run_result(
            conn,
            semantic_run_id=run["semantic_run_id"],
            facts=[
                {
                    "fact_type": "statement",
                    "content": "先创建一条事实",
                    "evidence_ids": [ev1["evidence_id"]],
                    "created_at": 11,
                    "updated_at": 11,
                }
            ],
            interpretations=[
                {
                    "fact_index": 9,
                    "kind": "explanation",
                    "content": "越界映射应触发回滚",
                    "created_at": 12,
                }
            ],
            completed_at=13,
        )

    assert _count(conn, "facts") == 0
    assert _count(conn, "fact_evidence") == 0
    assert _count(conn, "interpretations") == 0
    assert get_semantic_run(conn, "srun-1")["status"] == "running"


def test_persist_semantic_run_result_supports_nested_transaction_via_savepoint(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("act-zhang", "张伟", "[]", None, None, 0, 0.5, 11),
    )

    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_type": "request",
                "content": "张伟要求补复盘",
                "evidence_ids": [ev1["evidence_id"]],
                "actor_roles": [("act-zhang", "requester")],
                "created_at": 12,
                "updated_at": 12,
            }
        ],
        interpretations=[],
        completed_at=13,
    )
    conn.commit()

    assert persisted["semantic_run"]["status"] == "succeeded"
    assert _count(conn, "facts") == 1
    assert _count(conn, "fact_actors") == 1


def test_list_event_candidates_limits_active_events_and_recent_facts(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    _append_text(conn, blobs_root, evidence_id="ev-3", text="三", captured_at=3)

    for index in range(35):
        status = "resolved" if index == 34 else "active"
        event = create_event(
            conn,
            event_id=f"evt-{index:02d}",
            title=f"事项 {index:02d}",
            status=status,
            created_at=100 + index,
            updated_at=100 + index,
        )
        for fact_index in range(8):
            create_fact(
                conn,
                fact_id=f"fact-{index:02d}-{fact_index}",
                fact_type="statement",
                content=f"事项 {index:02d} 的事实 {fact_index}",
                evidence_ids=["ev-1" if fact_index % 3 == 0 else "ev-2"],
                event_id=event["event_id"],
                event_assignment="confirmed",
                created_at=1000 + fact_index,
                updated_at=1000 + fact_index,
            )

    candidates = list_event_candidates(conn)

    assert len(candidates) == 30
    assert all(item["event_id"] != "evt-34" for item in candidates)
    assert all(len(item["recent_facts"]) <= 6 for item in candidates)
    assert candidates[0]["event_id"] == "evt-33"
    assert candidates[0]["recent_facts"][0]["content"] == "事项 33 的事实 7"


def test_event_match_run_confirm_persists_safe_snapshot_with_real_fact_ids(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=10, updated_at=10)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-b",
                "fact_type": "request",
                "content": "请补一版渠道复盘",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 12,
                "updated_at": 12,
            },
            {
                "fact_id": "fact-a",
                "fact_type": "deadline_change",
                "content": "截止改到周五",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 13,
                "updated_at": 13,
            },
        ],
        interpretations=[],
        completed_at=14,
    )
    match_run = create_event_match_run(
        conn,
        event_match_run_id="mrun-1",
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=15,
    )

    result = persist_event_match_run_result(
        conn,
        event_match_run_id=match_run["event_match_run_id"],
        semantic_run_id=run["semantic_run_id"],
        routing_mode="confirm",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [1],
                    "target": "existing",
                    "event_id": "evt-1",
                    "proposed_title": None,
                    "confidence": 0.74,
                    "reason": "第二条像是在延续已有事项",
                },
                {
                    "fact_indexes": [0],
                    "target": "new",
                    "event_id": None,
                    "proposed_title": "补渠道复盘",
                    "confidence": 0.72,
                    "reason": "第一条也像独立新事项",
                },
            ],
            "ambiguities": ["可能涉及两件事"],
        },
        facts=persisted["facts"],
        completed_at=16,
    )

    assert result["status"] == "succeeded"
    assert result["routing_mode"] == "confirm"
    assert result["result"] == {
        "groups": [
            {
                "fact_ids": ["fact-a"],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.74,
                "reason": "第二条像是在延续已有事项",
            },
            {
                "fact_ids": ["fact-b"],
                "target": "new",
                "event_id": None,
                "proposed_title": "补渠道复盘",
                "confidence": 0.72,
                "reason": "第一条也像独立新事项",
            },
        ],
        "ambiguities": ["可能涉及两件事"],
    }
    fact_rows = conn.execute(
        """
        SELECT fact_id, event_id, event_assignment
        FROM facts
        WHERE semantic_run_id = ?
        ORDER BY fact_id ASC
        """,
        (run["semantic_run_id"],),
    ).fetchall()
    assert [(row["fact_id"], row["event_id"], row["event_assignment"]) for row in fact_rows] == [
        ("fact-a", None, "unassigned"),
        ("fact-b", None, "unassigned"),
    ]


def test_event_match_run_auto_existing_assigns_all_facts_atomically(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=10, updated_at=10)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "request",
                "content": "请补一版渠道复盘",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 12,
                "updated_at": 12,
            },
            {
                "fact_id": "fact-2",
                "fact_type": "deadline_change",
                "content": "截止改到周五",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 13,
                "updated_at": 13,
            },
        ],
        interpretations=[],
        completed_at=14,
    )
    match_run = create_event_match_run(
        conn,
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=15,
    )

    result = persist_event_match_run_result(
        conn,
        event_match_run_id=match_run["event_match_run_id"],
        semantic_run_id=run["semantic_run_id"],
        routing_mode="auto",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0, 1],
                    "target": "existing",
                    "event_id": "evt-1",
                    "proposed_title": None,
                    "confidence": 0.96,
                    "reason": "都属于渠道复盘事项",
                }
            ],
            "ambiguities": [],
        },
        facts=persisted["facts"],
        completed_at=16,
    )

    assert result["status"] == "succeeded"
    assert result["routing_mode"] == "auto"
    fact_rows = conn.execute(
        """
        SELECT fact_id, event_id, event_assignment, event_assignment_confidence
        FROM facts
        WHERE semantic_run_id = ?
        ORDER BY fact_id ASC
        """,
        (run["semantic_run_id"],),
    ).fetchall()
    assert [(row["fact_id"], row["event_id"], row["event_assignment"], row["event_assignment_confidence"]) for row in fact_rows] == [
        ("fact-1", "evt-1", "auto", 0.96),
        ("fact-2", "evt-1", "auto", 0.96),
    ]


def test_event_match_run_auto_new_creates_event_and_assigns_all_facts_atomically(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=11,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "request",
                "content": "请今天补签供应商合同",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 12,
                "updated_at": 12,
            }
        ],
        interpretations=[],
        completed_at=14,
    )
    match_run = create_event_match_run(
        conn,
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=15,
    )

    result = persist_event_match_run_result(
        conn,
        event_match_run_id=match_run["event_match_run_id"],
        semantic_run_id=run["semantic_run_id"],
        routing_mode="auto",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0],
                    "target": "new",
                    "event_id": None,
                    "proposed_title": "补签供应商合同",
                    "confidence": 0.95,
                    "reason": "是一件明确的新事项",
                }
            ],
            "ambiguities": [],
        },
        facts=persisted["facts"],
        completed_at=16,
    )

    assert result["status"] == "succeeded"
    assert result["result"]["groups"][0]["event_id"] is not None
    created_event = conn.execute(
        "SELECT title FROM events WHERE event_id = ?",
        (result["result"]["groups"][0]["event_id"],),
    ).fetchone()
    fact_row = conn.execute(
        """
        SELECT event_id, event_assignment, event_assignment_confidence
        FROM facts WHERE fact_id = ?
        """,
        ("fact-1",),
    ).fetchone()
    assert created_event["title"] == "补签供应商合同"
    assert fact_row["event_id"] == result["result"]["groups"][0]["event_id"]
    assert fact_row["event_assignment"] == "auto"
    assert fact_row["event_assignment_confidence"] == 0.95


def test_event_match_run_auto_rolls_back_on_protected_fact(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="旧事项", created_at=10, updated_at=10)
    create_event(conn, event_id="evt-2", title="新事项", created_at=11, updated_at=11)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=12,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "request",
                "content": "第一条事实",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 13,
                "updated_at": 13,
            },
            {
                "fact_id": "fact-2",
                "fact_type": "statement",
                "content": "第二条事实",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 14,
                "updated_at": 14,
            },
        ],
        interpretations=[],
        completed_at=15,
    )
    set_event_assignment_by_user(conn, fact_id="fact-2", event_id="evt-2", updated_at=16)
    match_run = create_event_match_run(
        conn,
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=17,
    )

    with pytest.raises(ProtectedFactError, match="confirmed event assignment"):
        persist_event_match_run_result(
            conn,
            event_match_run_id=match_run["event_match_run_id"],
            semantic_run_id=run["semantic_run_id"],
            routing_mode="auto",
            normalized_match={
                "groups": [
                    {
                        "fact_indexes": [0, 1],
                        "target": "existing",
                        "event_id": "evt-1",
                        "proposed_title": None,
                        "confidence": 0.95,
                        "reason": "都应归入旧事项",
                    }
                ],
                "ambiguities": [],
            },
            facts=persisted["facts"],
            completed_at=18,
        )

    event_count = conn.execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]
    fact_rows = conn.execute(
        """
        SELECT fact_id, event_id, event_assignment
        FROM facts
        WHERE semantic_run_id = ?
        ORDER BY fact_id ASC
        """,
        (run["semantic_run_id"],),
    ).fetchall()
    match_row = conn.execute(
        "SELECT status, routing_mode, result_json FROM event_match_runs WHERE event_match_run_id = ?",
        (match_run["event_match_run_id"],),
    ).fetchone()
    assert event_count == 2
    assert [(row["fact_id"], row["event_id"], row["event_assignment"]) for row in fact_rows] == [
        ("fact-1", None, "unassigned"),
        ("fact-2", "evt-2", "confirmed"),
    ]
    assert match_row["status"] == "running"
    assert match_row["routing_mode"] is None
    assert match_row["result_json"] is None


def test_review_event_match_run_assigns_existing_and_new_atomically(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=1, updated_at=1)
    ev1, run, _, reviewed_source = _prepare_reviewable_match_run(conn, blobs_root)

    reviewed = review_event_match_run_by_user(
        conn,
        evidence_id=ev1["evidence_id"],
        event_match_run_id=reviewed_source["event_match_run_id"],
        decisions=[
            {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
            {"group_index": 1, "choice": "new", "new_title": "整理客户回访"},
        ],
        reviewed_at=20,
    )

    fact_rows = conn.execute(
        "SELECT fact_id, event_id, event_assignment FROM facts ORDER BY fact_id ASC"
    ).fetchall()
    new_event = conn.execute(
        "SELECT event_id, title FROM events WHERE event_id != 'evt-1' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    assert reviewed["review_status"] == "completed"
    assert reviewed["reviewed_at"] == 20
    assert fact_rows[0]["event_id"] == "evt-1"
    assert fact_rows[0]["event_assignment"] == "confirmed"
    assert fact_rows[1]["event_id"] == new_event["event_id"]
    assert fact_rows[1]["event_assignment"] == "confirmed"
    assert new_event["title"] == "整理客户回访"
    assert run["semantic_run_id"] == "srun-confirm"


def test_review_event_match_run_needs_context_can_leave_group_unassigned(db_file, blobs_root):
    conn = init_db(db_file)
    ev1, _, _, reviewed_source = _prepare_reviewable_match_run(
        conn,
        blobs_root,
        routing_mode="needs_context",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0, 1],
                    "target": "unassigned",
                    "event_id": None,
                    "proposed_title": None,
                    "confidence": 0.0,
                    "reason": "上下文不足",
                }
            ],
            "ambiguities": [],
        },
    )

    reviewed = review_event_match_run_by_user(
        conn,
        evidence_id=ev1["evidence_id"],
        event_match_run_id=reviewed_source["event_match_run_id"],
        decisions=[{"group_index": 0, "choice": "unassigned"}],
        reviewed_at=21,
    )

    rows = conn.execute(
        "SELECT event_id, event_assignment FROM facts ORDER BY fact_id ASC"
    ).fetchall()
    assert reviewed["review_status"] == "completed"
    assert all(row["event_id"] is None and row["event_assignment"] == "unassigned" for row in rows)


def test_review_event_match_run_deduplicates_same_new_title_in_single_submit(db_file, blobs_root):
    conn = init_db(db_file)
    ev1, _, _, reviewed_source = _prepare_reviewable_match_run(
        conn,
        blobs_root,
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0],
                    "target": "new",
                    "event_id": None,
                    "proposed_title": "事项A",
                    "confidence": 0.7,
                    "reason": "第一组",
                },
                {
                    "fact_indexes": [1],
                    "target": "new",
                    "event_id": None,
                    "proposed_title": "事项B",
                    "confidence": 0.7,
                    "reason": "第二组",
                },
            ],
            "ambiguities": [],
        },
    )

    review_event_match_run_by_user(
        conn,
        evidence_id=ev1["evidence_id"],
        event_match_run_id=reviewed_source["event_match_run_id"],
        decisions=[
            {"group_index": 0, "choice": "new", "new_title": "  同一个事项  "},
            {"group_index": 1, "choice": "new", "new_title": "同一个事项"},
        ],
        reviewed_at=22,
    )

    events = conn.execute("SELECT event_id, title FROM events ORDER BY created_at ASC").fetchall()
    fact_rows = conn.execute("SELECT fact_id, event_id FROM facts ORDER BY fact_id ASC").fetchall()
    assert len(events) == 1
    assert events[0]["title"] == "同一个事项"
    assert fact_rows[0]["event_id"] == events[0]["event_id"]
    assert fact_rows[1]["event_id"] == events[0]["event_id"]


def test_review_event_match_run_rejects_stale_run(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=1, updated_at=1)
    ev1, _, _, first_review_source = _prepare_reviewable_match_run(conn, blobs_root)

    run2 = create_semantic_run(
        conn,
        semantic_run_id="srun-newer",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=30,
    )
    persisted2 = persist_semantic_run_result(
        conn,
        semantic_run_id=run2["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-newer",
                "fact_type": "statement",
                "content": "新的事实",
                "evidence_ids": [ev1["evidence_id"]],
                "created_at": 31,
                "updated_at": 31,
            }
        ],
        interpretations=[],
        completed_at=32,
    )
    match_run2 = create_event_match_run(
        conn,
        event_match_run_id="mrun-newer",
        semantic_run_id=run2["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=33,
    )
    persist_event_match_run_result(
        conn,
        event_match_run_id=match_run2["event_match_run_id"],
        semantic_run_id=run2["semantic_run_id"],
        routing_mode="confirm",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0],
                    "target": "existing",
                    "event_id": "evt-1",
                    "proposed_title": None,
                    "confidence": 0.8,
                    "reason": "更新后的建议",
                }
            ],
            "ambiguities": [],
        },
        facts=persisted2["facts"],
        completed_at=34,
    )

    with pytest.raises(SemanticStoreError, match="stale"):
        review_event_match_run_by_user(
            conn,
            evidence_id=ev1["evidence_id"],
            event_match_run_id=first_review_source["event_match_run_id"],
            decisions=[
                {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
                {"group_index": 1, "choice": "new", "new_title": "整理客户回访"},
            ],
            reviewed_at=35,
        )

    stale_run = conn.execute(
        "SELECT review_status FROM event_match_runs WHERE event_match_run_id = ?",
        (first_review_source["event_match_run_id"],),
    ).fetchone()
    assert stale_run["review_status"] == "pending"


def test_review_event_match_run_rejects_non_active_event_and_rolls_back(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=1, updated_at=1)
    create_event(conn, event_id="evt-2", title="旧事项", status="resolved", created_at=2, updated_at=2)
    ev1, _, _, reviewed_source = _prepare_reviewable_match_run(conn, blobs_root)

    with pytest.raises(SemanticStoreError, match="active event not found"):
        review_event_match_run_by_user(
            conn,
            evidence_id=ev1["evidence_id"],
            event_match_run_id=reviewed_source["event_match_run_id"],
            decisions=[
                {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
                {"group_index": 1, "choice": "existing", "event_id": "evt-2"},
            ],
            reviewed_at=36,
        )

    fact_rows = conn.execute(
        "SELECT fact_id, event_id, event_assignment FROM facts ORDER BY fact_id ASC"
    ).fetchall()
    reviewed_row = conn.execute(
        "SELECT review_status, reviewed_at FROM event_match_runs WHERE event_match_run_id = ?",
        (reviewed_source["event_match_run_id"],),
    ).fetchone()
    assert all(row["event_id"] is None and row["event_assignment"] == "unassigned" for row in fact_rows)
    assert reviewed_row["review_status"] == "pending"
    assert reviewed_row["reviewed_at"] is None


def test_review_event_match_run_cannot_be_submitted_twice_and_ai_cannot_override(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="渠道复盘", created_at=1, updated_at=1)
    ev1, _, _, reviewed_source = _prepare_reviewable_match_run(conn, blobs_root)

    review_event_match_run_by_user(
        conn,
        evidence_id=ev1["evidence_id"],
        event_match_run_id=reviewed_source["event_match_run_id"],
        decisions=[
            {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
            {"group_index": 1, "choice": "new", "new_title": "整理客户回访"},
        ],
        reviewed_at=37,
    )

    with pytest.raises(SemanticStoreError, match="already completed"):
        review_event_match_run_by_user(
            conn,
            evidence_id=ev1["evidence_id"],
            event_match_run_id=reviewed_source["event_match_run_id"],
            decisions=[
                {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
                {"group_index": 1, "choice": "new", "new_title": "整理客户回访"},
            ],
            reviewed_at=38,
        )
    with pytest.raises(ProtectedFactError, match="confirmed event assignment"):
        set_event_assignment_by_ai(
            conn,
            fact_id="fact-a",
            event_id="evt-1",
            assignment="suggested",
            event_assignment_confidence=0.2,
            updated_at=39,
        )


def test_event_match_run_lifecycle_failed_and_supersedes_history(db_file, blobs_root):
    conn = init_db(db_file)
    ev1 = _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    run1 = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=10,
    )
    mark_semantic_run_succeeded(conn, semantic_run_id=run1["semantic_run_id"], completed_at=11)
    first = create_event_match_run(
        conn,
        event_match_run_id="mrun-1",
        semantic_run_id=run1["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=12,
    )
    failed = mark_event_match_run_failed(
        conn,
        event_match_run_id=first["event_match_run_id"],
        failure_type="provider_invalid_response",
        completed_at=13,
    )

    run2 = create_semantic_run(
        conn,
        semantic_run_id="srun-2",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": ev1["evidence_id"], "position": 0}],
        created_at=14,
        supersedes_run_id=run1["semantic_run_id"],
    )
    mark_semantic_run_succeeded(conn, semantic_run_id=run2["semantic_run_id"], completed_at=15)
    second = create_event_match_run(
        conn,
        event_match_run_id="mrun-2",
        semantic_run_id=run2["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=16,
        supersedes_run_id=first["event_match_run_id"],
    )

    latest = get_latest_event_match_for_evidence(conn, ev1["evidence_id"])

    assert failed["status"] == "failed"
    assert failed["failure_type"] == "provider_invalid_response"
    assert second["supersedes_run_id"] == "mrun-1"
    assert latest["event_match_run_id"] == "mrun-2"


def test_event_change_run_lifecycle_dedupes_duplicates_and_tracks_latest_succeeded(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="事项一", created_at=1, updated_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="5月1日：采用方案A。", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="5月6日：我之前要求的是方案B。", captured_at=2)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[
            {"evidence_id": "ev-1", "position": 0},
            {"evidence_id": "ev-2", "position": 1},
        ],
        created_at=10,
    )
    persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "request",
                "content": "张三要求采用方案A。",
                "event_id": "evt-1",
                "event_assignment": "confirmed",
                "evidence_ids": ["ev-1"],
                "created_at": 11,
                "updated_at": 11,
            },
            {
                "fact_id": "fact-2",
                "fact_type": "statement",
                "content": "张三表示我之前要求的是方案B。",
                "event_id": "evt-1",
                "event_assignment": "confirmed",
                "evidence_ids": ["ev-2"],
                "created_at": 12,
                "updated_at": 12,
            },
        ],
        interpretations=[],
        completed_at=13,
    )

    first = create_event_change_run(
        conn,
        change_run_id="crun-1",
        event_id="evt-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        detector_version="1.0",
        created_at=20,
    )
    failed = mark_event_change_run_failed(
        conn,
        change_run_id=first["change_run_id"],
        failure_type="provider_timeout",
        completed_at=21,
    )
    second = create_event_change_run(
        conn,
        change_run_id="crun-2",
        event_id="evt-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        detector_version="1.0",
        created_at=22,
    )
    persisted = persist_event_change_run_result(
        conn,
        change_run_id=second["change_run_id"],
        event_id="evt-1",
        changes=[
            {
                "change_type": "contradiction",
                "earlier_fact_id": "fact-1",
                "later_fact_id": "fact-2",
                "summary": "前后记录对之前要求的方案表述不一致。",
                "confidence": 0.91,
            },
            {
                "change_type": "contradiction",
                "earlier_fact_id": "fact-1",
                "later_fact_id": "fact-2",
                "summary": "重复记录不应再次保存。",
                "confidence": 0.4,
            },
        ],
        completed_at=23,
    )

    latest = get_latest_event_change_run_for_event(conn, "evt-1")

    assert failed["status"] == "failed"
    assert failed["failure_type"] == "provider_timeout"
    assert persisted["status"] == "succeeded"
    assert len(persisted["changes"]) == 1
    assert persisted["changes"][0]["change_type"] == "contradiction"
    assert latest["change_run_id"] == "crun-2"


def test_event_match_operations_keep_verify_chain_clean(db_file, blobs_root):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    create_event(conn, event_id="evt-1", title="事件一", created_at=10)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[{"evidence_id": "ev-1", "position": 0}],
        created_at=11,
    )
    persisted = persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "statement",
                "content": "先记一条事实",
                "evidence_ids": ["ev-1"],
                "created_at": 12,
                "updated_at": 12,
            }
        ],
        interpretations=[],
        completed_at=13,
    )

    before = verify_chain(conn, blobs_root=blobs_root)
    match_run = create_event_match_run(
        conn,
        semantic_run_id=run["semantic_run_id"],
        provider="deepseek",
        model="deepseek-v4-flash",
        matcher_version="1.0",
        created_at=14,
    )
    persist_event_match_run_result(
        conn,
        event_match_run_id=match_run["event_match_run_id"],
        semantic_run_id=run["semantic_run_id"],
        routing_mode="auto",
        normalized_match={
            "groups": [
                {
                    "fact_indexes": [0],
                    "target": "existing",
                    "event_id": "evt-1",
                    "proposed_title": None,
                    "confidence": 0.95,
                    "reason": "延续旧事项",
                }
            ],
            "ambiguities": [],
        },
        facts=persisted["facts"],
        completed_at=15,
    )
    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)


def test_event_change_operations_keep_verify_chain_clean(db_file, blobs_root):
    conn = init_db(db_file)
    create_event(conn, event_id="evt-1", title="事件一", created_at=10)
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)
    run = create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[
            {"evidence_id": "ev-1", "position": 0},
            {"evidence_id": "ev-2", "position": 1},
        ],
        created_at=11,
    )
    persist_semantic_run_result(
        conn,
        semantic_run_id=run["semantic_run_id"],
        facts=[
            {
                "fact_id": "fact-1",
                "fact_type": "request",
                "content": "先做方案A。",
                "event_id": "evt-1",
                "event_assignment": "confirmed",
                "evidence_ids": ["ev-1"],
                "created_at": 12,
                "updated_at": 12,
            },
            {
                "fact_id": "fact-2",
                "fact_type": "request",
                "content": "改成方案B。",
                "event_id": "evt-1",
                "event_assignment": "confirmed",
                "evidence_ids": ["ev-2"],
                "created_at": 13,
                "updated_at": 13,
            },
        ],
        interpretations=[],
        completed_at=14,
    )

    before = verify_chain(conn, blobs_root=blobs_root)
    change_run = create_event_change_run(
        conn,
        event_id="evt-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        detector_version="1.0",
        created_at=15,
    )
    persist_event_change_run_result(
        conn,
        change_run_id=change_run["change_run_id"],
        event_id="evt-1",
        changes=[
            {
                "change_type": "requirement_change",
                "earlier_fact_id": "fact-1",
                "later_fact_id": "fact-2",
                "summary": "要求从方案A改为方案B。",
                "confidence": 0.89,
            }
        ],
        completed_at=16,
    )
    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)


def test_semantic_operations_keep_verify_chain_clean(db_file, blobs_root):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    _append_text(conn, blobs_root, evidence_id="ev-1", text="一", captured_at=1)
    _append_text(conn, blobs_root, evidence_id="ev-2", text="二", captured_at=2)

    before = verify_chain(conn, blobs_root=blobs_root)

    create_submission(conn, submission_id="sub-1", created_at=10, evidence_ids=["ev-2", "ev-1"])
    create_event(conn, event_id="evt-1", title="事件一", created_at=11)
    create_semantic_run(
        conn,
        semantic_run_id="srun-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        parser_version=SEMANTIC_PARSER_VERSION,
        inputs=[
            {"evidence_id": "ev-1", "position": 0},
            {"evidence_id": "ev-2", "position": 1},
        ],
        created_at=12,
    )
    create_fact(
        conn,
        fact_id="fact-1",
        fact_type="request",
        content="请补一份渠道复盘",
        evidence_ids=["ev-1", "ev-2"],
        semantic_run_id="srun-1",
        actor_roles=[("act-1", "requester"), ("act-2", "owner")],
        created_at=13,
        updated_at=13,
    )
    set_event_assignment_by_ai(
        conn,
        fact_id="fact-1",
        event_id="evt-1",
        assignment="auto",
        event_assignment_confidence=0.8,
        updated_at=14,
    )
    create_interpretation(
        conn,
        interpretation_id="itp-1",
        fact_id="fact-1",
        semantic_run_id="srun-1",
        kind="uncertainty",
        content="交付时间仍需确认",
        created_at=15,
    )
    correct_fact_by_user(
        conn,
        fact_id="fact-1",
        due_raw="下下周五",
        due_anchor_at=12,
        actor_roles=[("act-2", "owner")],
        updated_at=16,
    )
    correct_relative_due_dates_by_user(
        conn,
        evidence_id="ev-1",
        semantic_run_id="srun-1",
        due_updates=[
            {
                "fact_id": "fact-1",
                "due_at": due_date_to_millis("2026-08-21"),
                "due_anchor_at": due_date_to_millis("2026-08-07"),
            }
        ],
        updated_at=17,
    )
    mark_semantic_run_succeeded(conn, semantic_run_id="srun-1", completed_at=18)

    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)
