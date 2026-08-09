from __future__ import annotations

from pathlib import Path

import pytest

from app.semantic_llm import SEMANTIC_PARSER_VERSION
from evidence_core.db import init_db
from evidence_core.extraction_store import create_extraction
from evidence_core.semantic_store import (
    ProtectedFactError,
    SemanticStoreError,
    confirm_fact,
    create_event,
    create_event_match_run,
    create_fact,
    create_interpretation,
    create_semantic_run,
    create_submission,
    correct_fact_by_user,
    get_latest_event_match_for_evidence,
    get_semantic_run,
    get_latest_semantic_run_for_evidence,
    list_event_candidates,
    list_facts_for_semantic_run,
    list_interpretations_for_semantic_run,
    mark_event_match_run_failed,
    mark_semantic_run_failed,
    mark_semantic_run_succeeded,
    persist_event_match_run_result,
    persist_semantic_run_result,
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
    mark_semantic_run_succeeded(conn, semantic_run_id="srun-1", completed_at=17)

    after = verify_chain(conn, blobs_root=blobs_root)

    assert before == (True, None, None)
    assert after == (True, None, None)
