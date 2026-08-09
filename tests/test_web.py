from __future__ import annotations

import base64
import io
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from docx import Document
from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from app import main as main_module
from app.main import create_app
from evidence_core.chain import compute_content_hash
from evidence_core.db import init_db
from evidence_core.store import verify_chain


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII="
)


def _disable_external_ai(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)


def _build_png_bytes(width: int, height: int, color: tuple[int, int, int] = (32, 96, 160)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_pdf_bytes(text: str | None = None, *, image_only: bool = False) -> bytes:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("STSong-Light", 14)
    if text:
        pdf.drawString(72, 720, text)
    if image_only:
        pdf.drawImage(ImageReader(BytesIO(PNG_BYTES)), 72, 650, width=80, height=80)
    pdf.save()
    return buffer.getvalue()


def _build_docx_bytes(*parts: str) -> bytes:
    document = Document()
    for part in parts:
        document.add_paragraph(part)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _make_client(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))
    monkeypatch.setattr(
        "app.main.event_matcher.match_events",
        lambda facts, *, existing_events=None: {
            "groups": [
                {
                    "fact_indexes": list(range(len(facts))),
                    "target": "unassigned",
                    "event_id": None,
                    "proposed_title": None,
                    "confidence": 0.0,
                    "reason": "默认测试桩不做自动归档",
                }
            ]
            if facts
            else [],
            "ambiguities": [],
        },
    )
    client = TestClient(create_app())
    return client, demo_dir, sandbox_root


def _sandbox_db_path(client: TestClient, sandbox_root: Path) -> Path:
    sandbox_id = client.cookies.get("wc_sid")
    assert sandbox_id is not None
    return sandbox_root / sandbox_id / "workchain.db"


def _upload_png(
    client: TestClient,
    *,
    filename: str = "demo.png",
    text: str = "",
    source: str = "飞书",
    source_detail: str = "项目复盘群",
):
    return client.post(
        "/api/evidence",
        data={
            "text": text,
            "source": source,
            "source_detail": source_detail,
        },
        files={"file": (filename, PNG_BYTES, "image/png")},
    )


def _multipart_request(
    client: TestClient,
    *,
    data: dict[str, str],
    file_part: tuple[str, bytes, str] | None = None,
):
    boundary = "----workchain-boundary"
    body = bytearray()

    for key, value in data.items():
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    if file_part is not None:
        filename, payload, content_type = file_part
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("ascii"))
        body.extend(payload)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return client.post(
        "/api/evidence",
        content=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def _semantic_result(
    *,
    facts: list[dict[str, object]] | None = None,
    interpretations: list[dict[str, object]] | None = None,
    ambiguities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "facts": facts or [],
        "interpretations": interpretations or [],
        "ambiguities": ambiguities or [],
    }


def test_healthz_returns_ok_and_evidence_count_18(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "evidence_count": 18}


def test_diag_llm_without_api_key_returns_configured_false(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/api/diag/llm")

    assert response.status_code == 200
    assert response.json() == {
        "configured": False,
        "reachable": None,
        "detail": "DEEPSEEK_API_KEY not set",
    }


def test_diag_llm_with_connection_error_returns_reachable_false(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    fake_key = "sk-test-visible-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_key)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with patch("app.main.httpx.post", side_effect=httpx.ConnectError(f"boom {fake_key}")):
        with client:
            response = client.get("/api/diag/llm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["reachable"] is False
    assert payload["status_code"] is None
    assert payload["detail"]
    assert fake_key not in response.text
    assert fake_key not in payload["detail"]


def test_diag_llm_response_never_contains_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    fake_key = "sk-very-obvious-secret"
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_key)
    client, _, _ = _make_client(tmp_path, monkeypatch)
    mock_response = Mock(status_code=401)

    with patch("app.main.httpx.post", return_value=mock_response):
        with client:
            response = client.get("/api/diag/llm")

    assert response.status_code == 200
    assert fake_key not in response.text


def test_diag_llm_returns_404_when_diagnostics_disabled_and_does_not_probe(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKCHAIN_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with patch("app.main.httpx.post") as mock_post:
        with client:
            response = client.get("/api/diag/llm")

    assert response.status_code == 404
    mock_post.assert_not_called()


def test_diag_ocr_without_api_key_returns_configured_false(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/api/diag/ocr")

    assert response.status_code == 200
    assert response.json() == {
        "configured": False,
        "reachable": None,
        "detail": "DASHSCOPE_API_KEY not set",
    }


def test_diag_ocr_with_connection_error_returns_reachable_false_and_never_leaks_key(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    fake_key = "dashscope-visible-secret"
    monkeypatch.setenv("DASHSCOPE_API_KEY", fake_key)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = Mock(completions=Mock(create=Mock(side_effect=httpx.ConnectError(f"boom {fake_key}"))))

    with patch("app.ocr.OpenAI", FakeOpenAI):
        with client:
            response = client.get("/api/diag/ocr")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["reachable"] is False
    assert payload["status_code"] is None
    assert payload["model"] == "vanchin/deepseek-ocr"
    assert "无法连接图片识别服务:ConnectError" in payload["detail"]
    assert fake_key not in response.text
    assert fake_key not in payload["detail"]


def test_diag_ocr_returns_404_when_diagnostics_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKCHAIN_DIAGNOSTICS", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/api/diag/ocr")

    assert response.status_code == 404


def test_index_uses_single_column_home_layout_and_hides_secondary_sections(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert ">WC<" not in html
    assert html.count('href="/records"') == 2
    assert "把散落的沟通，整理成可以回看的事实。" in html
    assert "原始材料独立保存，AI 整理结果可以纠正。" in html
    assert "WorkChain 帮你还原发生了什么、解释难懂表达，并把同一件事串起来。" in html
    assert "我的事项" in html
    assert "记录" in html
    assert "我的词典" in html
    assert 'id="glossary-drawer"' in html
    assert 'data-open-glossary' in html
    assert 'id="mobile-nav-toggle"' in html
    assert 'id="mobile-nav-panel"' in html
    assert "补充信息（可选）" in html
    assert "开始整理" in html
    assert "放进记录" in html
    assert "看看示例" not in html
    assert "不知道放什么？试试看：" not in html
    assert 'data-fill-example=' not in html
    assert 'data-testid="thread-card"' not in html
    assert "历史事项" not in html
    assert "最近证据" not in html
    assert "导出完整举证包" not in html
    assert "参考信息" not in html
    assert 'lg:grid-cols-[300px_minmax(0,1fr)]' not in html
    assert 'name="counterpart"' not in html
    assert 'rows="8"' not in html
    assert 'min-h-[220px]' not in html
    assert 'rows="4"' in html
    assert 'min-h-[120px]' in html


def test_index_contains_glossary_drawer_shell_and_settings_controls(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    html = response.text
    assert 'id="glossary-drawer"' in html
    assert 'id="glossary-overlay"' in html
    assert 'id="glossary-list"' in html
    assert 'id="add-glossary-row"' in html
    assert 'id="save-settings"' in html
    assert 'id="new-glossary-form"' in html
    assert 'fetch("/api/settings")' in html
    assert 'fetch("/api/settings", {' in html
    assert "openGlossary" in html


def test_index_shows_real_event_in_my_events_without_secondary_home_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                index_response = client.get("/")

    assert create_response.status_code == 200
    html = index_response.text
    assert "我的事项" in html
    assert 'data-testid="event-card"' in html
    assert "补签供应商合同" in html
    assert "看看示例" not in html
    assert 'data-testid="thread-card"' not in html
    assert "历史事项" not in html
    assert "最近证据" not in html
    assert "导出完整举证包" not in html

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        event_row = conn.execute(
            """
            SELECT event_id, title
            FROM events
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert f'href="/event/{event_row["event_id"]}"' in html
        assert event_row["title"] == "补签供应商合同"
    finally:
        conn.close()


def test_event_detail_shows_facts_and_evidence_link(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        event_row = conn.execute(
            """
            SELECT event_id
            FROM events
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    with client:
        response = client.get(f"/event/{event_row['event_id']}")

    assert response.status_code == 200
    assert "事项详情" in response.text
    assert "请今天补签供应商合同。" in response.text
    assert "修改事项名称" in response.text
    assert "查看原始证据" in response.text
    assert f'href="/evidence/{evidence_id}"' in response.text


def test_evidence_detail_script_has_single_evidence_id_and_independent_initializers(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "deadline_change",
                "content": "请周五前补材料。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": "2026-08-14",
                "due_anchor_date": "2026-08-09",
                "occurred_date": None,
                "confidence": 0.9,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补材料",
                "confidence": 0.9,
                "reason": "这是一件新的补材料事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={
                        "text": "请周五前补材料。",
                        "source": "飞书",
                        "source_detail": "项目复盘群",
                        "record_date": "2026-08-09",
                    },
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    html = detail_response.text
    assert html.count("const evidenceId =") == 1
    assert 'const evidenceId = form.dataset.evidenceId;' not in html
    assert "const initRecordDateEditor = () => {" in html
    assert "const initEventAssignmentForm = () => {" in html
    assert "const initSlotsForm = () => {" in html
    assert "const initOcrEditor = () => {" in html
    assert "initRecordDateEditor();" in html
    assert "initEventAssignmentForm();" in html
    assert "initSlotsForm();" in html
    assert "initOcrEditor();" in html
    assert 'id="record-date-toggle"' in html
    assert "setRecordDateEditorOpen(true, elements);" in html


def test_evidence_detail_hides_legacy_slots_form_for_semantic_records(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert 'id="slots-form"' not in detail_response.text
    assert 'id="slot-deliverable"' not in detail_response.text
    assert 'id="slot-due-raw"' not in detail_response.text


def test_legacy_evidence_detail_keeps_slots_form_without_semantic_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=_semantic_result()):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "旧兼容记录。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        conn.execute(
            """
            DELETE FROM interpretations
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute(
            """
            DELETE FROM facts
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute(
            """
            DELETE FROM event_match_runs
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute("DELETE FROM semantic_run_inputs WHERE evidence_id = ?", (evidence_id,))
        conn.execute(
            """
            DELETE FROM semantic_runs
            WHERE semantic_run_id NOT IN (
                SELECT semantic_run_id FROM semantic_run_inputs
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM facts WHERE semantic_run_id IS NOT NULL
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM interpretations WHERE semantic_run_id IS NOT NULL
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM event_match_runs
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    with client:
        detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert 'id="slots-form"' in detail_response.text
    assert 'id="slot-deliverable"' in detail_response.text


def test_event_title_can_be_updated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        event_row = conn.execute(
            "SELECT event_id, title, updated_at FROM events ORDER BY created_at DESC, event_id DESC LIMIT 1"
        ).fetchone()
        original_updated_at = event_row["updated_at"]
    finally:
        conn.close()

    with client:
        update_response = client.post(
            f"/api/events/{event_row['event_id']}/title",
            json={"title": "  供应商合同改名  "},
        )
        detail_response = client.get(f"/event/{event_row['event_id']}")

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        updated_row = conn.execute(
            "SELECT title, updated_at FROM events WHERE event_id = ?",
            (event_row["event_id"],),
        ).fetchone()
    finally:
        conn.close()

    assert update_response.status_code == 200
    assert update_response.json()["title"] == "供应商合同改名"
    assert updated_row["title"] == "供应商合同改名"
    assert updated_row["updated_at"] >= original_updated_at
    assert "供应商合同改名" in detail_response.text


def test_correct_event_fact_marks_user_and_keeps_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        row = conn.execute(
            """
            SELECT f.fact_id, f.semantic_run_id, f.event_id, e.updated_at AS event_updated_at
            FROM facts f
            JOIN events e ON e.event_id = f.event_id
            ORDER BY f.created_at DESC, f.fact_id DESC
            LIMIT 1
            """
        ).fetchone()
        original_event_updated_at = row["event_updated_at"]
    finally:
        conn.close()

    with client:
        update_response = client.post(
            f"/api/events/{row['event_id']}/facts/{row['fact_id']}/correct",
            json={
                "content": "用户确认：供应商合同今天改为明天补签。",
                "fact_type": "deadline_change",
                "due_at": "2026-08-10",
            },
        )
        detail_response = client.get(f"/event/{row['event_id']}")

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        fact_row = conn.execute(
            """
            SELECT fact_type, content, due_at, origin, review_status, semantic_run_id
            FROM facts
            WHERE fact_id = ?
            """,
            (row["fact_id"],),
        ).fetchone()
        evidence_rows = conn.execute(
            "SELECT evidence_id FROM fact_evidence WHERE fact_id = ? ORDER BY evidence_id",
            (row["fact_id"],),
        ).fetchall()
        event_row = conn.execute(
            "SELECT updated_at FROM events WHERE event_id = ?",
            (row["event_id"],),
        ).fetchone()
    finally:
        conn.close()

    assert update_response.status_code == 200
    assert update_response.json()["origin"] == "user"
    assert update_response.json()["review_status"] == "corrected"
    assert fact_row["fact_type"] == "deadline_change"
    assert fact_row["content"] == "用户确认：供应商合同今天改为明天补签。"
    assert fact_row["due_at"] == main_module.llm.due_date_to_millis("2026-08-10")
    assert fact_row["origin"] == "user"
    assert fact_row["review_status"] == "corrected"
    assert fact_row["semantic_run_id"] == row["semantic_run_id"]
    assert [item["evidence_id"] for item in evidence_rows] == [evidence_id]
    assert event_row["updated_at"] >= original_event_updated_at
    assert "用户确认：供应商合同今天改为明天补签。" in detail_response.text
    assert "已纠正" in detail_response.text


def test_correct_event_fact_rejects_provenance_tampering_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        row = conn.execute(
            """
            SELECT f.fact_id, f.semantic_run_id, f.event_id, f.content, fe.evidence_id
            FROM facts f
            JOIN fact_evidence fe ON fe.fact_id = f.fact_id
            ORDER BY f.created_at DESC, f.fact_id DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    with client:
        update_response = client.post(
            f"/api/events/{row['event_id']}/facts/{row['fact_id']}/correct",
            json={
                "content": "有人想偷偷改 provenance。",
                "fact_type": "statement",
                "due_at": None,
                "semantic_run_id": "srun_hacked",
                "evidence_ids": ["ev_hacked"],
            },
        )

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        fact_row = conn.execute(
            "SELECT content, semantic_run_id FROM facts WHERE fact_id = ?",
            (row["fact_id"],),
        ).fetchone()
        evidence_rows = conn.execute(
            "SELECT evidence_id FROM fact_evidence WHERE fact_id = ?",
            (row["fact_id"],),
        ).fetchall()
    finally:
        conn.close()

    assert update_response.status_code == 400
    assert update_response.json()["detail"] == "unsupported fields: evidence_ids, semantic_run_id"
    assert fact_row["content"] == row["content"]
    assert fact_row["semantic_run_id"] == row["semantic_run_id"]
    assert [item["evidence_id"] for item in evidence_rows] == [row["evidence_id"]]


def test_home_event_card_prefers_semantic_fact_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "系统整理：补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "原文里只说请处理合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                index_response = client.get("/")

    assert create_response.status_code == 200
    assert "系统整理：补签供应商合同。" in index_response.text
    assert "补签供应商合同" in index_response.text
    assert "截止：" not in index_response.text


def test_home_page_contains_optional_meta_fields_and_mutually_exclusive_input(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="text-input-region"' in response.text
    assert 'id="file-picker-region"' in response.text
    assert 'id="text-mode-hint"' in response.text
    assert "清空文字后可改用文件" in response.text
    assert 'data-fill-example=' not in response.text
    assert "EXAMPLE_TEXTS" not in response.text
    assert "counterpart" not in response.text
    assert "syncInputMode();" in response.text
    assert 'min="1900-01-01"' in response.text
    assert 'max="2100-12-31"' in response.text
    assert 'textInputRegion.classList.toggle("hidden", hideTextInput);' in response.text
    assert 'filePickerRegion.classList.toggle("hidden", hasFile || hasTextInput);' in response.text
    assert 'textModeHint.classList.toggle("hidden", !hasTextInput || hasFile);' in response.text
    assert "fileInput.disabled = disableFileInput" in response.text
    assert 'name="record_date"' in response.text
    assert "记录发生日期" in response.text
    assert "有“今天、周五、下周”等相对时间时，补充日期可以换算得更准确。" in response.text
    assert "recordDateInput.reportValidity();" in response.text
    assert 'formData.append("counterpart"' not in response.text
    assert 'window.location.reload()' not in response.text
    assert 'window.location.href = `/evidence/${encodeURIComponent(parsed.value.evidence_id)}`;' in response.text


def test_records_page_lists_current_sandbox_evidence_and_links_to_detail(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client_a, _, _ = _make_client(tmp_path, monkeypatch)
    client_b, _, _ = _make_client(tmp_path, monkeypatch)

    with client_a:
        response_a = client_a.post(
            "/api/evidence",
            json={"text": "只在 A 里出现的记录", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id_a = response_a.json()["evidence_id"]
        records_a = client_a.get("/records")

    with client_b:
        response_b = client_b.post(
            "/api/evidence",
            json={"text": "只在 B 里出现的记录", "source": "Slack", "source_detail": "growth-sync"},
        )
        evidence_id_b = response_b.json()["evidence_id"]
        records_b = client_b.get("/records")

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert records_a.status_code == 200
    assert records_b.status_code == 200
    assert "导出完整记录(PDF)" in records_a.text
    assert 'href="/export/pdf?scope=mine"' in records_a.text
    assert "只在 A 里出现的记录" in records_a.text
    assert "只在 B 里出现的记录" not in records_a.text
    assert f'href="/evidence/{evidence_id_a}"' in records_a.text
    assert "来源：飞书 / 项目复盘群" in records_a.text
    assert "媒体类型：文字" in records_a.text
    assert "解析状态：" in records_a.text
    assert "只在 B 里出现的记录" in records_b.text
    assert "只在 A 里出现的记录" not in records_b.text
    assert f'href="/evidence/{evidence_id_b}"' in records_b.text


def test_image_evidence_detail_shows_processing_state_and_status_polling_script(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with patch(
        "app.main.get_image_extraction_startup",
        return_value={
            "supported": True,
            "configured": True,
            "requires_ocr_budget_on_start": False,
            "detail": None,
        },
    ):
        with patch("app.main._run_image_pipeline", return_value=None):
            with client:
                create_response = _upload_png(client)
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert create_response.json()["parse_status"] == "ocr_running"
    assert detail_response.status_code == 200
    html = detail_response.text
    assert "正在读取这份记录…" in html
    assert "刚刚这条记录" not in html
    assert f'const evidenceId = "{evidence_id}";' in html
    assert 'const STABLE_PARSE_STATUSES = new Set(["done", "failed", "unsupported"]);' in html
    assert 'fetch(`/api/evidence/${evidenceId}/status`)' in html
    assert 'statusPollTimer = window.setInterval(pollStatus, 1800);' in html
    assert "window.clearInterval(statusPollTimer);" in html
    assert "window.location.reload();" in html


def test_home_event_card_shows_due_text_when_event_has_due_at(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "张伟要求下周五前提交渠道复盘数据。",
                "actors": [],
                "due_raw": "下周五",
                "due_date": "2026-08-08",
                "due_anchor_date": "2026-08-08",
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "渠道复盘",
                "confidence": 0.95,
                "reason": "这是一件新的渠道复盘事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                response = client.post(
                    "/api/evidence",
                    json={"text": "张伟说下周五前提交渠道复盘数据。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                home_response = client.get("/")

    assert response.status_code == 200
    assert "渠道复盘" in home_response.text
    assert "截止：08-08" in home_response.text


def test_home_page_limits_active_event_cards_to_six(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        client.get("/")

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        for index in range(7):
            event_id = f"evt_limit_{index}"
            fact_id = f"fact_limit_{index}"
            evidence_id = f"ev_limit_{index}"
            timestamp = 1_000 + index
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, f"事项 {index}", "active", None, timestamp, timestamp),
            )
            conn.execute(
                """
                INSERT INTO evidence (
                    evidence_id, seq, thread_id, kind, media_type, blob_path, raw_text, source_hint,
                    slot_requester, slot_owner, slot_deliverable, slot_due, slot_due_raw, slot_direction,
                    slots_filled, plain_summary, caveats, occurred_at, captured_at,
                    content_hash, prev_hash, chain_hash
                ) VALUES (?, ?, NULL, 'reference', 'text', NULL, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, '[]', ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    100 + index,
                    f"原文 {index}",
                    "飞书-项目复盘群",
                    timestamp,
                    timestamp,
                    f"hash_{index}",
                    f"prev_{index}",
                    f"chain_{index}",
                ),
            )
            conn.execute(
                """
                INSERT INTO facts (
                    fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw, due_anchor_at,
                    confidence, event_assignment, event_assignment_confidence, origin, review_status,
                    semantic_run_id, created_at, updated_at
                ) VALUES (?, ?, 'request', ?, ?, NULL, NULL, NULL, NULL, 'confirmed', NULL, 'ai', 'unreviewed', NULL, ?, ?)
                """,
                (fact_id, event_id, f"事项 {index} 的摘要", timestamp, timestamp, timestamp),
            )
            conn.execute(
                "INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)",
                (fact_id, evidence_id),
            )
        conn.commit()
    finally:
        conn.close()

    with client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text.count('data-testid="event-card"') == 6
    assert "事项 6" in response.text
    assert "事项 0" not in response.text


def test_home_event_card_shows_only_latest_single_fact_summary(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        client.get("/")

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        conn.execute(
            """
            INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
            VALUES ('evt_summary', '事项摘要测试', 'active', NULL, 1, 1)
            """
        )
        for fact_index, content in enumerate(["更早的一条摘要", "最新的一条摘要"], start=1):
            evidence_id = f"ev_summary_{fact_index}"
            fact_id = f"fact_summary_{fact_index}"
            conn.execute(
                """
                INSERT INTO evidence (
                    evidence_id, seq, thread_id, kind, media_type, blob_path, raw_text, source_hint,
                    slot_requester, slot_owner, slot_deliverable, slot_due, slot_due_raw, slot_direction,
                    slots_filled, plain_summary, caveats, occurred_at, captured_at,
                    content_hash, prev_hash, chain_hash
                ) VALUES (?, ?, NULL, 'reference', 'text', NULL, ?, '飞书-项目复盘群', NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, '[]', ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    200 + fact_index,
                    content,
                    fact_index,
                    fact_index,
                    f"hash_s_{fact_index}",
                    f"prev_s_{fact_index}",
                    f"chain_s_{fact_index}",
                ),
            )
            conn.execute(
                """
                INSERT INTO facts (
                    fact_id, event_id, fact_type, content, occurred_at, due_at, due_raw, due_anchor_at,
                    confidence, event_assignment, event_assignment_confidence, origin, review_status,
                    semantic_run_id, created_at, updated_at
                ) VALUES (?, 'evt_summary', 'statement', ?, ?, NULL, NULL, NULL, NULL, 'confirmed', NULL, 'ai', 'unreviewed', NULL, ?, ?)
                """,
                (fact_id, content, fact_index, fact_index, fact_index),
            )
            conn.execute("INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (?, ?)", (fact_id, evidence_id))
        conn.commit()
    finally:
        conn.close()

    with client:
        response = client.get("/")

    assert response.status_code == 200
    assert "事项摘要测试" in response.text
    assert "最新的一条摘要" in response.text
    assert "更早的一条摘要" not in response.text


def test_user_facing_pages_hide_internal_semantic_field_names(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请周五前补材料。",
                "actors": [],
                "due_raw": "周五",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "uncertainty",
                "content": "由于缺少 anchor_date，无法将“周五”换算为具体日期。",
                "confidence": 0.6,
            }
        ],
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.72,
                "reason": "建议复查 event_assignment、fact_index 和 due_date。",
            }
        ],
        "ambiguities": [],
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "补材料", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                response = client.post(
                    "/api/evidence",
                    json={"text": "请周五前补材料。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = response.json()["evidence_id"]
                home_response = client.get("/")
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert response.status_code == 200
    friendly_copy = "还无法确定“周五”具体是哪一天。补充这段记录发生的日期后，可以换算成具体日期。"
    assert friendly_copy in detail_response.text
    home_visible = home_response.text.split("<script>", 1)[0]
    detail_visible = detail_response.text.split("<script>", 1)[0]
    for token in ("anchor_date", "due_date", "fact_index", "event_assignment"):
        assert token not in home_visible
        assert token not in detail_visible


def test_home_page_does_not_render_legacy_reference_content_without_event(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=_semantic_result()):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "旧兼容记录。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        conn.execute(
            """
            UPDATE evidence
            SET plain_summary = ?, slot_deliverable = ?, slot_due_raw = ?, caveats = ?
            WHERE evidence_id = ?
            """,
            ("旧摘要仍可展示", "旧交付物", "周五前", json.dumps(["旧兼容提示"], ensure_ascii=False), evidence_id),
        )
        conn.execute(
            """
            DELETE FROM interpretations
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute(
            """
            DELETE FROM facts
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute(
            """
            DELETE FROM event_match_runs
            WHERE semantic_run_id IN (
                SELECT semantic_run_id FROM semantic_run_inputs WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        )
        conn.execute("DELETE FROM semantic_run_inputs WHERE evidence_id = ?", (evidence_id,))
        conn.execute(
            """
            DELETE FROM semantic_runs
            WHERE semantic_run_id NOT IN (
                SELECT semantic_run_id FROM semantic_run_inputs
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM facts WHERE semantic_run_id IS NOT NULL
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM interpretations WHERE semantic_run_id IS NOT NULL
            )
              AND semantic_run_id NOT IN (
                SELECT semantic_run_id FROM event_match_runs
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    with client:
        response = client.get("/")

    assert response.status_code == 200
    home_visible = response.text.split("<script>", 1)[0]
    assert "旧摘要仍可展示" not in home_visible
    assert "旧交付物" not in home_visible
    assert "周五前" not in home_visible
    assert "旧兼容提示" not in home_visible


def test_thread_channel_page_contains_10_evidence_cards(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_channel")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 10
    assert "⚠️ 需求在这里发生了变更" in response.text
    assert "10 条记录" in response.text
    assert 'href="/help"' in response.text
    assert "导出这件事(PDF)" in response.text
    assert "导出完整举证包" not in response.text


def test_thread_userlist_page_contains_3_evidence_cards(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_userlist")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 3


def test_nonexistent_thread_returns_404(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_nonexistent")

    assert response.status_code == 404


def test_index_does_not_expose_hash_fields(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    html = response.text
    assert "chain_hash" not in html
    assert "content_hash" not in html
    assert "prev_hash" not in html
    assert 'href="/help"' in html


def test_search_returns_channel_records(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "渠道复盘"})

    assert response.status_code == 200
    assert "找到" in response.text
    assert "渠道复盘" in response.text


def test_search_actor_name_returns_actor_related_records(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    unique_text = "这条原文里没有那个名字，但和接口文档有关。"

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": unique_text, "source": "Jira", "source_detail": "WORK-238"},
        )

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        conn.execute(
            """
            UPDATE evidence
            SET slot_requester = ?, slot_owner = ?, plain_summary = ?
            WHERE evidence_id = ?
            """,
            ("act_wang", "act_self", "继续跟进接口文档。", create_response.json()["evidence_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    with client:
        response = client.get("/search", params={"q": "王强"})

    assert response.status_code == 200
    assert unique_text in response.text


def test_search_missing_term_returns_empty_state(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "根本不存在的词条123456"})

    assert response.status_code == 200
    assert "没有找到相关记录" in response.text


def test_search_source_filter_works(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "复盘", "source": "飞书"})

    assert response.status_code == 200
    platforms = re.findall(r'data-result-platform="([^"]+)"', response.text)
    assert platforms
    assert set(platforms) == {"飞书"}


def test_search_kind_filter_works(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "复盘", "kind": "change"})

    assert response.status_code == 200
    kinds = re.findall(r'data-result-kind="([^"]+)"', response.text)
    assert kinds
    assert set(kinds) == {"change"}


def test_search_date_range_filter_works(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get(
            "/search",
            params={"q": "渠道复盘", "start": "2026-03-05", "end": "2026-03-10"},
        )

    assert response.status_code == 200
    dates = re.findall(r'data-result-date="([^"]+)"', response.text)
    assert dates
    assert all("2026-03-05" <= item <= "2026-03-10" for item in dates)


def test_search_escapes_like_wildcards(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        percent_response = client.get("/search", params={"q": "%"})
        underscore_response = client.get("/search", params={"q": "_"})

    assert percent_response.status_code == 200
    assert underscore_response.status_code == 200
    assert "找到 0 条与「%」相关的记录" in percent_response.text
    assert "找到 0 条与「_」相关的记录" in underscore_response.text


def test_search_escapes_script_query(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "<script>"})

    assert response.status_code == 200
    assert "&lt;script&gt;" in response.text


def test_search_results_are_isolated_between_sandboxes(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        client_a.post(
            "/api/evidence",
            json={"text": "只在 A 的搜索里出现的独有文本。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        response_a = client_a.get("/search", params={"q": "独有文本"})
        response_b = client_b.get("/search", params={"q": "独有文本"})

    assert "没有找到相关记录" not in response_a.text
    assert "只在 A 的搜索里出现的" in response_a.text
    assert "<mark>独有文本</mark>" in response_a.text
    assert "没有找到相关记录" in response_b.text


def test_search_with_empty_query_shows_prompt(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/search", params={"q": "   "})

    assert response.status_code == 200
    assert "输入关键词开始搜索" in response.text


def test_search_hits_extracted_document_text(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    pdf_bytes = _build_pdf_bytes("独特关键词甲乙丙")

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            upload_response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("search.pdf", pdf_bytes, "application/pdf")},
            )
            search_response = client.get("/search", params={"q": "甲乙丙"})

    assert upload_response.status_code == 200
    assert search_response.status_code == 200
    assert "独特关键词<mark>甲乙丙</mark>" in search_response.text


def test_help_page_returns_200_and_contains_review_notes(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/help")

    assert response.status_code == 200
    html = response.text
    assert "无需登录即可体验" in html
    assert "每位访客都会拿到一个独立的匿名体验空间" in html
    assert "体验数据会在 24 小时后自动清理" in html
    assert "判断明确" in html
    assert "存在多个可能" in html
    assert "上下文不足" in html
    assert "AI 整理结果可以继续纠正" in html
    assert "即将开放" not in html
    assert "自动归并算法已完成设计但尚未实现" not in html
    assert "AUTO：" not in html
    assert "CONFIRM：" not in html
    assert "NEEDS_CONTEXT：" not in html
    assert "https://github.com/lelsie651-max/workchain" not in html


def test_html_500_page_does_not_expose_traceback_or_secret(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))
    secret = "sk-live-should-stay-hidden"
    client = TestClient(create_app(), raise_server_exceptions=False)

    with patch("app.main._fetch_index_data", side_effect=RuntimeError(f"boom {secret} traceback")):
        with client:
            response = client.get("/")

    assert response.status_code == 500
    assert "Internal Server Error" in response.text
    assert "traceback" not in response.text.lower()
    assert secret not in response.text


def test_startup_auto_generates_demo_data_when_directory_missing(tmp_path, monkeypatch):
    demo_dir = tmp_path / "missing_demo_data"
    sandbox_root = tmp_path / "missing_sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    assert not demo_dir.exists()

    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert demo_dir.exists()
    assert (demo_dir / "workchain.db").exists()

    conn = sqlite3.connect(demo_dir / "workchain.db")
    try:
        count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    finally:
        conn.close()

    assert count == 18


def test_concurrent_requests_all_return_200(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(5):
                index_response = client.get("/")
                thread_response = client.get("/thread/thr_channel")
                assert index_response.status_code == 200
                assert thread_response.status_code == 200
        except Exception as exc:  # pragma: no cover - only used on failure
            with lock:
                errors.append(str(exc))

    with client:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []


def test_healthz_is_stable_across_20_requests(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        for _ in range(20):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "evidence_count": 18}


def test_post_evidence_appends_reference_record(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/evidence",
            json={"text": "张总刚在群里又说先把原始明细留一下。", "source": "飞书", "source_detail": "项目复盘群"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["seq"] == 19
    assert data["evidence_id"].startswith("ev_")
    assert data["occurred_at"] is not None
    assert data["parse_status"] == "llm_running"
    assert _sandbox_db_path(client, sandbox_root).exists()


def test_index_keeps_home_focused_after_post_without_recent_records_list(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        client.post(
            "/api/evidence",
            json={"text": "李娜刚补了一句，说先不用发销售。", "source": "企业微信", "source_detail": "私聊-李娜"},
        )
        response = client.get("/")

    assert response.status_code == 200
    assert "最近证据" not in response.text
    assert "李娜刚补了一句" not in response.text
    assert "导出我的记录(PDF)" not in response.text
    assert "还没有事项。放进第一条记录后，WorkChain 会帮你把相关事实串起来。" in response.text


def test_index_does_not_dump_full_long_raw_text_on_home(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    long_text = "这是一条很长的原文。\\n" + ("后续细节" * 30)

    with client:
        client.post(
            "/api/evidence",
            json={"text": long_text, "source": "飞书", "source_detail": "项目复盘群"},
        )
        response = client.get("/")

    assert response.status_code == 200
    assert long_text not in response.text


def test_evidence_detail_returns_full_text_for_current_sandbox(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    full_text = "第一行\\n第二行\\n第三行，完整查看。"

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": full_text, "source": "企业微信", "source_detail": "私聊-李娜"},
        )
        evidence_id = create_response.json()["evidence_id"]
        response = client.get(f"/evidence/{evidence_id}")

    assert response.status_code == 200
    assert full_text in response.text
    assert evidence_id in response.text
    assert "不会影响完整性校验" in response.text


def test_evidence_detail_returns_404_for_other_sandbox(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        create_response = client_a.post(
            "/api/evidence",
            json={"text": "只有 A 能看到这条。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        response = client_b.get(f"/evidence/{evidence_id}")

    assert response.status_code == 404


def test_export_pdf_thread_route_returns_pdf(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/export/pdf", params={"thread_id": "thr_channel"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("attachment;")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages)
    assert "渠道复盘数据" in text


def test_export_pdf_mine_route_returns_pdf_for_user_records(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        client.post(
            "/api/evidence",
            json={"text": "这是我自己补充的一条记录。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        response = client.get("/export/pdf", params={"scope": "mine"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages)
    assert "这是我自己补充的一条记录。" in text


def test_export_package_route_returns_zip_and_verify_passes(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/export/package")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    zip_path = tmp_path / "package.zip"
    extract_dir = tmp_path / "unzipped"
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 18
    assert (extract_dir / "verify.py").exists()
    assert (extract_dir / "怎么验证这份材料.txt").exists()
    assert any(path.suffix == ".pdf" for path in extract_dir.iterdir())

    result = subprocess.run(
        [sys.executable, str(extract_dir / "verify.py"), "--dir", str(extract_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_export_package_ignores_thread_id_and_still_exports_full_chain(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/export/package", params={"thread_id": "thr_channel"})

    assert response.status_code == 200
    extract_dir = tmp_path / "ignored-thread-package"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(extract_dir)
    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 18


def test_export_pdf_thread_route_returns_404_across_sandboxes(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        client_a.post(
            "/api/evidence",
            json={"text": "只有 A 里能看到这条记录。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        db_path_a = _sandbox_db_path(client_a, sandbox_root)
        conn_a = init_db(db_path_a)
        try:
            conn_a.execute(
                """
                INSERT INTO threads (
                    thread_id, title, status, owner_actor_id, requester_actor_id,
                    current_deliverable, current_due, version, risk_flags,
                    last_activity_at, first_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "thr_only_in_a",
                    "只在 A 的事项",
                    "open",
                    "act_self",
                    "act_zhang",
                    "只在 A 的交付物",
                    None,
                    1,
                    "[]",
                    1723000000,
                    1723000000,
                ),
            )
            conn_a.commit()
        finally:
            conn_a.close()
        response = client_b.get("/export/pdf", params={"thread_id": "thr_only_in_a"})

    assert response.status_code == 404


def test_upload_png_returns_image_media_type_and_verify_chain_passes(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = _upload_png(client, filename="screen.png")

    assert response.status_code == 200
    payload = response.json()
    assert payload["media_type"] == "image"
    assert payload["parse_status"] == "unsupported"

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT media_type, raw_text FROM evidence WHERE evidence_id = ?",
            (payload["evidence_id"],),
        ).fetchone()
        assert row["media_type"] == "image"
        assert row["raw_text"] == "[图片] screen.png"
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_same_image_upload_twice_reuses_blob_but_creates_two_evidence_rows(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        first = _upload_png(client, filename="same.png")
        second = _upload_png(client, filename="same-again.png")

    assert first.status_code == 200
    assert second.status_code == 200

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT evidence_id, content_hash, blob_path
            FROM evidence
            WHERE evidence_id IN (?, ?)
            ORDER BY seq ASC
            """,
            (first.json()["evidence_id"], second.json()["evidence_id"]),
        ).fetchall()
        assert len(rows) == 2
        assert len({row["evidence_id"] for row in rows}) == 2
        assert len({row["content_hash"] for row in rows}) == 1
        assert len({row["blob_path"] for row in rows}) == 1
        blob_path = db_path.parent / "blobs" / rows[0]["blob_path"]
        assert blob_path.exists()
    finally:
        conn.close()


def test_fake_png_text_file_is_rejected_by_magic_bytes(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/evidence",
            data={"source": "飞书", "source_detail": "项目复盘群"},
            files={"file": ("fake.png", b"just plain text", "image/png")},
        )

    assert response.status_code == 400


def test_file_larger_than_8mb_is_rejected(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    oversized_pdf = b"%PDF-" + (b"0" * (8 * 1024 * 1024 + 1))

    with client:
        response = client.post(
            "/api/evidence",
            data={"source": "飞书", "source_detail": "项目复盘群"},
            files={"file": ("too-large.pdf", oversized_pdf, "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "单个文件不能超过 8 MB"


def test_unsupported_file_type_is_rejected(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/evidence",
            data={"source": "飞书", "source_detail": "项目复盘群"},
            files={"file": ("demo.exe", b"MZnot-supported", "application/octet-stream")},
        )

    assert response.status_code == 400


def test_blob_route_returns_content_type_and_nosniff_header(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = _upload_png(client, filename="preview.png")
        evidence_id = create_response.json()["evidence_id"]
        blob_response = client.get(f"/blob/{evidence_id}")

    assert blob_response.status_code == 200
    assert blob_response.headers["content-type"] == "image/png"
    assert blob_response.headers["x-content-type-options"] == "nosniff"
    assert blob_response.headers["content-disposition"].startswith("inline")
    assert blob_response.content == PNG_BYTES


def test_blob_route_returns_404_across_sandboxes(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        create_response = _upload_png(client_a, filename="only-a.png")
        evidence_id = create_response.json()["evidence_id"]
        blob_response = client_b.get(f"/blob/{evidence_id}")

    assert blob_response.status_code == 404


def test_image_upload_without_ocr_config_sets_unsupported_and_does_not_call_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics") as mock_extract:
        with client:
            create_response = _upload_png(client, filename="no-llm.png")
            evidence_id = create_response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert "图片识别未配置" in status_response.json()["detail"]
    mock_extract.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT media_type FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert row["media_type"] == "image"
    finally:
        conn.close()


def test_evidence_detail_prefers_extract_note_over_generic_unsupported_copy(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = _upload_png(client, filename="detail-note.png")
        evidence_id = create_response.json()["evidence_id"]
        detail_response = client.get(f"/evidence/{evidence_id}")

    assert detail_response.status_code == 200
    assert "图片识别未配置(DASHSCOPE_API_KEY 未设置)" in detail_response.text
    assert "这是一张图片/文档,系统暂不能自动读懂它的内容,但原件已完整保存,任何改动都会被发现。" not in detail_response.text


def test_evidence_detail_uses_generic_copy_when_no_extract_note_exists(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "只是留档。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        conn.execute(
            """
            INSERT INTO meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (f"parse_status:{evidence_id}", "unsupported"),
        )
        conn.execute(
            """
            INSERT INTO meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (f"parse_detail:{evidence_id}", ""),
        )
        conn.execute("DELETE FROM meta WHERE key = ?", (f"extract_note:{evidence_id}",))
        conn.commit()
    finally:
        conn.close()

    with client:
        detail_response = client.get(f"/evidence/{evidence_id}")

    assert detail_response.status_code == 200
    assert "这是一张图片/文档,系统暂不能自动读懂它的内容,但原件已完整保存,任何改动都会被发现。" in detail_response.text


def test_image_html_shows_parse_summary_before_collapsed_ocr_text(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "有人要求在周五前交付渠道复盘数据。",
                "actors": [{"name": "张总", "role": "requester"}],
                "due_raw": "周五前",
                "due_date": "2026-08-08",
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.91,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "explanation",
                "content": "这里的复盘指渠道复盘数据。",
                "confidence": 0.8,
            }
        ],
        ambiguities=["金额口径还需要人工核对。"],
    )

    with patch("app.extract.ocr.image_to_text", return_value=("审批通过,周五前交付渠道复盘数据", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="ordered.png")
                evidence_id = create_response.json()["evidence_id"]
                index_response = client.get("/")
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert index_response.status_code == 200
    assert detail_response.status_code == 200
    detail_details = re.search(r"<details[^>]*data-ocr-details[^>]*>", detail_response.text)
    assert detail_details is not None
    assert "open" not in detail_details.group(0)
    assert detail_response.text.index("事实整理") < detail_response.text.index("看看系统读到了什么")
    assert detail_response.text.index("有人要求在周五前交付渠道复盘数据。") < detail_response.text.index("看看系统读到了什么")
    assert "AI帮你理解" in detail_response.text
    assert "金额口径还需要人工核对。" in detail_response.text


def test_image_upload_without_ocr_does_not_consume_daily_parse_quota(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = _upload_png(client, filename="quota.png")

    assert response.status_code == 200

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        ocr_counter = conn.execute(
            "SELECT value FROM meta WHERE key LIKE 'ocr_count:%'"
        ).fetchone()
        parse_counter = conn.execute(
            "SELECT value FROM meta WHERE key LIKE 'parse_count:%'"
        ).fetchone()
        assert ocr_counter is None
        assert parse_counter is None
    finally:
        conn.close()


def test_image_upload_without_text_preserves_attachment_raw_text(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = _upload_png(client, filename="annotated.png", text="")

    assert response.status_code == 200
    evidence_id = response.json()["evidence_id"]
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT raw_text, plain_summary, media_type FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert row["media_type"] == "image"
        assert row["raw_text"] == "[图片] annotated.png"
        assert row["plain_summary"] is None
    finally:
        conn.close()


def test_pdf_upload_extracts_text_into_raw_text_and_preserves_original_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    pdf_bytes = _build_pdf_bytes("渠道复盘数据")

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("demo.pdf", pdf_bytes, "application/pdf")},
            )
            evidence_id = response.json()["evidence_id"]

    assert response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT raw_text, content_hash, media_type FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert row["media_type"] == "file"
        assert "渠道复盘数据" in row["raw_text"]
        assert row["content_hash"] == compute_content_hash(pdf_bytes)
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_docx_upload_extracts_paragraph_text(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    docx_bytes = _build_docx_bytes("第一段内容", "第二段内容")

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("demo.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
            evidence_id = response.json()["evidence_id"]

    assert response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        assert "第一段内容" in row["raw_text"]
        assert "第二段内容" in row["raw_text"]
    finally:
        conn.close()


def test_txt_upload_extracts_utf8_and_gbk_text(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            utf8_response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("utf8.txt", "渠道复盘数据".encode("utf-8"), "text/plain")},
            )
            gbk_response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("gbk.txt", "用户明细".encode("gbk"), "text/plain")},
            )

    assert utf8_response.status_code == 200
    assert gbk_response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT raw_text FROM evidence WHERE evidence_id IN (?, ?)",
            (utf8_response.json()["evidence_id"], gbk_response.json()["evidence_id"]),
        ).fetchall()
        joined = "\n".join(row["raw_text"] for row in rows)
        assert "渠道复盘数据" in joined
        assert "用户明细" in joined
    finally:
        conn.close()


def test_image_only_pdf_upload_marks_unsupported_and_stores_extract_note(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    image_pdf = _build_pdf_bytes(image_only=True)

    with patch("app.main.semantic_llm.extract_semantics") as mock_extract:
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("scan.pdf", image_pdf, "application/pdf")},
            )
            evidence_id = response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert "扫描件" in status_response.json()["detail"]
    mock_extract.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        note = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (f"extract_note:{evidence_id}",),
        ).fetchone()
        raw_row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        assert note["value"] == "这份 PDF 看起来是扫描件,没有可提取的文字"
        assert raw_row["raw_text"] == "[文件] scan.pdf"
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_broken_pdf_upload_still_succeeds_and_marks_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    broken_pdf = b"%PDF-broken"

    with patch("app.main.semantic_llm.extract_semantics") as mock_extract:
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("broken.pdf", broken_pdf, "application/pdf")},
            )
            evidence_id = response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert "PDF 暂时无法读取" in status_response.json()["detail"]
    mock_extract.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        assert row["raw_text"] == "[文件] broken.pdf"
    finally:
        conn.close()


def test_multipart_file_only_submission_returns_200(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = _multipart_request(
            client,
            data={"source": "飞书", "source_detail": "项目复盘群"},
            file_part=("x.png", PNG_BYTES, "image/png"),
        )

    assert response.status_code == 200
    assert response.json()["media_type"] == "image"


def test_multipart_text_and_file_submission_returns_400(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = _multipart_request(
            client,
            data={"text": "说明", "source": "飞书", "source_detail": "项目复盘群"},
            file_part=("x.png", PNG_BYTES, "image/png"),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "文字记录和文件不能同时提交，请二选一。"


def test_image_upload_with_mocked_ocr_text_enters_parse_pipeline_and_is_searchable(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": "张总",
        "owner_name": "我",
        "deliverable": "渠道复盘数据",
        "due_raw": "周五前",
        "due_date": "2026-08-08",
        "direction": "i_owe",
        "kind": "request",
        "plain_summary": "图片里写着周五前交付渠道复盘数据。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("审批通过,周五前交付渠道复盘数据", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed) as mock_extract:
            with client:
                response = _upload_png(client, filename="ocr-success.png")
                evidence_id = response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")
                search_response = client.get("/search", params={"q": "审批通过"})

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "done"
    mock_extract.assert_called_once()
    assert search_response.status_code == 200
    assert "审批通过" in search_response.text

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT raw_text FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert row["raw_text"].startswith("[图片] ocr-success.png\n\n")
        assert "审批通过,周五前交付渠道复盘数据" in row["raw_text"]
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_image_upload_defaults_to_ocr_when_provider_not_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    monkeypatch.delenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_raw": "周五前",
        "due_date": "2026-08-08",
        "direction": "none",
        "kind": "reference",
        "plain_summary": "默认走 OCR。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("默认 OCR 文字", "")) as mock_ocr:
        with patch("app.vision_provider.extract_visual_evidence") as mock_vision:
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                with client:
                    response = _upload_png(client, filename="default-provider.png")
                    evidence_id = response.json()["evidence_id"]
                    status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.json()["parse_status"] == "done"
    mock_ocr.assert_called_once()
    mock_vision.assert_not_called()


def test_ark_provider_success_uses_ark_only_and_persists_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "reference",
                "content": "画面文字提到渠道复盘数据。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.72,
            }
        ],
        ambiguities=["画面里只看到点赞反应,无法确认具体身份。"],
    )
    ark_extraction = {
        "transcript": "Ark transcript",
        "observations": [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": ["局部遮挡"],
    }

    with patch("app.vision_provider.extract_visual_evidence", return_value=ark_extraction) as mock_vision:
        with patch("app.extract.ocr.image_to_text") as mock_ocr:
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed) as mock_llm:
                with client:
                    response = _upload_png(client, filename="ark-success.png")
                    evidence_id = response.json()["evidence_id"]
                    status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.json()["parse_status"] == "done"
    mock_vision.assert_called_once()
    mock_ocr.assert_not_called()
    mock_llm.assert_called_once()
    assert mock_llm.call_args.args[0] == main_module._llm_input_text("Ark transcript")
    assert mock_llm.call_args.kwargs["observations"] == [
        {"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}
    ]
    assert mock_llm.call_args.kwargs["anchor_date"] is None

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            """
            SELECT raw_text FROM evidence WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        extraction_row = conn.execute(
            """
            SELECT provider, model, transcript, observations, warnings
            FROM evidence_extractions
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        assert row["raw_text"] == "[图片] ark-success.png\n\nArk transcript"
        assert extraction_row["provider"] == "doubao-ark"
        assert extraction_row["model"] == "doubao-seed-2-0-lite-260215"
        assert extraction_row["transcript"] == "Ark transcript"
        assert json.loads(extraction_row["observations"]) == [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}]
        assert json.loads(extraction_row["warnings"]) == ["局部遮挡"]
        semantic_run = conn.execute(
            """
            SELECT sr.provider, sr.model, sr.parser_version, sri.extraction_id
            FROM semantic_runs sr
            JOIN semantic_run_inputs sri ON sri.semantic_run_id = sr.semantic_run_id
            WHERE sri.evidence_id = ?
            ORDER BY sr.created_at DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        assert semantic_run["provider"] == "deepseek"
        assert semantic_run["model"] == "deepseek-v4-flash"
        assert semantic_run["parser_version"] == "2.3"
        assert semantic_run["extraction_id"] is not None
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_ark_provider_falls_back_to_ocr_and_consumes_budget_only_on_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_raw": "周五前",
        "due_date": "2026-08-08",
        "direction": "none",
        "kind": "reference",
        "plain_summary": "走 OCR fallback。",
        "caveats": [],
    }

    with patch("app.vision_provider.extract_visual_evidence", return_value=None) as mock_vision:
        with patch("app.extract.ocr.image_to_text", return_value=("Fallback OCR transcript", "")) as mock_ocr:
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                with client:
                    response = _upload_png(client, filename="ark-fallback.png")
                    evidence_id = response.json()["evidence_id"]
                    status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.json()["parse_status"] == "done"
    mock_vision.assert_called_once()
    mock_ocr.assert_called_once()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        extraction_row = conn.execute(
            """
            SELECT provider, model, transcript, warnings
            FROM evidence_extractions
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        ocr_counter = conn.execute(
            "SELECT value FROM meta WHERE key LIKE 'ocr_count:%'"
        ).fetchone()
        assert extraction_row["provider"] == "dashscope"
        assert extraction_row["model"] == "vanchin/deepseek-ocr"
        assert extraction_row["transcript"] == "Fallback OCR transcript"
        assert json.loads(extraction_row["warnings"]) == ["ark_vision_failed_fallback_to_ocr"]
        assert ocr_counter is not None
        assert ocr_counter["value"] == "1"
    finally:
        conn.close()


def test_ark_provider_failure_without_ocr_fallback_still_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.vision_provider.extract_visual_evidence", return_value=None):
        with patch("app.main.semantic_llm.extract_semantics") as mock_llm:
            with client:
                response = _upload_png(client, filename="ark-no-fallback.png")
                evidence_id = response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.json()["parse_status"] == "unsupported"
    assert "Ark Vision 提取失败" in status_response.json()["detail"]
    mock_llm.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        extraction_count = conn.execute(
            "SELECT COUNT(*) AS count FROM evidence_extractions WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()["count"]
        ocr_counter = conn.execute("SELECT value FROM meta WHERE key LIKE 'ocr_count:%'").fetchone()
        assert row["raw_text"] == "[图片] ark-no-fallback.png"
        assert extraction_count == 0
        assert ocr_counter is None
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_ark_provider_observations_only_enters_semantic_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    ark_extraction = {
        "transcript": None,
        "observations": [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": [],
    }

    with patch("app.vision_provider.extract_visual_evidence", return_value=ark_extraction):
        with patch(
            "app.main.semantic_llm.extract_semantics",
            return_value=_semantic_result(
                facts=[
                    {
                        "fact_type": "reference",
                        "content": "画面显示该消息存在点赞反应。",
                        "actors": [],
                        "due_raw": None,
                        "due_date": None,
                        "due_anchor_date": None,
                        "occurred_date": None,
                        "confidence": 0.7,
                    }
                ]
            ),
        ) as mock_llm:
            with client:
                response = _upload_png(client, filename="ark-observation-only.png")
                evidence_id = response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert status_response.json()["parse_status"] == "done"
    mock_llm.assert_called_once()
    assert mock_llm.call_args.args[0] is None
    assert mock_llm.call_args.kwargs["observations"] == [
        {"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}
    ]
    assert mock_llm.call_args.kwargs["anchor_date"] is None

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        extraction_row = conn.execute(
            "SELECT transcript, observations FROM evidence_extractions WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert row["raw_text"] == "[图片] ark-observation-only.png"
        assert extraction_row["transcript"] is None
        assert json.loads(extraction_row["observations"]) == [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}]
        fact_row = conn.execute(
            "SELECT content FROM facts WHERE semantic_run_id IS NOT NULL ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert fact_row["content"] == "画面显示该消息存在点赞反应。"
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_image_upload_ocr_timeout_still_succeeds_and_marks_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = Mock(completions=Mock(create=Mock(side_effect=httpx.TimeoutException("boom"))))

    with patch("app.ocr.OpenAI", FakeOpenAI):
        with patch("app.main.semantic_llm.extract_semantics") as mock_extract:
            with client:
                response = _upload_png(client, filename="ocr-timeout.png")
                evidence_id = response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert status_response.json()["detail"] == "图片识别超时"
    mock_extract.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        assert row["raw_text"] == "[图片] ocr-timeout.png"
    finally:
        conn.close()


def test_image_upload_with_short_ocr_text_marks_unsupported_and_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.extract.ocr.image_to_text", return_value=(None, "这张图里没有识别到文字,原件已完整保存")):
        with patch("app.main.semantic_llm.extract_semantics") as mock_extract:
            with client:
                response = _upload_png(client, filename="ocr-short.png")
                evidence_id = response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert status_response.json()["detail"] == "这张图里没有识别到文字,原件已完整保存"
    mock_extract.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute("SELECT raw_text FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        assert row["raw_text"] == "[图片] ocr-short.png"
    finally:
        conn.close()


def test_large_image_is_resized_for_ocr_but_original_blob_bytes_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    large_png = _build_png_bytes(4000, 3000)
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = Mock(completions=Mock(create=self._create))

        def _create(self, **kwargs):
            data_url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
            prepared_bytes = base64.b64decode(data_url.split(",", 1)[1])
            with Image.open(BytesIO(prepared_bytes)) as prepared_image:
                captured["prepared_size"] = prepared_image.size
                captured["prepared_format"] = prepared_image.format
            return Mock(choices=[Mock(message=Mock(content="渠道复盘数据"))])

    with patch("app.ocr.OpenAI", FakeOpenAI):
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("huge.png", large_png, "image/png")},
            )

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert captured["prepared_size"] == (2000, 1500)
    assert captured["prepared_format"] == "JPEG"

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT blob_path FROM evidence WHERE evidence_id = ?",
            (response.json()["evidence_id"],),
        ).fetchone()
        blob_path = db_path.parent / "blobs" / row["blob_path"]
        blob_bytes = blob_path.read_bytes()
        assert blob_bytes == large_png
        with Image.open(BytesIO(blob_bytes)) as original_image:
            assert original_image.size == (4000, 3000)
    finally:
        conn.close()


def test_image_upload_ocr_limit_marks_21st_record_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with patch("app.extract.ocr.image_to_text", return_value=("渠道复盘数据", "")):
        with client:
            for idx in range(20):
                response = _upload_png(client, filename=f"quota-{idx}.png")
                assert response.status_code == 200
                assert response.json()["parse_status"] == "ocr_running"

            last_response = _upload_png(client, filename="quota-21.png")
            evidence_id = last_response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert last_response.status_code == 200
    assert last_response.json()["parse_status"] == "unsupported"
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "unsupported"
    assert status_response.json()["detail"] == "今日图片识别次数已用完,原件已完整保存"


def test_image_upload_status_sequence_includes_ocr_running_then_llm_running(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "有人要求交付渠道复盘数据。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.83,
            }
        ]
    )
    seen_statuses: list[str] = []
    original_set_parse_status = main_module._set_parse_status

    def wrapped_set_parse_status(conn, evidence_id, status):
        seen_statuses.append(status)
        return original_set_parse_status(conn, evidence_id, status)

    with patch("app.main._set_parse_status", side_effect=wrapped_set_parse_status):
        with patch("app.extract.ocr.image_to_text", return_value=("审批通过,周五前交付渠道复盘数据", "")):
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                with client:
                    response = _upload_png(client, filename="sequence.png")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "ocr_running"
    assert "ocr_running" in seen_statuses
    assert "llm_running" in seen_statuses
    assert seen_statuses.index("ocr_running") < seen_statuses.index("llm_running")


def test_text_upload_status_sequence_does_not_pass_through_ocr_running(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result()
    seen_statuses: list[str] = []
    original_set_parse_status = main_module._set_parse_status

    def wrapped_set_parse_status(conn, evidence_id, status):
        seen_statuses.append(status)
        return original_set_parse_status(conn, evidence_id, status)

    with patch("app.main._set_parse_status", side_effect=wrapped_set_parse_status):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                response = client.post(
                    "/api/evidence",
                    json={"text": "先留个底。", "source": "飞书", "source_detail": "项目复盘群"},
                )

    assert response.status_code == 200
    assert response.json()["parse_status"] == "llm_running"
    assert "llm_running" in seen_statuses
    assert "ocr_running" not in seen_statuses


def test_multipart_text_only_submission_returns_200(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = _multipart_request(
            client,
            data={"text": "只有文字的 multipart 提交", "source": "飞书", "source_detail": "项目复盘群"},
        )

    assert response.status_code == 200
    assert response.json()["media_type"] == "text"
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT raw_text FROM evidence WHERE evidence_id = ?",
            (response.json()["evidence_id"],),
        ).fetchone()
        assert row["raw_text"] == "只有文字的 multipart 提交"
    finally:
        conn.close()


def test_patch_slots_updates_fields_and_recomputes_slots_filled(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "先存一条待修正记录。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        patch_response = client.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={
                "slot_deliverable": "渠道复盘数据",
                "slot_due_raw": "下周五",
                "slot_due_date": "2026-08-08",
                "slot_direction": "i_owe",
                "kind": "request",
                "plain_summary": "张总要你补渠道复盘。",
                "caveats": ["日期可能有变动"],
            },
        )

    assert patch_response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            """
            SELECT slot_deliverable, slot_due_raw, slot_due, slot_direction,
                   kind, plain_summary, caveats, slots_filled
            FROM evidence WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        assert row["slot_deliverable"] == "渠道复盘数据"
        assert row["slot_due_raw"] == "下周五"
        assert row["slot_due"] is not None
        assert row["slot_direction"] == "i_owe"
        assert row["kind"] == "request"
        assert row["plain_summary"] == "张总要你补渠道复盘。"
        assert row["slots_filled"] == 2
    finally:
        conn.close()


def test_patch_slots_keeps_verify_chain_valid(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "这条记录稍后会被手动修正。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        patch_response = client.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={
                "slot_deliverable": "复盘",
                "slot_due_raw": "周五",
                "slot_due_date": "2026-08-08",
                "slot_direction": "i_owe",
                "kind": "request",
                "plain_summary": "这是一条人工确认后的说明。",
                "caveats": ["先保留一个提醒"],
            },
        )

    assert patch_response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_patch_rejects_protected_fields(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "尝试改保护字段。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        response = client.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={"raw_text": "改原文", "content_hash": "x", "chain_hash": "y", "seq": 999},
        )

    assert response.status_code == 400


def test_patch_returns_404_for_other_sandbox(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        create_response = client_a.post(
            "/api/evidence",
            json={"text": "只有 A 可改。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        response = client_b.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={"slot_deliverable": "不该成功"},
        )

    assert response.status_code == 404


def test_patch_demo_record_returns_403(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.patch(
            "/api/evidence/ev_demo_01/slots",
            json={"slot_deliverable": "不该成功"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "演示记录不可修改"


def test_patch_sets_verified_flag_and_pages_show_badge(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "这条会被我人工确认。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        patch_response = client.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={
                "slot_deliverable": "复盘",
                "slot_due_raw": "周五",
                "slot_due_date": "2026-08-08",
                "slot_direction": "i_owe",
                "kind": "request",
                "plain_summary": "我已经核对过这条。",
                "caveats": [],
            },
        )
        detail_response = client.get(f"/evidence/{evidence_id}")
        index_response = client.get("/")

    assert patch_response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (f"verified:{evidence_id}",),
        ).fetchone()
        assert row["value"] == "1"
    finally:
        conn.close()

    assert "已确认" in detail_response.text
    assert "已确认" not in index_response.text


def test_patch_ocr_text_updates_raw_text_without_changing_hashes_and_keeps_verify_chain_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": "张总",
        "owner_name": "我",
        "deliverable": "渠道复盘数据",
        "due_raw": "周五前",
        "due_date": "2026-08-08",
        "direction": "i_owe",
        "kind": "request",
        "plain_summary": "图片里写着周五前交付渠道复盘数据。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="correctable.png")
                evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        before = conn.execute(
            "SELECT raw_text, content_hash, chain_hash FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
    finally:
        conn.close()

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed) as mock_extract:
        with client:
            patch_response = client.patch(
                f"/api/evidence/{evidence_id}/ocr_text",
                json={"text": "人工修正后的识别文字"},
            )
            status_response = client.get(f"/api/evidence/{evidence_id}/status")
            detail_response = client.get(f"/evidence/{evidence_id}")

    assert patch_response.status_code == 200
    assert patch_response.json()["parse_status"] == "llm_running"
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "done"
    mock_extract.assert_called_once()

    conn = init_db(db_path)
    try:
        after = conn.execute(
            "SELECT raw_text, content_hash, chain_hash FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        assert after["raw_text"] == "[图片] correctable.png\n\n人工修正后的识别文字"
        assert after["content_hash"] == before["content_hash"]
        assert after["chain_hash"] == before["chain_hash"]
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()

    assert "已人工校正" in detail_response.text


def test_patch_ocr_text_triggers_reparse_without_consuming_ocr_quota(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": "张总",
        "owner_name": "我",
        "deliverable": "渠道复盘数据",
        "due_raw": "周五前",
        "due_date": "2026-08-08",
        "direction": "i_owe",
        "kind": "request",
        "plain_summary": "图片里写着周五前交付渠道复盘数据。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="quota-stable.png")
                evidence_id = create_response.json()["evidence_id"]

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed) as mock_extract:
        with client:
            response = client.patch(
                f"/api/evidence/{evidence_id}/ocr_text",
                json={"text": "人工修正后的识别文字"},
            )

    assert response.status_code == 200
    mock_extract.assert_called_once()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        ocr_counter = conn.execute(
            "SELECT value FROM meta WHERE key LIKE 'ocr_count:%'"
        ).fetchone()
        assert ocr_counter is not None
        assert ocr_counter["value"] == "1"
    finally:
        conn.close()


def test_patch_ocr_text_returns_404_for_other_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))
    client_a = TestClient(create_app())
    client_b = TestClient(create_app())
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_raw": None,
        "due_date": None,
        "direction": "none",
        "kind": "reference",
        "plain_summary": "只是留档。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client_a, client_b:
                create_response = _upload_png(client_a, filename="cross-sandbox.png")
                evidence_id = create_response.json()["evidence_id"]
                response = client_b.patch(
                    f"/api/evidence/{evidence_id}/ocr_text",
                    json={"text": "不该成功"},
                )

    assert response.status_code == 404


def test_patch_ocr_text_returns_403_for_demo_record(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_raw": None,
        "due_date": None,
        "direction": "none",
        "kind": "reference",
        "plain_summary": "只是留档。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="demo-like.png")
                evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE evidence_extractions SET evidence_id = ? WHERE evidence_id = ?",
            ("ev_demo_ocr_text", evidence_id),
        )
        conn.execute("UPDATE evidence SET evidence_id = ? WHERE evidence_id = ?", ("ev_demo_ocr_text", evidence_id))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    finally:
        conn.close()

    with client:
        response = client.patch(
            "/api/evidence/ev_demo_ocr_text/ocr_text",
            json={"text": "不该成功"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "演示记录不可修改"


def test_image_ocr_success_persists_machine_extraction_history(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_date": "2026-08-08",
        "due_raw": "周五前",
        "direction": "none",
        "plain_summary": "审批通过,周五前交付渠道复盘数据",
        "caveats": [],
    }

    with client:
        with patch("app.extract.ocr.image_to_text", return_value=("审批通过,周五前交付渠道复盘数据", "")):
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                response = _upload_png(client, filename="tracked-ocr.png")

    assert response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        evidence_id = response.json()["evidence_id"]
        row = conn.execute(
            """
            SELECT origin, provider, model, transcript, observations, supersedes_extraction_id
            FROM evidence_extractions
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()

        assert row["origin"] == "machine"
        assert row["provider"] == "dashscope"
        assert row["model"] == "vanchin/deepseek-ocr"
        assert row["transcript"] == "审批通过,周五前交付渠道复盘数据"
        assert json.loads(row["observations"]) == []
        assert row["supersedes_extraction_id"] is None
    finally:
        conn.close()


def test_pdf_upload_persists_machine_extraction_history(tmp_path, monkeypatch):
    _disable_external_ai(monkeypatch)
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        with patch("app.main.semantic_llm.extract_semantics", return_value=None):
            response = client.post(
                "/api/evidence",
                data={"text": "", "source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("tracked.pdf", _build_pdf_bytes("渠道复盘数据"), "application/pdf")},
            )

    assert response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        evidence_id = response.json()["evidence_id"]
        row = conn.execute(
            """
            SELECT origin, provider, model, transcript, observations
            FROM evidence_extractions
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()

        assert row["origin"] == "machine"
        assert row["provider"] == "builtin"
        assert row["model"] is None
        assert "渠道复盘数据" in row["transcript"]
        assert json.loads(row["observations"]) == []
    finally:
        conn.close()


def test_text_evidence_creates_builtin_extraction_and_semantic_input_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=_semantic_result()):
        with client:
            response = client.post(
                "/api/evidence",
                json={"text": "直接粘贴的一段原始文字。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = response.json()["evidence_id"]

    assert response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        extraction_row = conn.execute(
            """
            SELECT extraction_id, origin, provider, model, transcript, observations
            FROM evidence_extractions
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        input_row = conn.execute(
            """
            SELECT extraction_id
            FROM semantic_run_inputs
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        assert extraction_row["origin"] == "machine"
        assert extraction_row["provider"] == "builtin"
        assert extraction_row["model"] is None
        assert extraction_row["transcript"] == "直接粘贴的一段原始文字。"
        assert json.loads(extraction_row["observations"]) == []
        assert input_row["extraction_id"] == extraction_row["extraction_id"]
    finally:
        conn.close()


def test_patch_ocr_text_persists_user_extraction_superseding_machine_history(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_date": "2026-08-08",
        "due_raw": "周五前",
        "direction": "none",
        "plain_summary": "审批通过,周五前交付渠道复盘数据",
        "caveats": [],
    }

    with client:
        with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                create_response = _upload_png(client, filename="tracked-correction.png")

        evidence_id = create_response.json()["evidence_id"]
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            patch_response = client.patch(
                f"/api/evidence/{evidence_id}/ocr_text",
                json={"text": "人工修正后的识别文字"},
            )

    assert patch_response.status_code == 200
    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT extraction_id, origin, provider, transcript, supersedes_extraction_id
            FROM evidence_extractions
            WHERE evidence_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (evidence_id,),
        ).fetchall()

        assert len(rows) == 2
        assert rows[0]["origin"] == "machine"
        assert rows[0]["provider"] == "dashscope"
        assert rows[0]["transcript"] == "原始识别文字"
        assert rows[1]["origin"] == "user"
        assert rows[1]["provider"] == "manual"
        assert rows[1]["transcript"] == "人工修正后的识别文字"
        assert rows[1]["supersedes_extraction_id"] == rows[0]["extraction_id"]
        runs = conn.execute(
            """
            SELECT semantic_run_id, status, supersedes_run_id
            FROM semantic_runs
            ORDER BY created_at ASC, semantic_run_id ASC
            """
        ).fetchall()
        assert len(runs) == 2
        assert runs[0]["status"] == "succeeded"
        assert runs[1]["status"] == "succeeded"
        assert runs[1]["supersedes_run_id"] == runs[0]["semantic_run_id"]
        match_count = conn.execute("SELECT COUNT(*) AS count FROM event_match_runs").fetchone()
        assert match_count["count"] == 0
    finally:
        conn.close()


def test_failed_reparse_keeps_previous_succeeded_semantic_run_and_detail_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    first_result = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "有人要求在周五前交付渠道复盘数据。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.9,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "uncertainty",
                "content": "金额口径仍需确认。",
                "confidence": 0.5,
            }
        ],
    )

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=first_result):
            with client:
                create_response = _upload_png(client, filename="reparse-fail.png")
                evidence_id = create_response.json()["evidence_id"]

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            patch_response = client.patch(
                f"/api/evidence/{evidence_id}/ocr_text",
                json={"text": "人工修正后的识别文字"},
            )
            status_response = client.get(f"/api/evidence/{evidence_id}/status")
            detail_response = client.get(f"/evidence/{evidence_id}")

    assert patch_response.status_code == 200
    assert status_response.json()["parse_status"] == "failed"
    assert "有人要求在周五前交付渠道复盘数据。" in detail_response.text
    assert "金额口径仍需确认。" in detail_response.text

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        runs = conn.execute(
            """
            SELECT semantic_run_id, status, supersedes_run_id
            FROM semantic_runs
            ORDER BY created_at ASC, semantic_run_id ASC
            """
        ).fetchall()
        facts = conn.execute("SELECT COUNT(*) AS count FROM facts").fetchone()
        assert len(runs) == 2
        assert runs[0]["status"] == "succeeded"
        assert runs[1]["status"] == "failed"
        assert runs[1]["supersedes_run_id"] == runs[0]["semantic_run_id"]
        assert facts["count"] == 1
    finally:
        conn.close()


def test_detail_page_keeps_legacy_summary_when_no_semantic_run_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "先存一条旧兼容记录。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        patch_response = client.patch(
            f"/api/evidence/{evidence_id}/slots",
            json={
                "slot_deliverable": "渠道复盘数据",
                "slot_due_raw": "下周五",
                "slot_due_date": "2026-08-08",
                "slot_direction": "i_owe",
                "kind": "request",
                "plain_summary": "这是旧兼容摘要。",
                "caveats": ["先按旧口径展示"],
            },
        )
        detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert patch_response.status_code == 200
    assert "这是旧兼容摘要。" in detail_response.text
    assert "渠道复盘数据" in detail_response.text
    assert "先按旧口径展示" in detail_response.text
    assert "事实整理" not in detail_response.text


def test_evidence_detail_shows_real_extraction_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_date": "2026-08-08",
        "due_raw": "周五前",
        "direction": "none",
        "plain_summary": "审批通过,周五前交付渠道复盘数据",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="provider-visible.png")
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert detail_response.status_code == 200
    assert "机器提取" in detail_response.text
    assert "DashScope" not in detail_response.text
    assert "vanchin/deepseek-ocr" not in detail_response.text


def test_evidence_diagnostics_off_does_not_expose_ui_or_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKCHAIN_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result()

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="diag-off.png")
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")
                llm_diag_response = client.get("/api/diag/llm")
                diag_response = client.get(f"/api/evidence/{evidence_id}/diagnostics")
                ocr_diag_response = client.get("/api/diag/ocr")
                ark_response = client.post(f"/api/evidence/{evidence_id}/diagnostics/ark-vision")
                deepseek_response = client.post(f"/api/evidence/{evidence_id}/diagnostics/deepseek-preflight")

    assert detail_response.status_code == 200
    assert "解析诊断" not in detail_response.text
    assert "Doubao Ark" not in detail_response.text
    assert "DeepSeek" not in detail_response.text
    assert "Parser v" not in detail_response.text
    assert "Run ID" not in detail_response.text
    assert "HTTP status" not in detail_response.text
    assert "latency" not in detail_response.text
    assert "timeout" not in detail_response.text
    assert "preflight" not in detail_response.text
    assert "用 Ark Vision 实验解析" not in detail_response.text
    assert "测试 DeepSeek 连接" not in detail_response.text
    assert llm_diag_response.status_code == 404
    assert diag_response.status_code == 404
    assert ocr_diag_response.status_code == 404
    assert ark_response.status_code == 404
    assert deepseek_response.status_code == 404


def test_evidence_diagnostics_on_returns_safe_actual_pipeline_info(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result()
    ark_extraction = {
        "transcript": "Ark 识别到的文字",
        "observations": [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": 0.74}],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": ["画面局部遮挡"],
    }
    semantic_diagnostic = {
        "success": True,
        "stage": "success",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-success",
        "latency_ms": 11,
        "timeout_seconds": 60.0,
        "thinking_mode": "disabled",
        "model": "deepseek-v4-flash",
    }

    with patch("app.vision_provider.extract_visual_evidence", return_value=ark_extraction) as mock_vision:
        with patch("app.extract.ocr.image_to_text") as mock_ocr:
            with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
                with patch("app.main.semantic_llm.pop_last_extract_diagnostic", return_value=semantic_diagnostic):
                    with client:
                        create_response = _upload_png(client, filename="diag-on.png")
                        evidence_id = create_response.json()["evidence_id"]
                        detail_response = client.get(f"/evidence/{evidence_id}")
                        diag_response = client.get(f"/api/evidence/{evidence_id}/diagnostics")

    assert detail_response.status_code == 200
    assert "解析诊断" in detail_response.text
    assert "Semantic Parser" in detail_response.text
    assert "测试 DeepSeek 连接" in detail_response.text
    assert "用 Ark Vision 实验解析" in detail_response.text
    assert diag_response.status_code == 200
    payload = diag_response.json()
    assert payload["parse_status"] == "done"
    assert payload["image_pipeline"] == {
        "configured_provider": "ark_vision",
        "configured_provider_label": "Doubao Ark Vision",
        "configured_model": "doubao-seed-2-0-lite-260215",
        "actual_provider": "doubao-ark",
        "actual_provider_label": "Doubao Ark",
        "actual_model": "doubao-seed-2-0-lite-260215",
        "fallback_used": False,
        "route": "app.main._run_image_pipeline -> app.evidence_extractor.run_production_image_extraction",
    }
    assert payload["text_llm"]["provider"] == "deepseek"
    assert payload["text_llm"]["model"] == main_module.get_text_model()
    assert payload["text_llm"]["parser_version"] == "2.3"
    assert payload["semantic_parser"]["run_status"] == "succeeded"
    assert payload["semantic_parser"]["failure_type"] is None
    assert payload["semantic_parser"]["diagnostic"]["success"] is True
    assert payload["semantic_parser"]["diagnostic"]["stage"] == "success"
    assert payload["semantic_parser"]["diagnostic"]["timeout_seconds"] == 60.0
    assert payload["semantic_parser"]["diagnostic"]["thinking_mode"] == "disabled"
    assert payload["extraction_history"] == [
        {
            "origin": "machine",
            "origin_label": "机器提取",
            "provider": "doubao-ark",
            "provider_label": "Doubao Ark",
            "model": "doubao-seed-2-0-lite-260215",
            "warnings": ["画面局部遮挡"],
            "created_at": payload["extraction_history"][0]["created_at"],
            "created_at_text": payload["extraction_history"][0]["created_at_text"],
        }
    ]
    assert isinstance(payload["extraction_history"][0]["created_at"], int)
    assert payload["extraction_history"][0]["created_at_text"]
    assert "Ark 识别到的文字" not in diag_response.text
    mock_vision.assert_called_once()
    mock_ocr.assert_not_called()


@pytest.mark.parametrize(
    ("diagnostic", "expected_failure_type"),
    [
        (
            {
                "success": False,
                "stage": "http",
                "status_code": 402,
                "error_code": "http_402",
                "error_type": "http_error",
                "safe_message": "DeepSeek API returned HTTP 402",
                "request_id": "req-402",
                "latency_ms": 23,
                "timeout_seconds": 60.0,
                "thinking_mode": "disabled",
                "model": "deepseek-v4-flash",
            },
            "provider_http_402",
        ),
        (
            {
                "success": False,
                "stage": "empty_content",
                "status_code": 200,
                "error_code": "empty_content",
                "error_type": "empty_content",
                "safe_message": "DeepSeek returned HTTP 200 but model content was empty",
                "request_id": "req-empty",
                "latency_ms": 14,
                "timeout_seconds": 60.0,
                "thinking_mode": "disabled",
                "model": "deepseek-v4-flash",
            },
            "provider_empty_content",
        ),
        (
            {
                "success": False,
                "stage": "model_json",
                "status_code": 200,
                "error_code": "invalid_semantic_json",
                "error_type": "semantic_invalid_json",
                "safe_message": "DeepSeek returned content, but Semantic Parser JSON was invalid",
                "request_id": "req-model-json",
                "latency_ms": 19,
                "timeout_seconds": 60.0,
                "thinking_mode": "disabled",
                "model": "deepseek-v4-flash",
            },
            "semantic_invalid_json",
        ),
    ],
)
def test_semantic_run_failure_type_uses_precise_safe_category(tmp_path, monkeypatch, diagnostic, expected_failure_type):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with patch("app.main.semantic_llm.pop_last_extract_diagnostic", return_value=diagnostic):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "张总:把材料补一下。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")
                diag_response = client.get(f"/api/evidence/{evidence_id}/diagnostics")

    assert create_response.status_code == 200
    assert detail_response.status_code == 200
    assert expected_failure_type in detail_response.text
    assert diagnostic["safe_message"] in detail_response.text
    payload = diag_response.json()
    assert payload["semantic_parser"]["failure_type"] == expected_failure_type
    assert payload["semantic_parser"]["diagnostic"]["stage"] == diagnostic["stage"]
    assert payload["semantic_parser"]["diagnostic"]["status_code"] == diagnostic["status_code"]
    assert payload["semantic_parser"]["diagnostic"]["safe_message"] == diagnostic["safe_message"]
    assert payload["semantic_parser"]["diagnostic"]["thinking_mode"] == "disabled"

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        run_row = conn.execute(
            """
            SELECT sr.failure_type
            FROM semantic_runs sr
            JOIN semantic_run_inputs sri ON sri.semantic_run_id = sr.semantic_run_id
            WHERE sri.evidence_id = ?
            ORDER BY sr.created_at DESC, sr.semantic_run_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        assert run_row["failure_type"] == expected_failure_type
    finally:
        conn.close()


def test_deepseek_preflight_does_not_write_db(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=_semantic_result()):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "只是留档。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        before_runs = conn.execute("SELECT COUNT(*) AS count FROM semantic_runs").fetchone()["count"]
        before_inputs = conn.execute("SELECT COUNT(*) AS count FROM semantic_run_inputs").fetchone()["count"]
        before_extractions = conn.execute("SELECT COUNT(*) AS count FROM evidence_extractions").fetchone()["count"]
    finally:
        conn.close()

    preflight = {
        "success": True,
        "stage": "success",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-preflight",
        "latency_ms": 7,
        "timeout_seconds": 60.0,
        "thinking_mode": "disabled",
        "model": "deepseek-v4-flash",
    }
    with patch("app.main.diagnose_deepseek_text_preflight", return_value=preflight):
        with client:
            response = client.post(f"/api/evidence/{evidence_id}/diagnostics/deepseek-preflight")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["deepseek_text_preflight"] == preflight

    conn = init_db(db_path)
    try:
        after_runs = conn.execute("SELECT COUNT(*) AS count FROM semantic_runs").fetchone()["count"]
        after_inputs = conn.execute("SELECT COUNT(*) AS count FROM semantic_run_inputs").fetchone()["count"]
        after_extractions = conn.execute("SELECT COUNT(*) AS count FROM evidence_extractions").fetchone()["count"]
        assert after_runs == before_runs
        assert after_inputs == before_inputs
        assert after_extractions == before_extractions
    finally:
        conn.close()


def test_ark_vision_diagnostic_uses_current_blob_and_does_not_mutate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": "渠道复盘数据",
        "due_date": "2026-08-08",
        "due_raw": "周五前",
        "direction": "none",
        "plain_summary": "审批通过,周五前交付渠道复盘数据",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="ark-diag.png")
                evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        before = conn.execute(
            """
            SELECT raw_text, content_hash, chain_hash, blob_path
            FROM evidence
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        before_extraction_count = conn.execute(
            "SELECT COUNT(*) AS count FROM evidence_extractions WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()["count"]
        before_parse_status = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (f"parse_status:{evidence_id}",),
        ).fetchone()["value"]
    finally:
        conn.close()

    expected_blob_bytes = (db_path.parent / "blobs" / before["blob_path"]).read_bytes()
    ark_result = {
        "transcript": "Ark 看到的文字",
        "observations": [
            {
                "kind": "reaction",
                "content": "有人对该消息显示👍反应,身份在画面中不可见",
                "confidence": 0.68,
            }
        ],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": ["画面右上角有轻微遮挡"],
    }
    preflight = {
        "success": True,
        "stage": "output_text",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-preflight",
        "latency_ms": 12,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 20.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": ["output_text"],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": "str",
        },
        "extraction": None,
    }
    diagnostic = {
        "success": True,
        "stage": "contract",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-vision",
        "latency_ms": 34,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 90.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": ["output_text"],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": "str",
        },
        "extraction": ark_result,
    }

    with patch("app.main.vision_provider.diagnose_text_preflight", return_value=preflight) as mock_preflight:
        with patch("app.main.vision_provider.diagnose_visual_evidence", return_value=diagnostic) as mock_vision:
            with patch("app.main._emit_structured_log") as mock_log:
                with client:
                    response = client.post(f"/api/evidence/{evidence_id}/diagnostics/ark-vision")

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline"]["provider"] == "dashscope"
    assert payload["baseline"]["summary"].startswith("机器提取")
    assert payload["ark_vision"] == {
        "status": "succeeded",
        "detail": "Ark Vision 实验解析完成,结果仅供对照,不会保存或影响当前记录",
        "extraction": ark_result,
        "text_preflight": preflight,
        "diagnostic": diagnostic,
    }
    assert "ark-test-key" not in response.text
    assert "data:image/" not in response.text
    assert str(db_path.parent) not in response.text

    mock_preflight.assert_called_once_with()
    mock_vision.assert_called_once()
    args, kwargs = mock_vision.call_args
    assert args == (expected_blob_bytes, "image/png")
    assert kwargs == {}
    mock_log.assert_called_once()
    assert mock_log.call_args.args[0] == "ark_vision_diagnostic"
    assert mock_log.call_args.args[1]["evidence_id"] == evidence_id
    assert mock_log.call_args.args[1]["provider"] == "doubao-ark"
    assert mock_log.call_args.args[1]["status"] == "succeeded"
    assert mock_log.call_args.args[1]["transcript_chars"] == len("Ark 看到的文字")
    assert mock_log.call_args.args[1]["observation_count"] == 1
    assert mock_log.call_args.args[1]["error_type"] is None

    conn = init_db(db_path)
    try:
        after = conn.execute(
            """
            SELECT raw_text, content_hash, chain_hash
            FROM evidence
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        after_extraction_count = conn.execute(
            "SELECT COUNT(*) AS count FROM evidence_extractions WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()["count"]
        after_parse_status = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (f"parse_status:{evidence_id}",),
        ).fetchone()["value"]
        assert after["raw_text"] == before["raw_text"]
        assert after["content_hash"] == before["content_hash"]
        assert after["chain_hash"] == before["chain_hash"]
        assert after_extraction_count == before_extraction_count
        assert after_parse_status == before_parse_status
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_ark_vision_diagnostic_rejects_non_image_and_missing_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        text_response = client.post(
            "/api/evidence",
            json={"text": "只是文字", "source": "飞书", "source_detail": "项目复盘群"},
        )
        text_evidence_id = text_response.json()["evidence_id"]
        non_image_response = client.post(f"/api/evidence/{text_evidence_id}/diagnostics/ark-vision")
        missing_response = client.post("/api/evidence/ev_missing/diagnostics/ark-vision")

    assert text_response.status_code == 200
    assert non_image_response.status_code == 400
    assert non_image_response.json()["detail"] == "只有图片记录支持 Ark Vision 实验解析"
    assert missing_response.status_code == 404


def test_ark_vision_diagnostic_returns_clear_failure_when_ark_key_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_date": None,
        "due_raw": None,
        "direction": "none",
        "plain_summary": "只是留档。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="ark-no-key.png")
                evidence_id = create_response.json()["evidence_id"]
                response = client.post(f"/api/evidence/{evidence_id}/diagnostics/ark-vision")

    assert response.status_code == 503
    payload = response.json()
    assert payload["ark_vision"]["status"] == "failed"
    assert payload["ark_vision"]["detail"] == "Ark text preflight 失败,优先排查 Key、Base URL、模型配置或模型开通状态。"
    assert payload["ark_vision"]["extraction"] is None
    assert payload["ark_vision"]["text_preflight"]["stage"] == "config"
    assert payload["ark_vision"]["text_preflight"]["error_code"] == "not_configured"
    assert payload["ark_vision"]["text_preflight"]["timeout_seconds"] == 20.0
    assert payload["ark_vision"]["text_preflight"]["thinking_mode"] == "disabled"
    assert payload["ark_vision"]["diagnostic"] is None


def test_ark_vision_diagnostic_distinguishes_text_preflight_success_and_vision_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_date": None,
        "due_raw": None,
        "direction": "none",
        "plain_summary": "只是留档。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="ark-provider-fail.png")
                evidence_id = create_response.json()["evidence_id"]

    preflight = {
        "success": True,
        "stage": "output_text",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-preflight",
        "latency_ms": 5,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 20.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": ["output_text"],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": "str",
        },
        "extraction": None,
    }
    diagnostic = {
        "success": False,
        "stage": "model_json",
        "status_code": 200,
        "error_code": "invalid_model_json",
        "error_type": "JSONDecodeError",
        "safe_message": "Ark 已正常返回,但 WorkChain 在 model_json 阶段无法解析结果",
        "request_id": "req-vision",
        "latency_ms": 19,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 90.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": ["output_text"],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": "str",
        },
        "extraction": None,
    }

    with patch("app.main.vision_provider.diagnose_text_preflight", return_value=preflight):
        with patch("app.main.vision_provider.diagnose_visual_evidence", return_value=diagnostic):
            with client:
                response = client.post(f"/api/evidence/{evidence_id}/diagnostics/ark-vision")

    assert response.status_code == 502
    payload = response.json()
    assert payload["ark_vision"]["status"] == "failed"
    assert payload["ark_vision"]["detail"] == "Ark 已正常返回,但 WorkChain 在 model_json 阶段无法解析结果"
    assert payload["ark_vision"]["extraction"] is None
    assert payload["ark_vision"]["text_preflight"]["success"] is True
    assert payload["ark_vision"]["diagnostic"]["stage"] == "model_json"
    assert payload["ark_vision"]["diagnostic"]["timeout_seconds"] == 90.0
    assert payload["ark_vision"]["diagnostic"]["thinking_mode"] == "disabled"


def test_ark_vision_timeout_response_reports_visual_timeout_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKCHAIN_DIAGNOSTICS", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test")
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_date": None,
        "due_raw": None,
        "direction": "none",
        "plain_summary": "只是留档。",
        "caveats": [],
    }

    with patch("app.extract.ocr.image_to_text", return_value=("原始识别文字", "")):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = _upload_png(client, filename="ark-timeout.png")
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    preflight = {
        "success": True,
        "stage": "output_text",
        "status_code": 200,
        "error_code": None,
        "error_type": None,
        "safe_message": None,
        "request_id": "req-preflight",
        "latency_ms": 4,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 20.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": ["output_text"],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": "str",
        },
        "extraction": None,
    }
    diagnostic = {
        "success": False,
        "stage": "http",
        "status_code": None,
        "error_code": "timeout",
        "error_type": "ReadTimeout",
        "safe_message": "请求超过当前超时上限 90 秒",
        "request_id": None,
        "latency_ms": 90123,
        "model": "doubao-seed-2-0-lite-260215",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "timeout_seconds": 90.0,
        "thinking_mode": "disabled",
        "response_shape": {
            "top_level_keys": [],
            "output_type": None,
            "output_item_types": [],
            "content_types": [],
            "output_text_type": None,
        },
        "extraction": None,
    }

    with patch("app.main.vision_provider.diagnose_text_preflight", return_value=preflight):
        with patch("app.main.vision_provider.diagnose_visual_evidence", return_value=diagnostic):
            with client:
                response = client.post(f"/api/evidence/{evidence_id}/diagnostics/ark-vision")

    assert detail_response.status_code == 200
    assert "超时上限" in detail_response.text
    assert "Thinking" in detail_response.text
    assert response.status_code == 502
    payload = response.json()
    assert payload["ark_vision"]["detail"] == "请求超过当前视觉超时上限 90 秒"
    assert payload["ark_vision"]["diagnostic"]["error_code"] == "timeout"
    assert payload["ark_vision"]["diagnostic"]["timeout_seconds"] == 90.0
    assert payload["ark_vision"]["diagnostic"]["thinking_mode"] == "disabled"


def test_post_evidence_rejects_empty_text_and_too_long_text(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        empty_response = client.post(
            "/api/evidence",
            json={"text": "   ", "source": "飞书", "source_detail": "项目复盘群"},
        )
        long_response = client.post(
            "/api/evidence",
            json={"text": "a" * 20001, "source": "飞书", "source_detail": "项目复盘群"},
        )

    assert empty_response.status_code == 400
    assert long_response.status_code == 400


def test_post_evidence_rejects_text_and_file_together(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = _multipart_request(
            client,
            data={"text": "这次既有文字也有文件", "source": "飞书", "source_detail": "项目复盘群"},
            file_part=("note.txt", "文件内容".encode("utf-8"), "text/plain"),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "文字记录和文件不能同时提交，请二选一。"


def test_two_clients_do_not_see_each_other_records(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        client_a.post(
            "/api/evidence",
            json={"text": "只属于 A 的内容。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        client_b.post(
            "/api/evidence",
            json={"text": "只属于 B 的内容。", "source": "Jira", "source_detail": "WORK-999"},
        )

        home_a = client_a.get("/")
        home_b = client_b.get("/")

    assert "只属于 A 的内容" not in home_a.text
    assert "只属于 B 的内容" not in home_a.text
    assert "只属于 B 的内容" not in home_b.text
    assert "只属于 A 的内容" not in home_b.text


def test_verify_chain_still_passes_after_posting_to_sandbox(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        client.post(
            "/api/evidence",
            json={"text": "这是新存的一条参考记录。", "source": "Slack", "source_detail": "growth-sync"},
        )

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_demo_database_mtime_does_not_change_after_post(tmp_path, monkeypatch):
    client, demo_dir, _ = _make_client(tmp_path, monkeypatch)

    with client:
        client.get("/")
        demo_db_path = demo_dir / "workchain.db"
        before = demo_db_path.stat().st_mtime_ns
        time.sleep(0.01)
        response = client.post(
            "/api/evidence",
            json={"text": "确认一下，演示库不能被写。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        after = demo_db_path.stat().st_mtime_ns

    assert response.status_code == 200
    assert before == after


def test_same_sandbox_hits_daily_rate_limit_on_51st_post(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        last_response = None
        for idx in range(51):
            last_response = client.post(
                "/api/evidence",
                json={"text": f"第 {idx + 1} 条临时记录", "source": "飞书", "source_detail": "项目复盘群"},
            )

    assert last_response is not None
    assert last_response.status_code == 429


def test_post_evidence_without_api_key_marks_parse_failed(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        create_response = client.post(
            "/api/evidence",
            json={"text": "张总:先把结果留档。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        evidence_id = create_response.json()["evidence_id"]
        status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "failed"


def test_parse_none_still_keeps_evidence_and_marks_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=None):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "这是一条需要解析失败的记录。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["parse_status"] == "failed"
    assert payload["slots_filled"] == 0


def test_successful_parse_persists_semantic_run_and_facts_and_chain_stays_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "张伟要求下周五前提交渠道复盘数据。",
                "actors": [{"name": "活爹", "role": "requester"}, {"name": "我", "role": "owner"}],
                "due_raw": "下周五",
                "due_date": "2026-08-08",
                "due_anchor_date": "2026-08-08",
                "occurred_date": None,
                "confidence": 0.95,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "explanation",
                "content": "这里的活爹是对张伟的别称。",
                "confidence": 0.8,
            }
        ],
        ambiguities=["没有看到明确交付格式。"],
    )

    with client:
        client.post(
            "/api/settings",
            json={
                "self_names": ["热心市民小李"],
                "glossary": [{"term": "活爹", "kind": "person", "meaning": "张伟"}],
            },
        )
    with patch("app.main.update_slots") as mock_update_slots:
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "活爹:下周五前把渠道复盘数据给我。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    payload = status_response.json()
    assert payload["parse_status"] == "done"
    assert payload["slots_filled"] == 0
    mock_update_slots.assert_not_called()

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        evidence_row = conn.execute(
            """
            SELECT kind, slot_direction
            FROM evidence WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        run_row = conn.execute(
            """
            SELECT sr.semantic_run_id, sr.status, sr.provider, sr.model, sr.parser_version,
                   sr.anchor_date, sri.extraction_id
            FROM semantic_runs sr
            JOIN semantic_run_inputs sri ON sri.semantic_run_id = sr.semantic_run_id
            WHERE sri.evidence_id = ?
            ORDER BY sr.created_at DESC, sr.semantic_run_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        fact_row = conn.execute(
            """
            SELECT fact_id, fact_type, content, due_at, due_raw, due_anchor_at,
                   event_assignment, origin, review_status, semantic_run_id
            FROM facts
            WHERE semantic_run_id = ?
            """,
            (run_row["semantic_run_id"],),
        ).fetchone()
        interpretation_rows = conn.execute(
            """
            SELECT kind, content, fact_id, evidence_id, semantic_run_id
            FROM interpretations
            WHERE semantic_run_id = ?
            ORDER BY kind ASC
            """,
            (run_row["semantic_run_id"],),
        ).fetchall()
        actor_row = conn.execute(
            """
            SELECT a.canonical_name, fa.role
            FROM fact_actors fa
            JOIN actors a ON a.actor_id = fa.actor_id
            WHERE fa.fact_id = ?
            ORDER BY fa.role ASC
            """,
            (fact_row["fact_id"],),
        ).fetchone()
        assert evidence_row["kind"] == "reference"
        assert evidence_row["slot_direction"] is None
        assert run_row["status"] == "succeeded"
        assert run_row["provider"] == "deepseek"
        assert run_row["model"] == "deepseek-v4-flash"
        assert run_row["parser_version"] == "2.3"
        assert run_row["anchor_date"] is None
        assert run_row["extraction_id"] is not None
        assert fact_row["fact_type"] == "request"
        assert fact_row["content"] == "张伟要求下周五前提交渠道复盘数据。"
        assert fact_row["due_raw"] == "下周五"
        assert fact_row["due_at"] == main_module.llm.due_date_to_millis("2026-08-08")
        assert fact_row["due_anchor_at"] == main_module.llm.due_date_to_millis("2026-08-08")
        assert fact_row["event_assignment"] == "unassigned"
        assert fact_row["origin"] == "ai"
        assert fact_row["review_status"] == "unreviewed"
        assert fact_row["semantic_run_id"] == run_row["semantic_run_id"]
        assert actor_row["canonical_name"] == "张伟"
        assert actor_row["role"] == "requester"
        assert {(row["kind"], row["semantic_run_id"]) for row in interpretation_rows} == {
            ("explanation", run_row["semantic_run_id"]),
            ("uncertainty", run_row["semantic_run_id"]),
        }
        assert any(row["fact_id"] == fact_row["fact_id"] for row in interpretation_rows if row["kind"] == "explanation")
        assert any(row["evidence_id"] == evidence_id for row in interpretation_rows if row["kind"] == "uncertainty")
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_semantic_run_with_zero_facts_does_not_call_event_matcher(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", return_value=_semantic_result()):
        with patch("app.main.event_matcher.match_events", side_effect=AssertionError("should not be called")):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "只是留档。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    assert status_response.json()["parse_status"] == "done"

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        count = conn.execute("SELECT COUNT(*) AS count FROM event_match_runs").fetchone()["count"]
        assert count == 0
    finally:
        conn.close()


def test_event_matcher_auto_existing_updates_facts_and_detail_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.95,
                "reason": "这是同一件已存在的渠道复盘事项",
            }
        ],
        "ambiguities": [],
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert "已自动归入事项：渠道复盘" in detail_response.text
    assert "这是同一件已存在的渠道复盘事项" in detail_response.text

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        fact_row = conn.execute(
            """
            SELECT event_id, event_assignment, event_assignment_confidence
            FROM facts
            ORDER BY created_at DESC, fact_id DESC
            LIMIT 1
            """
        ).fetchone()
        match_row = conn.execute(
            """
            SELECT status, routing_mode, result_json, review_status
            FROM event_match_runs
            ORDER BY created_at DESC, event_match_run_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert fact_row["event_id"] == "evt-1"
        assert fact_row["event_assignment"] == "auto"
        assert fact_row["event_assignment_confidence"] == 0.95
        assert match_row["status"] == "succeeded"
        assert match_row["routing_mode"] == "auto"
        assert match_row["review_status"] == "completed"
    finally:
        conn.close()


def test_event_matcher_auto_new_creates_event_and_updates_detail_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert "已自动归入事项：补签供应商合同" in detail_response.text
    assert "这是一件新的合同处理事项" in detail_response.text

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        event_row = conn.execute(
            """
            SELECT event_id, title
            FROM events
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """
        ).fetchone()
        fact_row = conn.execute(
            """
            SELECT event_id, event_assignment
            FROM facts
            ORDER BY created_at DESC, fact_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert event_row["title"] == "补签供应商合同"
        assert fact_row["event_id"] == event_row["event_id"]
        assert fact_row["event_assignment"] == "auto"
    finally:
        conn.close()


def test_event_matcher_single_group_confirm_copy_and_options(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            },
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.72,
                "reason": "像是在延续旧事项",
            }
        ],
        "ambiguities": [],
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘，客户回访也要整理。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert "AI认为这属于以下事项，请你确认。" in detail_response.text
    assert "这段记录可能涉及多个事项，请分别确认。" not in detail_response.text
    assert "渠道复盘" in detail_response.text
    assert "＋新建事项…" in detail_response.text
    assert "暂不归入" in detail_response.text

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        rows = conn.execute(
            """
            SELECT event_id, event_assignment
            FROM facts
            ORDER BY fact_id ASC
            """
        ).fetchall()
        assert all(row["event_id"] is None and row["event_assignment"] == "unassigned" for row in rows)
    finally:
        conn.close()


def test_event_matcher_multi_group_shows_groups_for_separate_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            },
            {
                "fact_type": "statement",
                "content": "客户回访也要整理。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.7,
            },
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.72,
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
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘，客户回访也要整理。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert "这段记录可能涉及多个事项，请分别确认。" in detail_response.text
    assert "第 1 组" in detail_response.text
    assert "第 2 组" in detail_response.text
    assert "整理客户回访" in detail_response.text
    assert "客户回访也要整理。" in detail_response.text


def test_event_matcher_needs_context_shows_existing_new_and_unassigned_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "unassigned",
                "event_id": None,
                "proposed_title": None,
                "confidence": 0.0,
                "reason": "上下文不足",
            }
        ],
        "ambiguities": [],
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert "暂时无法可靠判断属于哪件事，请选择归属。" in detail_response.text
    assert "渠道复盘" in detail_response.text
    assert "＋新建事项…" in detail_response.text
    assert "暂不归入" in detail_response.text


def test_event_assignment_confirmation_creates_event_and_marks_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            },
            {
                "fact_type": "statement",
                "content": "客户回访也要整理。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.7,
            },
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.72,
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
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘，客户回访也要整理。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]

        conn = init_db(db_path)
        try:
            match_run_id = conn.execute(
                """
                SELECT event_match_run_id
                FROM event_match_runs
                ORDER BY created_at DESC, event_match_run_id DESC
                LIMIT 1
                """
            ).fetchone()["event_match_run_id"]
        finally:
            conn.close()

        confirm_response = client.post(
            f"/api/evidence/{evidence_id}/event-assignment",
            json={
                "event_match_run_id": match_run_id,
                "groups": [
                    {"group_index": 0, "choice": "existing", "event_id": "evt-1"},
                    {"group_index": 1, "choice": "new", "new_title": "整理客户回访"},
                ],
            },
        )
        detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert confirm_response.status_code == 200
    assert detail_response.status_code == 200
    assert "事项归属已确认" in detail_response.text
    assert "已归入事项：渠道复盘" in detail_response.text
    assert "已归入事项：整理客户回访" in detail_response.text

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        match_row = conn.execute(
            """
            SELECT review_status
            FROM event_match_runs
            WHERE event_match_run_id = ?
            """,
            (match_run_id,),
        ).fetchone()
        fact_rows = conn.execute(
            """
            SELECT content, event_id, event_assignment
            FROM facts
            ORDER BY content ASC
            """
        ).fetchall()
        new_event = conn.execute(
            "SELECT title FROM events WHERE event_id != 'evt-1' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert match_row["review_status"] == "completed"
        assert fact_rows[0]["content"] == "客户回访也要整理。"
        assert fact_rows[0]["event_assignment"] == "confirmed"
        assert fact_rows[1]["content"] == "请补一版渠道复盘。"
        assert fact_rows[1]["event_id"] == "evt-1"
        assert fact_rows[1]["event_assignment"] == "confirmed"
        assert new_event["title"] == "整理客户回访"
    finally:
        conn.close()


def test_event_assignment_confirmation_rejects_tampered_fact_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "existing",
                "event_id": "evt-1",
                "proposed_title": None,
                "confidence": 0.72,
                "reason": "像是在延续旧事项",
            }
        ],
        "ambiguities": [],
    }

    with client:
        client.get("/")
        db_path = _sandbox_db_path(client, sandbox_root)
        conn = init_db(db_path)
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, title, status, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("evt-1", "渠道复盘", "active", None, 1, 1),
            )
            conn.commit()
        finally:
            conn.close()
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]

        conn = init_db(db_path)
        try:
            match_run_id = conn.execute(
                """
                SELECT event_match_run_id
                FROM event_match_runs
                ORDER BY created_at DESC, event_match_run_id DESC
                LIMIT 1
                """
            ).fetchone()["event_match_run_id"]
        finally:
            conn.close()

        confirm_response = client.post(
            f"/api/evidence/{evidence_id}/event-assignment",
            json={
                "event_match_run_id": match_run_id,
                "groups": [
                    {
                        "group_index": 0,
                        "choice": "existing",
                        "event_id": "evt-1",
                        "fact_ids": ["fact-hacked"],
                    }
                ],
            },
        )

    assert create_response.status_code == 200
    assert confirm_response.status_code == 400
    assert "unsupported group fields: fact_ids" in confirm_response.text


def test_event_matcher_failure_keeps_semantic_parse_done_and_shows_failed_status(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请补一版渠道复盘。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", side_effect=RuntimeError("boom")):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请补一版渠道复盘。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                status_response = client.get(f"/api/evidence/{evidence_id}/status")
                detail_response = client.get(f"/evidence/{evidence_id}")

    assert create_response.status_code == 200
    assert status_response.json()["parse_status"] == "done"
    assert "事项归属暂不可用，事实整理结果已保存。" in detail_response.text

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        semantic_run = conn.execute(
            """
            SELECT sr.status
            FROM semantic_runs sr
            JOIN semantic_run_inputs sri ON sri.semantic_run_id = sr.semantic_run_id
            WHERE sri.evidence_id = ?
            ORDER BY sr.created_at DESC, sr.semantic_run_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        event_match_run = conn.execute(
            """
            SELECT status, failure_type
            FROM event_match_runs
            ORDER BY created_at DESC, event_match_run_id DESC
            LIMIT 1
            """
        ).fetchone()
        assert semantic_run["status"] == "succeeded"
        assert event_match_run["status"] == "failed"
        assert event_match_run["failure_type"] == "persistence_error"
    finally:
        conn.close()


def test_unstable_pronouns_do_not_create_permanent_actor(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "有人要求补一份复盘。",
                "actors": [{"name": "我", "role": "owner"}, {"name": "对方", "role": "requester"}],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.66,
            }
        ]
    )

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "你把复盘给我。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        fact_count = conn.execute(
            "SELECT COUNT(*) AS count FROM facts WHERE semantic_run_id IS NOT NULL"
        ).fetchone()
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM fact_actors fa
            JOIN facts f ON f.fact_id = fa.fact_id
            JOIN fact_evidence fe ON fe.fact_id = f.fact_id
            WHERE fe.evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        assert fact_count["count"] == 1
        assert row["count"] == 0
    finally:
        conn.close()


def test_parse_timeout_exception_marks_failed_without_breaking_post(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with patch("app.main.semantic_llm.extract_semantics", side_effect=httpx.TimeoutException("boom")):
        with client:
            create_response = client.post(
                "/api/evidence",
                json={"text": "张总:把材料补一下。", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = create_response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert create_response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["parse_status"] == "failed"

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        run_row = conn.execute(
            """
            SELECT sr.failure_type
            FROM semantic_runs sr
            JOIN semantic_run_inputs sri ON sri.semantic_run_id = sr.semantic_run_id
            WHERE sri.evidence_id = ?
            ORDER BY sr.created_at DESC, sr.semantic_run_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        assert run_row["failure_type"] == "provider_timeout"
    finally:
        conn.close()


def test_parse_limit_marks_21st_record_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result()

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with client:
            for idx in range(20):
                response = client.post(
                    "/api/evidence",
                    json={"text": f"第 {idx + 1} 条要解析的记录", "source": "飞书", "source_detail": "项目复盘群"},
                )
                assert response.status_code == 200

            last_response = client.post(
                "/api/evidence",
                json={"text": "第 21 条要解析的记录", "source": "飞书", "source_detail": "项目复盘群"},
            )
            evidence_id = last_response.json()["evidence_id"]
            status_response = client.get(f"/api/evidence/{evidence_id}/status")

    assert last_response.status_code == 200
    assert status_response.json()["parse_status"] == "failed"
    assert status_response.json()["detail"] == "今日解析次数已用完,记录仍已保存"


def test_parse_pipeline_passes_glossary_source_hint_and_anchor_none(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}

    def fake_extract(text, *, observations=None, anchor_date=None, glossary=None, source_hint=None):
        captured["text"] = text
        captured["observations"] = observations
        captured["anchor_date"] = anchor_date
        captured["glossary"] = glossary
        captured["source_hint"] = source_hint
        return _semantic_result()

    with client:
        client.post(
            "/api/settings",
            json={
                "self_names": ["热心市民小李"],
                "glossary": [{"term": "活爹", "kind": "person", "meaning": "张伟"}],
            },
        )
        with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
            response = client.post(
                "/api/evidence",
                json={
                    "text": "活爹说先记一下。",
                    "source": "飞书",
                    "source_detail": "项目复盘群",
                    "counterpart": "冯云生(师父)",
                },
            )

    assert response.status_code == 200
    assert captured["text"] == "活爹说先记一下。"
    assert captured["observations"] == []
    assert captured["anchor_date"] is None
    assert captured["glossary"] == [{"term": "活爹", "kind": "person", "meaning": "张伟"}]
    assert captured["source_hint"] == "飞书-项目复盘群"


def test_parse_pipeline_uses_explicit_record_date_as_anchor_date(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}

    def fake_extract(text, *, observations=None, anchor_date=None, glossary=None, source_hint=None):
        captured["anchor_date"] = anchor_date
        return _semantic_result()

    with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
        with client:
            response = client.post(
                "/api/evidence",
                json={
                    "text": "小李（2026.8.9）：周五前补一版。",
                    "source": "飞书",
                    "source_detail": "项目复盘群",
                    "record_date": "2026-08-08",
                },
            )

    assert response.status_code == 200
    assert captured["anchor_date"] == "2026-08-08"


def test_parse_pipeline_infers_reliable_anchor_date_from_message_header(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}

    def fake_extract(text, *, observations=None, anchor_date=None, glossary=None, source_hint=None):
        captured["anchor_date"] = anchor_date
        return _semantic_result()

    with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
        with client:
            response = client.post(
                "/api/evidence",
                json={
                    "text": "小李（2026.8.9）：今天补你。",
                    "source": "飞书",
                    "source_detail": "项目复盘群",
                },
            )

    assert response.status_code == 200
    assert captured["anchor_date"] == "2026-08-09"


def test_parse_pipeline_does_not_infer_plain_body_delivery_date_as_anchor(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}

    def fake_extract(text, *, observations=None, anchor_date=None, glossary=None, source_hint=None):
        captured["anchor_date"] = anchor_date
        return _semantic_result()

    with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
        with client:
            response = client.post(
                "/api/evidence",
                json={
                    "text": "合同2026.8.20前交付。",
                    "source": "飞书",
                    "source_detail": "项目复盘群",
                },
            )

    assert response.status_code == 200
    assert captured["anchor_date"] is None


@pytest.mark.parametrize(
    ("record_date", "expected_detail"),
    [
        ("202600-08-09", "记录发生日期请按 YYYY-MM-DD 填写"),
        ("2200-01-01", "记录发生日期需在 1900-01-01 到 2100-12-31 之间"),
    ],
)
def test_record_date_rejects_invalid_or_out_of_range_year(tmp_path, monkeypatch, record_date, expected_detail):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/evidence",
            json={
                "text": "周五前补一版。",
                "source": "飞书",
                "source_detail": "项目复盘群",
                "record_date": record_date,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_evidence_detail_record_date_update_recomputes_due_without_calling_deepseek(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "deadline_change",
                "content": "请周五前补材料。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.8,
            }
        ],
        interpretations=[
            {
                "fact_index": 0,
                "kind": "uncertainty",
                "content": "由于缺少 anchor_date，无法将“周五”换算为具体日期。",
                "confidence": 0.6,
            },
            {
                "fact_index": 0,
                "kind": "uncertainty",
                "content": "负责人还不够明确。",
                "confidence": 0.5,
            },
        ],
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补材料",
                "confidence": 0.95,
                "reason": "这是一件新的补材料事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请周五前补材料。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]
                detail_before = client.get(f"/evidence/{evidence_id}")

    with patch("app.main.semantic_llm.extract_semantics", side_effect=AssertionError("should not call DeepSeek")):
        with client:
            update_response = client.post(
                f"/api/evidence/{evidence_id}/record-date",
                json={"record_date": "2026-08-09"},
            )
            detail_after = client.get(f"/evidence/{evidence_id}")
            home_after = client.get("/")

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        fact_row = conn.execute(
            """
            SELECT due_at, due_anchor_at, origin, review_status, event_id
            FROM facts
            WHERE fact_id IN (
                SELECT fact_id FROM fact_evidence WHERE evidence_id = ?
            )
            """,
            (evidence_id,),
        ).fetchone()
        anchor_rows = conn.execute(
            "SELECT key, value FROM meta WHERE key IN (?, ?)",
            (
                f"semantic_anchor_date:{evidence_id}",
                f"semantic_anchor_source:{evidence_id}",
            ),
        ).fetchall()
        anchor_meta = {row["key"]: row["value"] for row in anchor_rows}
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()

    friendly_copy = "还无法确定“周五”具体是哪一天。补充这段记录发生的日期后，可以换算成具体日期。"
    assert create_response.status_code == 200
    assert friendly_copy in detail_before.text
    assert update_response.status_code == 200
    assert update_response.json()["updated_fact_count"] == 1
    assert fact_row["due_at"] == main_module.llm.due_date_to_millis("2026-08-14")
    assert fact_row["due_anchor_at"] == main_module.llm.due_date_to_millis("2026-08-09")
    assert fact_row["origin"] == "user"
    assert fact_row["review_status"] == "corrected"
    assert anchor_meta[f"semantic_anchor_date:{evidence_id}"] == "2026-08-09"
    assert anchor_meta[f"semantic_anchor_source:{evidence_id}"] == "user"
    assert friendly_copy not in detail_after.text
    assert "负责人还不够明确。" in detail_after.text
    assert "记录日期：2026-08-09" in detail_after.text
    assert "来源：你填写的" in detail_after.text
    assert "已按 2026-08-09 换算相对日期。" in detail_after.text
    assert "截止：08-14" in home_after.text


def test_event_page_can_update_record_date_for_image_evidence_and_time_only_observation_needs_user_input(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORKCHAIN_IMAGE_EXTRACTION_PROVIDER", "ark_vision")
    monkeypatch.setenv("ARK_API_KEY", "ark-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "deadline_change",
                "content": "请周五前提交图片版材料。",
                "actors": [],
                "due_raw": "周五前",
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "图片补材料",
                "confidence": 0.95,
                "reason": "这是一件新的图片补材料事项",
            }
        ],
        "ambiguities": [],
    }
    ark_extraction = {
        "transcript": "请周五前提交图片版材料。",
        "observations": [{"kind": "message", "content": "19:21", "confidence": 0.81}],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": [],
    }

    with patch("app.vision_provider.extract_visual_evidence", return_value=ark_extraction):
        with patch("app.main.semantic_llm.extract_semantics", return_value=parsed) as mock_llm:
            with patch("app.main.event_matcher.match_events", return_value=normalized_match):
                with client:
                    create_response = _upload_png(client, filename="record-date-image.png")
                    evidence_id = create_response.json()["evidence_id"]

    assert create_response.status_code == 200
    assert mock_llm.call_args.kwargs["anchor_date"] is None

    conn = init_db(_sandbox_db_path(client, sandbox_root))
    try:
        row = conn.execute(
            """
            SELECT f.fact_id, f.event_id
            FROM facts f
            JOIN fact_evidence fe ON fe.fact_id = f.fact_id
            WHERE fe.evidence_id = ?
            ORDER BY f.created_at DESC, f.fact_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
    finally:
        conn.close()

    with client:
        detail_before = client.get(f"/event/{row['event_id']}")
        update_response = client.post(
            f"/api/evidence/{evidence_id}/record-date",
            json={"record_date": "2026-08-09"},
        )
        detail_after = client.get(f"/event/{row['event_id']}")

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        fact_row = conn.execute(
            "SELECT due_at, due_anchor_at, origin, review_status FROM facts WHERE fact_id = ?",
            (row["fact_id"],),
        ).fetchone()
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()

    assert "这条证据里有“今天 / 周五”等相对时间。" in detail_before.text
    assert f'data-evidence-id="{evidence_id}"' in detail_before.text
    assert update_response.status_code == 200
    assert update_response.json()["updated_fact_count"] == 1
    assert fact_row["due_at"] == main_module.llm.due_date_to_millis("2026-08-14")
    assert fact_row["due_anchor_at"] == main_module.llm.due_date_to_millis("2026-08-09")
    assert fact_row["origin"] == "user"
    assert fact_row["review_status"] == "corrected"
    assert "记录日期：2026-08-09" in detail_after.text
    assert "来源：你填写的" in detail_after.text
    assert "修改" in detail_after.text


def test_event_status_moves_between_active_and_history_without_touching_facts_or_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)
    parsed = _semantic_result(
        facts=[
            {
                "fact_type": "request",
                "content": "请今天补签供应商合同。",
                "actors": [],
                "due_raw": None,
                "due_date": None,
                "due_anchor_date": None,
                "occurred_date": None,
                "confidence": 0.95,
            }
        ]
    )
    normalized_match = {
        "groups": [
            {
                "fact_indexes": [0],
                "target": "new",
                "event_id": None,
                "proposed_title": "补签供应商合同",
                "confidence": 0.95,
                "reason": "这是一件新的合同处理事项",
            }
        ],
        "ambiguities": [],
    }

    with patch("app.main.semantic_llm.extract_semantics", return_value=parsed):
        with patch("app.main.event_matcher.match_events", return_value=normalized_match):
            with client:
                create_response = client.post(
                    "/api/evidence",
                    json={"text": "请今天补签供应商合同。", "source": "飞书", "source_detail": "项目复盘群"},
                )
                evidence_id = create_response.json()["evidence_id"]

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        row = conn.execute(
            """
            SELECT f.fact_id, f.event_id
            FROM facts f
            JOIN fact_evidence fe ON fe.fact_id = f.fact_id
            WHERE fe.evidence_id = ?
            ORDER BY f.created_at DESC, f.fact_id DESC
            LIMIT 1
            """,
            (evidence_id,),
        ).fetchone()
        before_counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM facts WHERE event_id = ?) AS fact_count,
                (SELECT COUNT(*) FROM fact_evidence fe JOIN facts f ON f.fact_id = fe.fact_id WHERE f.event_id = ?) AS link_count
            """,
            (row["event_id"], row["event_id"]),
        ).fetchone()
    finally:
        conn.close()

    with client:
        resolve_response = client.post(
            f"/api/events/{row['event_id']}/status",
            json={"status": "resolved"},
        )
        resolved_home = client.get("/")
        resolved_detail = client.get(f"/event/{row['event_id']}")
        reopen_response = client.post(
            f"/api/events/{row['event_id']}/status",
            json={"status": "active"},
        )
        reopened_home = client.get("/")

    conn = init_db(db_path)
    try:
        after_counts = conn.execute(
            """
            SELECT
                status,
                (SELECT COUNT(*) FROM facts WHERE event_id = ?) AS fact_count,
                (SELECT COUNT(*) FROM fact_evidence fe JOIN facts f ON f.fact_id = fe.fact_id WHERE f.event_id = ?) AS link_count
            FROM events
            WHERE event_id = ?
            """,
            (row["event_id"], row["event_id"], row["event_id"]),
        ).fetchone()
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()

    assert create_response.status_code == 200
    assert resolve_response.status_code == 200
    assert resolve_response.json()["status"] == "resolved"
    assert 'data-testid="event-card"' not in resolved_home.text
    assert "补签供应商合同" not in resolved_home.text
    assert "重新打开" in resolved_detail.text
    assert reopen_response.status_code == 200
    assert reopen_response.json()["status"] == "active"
    assert 'data-testid="event-card"' in reopened_home.text
    assert after_counts["status"] == "active"
    assert after_counts["fact_count"] == before_counts["fact_count"] == 1
    assert after_counts["link_count"] == before_counts["link_count"] == 1


def test_parse_pipeline_does_not_pass_self_names_or_counterpart(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}

    def fake_extract(text, **kwargs):
        captured["kwargs"] = kwargs
        return _semantic_result()

    with client:
        client.post(
            "/api/settings",
            json={"self_names": ["热心市民小李"], "glossary": []},
        )
        with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
            response = client.post(
                "/api/evidence",
                json={
                    "text": "先留个底。",
                    "source": "飞书",
                    "source_detail": "项目复盘群",
                    "counterpart": "冯云生(师父)",
                },
            )

    assert response.status_code == 200
    assert "self_names" not in captured["kwargs"]
    assert "counterpart" not in captured["kwargs"]
    assert captured["kwargs"]["glossary"] == []


def test_document_text_is_truncated_before_sending_to_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client, _, _ = _make_client(tmp_path, monkeypatch)
    captured = {}
    long_text = "很长的文档内容" * 1200

    def fake_extract(text, **kwargs):
        captured["text"] = text
        return None

    with patch("app.main.semantic_llm.extract_semantics", side_effect=fake_extract):
        with client:
            response = client.post(
                "/api/evidence",
                data={"source": "飞书", "source_detail": "项目复盘群"},
                files={"file": ("long.txt", long_text.encode("utf-8"), "text/plain")},
            )

    assert response.status_code == 200
    assert len(captured["text"]) <= 8000
    assert "以下内容较长,已截取前一部分" in captured["text"]


def test_settings_save_and_read_self_names(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        save_response = client.post(
            "/api/settings",
            json={"self_names": ["热心市民小李", "小李"], "glossary": []},
        )
        get_response = client.get("/api/settings")

    assert save_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["self_names"] == ["热心市民小李", "小李"]
    assert get_response.json()["has_self_names"] is True


def test_settings_reject_more_than_five_self_names(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/settings",
            json={"self_names": ["1", "2", "3", "4", "5", "6"], "glossary": []},
        )

    assert response.status_code == 400


def test_settings_glossary_save_and_limit(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    glossary = [{"term": "活爹", "kind": "person", "meaning": "张伟"}]

    with client:
        save_response = client.post(
            "/api/settings",
            json={"self_names": [], "glossary": glossary},
        )
        get_response = client.get("/api/settings")
        limit_response = client.post(
            "/api/settings",
            json={
                "self_names": [],
                "glossary": [{"term": str(i), "kind": "phrase", "meaning": "x"} for i in range(51)],
            },
        )

    assert save_response.status_code == 200
    assert get_response.json()["glossary"] == glossary
    assert limit_response.status_code == 400


def test_settings_are_isolated_between_sandboxes(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        client_a.post(
            "/api/settings",
            json={"self_names": ["小李"], "glossary": [{"term": "活爹", "kind": "person", "meaning": "张伟"}]},
        )
        response_a = client_a.get("/api/settings")
        response_b = client_b.get("/api/settings")

    assert response_a.json()["self_names"] == ["小李"]
    assert response_b.json()["self_names"] == []
    assert response_b.json()["glossary"] == []
