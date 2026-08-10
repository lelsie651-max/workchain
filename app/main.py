from __future__ import annotations

import html
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from urllib.parse import quote

import httpx
from starlette.background import BackgroundTask
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from app import change_detector, event_matcher, llm, ocr, semantic_llm, vision_provider
from app.ai_provider import (
    build_text_config_diagnostic,
    diagnose_deepseek_text_preflight,
    get_text_api_key,
    get_text_model,
    get_text_timeout_seconds,
)
from app.evidence_extractor import (
    ARK_FALLBACK_WARNING,
    get_image_extraction_provider,
    get_image_extraction_provider_label,
    get_image_extraction_startup,
    run_production_image_extraction,
)
from app.extract import extract_text
from app.labels import KIND, RISK, STATUS, source_badge_class, source_label, thread_headline
from app.labels import SOURCE_PRESETS
from app.pdf_report import PDF_FILENAME_PREFIX, build_evidence_pdf
from app.sandbox import (
    SandboxContext,
    apply_sandbox_cookie,
    cleanup_expired,
    get_glossary,
    get_sandbox,
    get_self_names,
    get_settings,
    save_glossary,
    save_self_names,
)
from evidence_core import chain
from evidence_core.db import init_db
from evidence_core.extraction_store import create_extraction, get_latest_extraction, list_extractions
from evidence_core.semantic_store import (
    ProtectedFactError,
    SemanticStoreError,
    create_source_review,
    create_submission,
    create_event_change_run,
    create_semantic_run,
    create_event_match_run,
    correct_fact_by_user,
    correct_relative_due_dates_by_user,
    get_effective_source_hint,
    get_latest_event_change_run_for_event,
    get_latest_event_match_for_evidence,
    get_latest_event_match_for_semantic_run,
    get_latest_semantic_run_for_evidence,
    get_latest_source_review,
    list_event_candidates,
    list_facts_for_semantic_run,
    list_interpretations_for_semantic_run,
    mark_event_change_run_failed,
    mark_event_match_run_failed,
    mark_semantic_run_failed,
    persist_event_change_run_result,
    persist_event_match_run_result,
    persist_semantic_run_result,
    review_event_match_run_by_user,
)
from evidence_core.export import export_evidence_package
from evidence_core.store import append_evidence, update_slots, verify_chain
from scripts.seed_demo import seed_demo_data


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
MAX_TEXT_LENGTH = 20_000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_SANDBOX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_LLM_TEXT_LENGTH = 8_000
MAX_OCR_TEXT_LENGTH = 50_000
MAX_USER_EVENT_TITLE_LENGTH = 80
MIN_RECORD_DATE = "1900-01-01"
MAX_RECORD_DATE = "2100-12-31"
WORKCHAIN_DIAGNOSTICS_ENV = "WORKCHAIN_DIAGNOSTICS"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PARSE_STATUS_OCR_RUNNING = "ocr_running"
PARSE_STATUS_LLM_RUNNING = "llm_running"
PARSE_STATUS_DONE = "done"
PARSE_STATUS_FAILED = "failed"
PARSE_STATUS_UNSUPPORTED = "unsupported"
PARSE_STATUS_CLARIFICATION_REQUIRED = "clarification_required"
SOURCE_GATE_PLATFORM_CONFIDENCE_THRESHOLD = 0.75
PACKAGE_README_TEXT = (
    "这个文件夹里的记录不能被偷偷修改。\n"
    "如果你想自己确认,在装有 Python 的电脑上,\n"
    "在本文件夹内运行:python verify.py --dir .\n"
    "显示 OK 表示所有记录都和当初保存时一模一样。\n"
    "显示 FAIL 表示有记录被改过,并会指出是第几条。\n"
)


def _format_datetime(value: int | None, fmt: str) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000).strftime(fmt)


def _parse_status_label(status: str | None) -> str:
    mapping = {
        PARSE_STATUS_OCR_RUNNING: "正在读取图片中的文字",
        PARSE_STATUS_LLM_RUNNING: "正在理解这段记录",
        PARSE_STATUS_DONE: "已完成",
        PARSE_STATUS_FAILED: "暂不可用",
        PARSE_STATUS_UNSUPPORTED: "当前不支持自动解析",
        PARSE_STATUS_CLARIFICATION_REQUIRED: "等待核实信息来源",
    }
    return mapping.get(status, status or "未知")


def _format_due_display(value: int | None) -> str | None:
    if value is None:
        return None
    due_dt = datetime.fromtimestamp(value / 1000)
    current_year = datetime.now().year
    if due_dt.year == current_year:
        return due_dt.strftime("%m-%d")
    return due_dt.strftime("%Y-%m-%d")


def _friendly_interpretation_label(kind: str | None) -> str:
    mapping = {
        "explanation": "补充说明",
        "term": "术语说明",
        "action_hint": "可以这样处理",
        "uncertainty": "建议确认",
    }
    return mapping.get(kind or "", "说明")


def _event_status_label(status: str | None) -> str:
    mapping = {
        "active": "进行中",
        "resolved": "已结档",
        "archived": "已归档",
    }
    return mapping.get(status or "", status or "未知状态")


def _event_fact_type_label(fact_type: str | None) -> str:
    mapping = {
        "request": "要求",
        "commitment": "承诺",
        "confirmation": "确认",
        "scope_change": "范围变化",
        "responsibility_change": "责任变化",
        "deadline_change": "截止变化",
        "delivery": "交付",
        "cancellation": "取消",
        "denial": "否认",
        "statement": "说明",
        "reference": "参考",
    }
    return mapping.get(fact_type or "", fact_type or "说明")


def _event_fact_type_options() -> list[dict[str, str]]:
    return [
        {"value": "request", "label": "要求"},
        {"value": "commitment", "label": "承诺"},
        {"value": "confirmation", "label": "确认"},
        {"value": "scope_change", "label": "范围变化"},
        {"value": "responsibility_change", "label": "责任变化"},
        {"value": "deadline_change", "label": "截止变化"},
        {"value": "delivery", "label": "交付"},
        {"value": "cancellation", "label": "取消"},
        {"value": "denial", "label": "否认"},
        {"value": "statement", "label": "说明"},
        {"value": "reference", "label": "参考"},
    ]


def _parse_due_date_input(value: Any) -> int | None:
    normalized = _normalize_record_date_input(value)
    if normalized is None:
        return None
    return llm.due_date_to_millis(normalized)


def _first_quoted_phrase(text: str) -> str | None:
    patterns = [
        r"[“\"]([^”\"]+)[”\"]",
        r"[‘']([^’']+)[’']",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip() or None
    return None


def _humanize_user_facing_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = text.strip()
    if not normalized:
        return normalized
    if "anchor_date" in normalized and "换算" in normalized:
        quoted = _first_quoted_phrase(normalized)
        if quoted:
            return f"还无法确定“{quoted}”具体是哪一天。补充这段记录发生的日期后，可以换算成具体日期。"
        return "还无法把这类相对日期换算成具体日期。补充这段记录发生的日期后，可以换算成具体日期。"

    replacements = {
        "anchor_date": "记录发生日期",
        "due_date": "具体日期",
        "due_anchor_at": "日期换算依据",
        "fact_index": "事实序号",
        "event_assignment": "事项归属",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _semantic_anchor_source_label(source: str | None) -> str | None:
    if source == "user":
        return "你填写的"
    if source == "content":
        return "从记录时间中识别"
    return None


def _is_date_resolution_message(text: str | None) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False
    if "anchor_date" in normalized and "换算" in normalized:
        return True
    if "缺少记录发生日期" in normalized:
        return True
    return (
        "还无法确定" in normalized and "具体是哪一天" in normalized
    ) or (
        "补充这段记录发生的日期后" in normalized and "换算" in normalized
    ) or (
        "无法换算" in normalized and "具体日期" in normalized
    )


def _normalize_record_date_input(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        parsed = datetime.strptime(normalized, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="记录发生日期请按 YYYY-MM-DD 填写") from exc
    if not 1900 <= parsed.year <= 2100:
        raise HTTPException(status_code=400, detail="记录发生日期需在 1900-01-01 到 2100-12-31 之间")
    return parsed.strftime("%Y-%m-%d")


def _decode_json_array(value: str | None) -> list[str]:
    if not value:
        return []
    return json.loads(value)


def _open_readonly_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA query_only = ON")
    return conn


def _get_demo_dir() -> Path:
    return Path(os.getenv("WORKCHAIN_DEMO_DIR", "demo_data"))


def _get_sandbox_root() -> Path:
    return Path(os.getenv("WORKCHAIN_SANDBOX_ROOT", "sandboxes"))


def get_conn(sandbox: SandboxContext = Depends(get_sandbox)) -> Generator[sqlite3.Connection, None, None]:
    conn = _open_readonly_connection(sandbox.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _open_write_connection(db_path: Path) -> sqlite3.Connection:
    return init_db(db_path)


def _diagnostics_enabled() -> bool:
    return os.getenv(WORKCHAIN_DIAGNOSTICS_ENV, "").strip() == "1"


def _emit_structured_log(event: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr,
    )


def _provider_label(provider: str | None) -> str:
    mapping = {
        "dashscope": "DashScope",
        "doubao-ark": "Doubao Ark",
        "ark_vision": "Doubao Ark Vision",
        "manual": "人工校正",
        "builtin": "Builtin",
        "deepseek": "DeepSeek",
    }
    if not provider:
        return "未知来源"
    return mapping.get(provider, provider)


def _extraction_origin_label(origin: str | None) -> str:
    if origin == "machine":
        return "机器提取"
    if origin == "user":
        return "人工校正"
    return "未知来源"


def _format_extraction_item(extraction: dict[str, Any]) -> dict[str, Any]:
    origin = extraction.get("origin")
    provider = extraction.get("provider")
    model = extraction.get("model")
    created_at = extraction.get("created_at")
    created_at_text = _format_datetime(created_at, "%Y-%m-%d %H:%M:%S") if isinstance(created_at, int) else None
    summary = _extraction_origin_label(origin)
    provider_label = _provider_label(provider)
    if model:
        summary = f"{summary} · {provider_label} / {model}"
    elif provider:
        summary = f"{summary} · {provider_label}"
    return {
        "origin": origin,
        "origin_label": _extraction_origin_label(origin),
        "provider": provider,
        "provider_label": provider_label,
        "model": model,
        "warnings": extraction.get("warnings") if isinstance(extraction.get("warnings"), list) else [],
        "created_at": created_at,
        "created_at_text": created_at_text,
        "summary": summary,
    }


def _production_image_pipeline_info(current_extraction: dict[str, Any] | None = None) -> dict[str, Any]:
    startup = get_image_extraction_startup()
    warnings = []
    if current_extraction is not None and isinstance(current_extraction.get("warnings"), list):
        warnings = current_extraction["warnings"]
    return {
        "configured_provider": startup["configured_provider"],
        "configured_provider_label": get_image_extraction_provider_label(startup["configured_provider"]),
        "configured_model": startup["configured_model"],
        "actual_provider": None if current_extraction is None else current_extraction.get("provider"),
        "actual_provider_label": None if current_extraction is None else _provider_label(current_extraction.get("provider")),
        "actual_model": None if current_extraction is None else current_extraction.get("model"),
        "fallback_used": ARK_FALLBACK_WARNING in warnings,
        "route": "app.main._run_image_pipeline -> app.evidence_extractor.run_production_image_extraction",
    }


def _truncate_text(value: str, limit: int = 100) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _start_of_current_day_ms() -> int:
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(day_start.timestamp() * 1000)


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _extract_filename(raw_text: str | None) -> str | None:
    if not raw_text or not (raw_text.startswith("[文件] ") or raw_text.startswith("[图片] ")):
        return None
    first_line = raw_text.splitlines()[0]
    return first_line[5:]


def _extract_file_text(raw_text: str | None) -> str | None:
    if not raw_text or not (raw_text.startswith("[文件] ") or raw_text.startswith("[图片] ")):
        return None
    parts = raw_text.split("\n\n", 1)
    if len(parts) != 2:
        return None
    extracted = parts[1].strip()
    return extracted or None


def _build_attachment_raw_text(media_type: str, filename: str | None, extracted_text: str | None = None) -> str:
    label = _build_attachment_label(media_type, filename)
    if extracted_text is None:
        return label
    extracted = extracted_text.strip()
    if not extracted:
        return label
    return f"{label}\n\n{extracted}"


def _is_upload_value(value: Any) -> bool:
    return bool(value) and hasattr(value, "filename") and callable(getattr(value, "read", None))


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            data.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def _detect_upload_type(upload: UploadFile, data: bytes) -> tuple[str, str]:
    content_type = (upload.content_type or "").lower().strip()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if content_type and content_type != "image/png":
            raise HTTPException(status_code=400, detail="文件类型与内容不一致")
        return "image", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        if content_type and content_type != "image/jpeg":
            raise HTTPException(status_code=400, detail="文件类型与内容不一致")
        return "image", "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        if content_type and content_type != "image/gif":
            raise HTTPException(status_code=400, detail="文件类型与内容不一致")
        return "image", "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        if content_type and content_type != "image/webp":
            raise HTTPException(status_code=400, detail="文件类型与内容不一致")
        return "image", "image/webp"
    if data.startswith(b"%PDF-"):
        if content_type and content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="文件类型与内容不一致")
        return "file", "application/pdf"
    if content_type == "text/plain" and _looks_like_text(data):
        return "file", "text/plain"
    if content_type == DOCX_MIME and data.startswith(b"PK\x03\x04"):
        return "file", DOCX_MIME
    raise HTTPException(status_code=400, detail="暂不支持这种文件类型")


def _detect_blob_content_type(blob_bytes: bytes, filename: str | None) -> str:
    if blob_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(blob_bytes) >= 12 and blob_bytes.startswith(b"RIFF") and blob_bytes[8:12] == b"WEBP":
        return "image/webp"
    if blob_bytes.startswith(b"%PDF-"):
        return "application/pdf"
    if _looks_like_text(blob_bytes):
        return "text/plain; charset=utf-8"
    if filename and filename.lower().endswith(".docx") and blob_bytes.startswith(b"PK\x03\x04"):
        return DOCX_MIME
    return "application/octet-stream"


def _current_upload_storage_bytes(conn: sqlite3.Connection, blobs_root: Path) -> int:
    rows = conn.execute(
        """
        SELECT DISTINCT blob_path
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
          AND media_type IN ('image', 'file')
          AND blob_path IS NOT NULL
        """
    ).fetchall()
    total = 0
    for row in rows:
        blob_path = blobs_root / row["blob_path"]
        if blob_path.exists():
            total += blob_path.stat().st_size
    return total


def _ensure_upload_budget(
    conn: sqlite3.Connection,
    blobs_root: Path,
    blob_payloads: Iterable[bytes],
) -> None:
    current_total = _current_upload_storage_bytes(conn, blobs_root)
    additional_bytes = 0
    seen_hashes: set[str] = set()
    for blob_bytes in blob_payloads:
        content_hash = chain.compute_content_hash(blob_bytes)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        blob_path = blobs_root / content_hash[:2] / f"{content_hash}.bin"
        if not blob_path.exists():
            additional_bytes += len(blob_bytes)
    if current_total + additional_bytes > MAX_SANDBOX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="这个沙箱累计上传已超过 50 MB")


def _build_attachment_label(media_type: str, filename: str | None) -> str:
    clean_name = (filename or "未命名文件").strip() or "未命名文件"
    prefix = "[图片]" if media_type == "image" else "[文件]"
    return f"{prefix} {clean_name}"


def _build_content_disposition(disposition: str, filename: str | None) -> str:
    if not filename:
        return disposition
    fallback = filename.encode("ascii", "ignore").decode("ascii") or "file"
    fallback = fallback.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
    return f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{quote(filename, safe="")}'


def _unsupported_detail(media_type: str) -> str:
    return f"这是一张图片/文档,系统暂不能自动读懂它的内容,但原件已完整保存,任何改动都会被发现。"


def _extract_note_key(evidence_id: str) -> str:
    return f"extract_note:{evidence_id}"


def _ocr_corrected_key(evidence_id: str) -> str:
    return f"ocr_corrected:{evidence_id}"


def _set_semantic_anchor(
    conn: sqlite3.Connection,
    evidence_id: str,
    anchor_date: str | None,
    source: str | None,
    *,
    commit: bool = True,
) -> None:
    date_key = _semantic_anchor_date_key(evidence_id)
    source_key = _semantic_anchor_source_key(evidence_id)
    if anchor_date is None:
        conn.execute("DELETE FROM meta WHERE key IN (?, ?)", (date_key, source_key))
        if commit:
            conn.commit()
        return
    _set_meta_value(conn, date_key, anchor_date)
    if source is not None:
        _set_meta_value(conn, source_key, source)
    else:
        conn.execute("DELETE FROM meta WHERE key = ?", (source_key,))
    if commit:
        conn.commit()


def _set_semantic_anchor_date(conn: sqlite3.Connection, evidence_id: str, anchor_date: str | None) -> None:
    _set_semantic_anchor(
        conn,
        evidence_id,
        anchor_date,
        _get_semantic_anchor_source(conn, evidence_id),
    )


def _get_semantic_anchor_date(conn: sqlite3.Connection, evidence_id: str) -> str | None:
    return _get_meta_value(conn, _semantic_anchor_date_key(evidence_id))


def _get_semantic_anchor_source(conn: sqlite3.Connection, evidence_id: str) -> str | None:
    return _get_meta_value(conn, _semantic_anchor_source_key(evidence_id))


def _set_extract_note(conn: sqlite3.Connection, evidence_id: str, note: str) -> None:
    _set_meta_value(conn, _extract_note_key(evidence_id), note)
    conn.commit()


def _get_extract_note(conn: sqlite3.Connection, evidence_id: str) -> str | None:
    return _get_meta_value(conn, _extract_note_key(evidence_id))


def _clear_extract_note(conn: sqlite3.Connection, evidence_id: str) -> None:
    conn.execute("DELETE FROM meta WHERE key = ?", (_extract_note_key(evidence_id),))


def _record_machine_extraction(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    transcript: str | None,
    observations: list[dict[str, Any]] | None,
    provider: str,
    model: str | None,
    warnings: list[str] | None = None,
    created_at: int | None = None,
) -> None:
    create_extraction(
        conn,
        evidence_id=evidence_id,
        origin="machine",
        provider=provider,
        model=model,
        transcript=transcript,
        observations=observations or [],
        warnings=warnings or [],
        created_at=created_at,
    )


def _record_user_extraction(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    transcript: str,
    created_at: int | None = None,
) -> None:
    latest = get_latest_extraction(conn, evidence_id)
    create_extraction(
        conn,
        evidence_id=evidence_id,
        origin="user",
        provider="manual",
        model=None,
        transcript=transcript,
        observations=[],
        warnings=[],
        created_at=created_at,
        supersedes_extraction_id=None if latest is None else latest["extraction_id"],
    )


def _set_ocr_corrected(conn: sqlite3.Connection, evidence_id: str, corrected: bool) -> None:
    _set_meta_value(conn, _ocr_corrected_key(evidence_id), "1" if corrected else "0")
    conn.commit()


def _is_ocr_corrected(conn: sqlite3.Connection, evidence_id: str) -> bool:
    return _get_meta_value(conn, _ocr_corrected_key(evidence_id)) == "1"


def _llm_input_text(text: str) -> str:
    if len(text) <= MAX_LLM_TEXT_LENGTH:
        return text
    prefix = "以下内容较长,已截取前一部分。\n\n"
    return prefix + text[: MAX_LLM_TEXT_LENGTH - len(prefix)]


def _saved_original_detail(note: str) -> str:
    if "原件已完整保存" in note:
        return note
    note = note.rstrip("。")
    return f"{note}。原件已完整保存。"


def _can_run_text_parse(transcript: str | None) -> bool:
    return isinstance(transcript, str) and bool(transcript.strip())


_UNSTABLE_ACTOR_NAMES = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "ta",
    "TA",
    "你们",
    "我们",
    "他们",
    "她们",
    "对方",
    "本人",
    "自己",
    "未知",
    "unknown",
    "actor unknown",
    "某人",
    "有人",
}


def _can_run_semantic_parse(transcript: str | None, observations: list[dict[str, Any]] | None) -> bool:
    return _can_run_text_parse(transcript) or bool(observations)


def _date_to_millis(value: str | None) -> int | None:
    return llm.due_date_to_millis(value)


def _normalize_user_source_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized not in SOURCE_PRESETS and len(normalized) > 20:
        raise HTTPException(status_code=400, detail="自定义来源不能超过 20 个字")
    return normalized


def _platform_detection_snapshot(
    *,
    declared_platform: str | None,
    observed_platform: str | None,
    source_consistency: str,
    platform_confidence: float | None,
) -> dict[str, Any]:
    return {
        "declared_platform": declared_platform,
        "observed_platform": observed_platform or "unknown",
        "source_consistency": source_consistency,
        "platform_confidence": platform_confidence,
    }


def _parse_platform_detection_observation(
    observations: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not isinstance(observations, list):
        return None
    for item in observations:
        if not isinstance(item, dict) or item.get("kind") != "platform_detection":
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        snapshot = _platform_detection_snapshot(
            declared_platform=payload.get("declared_platform"),
            observed_platform=payload.get("observed_platform"),
            source_consistency=str(payload.get("source_consistency") or "unknown"),
            platform_confidence=payload.get("platform_confidence"),
        )
        confidence = snapshot["platform_confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            snapshot["platform_confidence"] = None
        else:
            numeric = float(confidence)
            snapshot["platform_confidence"] = numeric if 0.0 <= numeric <= 1.0 else None
        return snapshot
    return None


def _effective_source_components(source_hint: str | None) -> tuple[str, str]:
    source_hint = source_hint or ""
    source, detail = source_label(source_hint)
    return source, detail


def _build_resolved_source_hint(current_source_hint: str | None, resolved_source: str) -> str:
    _, scene = _effective_source_components(current_source_hint)
    return resolved_source if not scene else f"{resolved_source}-{scene}"


def _effective_source_display(source_hint: str | None) -> dict[str, Any]:
    source, scene = _effective_source_components(source_hint)
    return {
        "source_hint": source_hint,
        "platform": source,
        "scene": scene,
        "platform_class": source_badge_class(source),
    }


def _source_gate_state(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    extraction: dict[str, Any] | None,
) -> dict[str, Any]:
    effective_source_hint = get_effective_source_hint(conn, evidence_id)
    effective_source = _effective_source_display(effective_source_hint)
    detection = None if extraction is None else _parse_platform_detection_observation(extraction.get("observations"))
    latest_review_for_extraction = None
    if extraction is not None:
        latest_review_for_extraction = get_latest_source_review(
            conn,
            evidence_id,
            extraction_id=extraction["extraction_id"],
        )

    if detection is None:
        detection = _platform_detection_snapshot(
            declared_platform=effective_source["platform"],
            observed_platform="unknown",
            source_consistency="unknown",
            platform_confidence=None,
        )

    declared_platform = detection.get("declared_platform")
    observed_platform = detection.get("observed_platform") or "unknown"
    confidence = detection.get("platform_confidence")
    requires_clarification = (
        isinstance(declared_platform, str)
        and declared_platform not in {"", "其他", "unknown"}
        and isinstance(observed_platform, str)
        and observed_platform not in {"", "其他", "unknown"}
        and observed_platform != declared_platform
        and isinstance(confidence, (int, float))
        and confidence >= SOURCE_GATE_PLATFORM_CONFIDENCE_THRESHOLD
        and not (
            latest_review_for_extraction is not None
            and latest_review_for_extraction["decision"] == "confirmed_declared"
        )
    )
    return {
        "requires_clarification": requires_clarification,
        "effective_source_hint": effective_source_hint,
        "effective_platform": effective_source["platform"],
        "effective_scene": effective_source["scene"],
        "declared_platform": declared_platform,
        "observed_platform": observed_platform,
        "source_consistency": detection.get("source_consistency") or "unknown",
        "platform_confidence": confidence,
        "reviewed_for_current_extraction": latest_review_for_extraction is not None,
        "latest_review_for_current_extraction": latest_review_for_extraction,
        "latest_review": get_latest_source_review(conn, evidence_id),
    }


def _clarification_detail(state: dict[str, Any]) -> str:
    declared = state.get("declared_platform") or state.get("effective_platform") or "当前来源"
    observed = state.get("observed_platform") or "unknown"
    return f"等待核实信息来源：你填写的是「{declared}」，机器识别为「{observed}」。"


def _build_source_gate_payload(
    state: dict[str, Any],
    *,
    reviewable: bool,
) -> dict[str, Any]:
    declared = state.get("declared_platform") or state.get("effective_platform") or "其他"
    observed = state.get("observed_platform") or "unknown"
    requires_clarification = bool(state.get("requires_clarification"))
    return {
        "requires_clarification": requires_clarification,
        "reviewable": reviewable,
        "effective_source_hint": state.get("effective_source_hint"),
        "effective_platform": state.get("effective_platform"),
        "effective_scene": state.get("effective_scene"),
        "declared_platform": declared,
        "observed_platform": observed,
        "source_consistency": state.get("source_consistency") or "unknown",
        "platform_confidence": state.get("platform_confidence"),
        "title": "信息来源可能需要核实" if requires_clarification else None,
        "detail": _clarification_detail(state) if requires_clarification else None,
        "prompt": "请确认这份记录实际来自哪里。" if requires_clarification else None,
    }


def _current_source_gate_payload(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    parse_status: str | None = None,
    reviewable: bool | None = None,
) -> dict[str, Any]:
    if parse_status is None:
        parse_status = _get_parse_status(conn, evidence_id)
    latest_extraction = get_latest_extraction(conn, evidence_id)
    state = _source_gate_state(
        conn,
        evidence_id=evidence_id,
        extraction=latest_extraction,
    )
    if reviewable is None:
        reviewable = parse_status == PARSE_STATUS_CLARIFICATION_REQUIRED and not evidence_id.startswith("ev_demo_")
    return _build_source_gate_payload(state, reviewable=reviewable)


def _build_relative_due_updates(
    facts: list[dict[str, Any]],
    *,
    anchor_date: str,
) -> list[dict[str, Any]]:
    anchor_millis = _date_to_millis(anchor_date)
    updates: list[dict[str, Any]] = []
    for fact in facts:
        due_raw = fact.get("due_raw")
        if not semantic_llm.is_relative_due_raw(due_raw):
            continue
        resolved_due_date = semantic_llm.resolve_due_date(due_raw, anchor_date)
        updates.append(
            {
                "fact_id": fact["fact_id"],
                "due_at": _date_to_millis(resolved_due_date),
                "due_anchor_at": anchor_millis if resolved_due_date is not None else None,
            }
        )
    return updates


def _person_glossary_map(glossary: list[dict[str, Any]] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in glossary or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "person":
            continue
        term = item.get("term")
        meaning = item.get("meaning")
        if isinstance(term, str) and term.strip() and isinstance(meaning, str) and meaning.strip():
            mapping[term.strip()] = meaning.strip()
    return mapping


def _normalize_semantic_actor_name(
    name: str | None,
    person_glossary: dict[str, str],
) -> tuple[str | None, str | None]:
    if not isinstance(name, str):
        return None, None
    raw = name.strip()
    if not raw:
        return None, None
    canonical = person_glossary.get(raw, raw).strip()
    lowered = canonical.lower()
    if raw in _UNSTABLE_ACTOR_NAMES or canonical in _UNSTABLE_ACTOR_NAMES:
        return None, None
    if lowered in _UNSTABLE_ACTOR_NAMES or "未知" in canonical or "unknown" in lowered:
        return None, None
    alias = raw if raw != canonical else None
    return canonical, alias


def _resolve_semantic_actor_id(
    conn: sqlite3.Connection,
    *,
    name: str | None,
    person_glossary: dict[str, str],
    created_at: int,
) -> str | None:
    canonical_name, alias = _normalize_semantic_actor_name(name, person_glossary)
    if canonical_name is None:
        return None

    rows = conn.execute(
        "SELECT actor_id, canonical_name, aliases FROM actors ORDER BY created_at ASC, actor_id ASC"
    ).fetchall()
    for row in rows:
        aliases = json.loads(row["aliases"]) if row["aliases"] else []
        if row["canonical_name"] == canonical_name or canonical_name in aliases or (alias and alias in aliases):
            actor_id = row["actor_id"]
            updated_aliases = list(aliases)
            for candidate in (alias, canonical_name):
                if candidate and candidate != row["canonical_name"] and candidate not in updated_aliases:
                    updated_aliases.append(candidate)
            if updated_aliases != aliases:
                conn.execute(
                    "UPDATE actors SET aliases = ? WHERE actor_id = ?",
                    (json.dumps(updated_aliases, ensure_ascii=False, separators=(",", ":")), actor_id),
                    )
            return actor_id

    actor_id = f"act_{uuid.uuid4().hex[:12]}"
    aliases = [] if alias is None else [alias]
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            actor_id,
            canonical_name,
            json.dumps(aliases, ensure_ascii=False, separators=(",", ":")),
            None,
            None,
            0,
            0.5,
            created_at,
        ),
    )
    return actor_id


def _semantic_actor_roles(
    conn: sqlite3.Connection,
    *,
    actors: list[dict[str, Any]] | None,
    glossary: list[dict[str, Any]] | None,
    created_at: int,
) -> list[tuple[str, str]]:
    person_glossary = _person_glossary_map(glossary)
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in actors or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            continue
        actor_id = _resolve_semantic_actor_id(
            conn,
            name=item.get("name"),
            person_glossary=person_glossary,
            created_at=created_at,
        )
        if actor_id is None:
            continue
        key = (actor_id, role.strip())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _semantic_fact_payloads(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    semantics: dict[str, Any],
    glossary: list[dict[str, Any]] | None,
    created_at: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for fact in semantics.get("facts", []):
        if not isinstance(fact, dict):
            continue
        payloads.append(
            {
                "fact_type": fact.get("fact_type"),
                "content": fact.get("content"),
                "evidence_ids": [evidence_id],
                "occurred_at": _date_to_millis(fact.get("occurred_date")),
                "due_at": _date_to_millis(fact.get("due_date")),
                "due_raw": fact.get("due_raw"),
                "due_anchor_at": _date_to_millis(fact.get("due_anchor_date")),
                "confidence": fact.get("confidence"),
                "event_assignment": "unassigned",
                "origin": "ai",
                "review_status": "unreviewed",
                "actor_roles": _semantic_actor_roles(
                    conn,
                    actors=fact.get("actors"),
                    glossary=glossary,
                    created_at=created_at,
                ),
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    return payloads


def _semantic_interpretation_payloads(
    evidence_id: str,
    semantics: dict[str, Any],
    *,
    created_at: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for interpretation in semantics.get("interpretations", []):
        if not isinstance(interpretation, dict):
            continue
        payloads.append(
            {
                "fact_index": interpretation.get("fact_index"),
                "kind": interpretation.get("kind"),
                "content": interpretation.get("content"),
                "confidence": interpretation.get("confidence"),
                "created_at": created_at,
            }
        )
    for ambiguity in semantics.get("ambiguities", []):
        if isinstance(ambiguity, str) and ambiguity.strip():
            payloads.append(
                {
                    "evidence_id": evidence_id,
                    "kind": "uncertainty",
                    "content": ambiguity.strip(),
                    "confidence": None,
                    "created_at": created_at,
                }
            )
    return payloads


def _should_hide_resolved_date_help_item(
    item: dict[str, Any],
    *,
    fact_map: dict[str, dict[str, Any]],
    has_relative_due_raw: bool,
    anchor_date: str | None,
) -> bool:
    if anchor_date is None or item.get("kind") != "uncertainty":
        return False
    raw_content = item.get("raw_content")
    if not _is_date_resolution_message(raw_content):
        return False
    fact_id = item.get("fact_id")
    if fact_id is None:
        return has_relative_due_raw
    related_fact = fact_map.get(fact_id)
    if related_fact is None:
        return has_relative_due_raw
    return semantic_llm.is_relative_due_raw(related_fact.get("due_raw"))


def _build_semantic_result(conn: sqlite3.Connection, evidence_id: str) -> dict[str, Any] | None:
    run = get_latest_semantic_run_for_evidence(conn, evidence_id, status="succeeded")
    if run is None:
        return None

    facts = list_facts_for_semantic_run(conn, run["semantic_run_id"], evidence_id=evidence_id)
    interpretations = list_interpretations_for_semantic_run(
        conn,
        run["semantic_run_id"],
        evidence_id=evidence_id,
    )
    fact_map = {fact["fact_id"]: fact for fact in facts}
    anchor_date = _get_semantic_anchor_date(conn, evidence_id)
    anchor_source = _get_semantic_anchor_source(conn, evidence_id)
    has_relative_due_raw = any(
        semantic_llm.is_relative_due_raw(fact.get("due_raw"))
        for fact in facts
    )
    grouped_help_items: dict[str, list[dict[str, Any]]] = {fact["fact_id"]: [] for fact in facts}
    help_items = []
    evidence_help_items = []
    filtered_stale_date_help_items = 0
    for item in interpretations:
        help_item = {
            "kind": item["kind"],
            "label": _friendly_interpretation_label(item["kind"]),
            "content": _humanize_user_facing_text(item["content"]) or "",
            "raw_content": item["content"],
            "confidence": item["confidence"],
            "is_uncertainty": item["kind"] == "uncertainty",
            "fact_id": item["fact_id"],
            "fact_content": None if item["fact_id"] is None else fact_map.get(item["fact_id"], {}).get("content"),
        }
        if _should_hide_resolved_date_help_item(
            help_item,
            fact_map=fact_map,
            has_relative_due_raw=has_relative_due_raw,
            anchor_date=anchor_date,
        ):
            filtered_stale_date_help_items += 1
            continue
        public_help_item = {
            "kind": help_item["kind"],
            "label": help_item["label"],
            "content": help_item["content"],
            "confidence": help_item["confidence"],
            "is_uncertainty": help_item["is_uncertainty"],
            "fact_id": help_item["fact_id"],
            "fact_content": help_item["fact_content"],
        }
        help_items.append(public_help_item)
        fact_id = help_item["fact_id"]
        if isinstance(fact_id, str) and fact_id in grouped_help_items:
            grouped_help_items[fact_id].append(public_help_item)
        else:
            evidence_help_items.append(public_help_item)
    event_match = _build_event_match_result(conn, run["semantic_run_id"])

    return {
        "semantic_run_id": run["semantic_run_id"],
        "provider": run["provider"],
        "provider_label": _provider_label(run["provider"]),
        "model": run["model"],
        "parser_version": run["parser_version"],
        "created_at": run["created_at"],
        "created_at_text": _format_datetime(run["created_at"], "%Y-%m-%d %H:%M:%S"),
        "facts": [
            {
                "fact_id": fact["fact_id"],
                "fact_type": fact["fact_type"],
                "fact_type_label": _event_fact_type_label(fact["fact_type"]),
                "content": fact["content"],
                "due_raw": fact["due_raw"],
                "due_date_value": _format_datetime(fact["due_at"], "%Y-%m-%d"),
                "help_items": grouped_help_items.get(fact["fact_id"], []),
            }
            for fact in facts
        ],
        "help_items": help_items,
        "evidence_help_items": evidence_help_items,
        "event_match": event_match,
        "has_relative_due_raw": has_relative_due_raw,
        "record_date": anchor_date,
        "record_date_source": anchor_source,
        "record_date_source_label": _semantic_anchor_source_label(anchor_source),
        "date_resolution_notice": None if not (anchor_date and has_relative_due_raw) else f"已按 {anchor_date} 换算相对日期。",
        "filtered_stale_date_help_items": filtered_stale_date_help_items,
    }


def _event_title(conn: sqlite3.Connection, event_id: str | None) -> str | None:
    if not event_id:
        return None
    row = conn.execute(
        "SELECT title FROM events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        return None
    return row["title"]


def _event_match_fact_items(
    conn: sqlite3.Connection,
    fact_ids: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fact_id in fact_ids:
        row = conn.execute(
            """
            SELECT f.fact_id, f.fact_type, f.content, f.event_id, f.event_assignment, ev.title AS event_title
            FROM facts f
            LEFT JOIN events ev ON ev.event_id = f.event_id
            WHERE f.fact_id = ?
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            continue
        items.append(
            {
                "fact_id": row["fact_id"],
                "fact_type": row["fact_type"],
                "fact_type_label": _event_fact_type_label(row["fact_type"]),
                "content": row["content"],
                "event_id": row["event_id"],
                "event_title": row["event_title"],
                "event_assignment": row["event_assignment"],
            }
        )
    return items


def _build_event_match_groups(
    conn: sqlite3.Connection,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_groups = result.get("groups")
    if not isinstance(raw_groups, list):
        return []

    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            continue
        fact_ids = group.get("fact_ids")
        if not isinstance(fact_ids, list):
            continue
        normalized_fact_ids = [item for item in fact_ids if isinstance(item, str) and item.strip()]
        event_id = group.get("event_id") if isinstance(group.get("event_id"), str) else None
        proposed_title = group.get("proposed_title") if isinstance(group.get("proposed_title"), str) else None
        target = group.get("target") if isinstance(group.get("target"), str) else "unassigned"
        default_choice = "unassigned"
        if target == "existing" and event_id:
            default_choice = "existing"
        elif target == "new":
            default_choice = "new"
        groups.append(
            {
                "group_index": group_index,
                "ai_title": _event_title(conn, event_id) or proposed_title or "AI 未给出标题",
                "reason": _humanize_user_facing_text(group.get("reason")) if isinstance(group.get("reason"), str) else "",
                "fact_items": _event_match_fact_items(conn, normalized_fact_ids),
                "default_choice": default_choice,
                "default_event_id": event_id,
                "default_new_title": proposed_title or "",
            }
        )
    return groups


def _build_completed_group_outcome(group: dict[str, Any]) -> dict[str, Any]:
    fact_items = group.get("fact_items", [])
    assigned_titles = []
    seen_titles = set()
    for item in fact_items:
        if not isinstance(item, dict):
            continue
        title = item.get("event_title")
        if isinstance(title, str) and title and title not in seen_titles:
            seen_titles.add(title)
            assigned_titles.append(title)
    if not assigned_titles:
        return {
            "group_index": group["group_index"],
            "label": "暂不归入事项",
        }
    return {
        "group_index": group["group_index"],
        "label": f"已归入事项：{' / '.join(assigned_titles)}",
    }


def _build_event_match_result(conn: sqlite3.Connection, semantic_run_id: str) -> dict[str, Any] | None:
    run = get_latest_event_match_for_semantic_run(conn, semantic_run_id)
    if run is None:
        return None
    result = run.get("result")
    if run["status"] == "failed":
        return {
            "status": "failed",
            "routing_mode": None,
            "review_status": "pending",
            "event_match_run_id": run["event_match_run_id"],
            "message": "事项归属暂不可用，事实整理结果已保存。",
            "reason": None,
            "suggestions": [],
            "groups": [],
            "completed_groups": [],
            "active_events": [],
        }
    if not isinstance(result, dict):
        return None

    groups = _build_event_match_groups(conn, result)
    active_events = [
        {"event_id": item["event_id"], "title": item["title"]}
        for item in list_event_candidates(conn, recent_facts_per_event=0)
    ]

    routing_mode = run.get("routing_mode")
    if routing_mode == "auto":
        auto_group = next(
            (
                group
                for group in groups
                if group["default_choice"] in {"existing", "new"}
            ),
            None,
        )
        return {
            "status": "succeeded",
            "routing_mode": "auto",
            "review_status": run.get("review_status"),
            "event_match_run_id": run["event_match_run_id"],
            "message": None if auto_group is None else f"已自动归入事项：{auto_group['ai_title']}",
            "reason": None if auto_group is None else auto_group.get("reason"),
            "suggestions": [],
            "groups": groups,
            "completed_groups": [],
            "active_events": active_events,
        }
    if run.get("review_status") == "completed" and routing_mode in {"confirm", "needs_context"}:
        return {
            "status": "succeeded",
            "routing_mode": routing_mode,
            "review_status": "completed",
            "event_match_run_id": run["event_match_run_id"],
            "message": "事项归属已确认",
            "reason": None,
            "suggestions": [],
            "groups": groups,
            "completed_groups": [_build_completed_group_outcome(group) for group in groups],
            "active_events": active_events,
        }
    if routing_mode == "confirm":
        return {
            "status": "succeeded",
            "routing_mode": "confirm",
            "review_status": run.get("review_status"),
            "event_match_run_id": run["event_match_run_id"],
            "message": "AI认为这属于以下事项，请你确认。"
            if len(groups) == 1
            else "这段记录可能涉及多个事项，请分别确认。",
            "reason": None,
            "suggestions": [],
            "groups": groups,
            "completed_groups": [],
            "active_events": active_events,
        }
    if routing_mode == "needs_context":
        return {
            "status": "succeeded",
            "routing_mode": "needs_context",
            "review_status": run.get("review_status"),
            "event_match_run_id": run["event_match_run_id"],
            "message": "暂时无法可靠判断属于哪件事，请选择归属。",
            "reason": None,
            "suggestions": [],
            "groups": groups,
            "completed_groups": [],
            "active_events": active_events,
        }
    return None


def _should_show_event_assignment_panel(event_match: dict[str, Any] | None) -> bool:
    if not isinstance(event_match, dict):
        return False
    return (
        event_match.get("review_status") == "pending"
        and event_match.get("routing_mode") in {"confirm", "needs_context"}
        and bool(event_match.get("groups"))
    )


def _render_event_assignment_panel_html(
    *,
    event_match: dict[str, Any] | None,
    evidence_id: str,
) -> str | None:
    if not _should_show_event_assignment_panel(event_match):
        return None
    return TEMPLATES.get_template("_event_assignment_panel.html").render(
        panel_event_match=event_match,
        panel_assignment_subject={"evidence_id": evidence_id},
    )


def _event_matcher_fact_payload(conn: sqlite3.Connection, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for fact in facts:
        actor_rows = conn.execute(
            """
            SELECT a.canonical_name, fa.role
            FROM fact_actors fa
            JOIN actors a ON a.actor_id = fa.actor_id
            WHERE fa.fact_id = ?
            ORDER BY fa.role ASC, a.canonical_name ASC
            """,
            (fact["fact_id"],),
        ).fetchall()
        payloads.append(
            {
                "fact_type": fact["fact_type"],
                "content": fact["content"],
                "confidence": fact["confidence"],
                "actors": [
                    {"name": row["canonical_name"], "role": row["role"]}
                    for row in actor_rows
                ],
                "occurred_date": None if fact["occurred_at"] is None else _format_datetime(fact["occurred_at"], "%Y-%m-%d"),
                "due_raw": fact["due_raw"],
                "due_date": None if fact["due_at"] is None else _format_datetime(fact["due_at"], "%Y-%m-%d"),
                "due_anchor_date": None if fact["due_anchor_at"] is None else _format_datetime(fact["due_anchor_at"], "%Y-%m-%d"),
            }
        )
    return payloads


def _event_change_fact_payload(
    conn: sqlite3.Connection,
    event_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    fact_rows = conn.execute(
        """
        WITH recent_facts AS (
            SELECT
                f.fact_id,
                f.fact_type,
                f.content,
                f.occurred_at,
                f.created_at,
                f.due_at,
                f.due_raw,
                (
                    SELECT fe.evidence_id
                    FROM fact_evidence fe
                    JOIN evidence e ON e.evidence_id = fe.evidence_id
                    WHERE fe.fact_id = f.fact_id
                    ORDER BY e.seq ASC, e.evidence_id ASC
                    LIMIT 1
                ) AS evidence_id
            FROM facts f
            WHERE f.event_id = ?
            ORDER BY COALESCE(f.occurred_at, f.created_at) DESC, f.created_at DESC, f.fact_id DESC
            LIMIT ?
        )
        SELECT *
        FROM recent_facts
        ORDER BY COALESCE(occurred_at, created_at) ASC, created_at ASC, fact_id ASC
        """,
        (event_id, limit),
    ).fetchall()

    payloads: list[dict[str, Any]] = []
    for row in fact_rows:
        actor_rows = conn.execute(
            """
            SELECT a.canonical_name, fa.role
            FROM fact_actors fa
            JOIN actors a ON a.actor_id = fa.actor_id
            WHERE fa.fact_id = ?
            ORDER BY fa.role ASC, a.canonical_name ASC
            """,
            (row["fact_id"],),
        ).fetchall()
        evidence_id = row["evidence_id"]
        if evidence_id is None:
            continue
        payloads.append(
            {
                "fact_id": row["fact_id"],
                "fact_type": row["fact_type"],
                "content": row["content"],
                "occurred_at": row["occurred_at"],
                "due_at": row["due_at"],
                "due_raw": row["due_raw"],
                "occurred_date": _format_datetime(row["occurred_at"], "%Y-%m-%d"),
                "due_date": _format_datetime(row["due_at"], "%Y-%m-%d"),
                "evidence_id": evidence_id,
                "actors": [
                    {"name": actor_row["canonical_name"], "role": actor_row["role"]}
                    for actor_row in actor_rows
                ],
            }
        )
    return payloads


def _maybe_run_event_change_detection(conn: sqlite3.Connection, event_id: str) -> None:
    fact_payloads = _event_change_fact_payload(conn, event_id, limit=20)
    evidence_ids = {item["evidence_id"] for item in fact_payloads if item.get("evidence_id")}
    if len(fact_payloads) < 2 or len(evidence_ids) < 2:
        return

    change_run = None
    try:
        change_run = create_event_change_run(
            conn,
            event_id=event_id,
            provider="deepseek",
            model=get_text_model(),
            detector_version=change_detector.CHANGE_DETECTOR_VERSION,
        )
        detection = change_detector.detect_changes_diagnostic_result(event_id, fact_payloads)
        if detection["result"] is None:
            mark_event_change_run_failed(
                conn,
                change_run_id=change_run["change_run_id"],
                failure_type=change_detector.change_failure_type_from_diagnostic(detection["diagnostic"]),
            )
            return
        persist_event_change_run_result(
            conn,
            change_run_id=change_run["change_run_id"],
            event_id=event_id,
            changes=detection["result"]["changes"],
        )
    except Exception:
        if change_run is not None:
            try:
                mark_event_change_run_failed(
                    conn,
                    change_run_id=change_run["change_run_id"],
                    failure_type="persistence_error",
                )
            except Exception:
                pass


def _event_change_type_label(change_type: str) -> str:
    return {
        "requirement_change": "要求发生变化",
        "deadline_change": "截止时间发生变化",
        "responsibility_change": "负责人发生变化",
        "contradiction": "前后记录存在不一致",
    }.get(change_type, "记录发生变化")


def _export_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _pdf_download_name() -> str:
    return f"{PDF_FILENAME_PREFIX}-{_export_timestamp()}.pdf"


def _package_download_name() -> str:
    return f"{PDF_FILENAME_PREFIX}-{_export_timestamp()}.zip"


def _cleanup_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _mine_evidence_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT evidence_id
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
        ORDER BY occurred_at ASC, seq ASC
        """
    ).fetchall()
    return [row["evidence_id"] for row in rows]


def _thread_exists(conn: sqlite3.Connection, thread_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    return row is not None


async def _parse_evidence_input(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploads: list[dict[str, Any]] = []
        for form_value in form.getlist("file"):
            upload = form_value if _is_upload_value(form_value) else None
            if upload is None:
                continue
            file_bytes = await upload.read()
            if not file_bytes and not upload.filename:
                continue
            if len(file_bytes) > MAX_FILE_BYTES:
                raise HTTPException(status_code=400, detail="单个文件不能超过 8 MB")
            detected_media_type, detected_content_type = _detect_upload_type(upload, file_bytes)
            uploads.append(
                {
                    "upload": upload,
                    "file_bytes": file_bytes,
                    "media_type": detected_media_type,
                    "file_content_type": detected_content_type,
                    "filename": upload.filename or None,
                }
            )
        return {
            "text": str(form.get("text", "")).strip(),
            "source": str(form.get("source", "")).strip(),
            "source_detail": str(form.get("source_detail", "")).strip() or None,
            "counterpart": str(form.get("counterpart", "")).strip() or None,
            "record_date": _normalize_record_date_input(form.get("record_date")),
            "uploads": uploads,
        }

    payload = await request.json()
    return {
        "text": str(payload.get("text", "")).strip(),
        "source": str(payload.get("source", "")).strip(),
        "source_detail": None if payload.get("source_detail") is None else str(payload.get("source_detail")).strip() or None,
        "counterpart": None if payload.get("counterpart") is None else str(payload.get("counterpart")).strip() or None,
        "record_date": _normalize_record_date_input(payload.get("record_date")),
        "uploads": [],
    }


def _safe_diag_detail(message: str, api_key: str) -> str:
    detail = message.strip() or "unknown error"
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    return detail


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()


def _open_meta_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_meta_table(conn)
    return conn


def _global_meta_db_path() -> Path:
    return _get_sandbox_root() / "_global_meta.db"


def _set_meta_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _get_meta_value(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def _parse_status_key(evidence_id: str) -> str:
    return f"parse_status:{evidence_id}"


def _parse_detail_key(evidence_id: str) -> str:
    return f"parse_detail:{evidence_id}"


def _semantic_diagnostic_key(semantic_run_id: str) -> str:
    return f"semantic_diagnostic:{semantic_run_id}"


def _semantic_anchor_date_key(evidence_id: str) -> str:
    return f"semantic_anchor_date:{evidence_id}"


def _semantic_anchor_source_key(evidence_id: str) -> str:
    return f"semantic_anchor_source:{evidence_id}"


def _verified_key(evidence_id: str) -> str:
    return f"verified:{evidence_id}"


def _parse_count_key(today: str) -> str:
    return f"parse_count:{today}"


def _global_parse_count_key(today: str) -> str:
    return f"global_parse_count:{today}"


def _ocr_count_key(today: str) -> str:
    return f"ocr_count:{today}"


def _global_ocr_count_key(today: str) -> str:
    return f"global_ocr_count:{today}"


def _set_parse_status(conn: sqlite3.Connection, evidence_id: str, status: str) -> None:
    _set_meta_value(conn, _parse_status_key(evidence_id), status)
    conn.commit()


def _set_parse_detail(conn: sqlite3.Connection, evidence_id: str, detail: str) -> None:
    _set_meta_value(conn, _parse_detail_key(evidence_id), detail)
    conn.commit()


def _get_parse_status(conn: sqlite3.Connection, evidence_id: str) -> str:
    return _get_meta_value(conn, _parse_status_key(evidence_id)) or PARSE_STATUS_FAILED


def _get_parse_detail(conn: sqlite3.Connection, evidence_id: str) -> str | None:
    return _get_meta_value(conn, _parse_detail_key(evidence_id))


def _set_semantic_diagnostic(
    conn: sqlite3.Connection,
    semantic_run_id: str,
    diagnostic: dict[str, Any] | None,
) -> None:
    if diagnostic is None:
        return
    safe_diagnostic = {
        "success": bool(diagnostic.get("success")),
        "stage": diagnostic.get("stage"),
        "status_code": diagnostic.get("status_code"),
        "error_code": diagnostic.get("error_code"),
        "error_type": diagnostic.get("error_type"),
        "safe_message": diagnostic.get("safe_message"),
        "request_id": diagnostic.get("request_id"),
        "latency_ms": diagnostic.get("latency_ms"),
        "timeout_seconds": diagnostic.get("timeout_seconds"),
        "thinking_mode": diagnostic.get("thinking_mode"),
        "model": diagnostic.get("model"),
    }
    _set_meta_value(
        conn,
        _semantic_diagnostic_key(semantic_run_id),
        json.dumps(safe_diagnostic, ensure_ascii=False, separators=(",", ":")),
    )
    conn.commit()


def _get_semantic_diagnostic(conn: sqlite3.Connection, semantic_run_id: str | None) -> dict[str, Any] | None:
    if not semantic_run_id:
        return None
    raw = _get_meta_value(conn, _semantic_diagnostic_key(semantic_run_id))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _semantic_failure_type_from_diagnostic(diagnostic: dict[str, Any] | None) -> str:
    if not diagnostic:
        return "provider_invalid_response"
    stage = diagnostic.get("stage")
    status_code = diagnostic.get("status_code")
    if stage == "config":
        return "provider_config"
    if stage == "network":
        return "provider_network"
    if stage == "timeout":
        return "provider_timeout"
    if stage == "http":
        if isinstance(status_code, int):
            return f"provider_http_{status_code}"
        return "provider_http"
    if stage == "empty_content":
        return "provider_empty_content"
    if stage == "model_json":
        return "semantic_invalid_json"
    if stage in {"response_json", "output_text"}:
        return "provider_invalid_response"
    return "provider_invalid_response"


def _semantic_failure_detail(diagnostic: dict[str, Any] | None) -> str:
    if not diagnostic:
        return "解析暂不可用,记录已完整保存"
    safe_message = diagnostic.get("safe_message")
    if isinstance(safe_message, str) and safe_message.strip():
        return safe_message
    failure_type = _semantic_failure_type_from_diagnostic(diagnostic)
    if failure_type == "provider_timeout":
        timeout_seconds = diagnostic.get("timeout_seconds")
        if isinstance(timeout_seconds, (int, float)):
            return f"DeepSeek 请求超时（{timeout_seconds:g} 秒）"
    if failure_type.startswith("provider_http_"):
        return f"DeepSeek 接口返回 {failure_type.removeprefix('provider_http_')}"
    if failure_type == "provider_empty_content":
        return "DeepSeek 返回成功,但内容为空"
    if failure_type == "semantic_invalid_json":
        return "DeepSeek 已返回内容,但 Semantic JSON 无法解析"
    return "解析暂不可用,记录已完整保存"


def _set_verified(conn: sqlite3.Connection, evidence_id: str) -> None:
    _set_meta_value(conn, _verified_key(evidence_id), "1")
    conn.commit()


def _is_verified(conn: sqlite3.Connection, evidence_id: str) -> bool:
    return _get_meta_value(conn, _verified_key(evidence_id)) == "1"


def _get_verified_map(conn: sqlite3.Connection, evidence_ids: Iterable[str]) -> dict[str, bool]:
    ids = [evidence_id for evidence_id in evidence_ids]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    keys = [_verified_key(evidence_id) for evidence_id in ids]
    rows = conn.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})",
        keys,
    ).fetchall()
    result = {row["key"].split(":", 1)[1]: row["value"] == "1" for row in rows}
    for evidence_id in ids:
        result.setdefault(evidence_id, False)
    return result


def _get_parse_status_map(conn: sqlite3.Connection, evidence_ids: Iterable[str]) -> dict[str, str]:
    ids = [evidence_id for evidence_id in evidence_ids]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    keys = [_parse_status_key(evidence_id) for evidence_id in ids]
    rows = conn.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})",
        keys,
    ).fetchall()
    result = {row["key"].split(":", 1)[1]: row["value"] for row in rows}
    for evidence_id in ids:
        result.setdefault(evidence_id, PARSE_STATUS_FAILED)
    return result


def _get_parse_detail_map(conn: sqlite3.Connection, evidence_ids: Iterable[str]) -> dict[str, str | None]:
    ids = [evidence_id for evidence_id in evidence_ids]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    keys = [_parse_detail_key(evidence_id) for evidence_id in ids]
    rows = conn.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})",
        keys,
    ).fetchall()
    result = {row["key"].split(":", 1)[1]: row["value"] for row in rows}
    for evidence_id in ids:
        result.setdefault(evidence_id, None)
    return result


def _get_extract_note_map(conn: sqlite3.Connection, evidence_ids: Iterable[str]) -> dict[str, str | None]:
    ids = [evidence_id for evidence_id in evidence_ids]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    keys = [_extract_note_key(evidence_id) for evidence_id in ids]
    rows = conn.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})",
        keys,
    ).fetchall()
    result = {row["key"].split(":", 1)[1]: row["value"] for row in rows}
    for evidence_id in ids:
        result.setdefault(evidence_id, None)
    return result


def _try_increment_meta_counter(conn: sqlite3.Connection, key: str, limit: int) -> bool:
    conn.execute("BEGIN IMMEDIATE")
    try:
        current_raw = _get_meta_value(conn, key)
        current = 0 if current_raw is None else int(current_raw)
        if current >= limit:
            conn.rollback()
            return False
        _set_meta_value(conn, key, str(current + 1))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def _rollback_meta_counter(db_path: Path, key: str, *, global_meta: bool = False) -> None:
    conn = _open_meta_connection(db_path) if global_meta else init_db(db_path)
    try:
        current_raw = _get_meta_value(conn, key)
        current = 0 if current_raw is None else int(current_raw)
        _set_meta_value(conn, key, str(max(0, current - 1)))
        conn.commit()
    finally:
        conn.close()


def _consume_daily_budget(
    sandbox_db_path: Path,
    global_meta_db_path: Path,
    *,
    sandbox_key: str,
    global_key: str,
    sandbox_limit: int,
    global_limit: int,
    failure_detail: str,
) -> tuple[bool, str | None]:
    sandbox_conn = init_db(sandbox_db_path)
    try:
        if not _try_increment_meta_counter(sandbox_conn, sandbox_key, sandbox_limit):
            return False, failure_detail
    finally:
        sandbox_conn.close()

    global_conn = _open_meta_connection(global_meta_db_path)
    try:
        if not _try_increment_meta_counter(global_conn, global_key, global_limit):
            _rollback_meta_counter(sandbox_db_path, sandbox_key)
            return False, failure_detail
    finally:
        global_conn.close()

    return True, None


def _consume_parse_budget(sandbox_db_path: Path, global_meta_db_path: Path) -> tuple[bool, str | None]:
    today = _today_str()
    return _consume_daily_budget(
        sandbox_db_path,
        global_meta_db_path,
        sandbox_key=_parse_count_key(today),
        global_key=_global_parse_count_key(today),
        sandbox_limit=20,
        global_limit=300,
        failure_detail="今日解析次数已用完,记录仍已保存",
    )


def _consume_ocr_budget(sandbox_db_path: Path, global_meta_db_path: Path) -> tuple[bool, str | None]:
    today = _today_str()
    return _consume_daily_budget(
        sandbox_db_path,
        global_meta_db_path,
        sandbox_key=_ocr_count_key(today),
        global_key=_global_ocr_count_key(today),
        sandbox_limit=20,
        global_limit=300,
        failure_detail="今日图片识别次数已用完,原件已完整保存",
    )


def _final_kind(parsed: dict[str, Any], slot_requester: str | None, slot_owner: str | None, slot_due: int | None) -> str:
    filled = sum(
        value is not None
        for value in (slot_requester, slot_owner, parsed.get("deliverable"), slot_due)
    )
    if filled >= 3 and parsed.get("direction") != "none":
        return parsed.get("kind", "reference")
    return "reference"


def _run_parse_pipeline(
    sandbox_db_path: Path,
    global_meta_db_path: Path,
    evidence_id: str,
) -> None:
    conn = init_db(sandbox_db_path)
    semantic_run_id: str | None = None
    parse_start = time.perf_counter()
    evidence_row = None
    extraction = None
    transcript = None
    observations: list[dict[str, Any]] = []
    glossary: list[dict[str, Any]] = []
    anchor_date: str | None = None
    anchor_source: str | None = None
    effective_source_hint: str | None = None
    try:
        evidence_row = conn.execute(
            "SELECT evidence_id FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        extraction = get_latest_extraction(conn, evidence_id)
        if evidence_row is not None:
            glossary = get_settings(sandbox_db_path).get("glossary", [])
            stored_anchor_date = _get_semantic_anchor_date(conn, evidence_id)
            stored_anchor_source = _get_semantic_anchor_source(conn, evidence_id)
            effective_source_hint = get_effective_source_hint(conn, evidence_id)
        if extraction is not None:
            transcript = extraction.get("transcript")
            observations = extraction.get("observations") if isinstance(extraction.get("observations"), list) else []
        if evidence_row is not None:
            if stored_anchor_source == "user" and stored_anchor_date is not None:
                anchor_date = stored_anchor_date
                anchor_source = "user"
            else:
                anchor_date = semantic_llm.infer_reliable_anchor_date(transcript, observations=observations)
                anchor_source = None if anchor_date is None else "content"
                _set_semantic_anchor(conn, evidence_id, anchor_date, anchor_source)
    finally:
        conn.close()

    log_base = {
        "evidence_id": evidence_id,
        "provider": "deepseek",
        "model": get_text_model(),
        "input_chars": len(transcript or ""),
        "observation_count": len(observations),
    }
    if evidence_row is None:
        return
    should_call_model = _can_run_semantic_parse(transcript, observations)

    previous_run = None
    conn = init_db(sandbox_db_path)
    try:
        previous_run = get_latest_semantic_run_for_evidence(conn, evidence_id)
        if extraction is None:
            _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
            _set_parse_detail(conn, evidence_id, "解析缺少提取版本,记录已完整保存")
            return

        semantic_run = create_semantic_run(
            conn,
            provider="deepseek",
            model=get_text_model(),
            parser_version=semantic_llm.SEMANTIC_PARSER_VERSION,
            anchor_date=anchor_date,
            supersedes_run_id=None if previous_run is None else previous_run["semantic_run_id"],
            inputs=[
                {
                    "evidence_id": evidence_id,
                    "extraction_id": extraction["extraction_id"],
                    "position": 0,
                }
            ],
        )
        semantic_run_id = semantic_run["semantic_run_id"]
    finally:
        conn.close()

    _emit_structured_log(
        "semantic_parse",
        {
            **log_base,
            "semantic_run_id": semantic_run_id,
            "status": "started",
            "latency_ms": 0,
            "parse_success": False,
            "failure_type": None,
        },
    )
    if not should_call_model:
        conn = init_db(sandbox_db_path)
        try:
            if semantic_run_id is not None:
                _set_semantic_diagnostic(
                    conn,
                    semantic_run_id,
                    {
                        "success": True,
                        "stage": "success",
                        "status_code": None,
                        "error_code": None,
                        "error_type": None,
                        "safe_message": "No transcript or visual observations; semantic parse short-circuited",
                        "request_id": None,
                        "latency_ms": 0,
                        "timeout_seconds": get_text_timeout_seconds(),
                        "thinking_mode": "disabled",
                        "model": get_text_model(),
                    },
                )
            fact_payloads = _semantic_fact_payloads(
                conn,
                evidence_id=evidence_id,
                semantics={"facts": [], "interpretations": [], "ambiguities": []},
                glossary=glossary,
                created_at=int(time.time() * 1000),
            )
            interpretation_payloads = _semantic_interpretation_payloads(
                evidence_id,
                {"facts": [], "interpretations": [], "ambiguities": []},
                created_at=int(time.time() * 1000),
            )
            persist_semantic_run_result(
                conn,
                semantic_run_id=semantic_run_id,
                facts=fact_payloads,
                interpretations=interpretation_payloads,
            )
            _set_parse_status(conn, evidence_id, PARSE_STATUS_DONE)
            _set_parse_detail(conn, evidence_id, "")
        except Exception:
            try:
                if semantic_run_id is not None:
                    mark_semantic_run_failed(
                        conn,
                        semantic_run_id=semantic_run_id,
                        failure_type="persistence_error",
                    )
            except Exception:
                pass
            _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
            _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
            _emit_structured_log(
                "semantic_parse",
                {
                    **log_base,
                    "semantic_run_id": semantic_run_id,
                    "status": "failed",
                    "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                    "parse_success": False,
                    "failure_type": "persistence_error",
                },
            )
        else:
            _emit_structured_log(
                "semantic_parse",
                {
                    **log_base,
                    "semantic_run_id": semantic_run_id,
                    "status": "succeeded",
                    "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                    "parse_success": True,
                    "failure_type": None,
                },
            )
        finally:
            conn.close()
        return

    if not get_text_api_key():
        diagnostic = build_text_config_diagnostic()
        conn = init_db(sandbox_db_path)
        try:
            if semantic_run_id is not None:
                _set_semantic_diagnostic(conn, semantic_run_id, diagnostic)
                try:
                    mark_semantic_run_failed(
                        conn,
                        semantic_run_id=semantic_run_id,
                        failure_type=_semantic_failure_type_from_diagnostic(diagnostic),
                    )
                except Exception:
                    pass
            _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
            _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        _emit_structured_log(
            "semantic_parse",
            {
                **log_base,
                "semantic_run_id": semantic_run_id,
                "status": "failed",
                "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                "parse_success": False,
                "failure_type": _semantic_failure_type_from_diagnostic(diagnostic),
            },
        )
        return

    allowed, reason = _consume_parse_budget(sandbox_db_path, global_meta_db_path)
    if not allowed:
        conn = init_db(sandbox_db_path)
        try:
            if semantic_run_id is not None:
                try:
                    mark_semantic_run_failed(
                        conn,
                        semantic_run_id=semantic_run_id,
                        failure_type="budget_exhausted",
                    )
                except Exception:
                    pass
            _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
            _set_parse_detail(conn, evidence_id, reason or "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        _emit_structured_log(
            "semantic_parse",
            {
                **log_base,
                "semantic_run_id": semantic_run_id,
                "status": "failed",
                "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                "parse_success": False,
                "failure_type": "budget_exhausted",
            },
        )
        return

    diagnostic = None
    try:
        parsed = semantic_llm.extract_semantics(
            _llm_input_text(transcript) if transcript is not None else None,
            observations=observations,
            anchor_date=anchor_date,
            glossary=glossary,
            source_hint=effective_source_hint,
        )
        diagnostic = semantic_llm.pop_last_extract_diagnostic()
    except Exception as exc:
        diagnostic = semantic_llm.pop_last_extract_diagnostic()
        if diagnostic is None:
            if isinstance(exc, httpx.TimeoutException):
                diagnostic = {
                    "success": False,
                    "stage": "timeout",
                    "status_code": None,
                    "error_code": "timeout",
                    "error_type": type(exc).__name__,
                    "safe_message": f"DeepSeek request timed out after {get_text_timeout_seconds():g} seconds",
                    "request_id": None,
                    "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                    "timeout_seconds": get_text_timeout_seconds(),
                    "thinking_mode": "disabled",
                    "model": get_text_model(),
                }
            else:
                diagnostic = {
                    "success": False,
                    "stage": "network",
                    "status_code": None,
                    "error_code": "request_error",
                    "error_type": type(exc).__name__,
                    "safe_message": "DeepSeek request failed before a safe provider diagnostic was available",
                    "request_id": None,
                    "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                    "timeout_seconds": get_text_timeout_seconds(),
                    "thinking_mode": "disabled",
                    "model": get_text_model(),
                }
        parsed = None
    if parsed is None:
        failure_type = _semantic_failure_type_from_diagnostic(diagnostic)
        conn = init_db(sandbox_db_path)
        try:
            if semantic_run_id is not None:
                _set_semantic_diagnostic(conn, semantic_run_id, diagnostic)
                try:
                    mark_semantic_run_failed(
                        conn,
                        semantic_run_id=semantic_run_id,
                        failure_type=failure_type,
                    )
                except Exception:
                    pass
            _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
            _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        _emit_structured_log(
            "semantic_parse",
            {
                **log_base,
                "semantic_run_id": semantic_run_id,
                "status": "failed",
                "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                "parse_success": False,
                "failure_type": failure_type,
            },
        )
        return

    conn = init_db(sandbox_db_path)
    try:
        if semantic_run_id is not None:
            _set_semantic_diagnostic(conn, semantic_run_id, diagnostic)
        fact_payloads = _semantic_fact_payloads(
            conn,
            evidence_id=evidence_id,
            semantics=parsed,
            glossary=glossary,
            created_at=int(time.time() * 1000),
        )
        interpretation_payloads = _semantic_interpretation_payloads(
            evidence_id,
            parsed,
            created_at=int(time.time() * 1000),
        )
        persisted = persist_semantic_run_result(
            conn,
            semantic_run_id=semantic_run_id,
            facts=fact_payloads,
            interpretations=interpretation_payloads,
        )
        if semantic_run_id is not None and persisted["facts"]:
            previous_match_run = get_latest_event_match_for_evidence(conn, evidence_id)
            event_match_run = create_event_match_run(
                conn,
                semantic_run_id=semantic_run_id,
                provider="deepseek",
                model=get_text_model(),
                matcher_version=event_matcher.EVENT_MATCHER_VERSION,
                supersedes_run_id=None if previous_match_run is None else previous_match_run["event_match_run_id"],
            )
            try:
                normalized_match = event_matcher.match_events(
                    _event_matcher_fact_payload(conn, persisted["facts"]),
                    existing_events=list_event_candidates(conn),
                )
                if normalized_match is None:
                    raise ValueError("matcher returned no normalized result")
                stored_match = persist_event_match_run_result(
                    conn,
                    event_match_run_id=event_match_run["event_match_run_id"],
                    semantic_run_id=semantic_run_id,
                    routing_mode=event_matcher.decide_assignment_mode(normalized_match),
                    normalized_match=normalized_match,
                    facts=persisted["facts"],
                )
                if stored_match["routing_mode"] == "auto":
                    groups = stored_match.get("result", {}).get("groups") or []
                    target_event_ids = {
                        group.get("event_id")
                        for group in groups
                        if group.get("target") in {"existing", "new"} and group.get("event_id")
                    }
                    for target_event_id in target_event_ids:
                        _maybe_run_event_change_detection(conn, target_event_id)
            except ProtectedFactError:
                mark_event_match_run_failed(
                    conn,
                    event_match_run_id=event_match_run["event_match_run_id"],
                    failure_type="protected_fact",
                )
            except ValueError:
                mark_event_match_run_failed(
                    conn,
                    event_match_run_id=event_match_run["event_match_run_id"],
                    failure_type="provider_invalid_response",
                )
            except Exception:
                mark_event_match_run_failed(
                    conn,
                    event_match_run_id=event_match_run["event_match_run_id"],
                    failure_type="persistence_error",
                )
        _set_parse_status(conn, evidence_id, PARSE_STATUS_DONE)
        _set_parse_detail(conn, evidence_id, "")
        _emit_structured_log(
            "semantic_parse",
            {
                **log_base,
                "semantic_run_id": semantic_run_id,
                "status": "succeeded",
                "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                "parse_success": True,
                "failure_type": None,
            },
        )
    except Exception:
        try:
            if semantic_run_id is not None:
                mark_semantic_run_failed(
                    conn,
                    semantic_run_id=semantic_run_id,
                    failure_type="persistence_error",
                )
        except Exception:
            pass
        _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
        _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
        _emit_structured_log(
            "semantic_parse",
            {
                **log_base,
                "semantic_run_id": semantic_run_id,
                "status": "failed",
                "latency_ms": int((time.perf_counter() - parse_start) * 1000),
                "parse_success": False,
                "failure_type": "persistence_error",
            },
        )
    finally:
        conn.close()


def _run_image_pipeline(
    sandbox_db_path: Path,
    global_meta_db_path: Path,
    evidence_id: str,
    image_bytes: bytes,
    filename: str | None,
    counterpart: str | None,
) -> None:
    mime_type = _detect_blob_content_type(image_bytes, filename)
    selected_provider = get_image_extraction_provider()
    conn = init_db(sandbox_db_path)
    try:
        source_hint = get_effective_source_hint(conn, evidence_id)
    finally:
        conn.close()
    extraction_result = run_production_image_extraction(
        image_bytes,
        mime_type,
        provider=selected_provider,
        source_hint=source_hint,
        allow_ocr_fallback=selected_provider != "ocr",
        consume_ocr_fallback_budget=(
            None
            if selected_provider == "ocr"
            else lambda: _consume_ocr_budget(sandbox_db_path, global_meta_db_path)
        ),
    )
    extraction = extraction_result.get("extraction")
    transcript = None if extraction is None else extraction.get("transcript")
    observations = [] if extraction is None else extraction.get("observations", [])
    warnings = [] if extraction is None else extraction.get("warnings", [])
    conn = init_db(sandbox_db_path)
    try:
        if extraction is None:
            detail = extraction_result.get("detail") or "图片提取暂不可用,原件已完整保存"
            _set_parse_status(conn, evidence_id, PARSE_STATUS_UNSUPPORTED)
            _set_parse_detail(conn, evidence_id, _saved_original_detail(detail))
            _set_extract_note(conn, evidence_id, detail)
            return

        _record_machine_extraction(
            conn,
            evidence_id=evidence_id,
            transcript=transcript,
            observations=observations,
            provider=extraction.get("provider") or "unknown",
            model=extraction.get("model"),
            warnings=warnings,
        )
        conn.execute(
            "UPDATE evidence SET raw_text = ? WHERE evidence_id = ?",
            (_build_attachment_raw_text("image", filename, transcript), evidence_id),
        )
        latest_extraction = get_latest_extraction(conn, evidence_id)
        _clear_extract_note(conn, evidence_id)
        if not _can_run_semantic_parse(transcript, observations):
            note = "图片提取未产生可供语义解析的 transcript 或 observations"
            _set_parse_status(conn, evidence_id, PARSE_STATUS_UNSUPPORTED)
            _set_parse_detail(conn, evidence_id, _saved_original_detail(note))
            _set_extract_note(conn, evidence_id, note)
            conn.commit()
            return

        gate_state = _source_gate_state(
            conn,
            evidence_id=evidence_id,
            extraction=latest_extraction,
        )
        if gate_state["requires_clarification"]:
            _set_parse_status(conn, evidence_id, PARSE_STATUS_CLARIFICATION_REQUIRED)
            _set_parse_detail(conn, evidence_id, _clarification_detail(gate_state))
            conn.commit()
            return

        _clear_extract_note(conn, evidence_id)
        _set_parse_status(conn, evidence_id, PARSE_STATUS_LLM_RUNNING)
        _set_parse_detail(conn, evidence_id, "")
        conn.commit()
    finally:
        conn.close()

    _run_parse_pipeline(
        sandbox_db_path,
        global_meta_db_path,
        evidence_id,
    )


def _mark_pipeline_exception(
    sandbox_db_path: Path,
    evidence_id: str,
    *,
    detail: str = "解析暂不可用,记录已完整保存",
) -> None:
    conn = init_db(sandbox_db_path)
    try:
        _set_parse_status(conn, evidence_id, PARSE_STATUS_FAILED)
        _set_parse_detail(conn, evidence_id, detail)
        _set_extract_note(conn, evidence_id, detail)
    finally:
        conn.close()


def _run_multi_image_pipeline(
    sandbox_db_path: Path,
    global_meta_db_path: Path,
    image_items: list[dict[str, Any]],
) -> None:
    for item in image_items:
        try:
            _run_image_pipeline(
                sandbox_db_path,
                global_meta_db_path,
                item["evidence_id"],
                item["image_bytes"],
                item.get("filename"),
                None,
            )
        except Exception:
            _mark_pipeline_exception(sandbox_db_path, item["evidence_id"])


def _prepare_thread_card(row: sqlite3.Row) -> dict[str, Any]:
    risk_flags = _decode_json_array(row["risk_flags"])
    source_platforms = []
    if row["source_platforms"]:
        source_platforms = [
            {"platform": platform, "platform_class": source_badge_class(platform)}
            for platform in row["source_platforms"].split(",")
            if platform
        ]
    thread = {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "status": row["status"],
        "risk_flags": risk_flags,
    }
    return {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "status_label": STATUS[row["status"]],
        "version": row["version"],
        "headline": thread_headline(thread),
        "risk_labels": [RISK.get(flag, flag) for flag in risk_flags],
        "source_platforms": source_platforms,
        "evidence_count": row["evidence_count"],
        "last_activity_text": _format_datetime(row["last_activity_at"], "%m-%d"),
    }


def _prepare_reference_row(row: sqlite3.Row) -> dict[str, Any]:
    platform, scene = source_label(row["source_hint"])
    return {
        "seq": row["seq"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": row["raw_text"],
    }


def _prepare_recent_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    sandbox: SandboxContext,
) -> dict[str, Any]:
    effective_source_hint = get_effective_source_hint(conn, row["evidence_id"])
    platform, scene = source_label(effective_source_hint)
    raw_text = row["raw_text"] or ""
    filename = _extract_filename(raw_text)
    extracted_text = _extract_file_text(raw_text)
    display_text = extracted_text if extracted_text is not None else raw_text
    semantic_result = _build_semantic_result(conn, row["evidence_id"])
    related_event_rows = conn.execute(
        """
        SELECT DISTINCT ev.event_id, ev.title
        FROM fact_evidence fe
        JOIN facts f ON f.fact_id = fe.fact_id
        JOIN events ev ON ev.event_id = f.event_id
        WHERE fe.evidence_id = ?
        ORDER BY ev.updated_at DESC, ev.event_id DESC
        LIMIT 3
        """,
        (row["evidence_id"],),
    ).fetchall()
    return {
        "evidence_id": row["evidence_id"],
        "effective_source_hint": effective_source_hint,
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "captured_at_text": _format_datetime(row.get("captured_at"), "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": raw_text,
        "display_text": display_text,
        "raw_text_preview": _truncate_text(display_text),
        "is_long_text": len(display_text) > 100,
        "parse_status": row["parse_status"],
        "parse_status_label": _parse_status_label(row["parse_status"]),
        "parse_detail": _humanize_user_facing_text(row["parse_detail"]) or row["parse_detail"],
        "extract_note": _humanize_user_facing_text(row.get("extract_note")) if row.get("extract_note") else None,
        "plain_summary": row["plain_summary"],
        "deliverable": row["slot_deliverable"],
        "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d"),
        "caveats": [_humanize_user_facing_text(item) or item for item in _decode_json_array(row["caveats"])],
        "semantic_fact_preview": [] if semantic_result is None else semantic_result["facts"][:2],
        "event_match": None if semantic_result is None else semantic_result.get("event_match"),
        "assignment_subject": {"evidence_id": row["evidence_id"]},
        "is_verified": bool(row["is_verified"]),
        "media_type": row["media_type"],
        "media_type_label": {"text": "文字", "image": "图片", "file": "文件"}.get(row["media_type"], row["media_type"]),
        "blob_url": f"/blob/{row['evidence_id']}" if row["media_type"] in {"image", "file"} else None,
        "filename": filename,
        "is_image": row["media_type"] == "image",
        "is_file": row["media_type"] == "file",
        "has_extracted_text": extracted_text is not None,
        "related_events": [
            {"event_id": item["event_id"], "title": item["title"]}
            for item in related_event_rows
        ],
    }


def _prepare_event_card(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    summary_rows = conn.execute(
        """
        SELECT f.fact_type, f.content
        FROM facts f
        WHERE f.event_id = ?
          AND EXISTS (
              SELECT 1
              FROM fact_evidence fe
              JOIN evidence e ON e.evidence_id = fe.evidence_id
              WHERE fe.fact_id = f.fact_id
                AND e.evidence_id NOT LIKE 'ev_demo_%'
          )
        ORDER BY COALESCE(f.occurred_at, f.updated_at, f.created_at) DESC, f.fact_id DESC
        LIMIT 1
        """,
        (row["event_id"],),
    ).fetchall()
    return {
        "event_id": row["event_id"],
        "title": row["title"],
        "status": row["status"],
        "status_label": _event_status_label(row["status"]),
        "fact_count": row["fact_count"],
        "last_activity_text": _format_datetime(row["last_activity_at"], "%m-%d %H:%M"),
        "due_text": _format_due_display(row["due_at"]),
        "recent_facts": [
            {
                "fact_type": item["fact_type"],
                "fact_type_label": _event_fact_type_label(item["fact_type"]),
                "content": item["content"],
            }
            for item in summary_rows
        ],
    }


def _prepare_pending_event_card(
    conn: sqlite3.Connection,
    *,
    evidence_id: str,
    semantic_result: dict[str, Any],
) -> dict[str, Any] | None:
    event_match = semantic_result.get("event_match")
    if not _should_show_event_assignment_panel(event_match):
        return None

    groups = event_match.get("groups") or []
    if not groups:
        return None

    latest_fact_row = conn.execute(
        """
        SELECT f.fact_type, f.content
        FROM facts f
        JOIN fact_evidence fe ON fe.fact_id = f.fact_id
        WHERE f.semantic_run_id = ?
          AND fe.evidence_id = ?
        ORDER BY COALESCE(f.occurred_at, f.updated_at, f.created_at) DESC, f.fact_id DESC
        LIMIT 1
        """,
        (semantic_result["semantic_run_id"], evidence_id),
    ).fetchone()

    if len(groups) == 1:
        title = groups[0].get("ai_title") or "AI 未给出标题"
    else:
        title = f"可能涉及 {len(groups)} 个事项"

    fact_preview = None
    if latest_fact_row is not None:
        fact_preview = {
            "fact_type": latest_fact_row["fact_type"],
            "fact_type_label": _event_fact_type_label(latest_fact_row["fact_type"]),
            "content": latest_fact_row["content"],
        }

    return {
        "evidence_id": evidence_id,
        "label": "待确认",
        "title": title,
        "fact_preview": fact_preview,
        "routing_mode": event_match.get("routing_mode"),
    }


def _fetch_pending_event_cards(
    conn: sqlite3.Connection,
    sandbox: SandboxContext,
) -> list[dict[str, Any]]:
    evidence_rows = conn.execute(
        """
        SELECT evidence_id
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
        ORDER BY seq DESC
        """
    ).fetchall()

    pending_cards: list[dict[str, Any]] = []
    for row in evidence_rows:
        semantic_result = _build_semantic_result(conn, row["evidence_id"])
        if semantic_result is None:
            continue
        card = _prepare_pending_event_card(
            conn,
            evidence_id=row["evidence_id"],
            semantic_result=semantic_result,
        )
        if card is not None:
            pending_cards.append(card)
    return pending_cards


def _fetch_recent_records_data(
    conn: sqlite3.Connection,
    sandbox: SandboxContext,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    recent_rows = conn.execute(
        """
        SELECT evidence_id, occurred_at, captured_at, source_hint, raw_text, media_type,
               plain_summary, slot_deliverable, slot_due, slot_due_raw, caveats
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
        ORDER BY seq DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    evidence_ids = [row["evidence_id"] for row in recent_rows]
    parse_status_map = _get_parse_status_map(conn, evidence_ids)
    parse_detail_map = _get_parse_detail_map(conn, evidence_ids)
    extract_note_map = _get_extract_note_map(conn, evidence_ids)
    verified_map = _get_verified_map(conn, evidence_ids)
    decorated_recent_rows = []
    for row in recent_rows:
        row_dict = dict(row)
        row_dict["parse_status"] = parse_status_map.get(row["evidence_id"], "failed")
        row_dict["parse_detail"] = parse_detail_map.get(row["evidence_id"])
        row_dict["extract_note"] = extract_note_map.get(row["evidence_id"])
        row_dict["is_verified"] = verified_map.get(row["evidence_id"], False)
        decorated_recent_rows.append(row_dict)
    return [_prepare_recent_row(conn, row, sandbox) for row in decorated_recent_rows]


def _settings_payload(db_path: Path) -> dict[str, Any]:
    settings = get_settings(db_path)
    return {
        "self_names": settings["self_names"],
        "glossary": settings["glossary"],
        "has_self_names": bool(settings["self_names"]),
    }


def _escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _parse_date_start(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _parse_date_end(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000) + 86_399_999


def _source_platform_expr(column: str = "e.source_hint") -> str:
    return (
        f"CASE WHEN instr({column}, '-') > 0 "
        f"THEN substr({column}, 1, instr({column}, '-') - 1) "
        f"ELSE {column} END"
    )


def _highlight_text(text: str | None, query: str) -> str:
    text = text or ""
    if not query:
        return html.escape(text)

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    parts: list[str] = []
    last_index = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        parts.append(html.escape(text[last_index:start]))
        parts.append(f"<mark>{html.escape(text[start:end])}</mark>")
        last_index = end
    parts.append(html.escape(text[last_index:]))
    return "".join(parts)


def _search_evidence(
    conn: sqlite3.Connection,
    *,
    q: str,
    source: str | None,
    kind: str | None,
    start: str | None,
    end: str | None,
) -> list[dict[str, Any]]:
    trimmed_q = q.strip()[:100]
    if not trimmed_q:
        return []

    like_pattern = f"%{_escape_like(trimmed_q)}%"
    actor_rows = conn.execute(
        """
        SELECT actor_id
        FROM actors
        WHERE canonical_name LIKE ? ESCAPE '!'
           OR aliases LIKE ? ESCAPE '!'
        """,
        (like_pattern, like_pattern),
    ).fetchall()
    actor_ids = [row["actor_id"] for row in actor_rows]

    text_match_clauses = [
        "e.raw_text LIKE ? ESCAPE '!'",
        "e.plain_summary LIKE ? ESCAPE '!'",
        "e.slot_deliverable LIKE ? ESCAPE '!'",
        "e.source_hint LIKE ? ESCAPE '!'",
        "sr.canonical_name LIKE ? ESCAPE '!'",
        "sr.aliases LIKE ? ESCAPE '!'",
        "so.canonical_name LIKE ? ESCAPE '!'",
        "so.aliases LIKE ? ESCAPE '!'",
    ]
    params: list[Any] = [like_pattern] * 8

    if actor_ids:
        placeholders = ",".join("?" for _ in actor_ids)
        text_match_clauses.append(f"e.slot_requester IN ({placeholders})")
        text_match_clauses.append(f"e.slot_owner IN ({placeholders})")
        params.extend(actor_ids)
        params.extend(actor_ids)

    filters = [f"({' OR '.join(text_match_clauses)})"]

    if source:
        filters.append(f"{_source_platform_expr()} = ?")
        params.append(source)
    if kind and kind in KIND:
        filters.append("e.kind = ?")
        params.append(kind)

    start_ms = _parse_date_start(start)
    end_ms = _parse_date_end(end)
    if start_ms is not None:
        filters.append("e.occurred_at >= ?")
        params.append(start_ms)
    if end_ms is not None:
        filters.append("e.occurred_at <= ?")
        params.append(end_ms)

    rows = conn.execute(
        f"""
        SELECT
            e.evidence_id,
            e.occurred_at,
            e.raw_text,
            e.plain_summary,
            e.source_hint,
            e.kind,
            t.title AS thread_title
        FROM evidence AS e
        LEFT JOIN threads AS t ON t.thread_id = e.thread_id
        LEFT JOIN actors AS sr ON sr.actor_id = e.slot_requester
        LEFT JOIN actors AS so ON so.actor_id = e.slot_owner
        WHERE {' AND '.join(filters)}
        ORDER BY e.occurred_at DESC, e.seq DESC
        LIMIT 100
        """,
        params,
    ).fetchall()

    results = []
    for row in rows:
        platform, scene = source_label(row["source_hint"])
        results.append(
            {
                "evidence_id": row["evidence_id"],
                "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
                "occurred_date": _format_datetime(row["occurred_at"], "%Y-%m-%d"),
                "platform": platform,
                "platform_class": source_badge_class(platform),
                "scene": scene,
                "kind": row["kind"],
                "kind_label": KIND[row["kind"]],
                "raw_text_html": _highlight_text(row["raw_text"], trimmed_q),
                "plain_summary": row["plain_summary"],
                "thread_title": row["thread_title"],
            }
        )
    return results


def _prepare_timeline_row(row: sqlite3.Row) -> dict[str, Any]:
    caveats = _decode_json_array(row["caveats"])
    platform, scene = source_label(row["source_hint"])
    return {
        "seq": row["seq"],
        "kind": row["kind"],
        "kind_label": KIND[row["kind"]],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": row["raw_text"],
        "plain_summary": row["plain_summary"],
        "deliverable": row["slot_deliverable"],
        "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d %H:%M"),
        "caveats": caveats,
        "is_change": row["kind"] == "change",
    }


def _fetch_index_data(conn: sqlite3.Connection, sandbox: SandboxContext) -> dict[str, Any]:
    event_rows = conn.execute(
        """
        SELECT
            ev.event_id,
            ev.title,
            ev.status,
            COUNT(DISTINCT f.fact_id) AS fact_count,
            MIN(f.due_at) AS due_at,
            MAX(COALESCE(f.occurred_at, f.updated_at, f.created_at, ev.updated_at)) AS last_activity_at
        FROM events AS ev
        JOIN facts AS f ON f.event_id = ev.event_id
        WHERE EXISTS (
            SELECT 1
            FROM fact_evidence fe
            JOIN evidence e ON e.evidence_id = fe.evidence_id
            WHERE fe.fact_id = f.fact_id
              AND e.evidence_id NOT LIKE 'ev_demo_%'
        )
        GROUP BY ev.event_id, ev.title, ev.status
        ORDER BY last_activity_at DESC, ev.updated_at DESC, ev.event_id DESC
        """
    ).fetchall()
    demo_thread_rows = conn.execute(
        """
        SELECT
            t.thread_id,
            t.title,
            t.status,
            t.version,
            t.risk_flags,
            t.last_activity_at,
            COUNT(e.evidence_id) AS evidence_count,
            GROUP_CONCAT(DISTINCT CASE
                WHEN instr(e.source_hint, '-') > 0 THEN substr(e.source_hint, 1, instr(e.source_hint, '-') - 1)
                ELSE e.source_hint
            END) AS source_platforms
        FROM threads AS t
        LEFT JOIN evidence AS e ON e.thread_id = t.thread_id
        WHERE EXISTS (
            SELECT 1
            FROM evidence demo_e
            WHERE demo_e.thread_id = t.thread_id
              AND demo_e.evidence_id LIKE 'ev_demo_%'
        )
        GROUP BY
            t.thread_id, t.title, t.status, t.version, t.risk_flags, t.last_activity_at
        ORDER BY t.last_activity_at DESC, t.thread_id ASC
        """
    ).fetchall()
    reference_rows = conn.execute(
        """
        SELECT seq, occurred_at, source_hint, raw_text
        FROM evidence
        WHERE thread_id IS NULL
        ORDER BY seq ASC
        """
    ).fetchall()
    prepared_events = [_prepare_event_card(conn, row) for row in event_rows]
    return {
        "my_events": [item for item in prepared_events if item["status"] == "active"][:6],
        "pending_event_cards": _fetch_pending_event_cards(conn, sandbox),
        "history_events": [item for item in prepared_events if item["status"] == "resolved"],
        "demo_threads": [_prepare_thread_card(row) for row in demo_thread_rows],
        "references": [_prepare_reference_row(row) for row in reference_rows],
        "recent_records": _fetch_recent_records_data(conn, sandbox, limit=20),
    }


def _fetch_event_detail(conn: sqlite3.Connection, event_id: str) -> dict[str, Any] | None:
    def build_record_date_state(evidence_id: str) -> dict[str, Any] | None:
        latest_run = get_latest_semantic_run_for_evidence(conn, evidence_id, status="succeeded")
        if latest_run is None:
            return None
        facts_for_evidence = list_facts_for_semantic_run(
            conn,
            latest_run["semantic_run_id"],
            evidence_id=evidence_id,
        )
        if not any(semantic_llm.is_relative_due_raw(item.get("due_raw")) for item in facts_for_evidence):
            return None
        record_date = _get_semantic_anchor_date(conn, evidence_id)
        record_date_source = _get_semantic_anchor_source(conn, evidence_id)
        return {
            "evidence_id": evidence_id,
            "record_date": record_date,
            "record_date_source": record_date_source,
            "record_date_source_label": _semantic_anchor_source_label(record_date_source),
        }

    def build_related_evidence_card(row: sqlite3.Row) -> dict[str, Any]:
        effective_source_hint = get_effective_source_hint(conn, row["evidence_id"])
        platform, scene = source_label(effective_source_hint)
        raw_text = row["raw_text"] or ""
        filename = _extract_filename(raw_text)
        extracted_text = _extract_file_text(raw_text)
        preview_text = raw_text if row["media_type"] == "text" else extracted_text
        return {
            "evidence_id": row["evidence_id"],
            "href": f"/evidence/{row['evidence_id']}",
            "blob_url": f"/blob/{row['evidence_id']}" if row["media_type"] in {"image", "file"} else None,
            "media_type": row["media_type"],
            "media_type_label": {
                "image": "图片记录",
                "text": "文字记录",
                "file": "文件记录",
            }.get(row["media_type"], "记录"),
            "filename": filename,
            "effective_source_hint": effective_source_hint,
            "preview_text": preview_text,
            "platform": platform,
            "scene": scene,
            "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
            "captured_at_text": _format_datetime(row["captured_at"], "%m-%d %H:%M"),
            "record_date_state": build_record_date_state(row["evidence_id"]),
        }

    event_row = conn.execute(
        """
        SELECT event_id, title, status, summary, created_at, updated_at
        FROM events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if event_row is None:
        return None

    fact_rows = conn.execute(
        """
        SELECT
            f.fact_id,
            f.fact_type,
            f.content,
            f.occurred_at,
            f.created_at,
            f.due_at,
            f.origin,
            f.review_status,
            (
                SELECT fe.evidence_id
                FROM fact_evidence fe
                JOIN evidence e ON e.evidence_id = fe.evidence_id
                WHERE fe.fact_id = f.fact_id
                ORDER BY e.seq ASC, e.evidence_id ASC
                LIMIT 1
            ) AS evidence_id
        FROM facts f
        WHERE f.event_id = ?
        ORDER BY COALESCE(f.occurred_at, f.created_at) ASC, f.created_at ASC, f.fact_id ASC
        """,
        (event_id,),
    ).fetchall()
    related_evidence_rows = conn.execute(
        """
        SELECT DISTINCT
            e.evidence_id, e.seq, e.media_type, e.raw_text, e.source_hint, e.occurred_at, e.captured_at
        FROM facts f
        JOIN fact_evidence fe ON fe.fact_id = f.fact_id
        JOIN evidence e ON e.evidence_id = fe.evidence_id
        WHERE f.event_id = ?
        ORDER BY e.occurred_at ASC, e.seq ASC, e.evidence_id ASC
        """,
        (event_id,),
    ).fetchall()

    facts: list[dict[str, Any]] = []
    for row in fact_rows:
        facts.append(
            {
                "fact_id": row["fact_id"],
                "fact_type": row["fact_type"],
                "fact_type_label": _event_fact_type_label(row["fact_type"]),
                "content": row["content"],
                "due_text": _format_due_display(row["due_at"]),
                "due_date_value": _format_datetime(row["due_at"], "%Y-%m-%d"),
                "origin": row["origin"],
                "review_status": row["review_status"],
                "evidence_id": row["evidence_id"],
                "occurred_at_text": _format_datetime(
                    row["occurred_at"] if row["occurred_at"] is not None else row["created_at"],
                    "%m-%d %H:%M",
                ),
            }
        )

    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    latest_change_run = get_latest_event_change_run_for_event(conn, event_id, status="succeeded")
    event_changes: list[dict[str, Any]] = []
    if latest_change_run is not None:
        for item in latest_change_run.get("changes", []):
            earlier_fact = facts_by_id.get(item["earlier_fact_id"])
            later_fact = facts_by_id.get(item["later_fact_id"])
            if earlier_fact is None or later_fact is None:
                continue
            event_changes.append(
                {
                    "change_type": item["change_type"],
                    "title": _event_change_type_label(item["change_type"]),
                    "summary": item["summary"],
                    "is_contradiction": item["change_type"] == "contradiction",
                    "earlier": {
                        "occurred_at_text": earlier_fact["occurred_at_text"],
                        "content": earlier_fact["content"],
                        "href": (
                            f"/evidence/{earlier_fact['evidence_id']}"
                            if earlier_fact.get("evidence_id")
                            else None
                        ),
                    },
                    "later": {
                        "occurred_at_text": later_fact["occurred_at_text"],
                        "content": later_fact["content"],
                        "href": (
                            f"/evidence/{later_fact['evidence_id']}"
                            if later_fact.get("evidence_id")
                            else None
                        ),
                    },
                }
            )

    return {
        "event": {
            "event_id": event_row["event_id"],
            "title": event_row["title"],
            "status": event_row["status"],
            "status_label": _event_status_label(event_row["status"]),
            "can_resolve": event_row["status"] == "active",
            "can_reopen": event_row["status"] == "resolved",
            "summary": event_row["summary"],
            "fact_count": len(facts),
            "updated_at_text": _format_datetime(event_row["updated_at"], "%m-%d %H:%M"),
        },
        "related_evidence": [build_related_evidence_card(row) for row in related_evidence_rows],
        "event_changes": event_changes,
        "facts": facts,
        "fact_type_options": _event_fact_type_options(),
    }


def _fetch_thread_detail(conn: sqlite3.Connection, thread_id: str) -> dict[str, Any] | None:
    thread_row = conn.execute(
        """
        SELECT
            thread_id, title, status, version, risk_flags,
            current_deliverable, current_due, last_activity_at
        FROM threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if thread_row is None:
        return None

    evidence_rows = conn.execute(
        """
        SELECT
            seq, kind, occurred_at, source_hint, raw_text,
            plain_summary, slot_deliverable, slot_due, slot_due_raw, caveats
        FROM evidence
        WHERE thread_id = ?
        ORDER BY seq ASC
        """,
        (thread_id,),
    ).fetchall()
    occurred_values = [row["occurred_at"] for row in evidence_rows if row["occurred_at"] is not None]
    change_count = sum(1 for row in evidence_rows if row["kind"] == "change")
    return {
        "thread": {
            "thread_id": thread_row["thread_id"],
            "title": thread_row["title"],
            "status_label": STATUS[thread_row["status"]],
            "version": thread_row["version"],
            "headline": thread_headline(
                {"status": thread_row["status"], "risk_flags": _decode_json_array(thread_row["risk_flags"])}
            ),
            "risk_labels": [RISK.get(flag, flag) for flag in _decode_json_array(thread_row["risk_flags"])],
            "current_deliverable": thread_row["current_deliverable"],
            "current_due_text": _format_datetime(thread_row["current_due"], "%m-%d %H:%M"),
            "last_activity_text": _format_datetime(thread_row["last_activity_at"], "%m-%d %H:%M"),
            "evidence_count": len(evidence_rows),
            "summary_text": (
                f"这件事从 {_format_datetime(min(occurred_values), '%m-%d')} 到 "
                f"{_format_datetime(max(occurred_values), '%m-%d')},共 {len(evidence_rows)} 条记录,期间需求变更 {change_count} 次"
            ),
        },
        "entries": [_prepare_timeline_row(row) for row in evidence_rows],
    }


def _fetch_evidence_detail(conn: sqlite3.Connection, evidence_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            evidence_id, seq, occurred_at, captured_at, source_hint, raw_text,
            plain_summary, slot_deliverable, slot_due, slot_due_raw, caveats,
            content_hash, slots_filled, kind, slot_direction, media_type, blob_path
        FROM evidence
        WHERE evidence_id = ?
        """,
        (evidence_id,),
    ).fetchone()
    if row is None:
        return None

    effective_source_hint = get_effective_source_hint(conn, evidence_id)
    platform, scene = source_label(effective_source_hint)
    raw_text = row["raw_text"] or ""
    filename = _extract_filename(raw_text)
    extracted_text = _extract_file_text(raw_text)
    raw_extractions = list_extractions(conn, evidence_id)
    extraction_history = [_format_extraction_item(item) for item in raw_extractions]
    current_extraction = None
    if raw_extractions:
        latest_extraction = raw_extractions[-1]
        current_extraction = {
            **_format_extraction_item(latest_extraction),
            "transcript": latest_extraction.get("transcript"),
            "observations": latest_extraction.get("observations") if isinstance(latest_extraction.get("observations"), list) else [],
            "warnings": latest_extraction.get("warnings") if isinstance(latest_extraction.get("warnings"), list) else [],
        }
    semantic_result = _build_semantic_result(conn, evidence_id)
    return {
        "evidence_id": row["evidence_id"],
        "effective_source_hint": effective_source_hint,
        "seq": row["seq"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "captured_at_text": _format_datetime(row["captured_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": raw_text,
        "display_text": extracted_text if extracted_text is not None else raw_text,
        "extracted_text": extracted_text,
        "plain_summary": row["plain_summary"],
        "deliverable": row["slot_deliverable"],
        "due_date_value": _format_datetime(row["slot_due"], "%Y-%m-%d"),
        "due_text": row["slot_due_raw"] or _format_due_display(row["slot_due"]),
        "caveats": [_humanize_user_facing_text(item) or item for item in _decode_json_array(row["caveats"])],
        "content_hash_prefix": (row["content_hash"] or "")[:12],
        "slots_filled": row["slots_filled"],
        "kind": row["kind"],
        "slot_direction": row["slot_direction"] or "none",
        "media_type": row["media_type"],
        "blob_path": row["blob_path"],
        "filename": filename,
        "is_image": row["media_type"] == "image",
        "is_file": row["media_type"] == "file",
        "blob_url": f"/blob/{row['evidence_id']}" if row["media_type"] in {"image", "file"} else None,
        "has_extracted_text": extracted_text is not None,
        "current_extraction": current_extraction,
        "extraction_history": extraction_history,
        "semantic_result": semantic_result,
        "assignment_subject": {"evidence_id": row["evidence_id"]},
    }


def _has_legacy_semantic_data(evidence: dict[str, Any]) -> bool:
    if evidence.get("slots_filled", 0):
        return True
    if evidence.get("plain_summary"):
        return True
    if evidence.get("deliverable") or evidence.get("due_date_value") or evidence.get("due_text"):
        return True
    if evidence.get("caveats"):
        return True
    if evidence.get("kind") != "reference":
        return True
    return evidence.get("slot_direction") not in {None, "none"}


def _should_show_legacy_editor(
    *,
    evidence: dict[str, Any],
    parse_status: str,
) -> bool:
    if evidence.get("semantic_result") is not None:
        return False
    if parse_status in {
        PARSE_STATUS_OCR_RUNNING,
        PARSE_STATUS_LLM_RUNNING,
        PARSE_STATUS_CLARIFICATION_REQUIRED,
        "pending",
    }:
        return False
    return _has_legacy_semantic_data(evidence)


def _build_evidence_diagnostics(
    conn: sqlite3.Connection,
    *,
    evidence: dict[str, Any],
    parse_status: str,
    parse_detail: str | None,
) -> dict[str, Any]:
    latest_run = get_latest_semantic_run_for_evidence(conn, evidence["evidence_id"])
    semantic_diagnostic = None if latest_run is None else _get_semantic_diagnostic(conn, latest_run["semantic_run_id"])
    return {
        "evidence_id": evidence["evidence_id"],
        "parse_status": parse_status,
        "parse_detail": parse_detail,
        "extraction_history": [
            {
                "origin": item["origin"],
                "origin_label": item["origin_label"],
                "provider": item["provider"],
                "provider_label": item["provider_label"],
                "model": item["model"],
                "warnings": item.get("warnings", []),
                "created_at": item["created_at"],
                "created_at_text": item["created_at_text"],
            }
            for item in evidence["extraction_history"]
        ],
        "image_pipeline": _production_image_pipeline_info(evidence.get("current_extraction")),
        "text_llm": {
            "provider": "deepseek",
            "provider_label": _provider_label("deepseek"),
            "model": get_text_model(),
            "parser_version": semantic_llm.SEMANTIC_PARSER_VERSION,
        },
        "semantic_parser": None if latest_run is None else {
            "semantic_run_id": latest_run["semantic_run_id"],
            "run_status": latest_run["status"],
            "provider": latest_run["provider"],
            "provider_label": _provider_label(latest_run["provider"]),
            "model": latest_run["model"],
            "parser_version": latest_run["parser_version"],
            "failure_type": latest_run["failure_type"],
            "diagnostic": semantic_diagnostic,
        },
    }


def _build_extraction_baseline_summary(extraction: dict[str, Any] | None) -> dict[str, Any] | None:
    if extraction is None:
        return None
    created_at = extraction.get("created_at")
    created_at_text = _format_datetime(created_at, "%Y-%m-%d %H:%M:%S") if isinstance(created_at, int) else None
    provider = extraction.get("provider")
    model = extraction.get("model")
    observations = extraction.get("observations")
    normalized_observations = observations if isinstance(observations, list) else []
    return {
        "origin": extraction.get("origin"),
        "origin_label": _extraction_origin_label(extraction.get("origin")),
        "provider": provider,
        "provider_label": _provider_label(provider),
        "model": model,
        "created_at": created_at,
        "created_at_text": created_at_text,
        "summary": _format_extraction_item(extraction)["summary"],
        "transcript_chars": len(extraction.get("transcript") or ""),
        "observation_count": len(normalized_observations),
    }


def _ark_diagnostic_response(
    *,
    baseline: dict[str, Any] | None,
    status: str,
    detail: str,
    extraction: dict[str, Any] | None,
    text_preflight: dict[str, Any] | None,
    diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "ark_vision": {
            "status": status,
            "detail": detail,
            "extraction": extraction,
            "text_preflight": text_preflight,
            "diagnostic": diagnostic,
        },
    }


def _log_ark_diagnostic_attempt(
    *,
    evidence_id: str,
    model: str | None,
    status: str,
    latency_ms: int,
    extraction: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> None:
    observations = []
    transcript = None
    if extraction is not None:
        maybe_observations = extraction.get("observations")
        if isinstance(maybe_observations, list):
            observations = maybe_observations
        transcript = extraction.get("transcript")
    _emit_structured_log(
        "ark_vision_diagnostic",
        {
            "evidence_id": evidence_id,
            "provider": vision_provider.ARK_PROVIDER_NAME,
            "model": model,
            "status": status,
            "latency_ms": latency_ms,
            "transcript_chars": len(transcript or ""),
            "observation_count": len(observations),
            "error_type": error_type,
        },
    )


def _ark_diagnostic_detail(preflight: dict[str, Any], diagnostic: dict[str, Any] | None) -> str:
    if not preflight.get("success"):
        return (
            "Ark text preflight 失败,优先排查 Key、Base URL、模型配置或模型开通状态。"
        )
    if diagnostic is None:
        return "Ark text preflight 成功,但未执行图片实验解析。"
    if diagnostic.get("success"):
        return "Ark Vision 实验解析完成,结果仅供对照,不会保存或影响当前记录"
    stage = diagnostic.get("stage")
    if stage == "model_json":
        return "Ark 已正常返回,但 WorkChain 在 model_json 阶段无法解析结果"
    if stage == "contract":
        return "Ark 已正常返回,但 WorkChain 在 contract 阶段无法解析结果"
    if stage == "output_text":
        return "Ark 已正常返回,但 WorkChain 在 output_text 阶段找不到模型文本"
    if stage == "http" and diagnostic.get("error_code") == "timeout":
        timeout_seconds = diagnostic.get("timeout_seconds")
        if isinstance(timeout_seconds, (int, float)):
            return f"请求超过当前视觉超时上限 {timeout_seconds:g} 秒"
        return "请求超过当前视觉超时上限"
    safe_message = diagnostic.get("safe_message")
    if isinstance(safe_message, str) and safe_message.strip():
        return safe_message
    return "Ark Vision 实验解析失败"


def _ark_diagnostic_http_status(preflight: dict[str, Any], diagnostic: dict[str, Any] | None) -> int:
    if not preflight.get("success") and preflight.get("stage") == "config":
        return 503
    if diagnostic is None:
        return 502
    if not diagnostic.get("success") and diagnostic.get("stage") == "config":
        return 503
    return 502


def _deepseek_preflight_detail(preflight: dict[str, Any]) -> str:
    if preflight.get("success"):
        return "DeepSeek text preflight 成功；若当前 Semantic Parser 仍失败，可继续聚焦真实 Prompt、模型 JSON 或 parser。"
    stage = preflight.get("stage")
    if stage == "config":
        return "DeepSeek text preflight 失败，优先排查 API Key、模型配置或环境变量。"
    if stage == "http":
        return "DeepSeek text preflight 失败，优先排查余额、模型权限或上游接口状态。"
    if stage == "timeout":
        return "DeepSeek text preflight 超时，优先排查网络与上游响应时延。"
    safe_message = preflight.get("safe_message")
    if isinstance(safe_message, str) and safe_message.strip():
        return safe_message
    return "DeepSeek text preflight 失败。"


def _deepseek_preflight_http_status(preflight: dict[str, Any]) -> int:
    if preflight.get("success"):
        return 200
    stage = preflight.get("stage")
    if stage == "config":
        return 503
    if stage == "http" and preflight.get("status_code") in {401, 402, 422, 429}:
        return 502
    if stage in {"network", "timeout", "http"}:
        return 502
    return 500


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"starting workchain, port={os.environ.get('PORT')}, cwd={os.getcwd()}")
        demo_dir = _get_demo_dir()
        sandbox_root = _get_sandbox_root()
        global_meta_db_path = _global_meta_db_path()
        db_path = demo_dir / "workchain.db"
        try:
            if not demo_dir.exists() or not db_path.exists():
                seed_demo_data(demo_dir)

            cleanup_expired(sandbox_root)
            global_meta_db_path.parent.mkdir(parents=True, exist_ok=True)
            global_meta_conn = _open_meta_connection(global_meta_db_path)
            global_meta_conn.close()
            conn = _open_readonly_connection(db_path)
            try:
                evidence_count = conn.execute("SELECT COUNT(*) AS count FROM evidence").fetchone()["count"]
            finally:
                conn.close()
            print(f"demo data ready: {evidence_count} records")
        except Exception:
            traceback.print_exc()
            raise

        app.state.demo_dir = demo_dir
        app.state.db_path = db_path
        app.state.sandbox_root = sandbox_root
        app.state.global_meta_db_path = global_meta_db_path
        yield

    app = FastAPI(title="WorkChain", lifespan=lifespan)

    @app.get("/healthz")
    def healthz(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        evidence_count = conn.execute("SELECT COUNT(*) AS count FROM evidence").fetchone()["count"]
        return {"status": "ok", "evidence_count": evidence_count}

    @app.get("/api/diag/llm")
    def diag_llm() -> dict[str, Any]:
        if not _diagnostics_enabled():
            raise HTTPException(status_code=404, detail="diagnostics disabled")
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return {
                "configured": False,
                "reachable": None,
                "detail": "DEEPSEEK_API_KEY not set",
            }

        url = "https://api.deepseek.com/chat/completions"
        start = time.perf_counter()
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm.get_deepseek_model(),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=10.0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            reachable = response.status_code == 200
            if reachable:
                detail = "DeepSeek API reachable; this probe consumed a very small number of tokens"
            else:
                detail = (
                    f"HTTP {response.status_code} from DeepSeek API; "
                    "this probe consumed a very small number of tokens"
                )
            return {
                "configured": True,
                "reachable": reachable,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "detail": _safe_diag_detail(detail, api_key),
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            detail = (
                f"{type(exc).__name__}: {exc}; "
                "this probe consumed a very small number of tokens"
            )
            return {
                "configured": True,
                "reachable": False,
                "status_code": None,
                "latency_ms": latency_ms,
                "detail": _safe_diag_detail(detail, api_key),
            }

    @app.get("/api/diag/ocr")
    def diag_ocr() -> dict[str, Any]:
        if not _diagnostics_enabled():
            raise HTTPException(status_code=404, detail="diagnostics disabled")
        return ocr.diagnose_ocr()

    @app.post("/api/evidence/{evidence_id}/diagnostics/ark-vision")
    def evidence_ark_vision_diagnostic(
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        if not _diagnostics_enabled():
            raise HTTPException(status_code=404, detail="diagnostics disabled")

        conn = init_db(sandbox.db_path)
        try:
            evidence = _fetch_evidence_detail(conn, evidence_id)
            if evidence is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            if evidence["media_type"] != "image":
                raise HTTPException(status_code=400, detail="只有图片记录支持 Ark Vision 实验解析")
            if not evidence["blob_path"]:
                raise HTTPException(status_code=404, detail="blob not found")

            blob_path = sandbox.blobs_root / evidence["blob_path"]
            if not blob_path.exists():
                raise HTTPException(status_code=404, detail="blob not found")

            latest_extraction = get_latest_extraction(conn, evidence_id)
            evidence_source_hint_row = conn.execute(
                "SELECT source_hint FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            evidence_source_hint = (
                None if evidence_source_hint_row is None else evidence_source_hint_row["source_hint"]
            )
            baseline = _build_extraction_baseline_summary(latest_extraction)
        finally:
            conn.close()

        image_bytes = blob_path.read_bytes()
        mime_type = _detect_blob_content_type(image_bytes, evidence.get("filename"))
        text_preflight = vision_provider.diagnose_text_preflight()
        if not text_preflight["success"]:
            detail = _ark_diagnostic_detail(text_preflight, None)
            _log_ark_diagnostic_attempt(
                evidence_id=evidence_id,
                model=text_preflight.get("model"),
                status="failed",
                latency_ms=int(text_preflight.get("latency_ms") or 0),
                error_type=text_preflight.get("error_type") or text_preflight.get("error_code") or text_preflight.get("stage"),
            )
            return JSONResponse(
                status_code=_ark_diagnostic_http_status(text_preflight, None),
                content=_ark_diagnostic_response(
                    baseline=baseline,
                    status="failed",
                    detail=detail,
                    extraction=None,
                    text_preflight=text_preflight,
                    diagnostic=None,
                ),
            )

        diagnostic = vision_provider.diagnose_visual_evidence(
            image_bytes,
            mime_type,
            source_hint=evidence_source_hint,
        )
        detail = _ark_diagnostic_detail(text_preflight, diagnostic)
        if not diagnostic["success"]:
            _log_ark_diagnostic_attempt(
                evidence_id=evidence_id,
                model=diagnostic.get("model"),
                status="failed",
                latency_ms=int(diagnostic.get("latency_ms") or 0),
                error_type=diagnostic.get("error_type") or diagnostic.get("error_code") or diagnostic.get("stage"),
            )
            return JSONResponse(
                status_code=_ark_diagnostic_http_status(text_preflight, diagnostic),
                content=_ark_diagnostic_response(
                    baseline=baseline,
                    status="failed",
                    detail=detail,
                    extraction=None,
                    text_preflight=text_preflight,
                    diagnostic=diagnostic,
                ),
            )

        extraction = diagnostic.get("extraction")
        _log_ark_diagnostic_attempt(
            evidence_id=evidence_id,
            model=diagnostic.get("model"),
            status="succeeded",
            latency_ms=int(diagnostic.get("latency_ms") or 0),
            extraction=extraction,
            error_type=None,
        )
        return JSONResponse(
            _ark_diagnostic_response(
                baseline=baseline,
                status="succeeded",
                detail=detail,
                extraction=extraction,
                text_preflight=text_preflight,
                diagnostic=diagnostic,
            )
        )

    @app.post("/api/evidence/{evidence_id}/diagnostics/deepseek-preflight")
    def evidence_deepseek_preflight(
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        if not _diagnostics_enabled():
            raise HTTPException(status_code=404, detail="diagnostics disabled")

        conn = init_db(sandbox.db_path)
        try:
            evidence = _fetch_evidence_detail(conn, evidence_id)
            if evidence is None:
                raise HTTPException(status_code=404, detail="evidence not found")
        finally:
            conn.close()

        preflight = diagnose_deepseek_text_preflight()
        return JSONResponse(
            status_code=_deepseek_preflight_http_status(preflight),
            content={
                "status": "succeeded" if preflight.get("success") else "failed",
                "detail": _deepseek_preflight_detail(preflight),
                "deepseek_text_preflight": preflight,
            },
        )

    @app.get("/api/evidence/{evidence_id}/diagnostics")
    def evidence_diagnostics(
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> dict[str, Any]:
        if not _diagnostics_enabled():
            raise HTTPException(status_code=404, detail="diagnostics disabled")

        conn = init_db(sandbox.db_path)
        try:
            evidence = _fetch_evidence_detail(conn, evidence_id)
            if evidence is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            extract_note = _get_extract_note(conn, evidence_id)
            parse_detail = extract_note or _get_parse_detail(conn, evidence_id)
            parse_status = _get_parse_status(conn, evidence_id)
            return _build_evidence_diagnostics(
                conn,
                evidence=evidence,
                parse_status=parse_status,
                parse_detail=parse_detail,
            )
        finally:
            conn.close()

    @app.get("/api/evidence/{evidence_id}/status")
    def evidence_status(
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> dict[str, Any]:
        conn = init_db(sandbox.db_path)
        try:
            row = conn.execute(
                """
                SELECT slots_filled, plain_summary, slot_deliverable,
                       slot_due, slot_due_raw, caveats, media_type
                FROM evidence
                WHERE evidence_id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            parse_status = _get_parse_status(conn, evidence_id)
            extract_note = _get_extract_note(conn, evidence_id)
            parse_detail = extract_note or _get_parse_detail(conn, evidence_id)
            semantic_result = _build_semantic_result(conn, evidence_id)
            event_match = None if semantic_result is None else semantic_result.get("event_match")
            source_gate = (
                _current_source_gate_payload(
                    conn,
                    evidence_id=evidence_id,
                    parse_status=parse_status,
                    reviewable=parse_status == PARSE_STATUS_CLARIFICATION_REQUIRED and not evidence_id.startswith("ev_demo_"),
                )
                if row["media_type"] == "image"
                else None
            )
            return {
                "parse_status": parse_status,
                "parse_status_label": _parse_status_label(parse_status),
                "slots_filled": row["slots_filled"],
                "plain_summary": row["plain_summary"],
                "deliverable": row["slot_deliverable"],
                "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d"),
                "caveats": [_humanize_user_facing_text(item) or item for item in _decode_json_array(row["caveats"])],
                "semantic_fact_preview": [] if semantic_result is None else semantic_result["facts"][:2],
                "event_match": event_match,
                "event_assignment_panel_html": _render_event_assignment_panel_html(
                    event_match=event_match,
                    evidence_id=evidence_id,
                ),
                "detail": _humanize_user_facing_text(parse_detail) or parse_detail,
                "is_verified": _is_verified(conn, evidence_id),
                "media_type": row["media_type"],
                "is_ocr_corrected": _is_ocr_corrected(conn, evidence_id),
                "source_gate": source_gate,
            }
        finally:
            conn.close()

    @app.post("/api/evidence/{evidence_id}/source-review")
    async def submit_source_review(
        evidence_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_fields = sorted(set(payload) - {"decision", "resolved_source"})
        if unexpected_fields:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported fields: {', '.join(unexpected_fields)}",
            )

        decision = str(payload.get("decision", "")).strip()
        if decision not in {"confirmed_declared", "corrected"}:
            raise HTTPException(status_code=400, detail="decision must be confirmed_declared or corrected")

        resolved_source = _normalize_user_source_value(payload.get("resolved_source"))
        conn = init_db(sandbox.db_path)
        rerun_image_bytes: bytes | None = None
        rerun_filename: str | None = None
        response_payload: dict[str, Any]
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT evidence_id, media_type, raw_text, blob_path
                FROM evidence
                WHERE evidence_id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            if evidence_id.startswith("ev_demo_"):
                raise HTTPException(status_code=403, detail="演示记录不可修改")
            if row["media_type"] != "image":
                raise HTTPException(status_code=400, detail="只有图片记录支持核实信息来源")

            parse_status = _get_meta_value(conn, _parse_status_key(evidence_id)) or PARSE_STATUS_FAILED
            if parse_status != PARSE_STATUS_CLARIFICATION_REQUIRED:
                raise HTTPException(status_code=409, detail="当前记录不处于待核实来源状态")

            latest_extraction = get_latest_extraction(conn, evidence_id)
            if latest_extraction is None:
                raise HTTPException(status_code=400, detail="当前记录缺少可核实的提取版本")

            gate_state = _source_gate_state(
                conn,
                evidence_id=evidence_id,
                extraction=latest_extraction,
            )
            if not gate_state["requires_clarification"]:
                raise HTTPException(status_code=409, detail="当前记录不需要再次核实来源")

            current_effective_source_hint = gate_state["effective_source_hint"]
            if decision == "confirmed_declared":
                resolved_source_hint = current_effective_source_hint
                next_status = PARSE_STATUS_LLM_RUNNING
            else:
                if resolved_source is None:
                    raise HTTPException(status_code=400, detail="请选择或填写修正后的信息来源")
                resolved_source_hint = _build_resolved_source_hint(current_effective_source_hint, resolved_source)
                next_status = PARSE_STATUS_OCR_RUNNING
                if row["blob_path"] is None:
                    raise HTTPException(status_code=400, detail="当前图片缺少原始文件")
                blob_path = sandbox.blobs_root / row["blob_path"]
                if not blob_path.exists():
                    raise HTTPException(status_code=404, detail="blob not found")
                rerun_image_bytes = blob_path.read_bytes()
                rerun_filename = _extract_filename(row["raw_text"])

            create_source_review(
                conn,
                evidence_id=evidence_id,
                extraction_id=latest_extraction["extraction_id"],
                original_source_hint=current_effective_source_hint,
                observed_platform=gate_state["observed_platform"],
                resolved_source_hint=resolved_source_hint,
                decision=decision,
            )
            _clear_extract_note(conn, evidence_id)
            _set_meta_value(conn, _parse_status_key(evidence_id), next_status)
            _set_meta_value(conn, _parse_detail_key(evidence_id), "")
            conn.commit()
            response_payload = {
                "evidence_id": evidence_id,
                "decision": decision,
                "parse_status": next_status,
                "source_gate": _current_source_gate_payload(
                    conn,
                    evidence_id=evidence_id,
                    parse_status=next_status,
                    reviewable=False,
                ),
            }
        except HTTPException:
            if conn.in_transaction:
                conn.rollback()
            raise
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

        if decision == "confirmed_declared":
            background_tasks.add_task(
                _run_parse_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                evidence_id,
            )
        else:
            background_tasks.add_task(
                _run_image_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                evidence_id,
                rerun_image_bytes,
                rerun_filename,
                None,
            )

        response = JSONResponse(response_payload)
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/evidence/{evidence_id}/record-date")
    async def update_evidence_record_date(
        evidence_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_fields = sorted(set(payload) - {"record_date"})
        if unexpected_fields:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported fields: {', '.join(unexpected_fields)}",
            )

        record_date = _normalize_record_date_input(payload.get("record_date"))
        if record_date is None:
            raise HTTPException(status_code=400, detail="请选择记录发生日期")

        conn = init_db(sandbox.db_path)
        try:
            evidence_row = conn.execute(
                "SELECT evidence_id FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if evidence_row is None:
                raise HTTPException(status_code=404, detail="evidence not found")

            latest_run = get_latest_semantic_run_for_evidence(conn, evidence_id, status="succeeded")
            relative_due_updates: list[dict[str, Any]] = []
            if latest_run is not None:
                latest_facts = list_facts_for_semantic_run(
                    conn,
                    latest_run["semantic_run_id"],
                    evidence_id=evidence_id,
                )
                relative_due_updates = _build_relative_due_updates(
                    latest_facts,
                    anchor_date=record_date,
                )

            updated_at = int(time.time() * 1000)
            conn.execute("BEGIN IMMEDIATE")
            _set_semantic_anchor(conn, evidence_id, record_date, "user", commit=False)
            if latest_run is not None and relative_due_updates:
                correct_relative_due_dates_by_user(
                    conn,
                    evidence_id=evidence_id,
                    semantic_run_id=latest_run["semantic_run_id"],
                    due_updates=relative_due_updates,
                    updated_at=updated_at,
                )
            conn.commit()
        except HTTPException:
            if conn.in_transaction:
                conn.rollback()
            raise
        except SemanticStoreError as exc:
            if conn.in_transaction:
                conn.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

        response = JSONResponse(
            {
                "evidence_id": evidence_id,
                "record_date": record_date,
                "record_date_source": "user",
                "record_date_source_label": _semantic_anchor_source_label("user"),
                "updated_fact_count": len(relative_due_updates),
            }
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/evidence/{evidence_id}/event-assignment")
    async def confirm_event_assignment(
        evidence_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_top_level = sorted(set(payload) - {"event_match_run_id", "groups"})
        if unexpected_top_level:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported fields: {', '.join(unexpected_top_level)}",
            )

        event_match_run_id = payload.get("event_match_run_id")
        if not isinstance(event_match_run_id, str) or not event_match_run_id.strip():
            raise HTTPException(status_code=400, detail="event_match_run_id is required")
        groups = payload.get("groups")
        if not isinstance(groups, list) or not groups:
            raise HTTPException(status_code=400, detail="groups must be a non-empty list")

        normalized_groups: list[dict[str, Any]] = []
        for item in groups:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="each group decision must be an object")
            unexpected_group_fields = sorted(set(item) - {"group_index", "choice", "event_id", "new_title"})
            if unexpected_group_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"unsupported group fields: {', '.join(unexpected_group_fields)}",
                )
            normalized_groups.append(item)

        conn = init_db(sandbox.db_path)
        try:
            evidence_row = conn.execute(
                "SELECT evidence_id FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if evidence_row is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            try:
                reviewed_run = review_event_match_run_by_user(
                    conn,
                    evidence_id=evidence_id,
                    event_match_run_id=event_match_run_id.strip(),
                    decisions=normalized_groups,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            affected_event_rows = conn.execute(
                """
                SELECT DISTINCT event_id
                FROM facts
                WHERE semantic_run_id = ?
                  AND event_id IS NOT NULL
                ORDER BY event_id ASC
                """,
                (reviewed_run["semantic_run_id"],),
            ).fetchall()
            for row in affected_event_rows:
                _maybe_run_event_change_detection(conn, row["event_id"])
            return JSONResponse(
                {
                    "event_match_run_id": reviewed_run["event_match_run_id"],
                    "review_status": reviewed_run["review_status"],
                    "reviewed_at": reviewed_run["reviewed_at"],
                }
            )
        finally:
            conn.close()

    @app.patch("/api/evidence/{evidence_id}/slots")
    async def patch_evidence_slots(
        evidence_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        allowed_fields = {
            "slot_deliverable",
            "slot_due_raw",
            "slot_due_date",
            "slot_direction",
            "kind",
            "plain_summary",
            "caveats",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise HTTPException(status_code=400, detail="包含不允许修改的字段")

        conn = init_db(sandbox.db_path)
        try:
            row = conn.execute(
                """
                SELECT evidence_id, raw_text, source_hint, seq, content_hash, chain_hash,
                       prev_hash, occurred_at, captured_at, media_type
                FROM evidence
                WHERE evidence_id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            if evidence_id.startswith("ev_demo_"):
                raise HTTPException(status_code=403, detail="演示记录不可修改")

            slot_due_value = payload.get("slot_due_date")
            slot_due = None
            if slot_due_value:
                slot_due = llm.due_date_to_millis(str(slot_due_value))

            caveats = payload.get("caveats", [])
            if not isinstance(caveats, list):
                caveats = []
            caveats = [str(item).strip() for item in caveats if str(item).strip()]

            slot_direction = payload.get("slot_direction")
            if slot_direction not in {"i_owe", "owed_to_me", "none"}:
                slot_direction = "none"

            kind = payload.get("kind")
            if kind not in {"request", "confirm", "change", "deliver", "dispute", "reference"}:
                kind = "reference"

            updated = update_slots(
                conn,
                evidence_id,
                slot_deliverable=str(payload.get("slot_deliverable", "")).strip() or None,
                slot_due=slot_due,
                slot_due_raw=str(payload.get("slot_due_raw", "")).strip() or None,
                slot_direction=slot_direction,
                plain_summary=str(payload.get("plain_summary", "")).strip() or None,
                caveats=caveats,
            )
            conn.execute("UPDATE evidence SET kind = ? WHERE evidence_id = ?", (kind, evidence_id))
            _set_verified(conn, evidence_id)
            updated = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
            conn.commit()
            response = JSONResponse(dict(updated))
            apply_sandbox_cookie(response, sandbox)
            return response
        finally:
            conn.close()

    @app.patch("/api/evidence/{evidence_id}/ocr_text")
    async def patch_evidence_ocr_text(
        evidence_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        corrected_text = str(payload.get("text", "")).strip()
        if len(corrected_text) > MAX_OCR_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail="识别文字过长,请控制在 50000 字以内")

        conn = init_db(sandbox.db_path)
        try:
            row = conn.execute(
                """
                SELECT evidence_id, raw_text, media_type, content_hash, chain_hash
                FROM evidence
                WHERE evidence_id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            if evidence_id.startswith("ev_demo_"):
                raise HTTPException(status_code=403, detail="演示记录不可修改")
            if row["media_type"] != "image":
                raise HTTPException(status_code=400, detail="只有图片记录支持校正识别文字")

            filename = _extract_filename(row["raw_text"])
            conn.execute(
                "UPDATE evidence SET raw_text = ? WHERE evidence_id = ?",
                (_build_attachment_raw_text("image", filename, corrected_text), evidence_id),
            )
            _record_user_extraction(
                conn,
                evidence_id=evidence_id,
                transcript=corrected_text,
            )
            _clear_extract_note(conn, evidence_id)
            _set_ocr_corrected(conn, evidence_id, True)
            latest_extraction = get_latest_extraction(conn, evidence_id)
            gate_state = _source_gate_state(
                conn,
                evidence_id=evidence_id,
                extraction=latest_extraction,
            )
            if gate_state["requires_clarification"]:
                _set_parse_status(conn, evidence_id, PARSE_STATUS_CLARIFICATION_REQUIRED)
                _set_parse_detail(conn, evidence_id, _clarification_detail(gate_state))
            else:
                _set_parse_status(conn, evidence_id, PARSE_STATUS_LLM_RUNNING)
                _set_parse_detail(conn, evidence_id, "")
            conn.commit()
        finally:
            conn.close()

        if gate_state["requires_clarification"]:
            parse_status = PARSE_STATUS_CLARIFICATION_REQUIRED
        else:
            background_tasks.add_task(
                _run_parse_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                evidence_id,
            )
            parse_status = PARSE_STATUS_LLM_RUNNING
        response = JSONResponse(
            {
                "evidence_id": evidence_id,
                "parse_status": parse_status,
            }
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/api/settings")
    def settings_get(sandbox: SandboxContext = Depends(get_sandbox)) -> dict[str, Any]:
        return _settings_payload(sandbox.db_path)

    @app.post("/api/settings")
    async def settings_save(
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        self_names = payload.get("self_names", [])
        glossary = payload.get("glossary", [])

        if isinstance(self_names, list) and len(self_names) > 5:
            raise HTTPException(status_code=400, detail="你在对话里的称呼最多填 5 个")
        if isinstance(glossary, list) and len(glossary) > 50:
            raise HTTPException(status_code=400, detail="我的词典最多保留 50 条")

        try:
            saved_self_names = save_self_names(sandbox.db_path, self_names if isinstance(self_names, list) else [])
            saved_glossary = save_glossary(sandbox.db_path, glossary if isinstance(glossary, list) else [])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response = JSONResponse(
            {
                "self_names": saved_self_names,
                "glossary": saved_glossary,
                "has_self_names": bool(saved_self_names),
            }
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/search", response_class=HTMLResponse)
    def search_page(
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        q = str(request.query_params.get("q", "")).strip()[:100]
        source = str(request.query_params.get("source", "")).strip() or None
        kind = str(request.query_params.get("kind", "")).strip() or None
        start = str(request.query_params.get("start", "")).strip() or None
        end = str(request.query_params.get("end", "")).strip() or None
        results = _search_evidence(conn, q=q, source=source, kind=kind, start=start, end=end) if q else []

        response = TEMPLATES.TemplateResponse(
            request,
            "search.html",
            {
                "page_title": "搜索",
                "current_search_q": q,
                "query": q,
                "selected_source": source or "",
                "selected_kind": kind or "",
                "selected_start": start or "",
                "selected_end": end or "",
                "results": results,
                "result_count": len(results),
                "source_presets": SOURCE_PRESETS,
                "kind_options": KIND,
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/records", response_class=HTMLResponse)
    def records_page(
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        records = _fetch_recent_records_data(conn, sandbox, limit=100)
        response = TEMPLATES.TemplateResponse(
            request,
            "records.html",
            {
                "page_title": "记录",
                "records": records,
                "record_limit": 100,
                "current_search_q": "",
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_index_data(conn, sandbox)
        response = TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "page_title": "WorkChain",
                "my_events": context["my_events"],
                "pending_event_cards": context["pending_event_cards"],
                "history_events": context["history_events"],
                "demo_threads": context["demo_threads"],
                "diagnostics_enabled": _diagnostics_enabled(),
                "references": context["references"],
                "recent_records": context["recent_records"],
                "source_presets": SOURCE_PRESETS,
                "settings": _settings_payload(sandbox.db_path),
                "current_search_q": "",
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/events/{event_id}/title")
    async def update_event_title(
        event_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_fields = sorted(set(payload) - {"title"})
        if unexpected_fields:
            raise HTTPException(status_code=400, detail=f"unsupported fields: {', '.join(unexpected_fields)}")
        raw_title = payload.get("title")
        if not isinstance(raw_title, str):
            raise HTTPException(status_code=400, detail="事项名称不能为空")
        title = " ".join(raw_title.strip().split())
        if not title:
            raise HTTPException(status_code=400, detail="事项名称不能为空")
        if len(title) > MAX_USER_EVENT_TITLE_LENGTH:
            raise HTTPException(status_code=400, detail=f"事项名称不能超过 {MAX_USER_EVENT_TITLE_LENGTH} 个字")

        updated_at = int(time.time() * 1000)
        conn = init_db(sandbox.db_path)
        try:
            row = conn.execute("SELECT event_id FROM events WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="event not found")
            conn.execute(
                "UPDATE events SET title = ?, updated_at = ? WHERE event_id = ?",
                (title, updated_at, event_id),
            )
            conn.commit()
        finally:
            conn.close()
        response = JSONResponse({"event_id": event_id, "title": title, "updated_at": updated_at})
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/events/{event_id}/status")
    async def update_event_status(
        event_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_fields = sorted(set(payload) - {"status"})
        if unexpected_fields:
            raise HTTPException(status_code=400, detail=f"unsupported fields: {', '.join(unexpected_fields)}")
        target_status = str(payload.get("status", "")).strip()
        if target_status not in {"active", "resolved"}:
            raise HTTPException(status_code=400, detail="status must be active or resolved")

        updated_at = int(time.time() * 1000)
        conn = init_db(sandbox.db_path)
        try:
            row = conn.execute(
                "SELECT status FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="event not found")
            current_status = row["status"]
            allowed = {("active", "resolved"), ("resolved", "active")}
            if (current_status, target_status) not in allowed:
                raise HTTPException(status_code=400, detail="unsupported status transition")
            conn.execute(
                "UPDATE events SET status = ?, updated_at = ? WHERE event_id = ?",
                (target_status, updated_at, event_id),
            )
            conn.commit()
        finally:
            conn.close()
        response = JSONResponse(
            {
                "event_id": event_id,
                "status": target_status,
                "status_label": _event_status_label(target_status),
                "updated_at": updated_at,
            }
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/events/{event_id}/facts/{fact_id}/correct")
    async def correct_event_fact(
        event_id: str,
        fact_id: str,
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid request body")
        unexpected_fields = sorted(set(payload) - {"content", "fact_type", "due_at"})
        if unexpected_fields:
            raise HTTPException(status_code=400, detail=f"unsupported fields: {', '.join(unexpected_fields)}")

        content = str(payload.get("content", "")).strip()
        fact_type = str(payload.get("fact_type", "")).strip()
        if not content:
            raise HTTPException(status_code=400, detail="事实内容不能为空")
        if fact_type not in {item["value"] for item in _event_fact_type_options()}:
            raise HTTPException(status_code=400, detail="fact_type 不合法")
        due_at = _parse_due_date_input(payload.get("due_at"))
        updated_at = int(time.time() * 1000)

        conn = init_db(sandbox.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            fact_row = conn.execute(
                "SELECT event_id FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if fact_row is None:
                raise HTTPException(status_code=404, detail="fact not found")
            if fact_row["event_id"] != event_id:
                raise HTTPException(status_code=400, detail="fact does not belong to event")
            try:
                corrected = correct_fact_by_user(
                    conn,
                    fact_id=fact_id,
                    fact_type=fact_type,
                    content=content,
                    due_at=due_at,
                    due_raw=None,
                    due_anchor_at=None,
                    updated_at=updated_at,
                )
            except SemanticStoreError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            conn.execute("UPDATE events SET updated_at = ? WHERE event_id = ?", (updated_at, event_id))
            conn.commit()
        except HTTPException:
            if conn.in_transaction:
                conn.rollback()
            raise
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()
        response = JSONResponse(
            {
                "event_id": event_id,
                "fact_id": fact_id,
                "origin": corrected["origin"],
                "review_status": corrected["review_status"],
                "due_at": corrected["due_at"],
            }
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/event/{event_id}", response_class=HTMLResponse)
    def event_detail(
        request: Request,
        event_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_event_detail(conn, event_id)
        if context is None:
            raise HTTPException(status_code=404, detail="event not found")
        response = TEMPLATES.TemplateResponse(
            request,
            "event.html",
            {
                "page_title": context["event"]["title"],
                "event": context["event"],
                "related_evidence": context["related_evidence"],
                "event_changes": context["event_changes"],
                "facts": context["facts"],
                "fact_type_options": context["fact_type_options"],
                "current_search_q": "",
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/evidence/{evidence_id}", response_class=HTMLResponse)
    def evidence_detail(
        request: Request,
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_evidence_detail(conn, evidence_id)
        if context is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        status_conn = init_db(sandbox.db_path)
        try:
            parse_status = _get_parse_status(status_conn, evidence_id)
            extract_note = _get_extract_note(status_conn, evidence_id)
            parse_detail = _humanize_user_facing_text(extract_note or _get_parse_detail(status_conn, evidence_id)) or (
                extract_note or _get_parse_detail(status_conn, evidence_id)
            )
            is_verified = _is_verified(status_conn, evidence_id)
            is_ocr_corrected = _is_ocr_corrected(status_conn, evidence_id)
            source_gate = (
                _current_source_gate_payload(
                    status_conn,
                    evidence_id=evidence_id,
                    parse_status=parse_status,
                    reviewable=parse_status == PARSE_STATUS_CLARIFICATION_REQUIRED and not evidence_id.startswith("ev_demo_"),
                )
                if context["media_type"] == "image"
                else None
            )
            diagnostics = None
            diagnostics_json = None
            if _diagnostics_enabled():
                diagnostics = _build_evidence_diagnostics(
                    status_conn,
                    evidence=context,
                    parse_status=parse_status,
                    parse_detail=parse_detail,
                )
                diagnostics_json = json.dumps(diagnostics, ensure_ascii=False, indent=2)
        finally:
            status_conn.close()

        response = TEMPLATES.TemplateResponse(
            request,
            "evidence.html",
            {
                "page_title": "记录详情",
                "evidence": context,
                "show_legacy_editor": _should_show_legacy_editor(
                    evidence=context,
                    parse_status=parse_status,
                ),
                "parse_status": parse_status,
                "parse_status_label": _parse_status_label(parse_status),
                "parse_detail": parse_detail,
                "is_verified": is_verified,
                "is_ocr_corrected": is_ocr_corrected,
                "source_gate": source_gate,
                "source_presets": SOURCE_PRESETS,
                "diagnostics_enabled": _diagnostics_enabled(),
                "diagnostics": diagnostics,
                "diagnostics_json": diagnostics_json,
                "current_search_q": "",
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/blob/{evidence_id}")
    def blob_detail(
        evidence_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> Response:
        row = conn.execute(
            """
            SELECT evidence_id, media_type, blob_path, raw_text
            FROM evidence
            WHERE evidence_id = ?
            """,
            (evidence_id,),
        ).fetchone()
        if row is None or row["blob_path"] is None:
            raise HTTPException(status_code=404, detail="blob not found")

        blob_path = sandbox.blobs_root / row["blob_path"]
        if not blob_path.exists():
            raise HTTPException(status_code=404, detail="blob not found")

        blob_bytes = blob_path.read_bytes()
        filename = _extract_filename(row["raw_text"])
        content_type = _detect_blob_content_type(blob_bytes, filename)
        disposition = "inline" if row["media_type"] == "image" else "attachment"
        disposition = _build_content_disposition(disposition, filename)

        response = Response(content=blob_bytes, media_type=content_type)
        response.headers["Content-Disposition"] = disposition
        response.headers["X-Content-Type-Options"] = "nosniff"
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/export/pdf")
    def export_pdf(
        thread_id: str | None = None,
        scope: str | None = None,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> FileResponse:
        if thread_id:
            if not _thread_exists(conn, thread_id):
                raise HTTPException(status_code=404, detail="thread not found")
        elif scope != "mine":
            raise HTTPException(status_code=400, detail="请指定 thread_id 或 scope=mine")
        elif not _mine_evidence_ids(conn):
            raise HTTPException(status_code=404, detail="no evidence to export")

        temp_dir = Path(tempfile.mkdtemp(prefix="workchain-pdf-"))
        pdf_name = _pdf_download_name()
        pdf_path = temp_dir / pdf_name
        build_evidence_pdf(
            conn,
            blobs_root=sandbox.blobs_root,
            thread_id=thread_id,
            scope=scope,
            out_path=pdf_path,
        )
        response = FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=pdf_name,
            background=BackgroundTask(_cleanup_temp_dir, temp_dir),
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/export/package")
    def export_package(
        thread_id: str | None = None,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> FileResponse:
        temp_dir = Path(tempfile.mkdtemp(prefix="workchain-package-"))
        package_dir = temp_dir / "package"
        package_dir.mkdir(parents=True, exist_ok=True)
        pdf_name = _pdf_download_name()
        build_evidence_pdf(
            conn,
            blobs_root=sandbox.blobs_root,
            scope="all",
            out_path=package_dir / pdf_name,
        )
        export_evidence_package(
            conn,
            blobs_root=sandbox.blobs_root,
            out_dir=package_dir,
        )
        (package_dir / "怎么验证这份材料.txt").write_text(PACKAGE_README_TEXT, encoding="utf-8")

        zip_name = _package_download_name()
        zip_path = temp_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(package_dir))
        response = FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=zip_name,
            background=BackgroundTask(_cleanup_temp_dir, temp_dir),
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/evidence")
    async def create_evidence(
        request: Request,
        background_tasks: BackgroundTasks,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await _parse_evidence_input(request)
        text = payload["text"]
        record_date = payload["record_date"]
        uploads = payload["uploads"]
        if not text and not uploads:
            raise HTTPException(status_code=400, detail="请输入内容或选择文件")
        if text and uploads:
            raise HTTPException(status_code=400, detail="文字记录和文件不能同时提交，请二选一。")
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail="内容过长,请控制在 20000 字以内")

        source = payload["source"]
        if not source:
            raise HTTPException(status_code=400, detail="请选择或填写来源")
        if source not in SOURCE_PRESETS and len(source) > 20:
            raise HTTPException(status_code=400, detail="自定义来源不能超过 20 个字")

        source_detail = payload["source_detail"]
        source_hint = source if not source_detail else f"{source}-{source_detail}"
        submission_size = 1 if text else len(uploads)

        if len(uploads) > 1:
            if record_date is not None:
                raise HTTPException(
                    status_code=400,
                    detail="多张图片不能同时补充同一个记录日期，请逐张上传或清空日期。",
                )
            if any(item["media_type"] != "image" for item in uploads):
                raise HTTPException(
                    status_code=400,
                    detail="一次上传多个文件时，只支持多张图片；文档一次只能传一个。",
                )

        conn = _open_write_connection(sandbox.db_path)
        try:
            today_count = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM evidence
                WHERE evidence_id NOT LIKE 'ev_demo_%' AND captured_at >= ?
                """,
                (_start_of_current_day_ms(),),
            ).fetchone()["count"]
            if today_count + submission_size > 50:
                if today_count >= 50:
                    raise HTTPException(status_code=429, detail="你今天已经存了 50 条,明天再来继续")
                raise HTTPException(status_code=429, detail="本次提交后今天会超过 50 条，请减少图片数量后再试")

            if uploads:
                _ensure_upload_budget(
                    conn,
                    sandbox.blobs_root,
                    [item["file_bytes"] for item in uploads],
                )

            now_ms = int(time.time() * 1000)
            prepared_items: list[dict[str, Any]] = []
            if text:
                prepared_items.append(
                    {
                        "media_type": "text",
                        "append_payload": text,
                        "raw_text_override": None,
                        "parse_status": PARSE_STATUS_LLM_RUNNING,
                        "parse_detail": "",
                        "extract_note": None,
                        "extracted_transcript": text,
                        "filename": None,
                        "image_pipeline_item": None,
                    }
                )
            else:
                for upload_item in uploads:
                    media_type = upload_item["media_type"]
                    file_bytes = upload_item["file_bytes"]
                    filename = upload_item["filename"]
                    prepared_item = {
                        "media_type": media_type,
                        "append_payload": file_bytes,
                        "raw_text_override": _build_attachment_raw_text(media_type, filename),
                        "parse_status": PARSE_STATUS_LLM_RUNNING,
                        "parse_detail": "",
                        "extract_note": None,
                        "extracted_transcript": None,
                        "filename": filename,
                        "image_pipeline_item": None,
                    }
                    if media_type == "image":
                        image_startup = get_image_extraction_startup()
                        if not image_startup["supported"] or not image_startup["configured"]:
                            prepared_item["parse_status"] = PARSE_STATUS_UNSUPPORTED
                            prepared_item["extract_note"] = image_startup["detail"] or "图片提取暂不可用"
                            prepared_item["parse_detail"] = _saved_original_detail(prepared_item["extract_note"])
                        elif image_startup["requires_ocr_budget_on_start"]:
                            allowed, reason = _consume_ocr_budget(
                                sandbox.db_path,
                                request.app.state.global_meta_db_path,
                            )
                            if allowed:
                                prepared_item["parse_status"] = PARSE_STATUS_OCR_RUNNING
                                prepared_item["image_pipeline_item"] = {
                                    "image_bytes": file_bytes,
                                    "filename": filename,
                                }
                            else:
                                prepared_item["parse_status"] = PARSE_STATUS_UNSUPPORTED
                                prepared_item["extract_note"] = reason
                                prepared_item["parse_detail"] = _saved_original_detail(
                                    reason or "图片识别暂不可用"
                                )
                        else:
                            prepared_item["parse_status"] = PARSE_STATUS_OCR_RUNNING
                            prepared_item["image_pipeline_item"] = {
                                "image_bytes": file_bytes,
                                "filename": filename,
                            }
                    else:
                        extracted_text, extract_status = extract_text(file_bytes, media_type, filename or "")
                        if extracted_text is not None:
                            prepared_item["raw_text_override"] = _build_attachment_raw_text(
                                media_type,
                                filename,
                                extracted_text,
                            )
                            prepared_item["extracted_transcript"] = extracted_text
                        else:
                            prepared_item["parse_status"] = PARSE_STATUS_UNSUPPORTED
                            prepared_item["extract_note"] = extract_status
                            prepared_item["parse_detail"] = _saved_original_detail(extract_status)
                    prepared_items.append(prepared_item)

            created_items: list[dict[str, Any]] = []
            conn.execute("BEGIN IMMEDIATE")
            try:
                for prepared_item in prepared_items:
                    row = append_evidence(
                        conn,
                        blobs_root=sandbox.blobs_root,
                        media_type=prepared_item["media_type"],
                        payload=prepared_item["append_payload"],
                        captured_at=now_ms,
                        occurred_at=now_ms,
                        source_hint=source_hint,
                        kind="reference",
                    )
                    if prepared_item["media_type"] == "text" and text:
                        _record_machine_extraction(
                            conn,
                            evidence_id=row["evidence_id"],
                            transcript=text,
                            observations=[],
                            provider="builtin",
                            model=None,
                            warnings=[],
                            created_at=now_ms,
                        )
                    if prepared_item["raw_text_override"] is not None:
                        conn.execute(
                            "UPDATE evidence SET raw_text = ?, plain_summary = ? WHERE evidence_id = ?",
                            (prepared_item["raw_text_override"], text or None, row["evidence_id"]),
                        )
                        if (
                            prepared_item["media_type"] != "image"
                            and prepared_item["extracted_transcript"]
                        ):
                            _record_machine_extraction(
                                conn,
                                evidence_id=row["evidence_id"],
                                transcript=prepared_item["extracted_transcript"],
                                observations=[],
                                provider="builtin",
                                model=None,
                                warnings=[],
                                created_at=now_ms,
                            )
                        row = conn.execute(
                            "SELECT * FROM evidence WHERE evidence_id = ?",
                            (row["evidence_id"],),
                        ).fetchone()
                        row = dict(row)
                    _set_parse_status(conn, row["evidence_id"], prepared_item["parse_status"])
                    _set_parse_detail(conn, row["evidence_id"], prepared_item["parse_detail"])
                    if prepared_item["extract_note"] is not None:
                        _set_extract_note(conn, row["evidence_id"], prepared_item["extract_note"])
                    if record_date is not None:
                        _set_semantic_anchor(conn, row["evidence_id"], record_date, "user", commit=False)
                    created_items.append(
                        {
                            "row": row,
                            "parse_status": prepared_item["parse_status"],
                            "image_pipeline_item": prepared_item["image_pipeline_item"],
                        }
                    )

                submission = create_submission(
                    conn,
                    evidence_ids=[item["row"]["evidence_id"] for item in created_items],
                    created_at=now_ms,
                    source_hint=source_hint,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

        image_pipeline_items = [
            {
                "evidence_id": item["row"]["evidence_id"],
                "image_bytes": item["image_pipeline_item"]["image_bytes"],
                "filename": item["image_pipeline_item"]["filename"],
            }
            for item in created_items
            if item["parse_status"] == PARSE_STATUS_OCR_RUNNING and item["image_pipeline_item"] is not None
        ]
        llm_pipeline_evidence_ids = [
            item["row"]["evidence_id"]
            for item in created_items
            if item["parse_status"] == PARSE_STATUS_LLM_RUNNING
        ]

        for evidence_id in llm_pipeline_evidence_ids:
            background_tasks.add_task(
                _run_parse_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                evidence_id,
            )
        if len(image_pipeline_items) == 1:
            image_item = image_pipeline_items[0]
            background_tasks.add_task(
                _run_image_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                image_item["evidence_id"],
                image_item["image_bytes"],
                image_item["filename"],
                None,
            )
        elif len(image_pipeline_items) > 1:
            background_tasks.add_task(
                _run_multi_image_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                image_pipeline_items,
            )

        response_payload: dict[str, Any] = {
            "submission_id": submission["submission_id"],
        }
        if len(created_items) == 1:
            row = created_items[0]["row"]
            response_payload.update(
                {
                    "evidence_id": row["evidence_id"],
                    "seq": row["seq"],
                    "occurred_at": row["occurred_at"],
                    "parse_status": created_items[0]["parse_status"],
                    "media_type": row["media_type"],
                }
            )
        else:
            response_payload.update(
                {
                    "evidence_ids": [item["row"]["evidence_id"] for item in created_items],
                    "items": [
                        {
                            "evidence_id": item["row"]["evidence_id"],
                            "seq": item["row"]["seq"],
                            "occurred_at": item["row"]["occurred_at"],
                            "parse_status": item["parse_status"],
                            "media_type": item["row"]["media_type"],
                        }
                        for item in created_items
                    ],
                }
            )

        response = JSONResponse(response_payload)
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.get("/help", response_class=HTMLResponse)
    def help_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "help.html",
            {
                "page_title": "使用说明",
                "current_search_q": "",
            },
        )

    @app.get("/thread/{thread_id}", response_class=HTMLResponse)
    def thread_detail(
        request: Request,
        thread_id: str,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_thread_detail(conn, thread_id)
        if context is None:
            raise HTTPException(status_code=404, detail="thread not found")
        response = TEMPLATES.TemplateResponse(
            request,
            "thread.html",
            {
                "page_title": context["thread"]["title"],
                "thread": context["thread"],
                "entries": context["entries"],
                "current_search_q": "",
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
