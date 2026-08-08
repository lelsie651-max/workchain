from __future__ import annotations

import html
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import traceback
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

from app import llm
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
from evidence_core.export import export_evidence_package
from evidence_core.store import append_evidence, update_slots, verify_chain
from scripts.seed_demo import seed_demo_data


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
MAX_TEXT_LENGTH = 20_000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_SANDBOX_UPLOAD_BYTES = 50 * 1024 * 1024
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
    if not raw_text or not raw_text.startswith("[文件] "):
        return None
    return raw_text[5:]


def _is_upload_value(value: Any) -> bool:
    return bool(value) and hasattr(value, "filename") and callable(getattr(value, "read", None))


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


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


def _ensure_upload_budget(conn: sqlite3.Connection, blobs_root: Path, blob_bytes: bytes) -> None:
    content_hash = chain.compute_content_hash(blob_bytes)
    blob_path = blobs_root / content_hash[:2] / f"{content_hash}.bin"
    additional_bytes = 0 if blob_path.exists() else len(blob_bytes)
    current_total = _current_upload_storage_bytes(conn, blobs_root)
    if current_total + additional_bytes > MAX_SANDBOX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="这个沙箱累计上传已超过 50 MB")


def _build_file_label(filename: str | None) -> str:
    clean_name = (filename or "未命名文件").strip() or "未命名文件"
    return f"[文件] {clean_name}"


def _build_content_disposition(disposition: str, filename: str | None) -> str:
    if not filename:
        return disposition
    fallback = filename.encode("ascii", "ignore").decode("ascii") or "file"
    fallback = fallback.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
    return f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{quote(filename, safe="")}'


def _unsupported_detail(media_type: str) -> str:
    return f"这是一张图片/文档,系统暂不能自动读懂它的内容,但原件已完整保存,任何改动都会被发现。"


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
        file_value = form.get("file")
        upload = file_value if _is_upload_value(file_value) else None
        file_bytes = None
        detected_media_type = None
        detected_content_type = None
        if upload is not None and upload.filename:
            file_bytes = await upload.read()
            if len(file_bytes) > MAX_FILE_BYTES:
                raise HTTPException(status_code=400, detail="单个文件不能超过 8 MB")
            detected_media_type, detected_content_type = _detect_upload_type(upload, file_bytes)
        return {
            "text": str(form.get("text", "")).strip(),
            "source": str(form.get("source", "")).strip(),
            "source_detail": str(form.get("source_detail", "")).strip() or None,
            "counterpart": str(form.get("counterpart", "")).strip() or None,
            "upload": upload,
            "file_bytes": file_bytes,
            "media_type": detected_media_type,
            "file_content_type": detected_content_type,
        }

    payload = await request.json()
    return {
        "text": str(payload.get("text", "")).strip(),
        "source": str(payload.get("source", "")).strip(),
        "source_detail": None if payload.get("source_detail") is None else str(payload.get("source_detail")).strip() or None,
        "counterpart": None if payload.get("counterpart") is None else str(payload.get("counterpart")).strip() or None,
        "upload": None,
        "file_bytes": None,
        "media_type": None,
        "file_content_type": None,
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


def _verified_key(evidence_id: str) -> str:
    return f"verified:{evidence_id}"


def _parse_count_key(today: str) -> str:
    return f"parse_count:{today}"


def _global_parse_count_key(today: str) -> str:
    return f"global_parse_count:{today}"


def _set_parse_status(conn: sqlite3.Connection, evidence_id: str, status: str) -> None:
    _set_meta_value(conn, _parse_status_key(evidence_id), status)
    conn.commit()


def _set_parse_detail(conn: sqlite3.Connection, evidence_id: str, detail: str) -> None:
    _set_meta_value(conn, _parse_detail_key(evidence_id), detail)
    conn.commit()


def _get_parse_status(conn: sqlite3.Connection, evidence_id: str) -> str:
    return _get_meta_value(conn, _parse_status_key(evidence_id)) or "failed"


def _get_parse_detail(conn: sqlite3.Connection, evidence_id: str) -> str | None:
    return _get_meta_value(conn, _parse_detail_key(evidence_id))


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
        result.setdefault(evidence_id, "failed")
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


def _consume_parse_budget(sandbox_db_path: Path, global_meta_db_path: Path) -> tuple[bool, str | None]:
    today = _today_str()
    sandbox_conn = init_db(sandbox_db_path)
    try:
        if not _try_increment_meta_counter(sandbox_conn, _parse_count_key(today), 20):
            return False, "今日解析次数已用完,记录仍已保存"
    finally:
        sandbox_conn.close()

    global_conn = _open_meta_connection(global_meta_db_path)
    try:
        if not _try_increment_meta_counter(global_conn, _global_parse_count_key(today), 300):
            sandbox_rollback_conn = init_db(sandbox_db_path)
            try:
                sandbox_conn_value = _get_meta_value(sandbox_rollback_conn, _parse_count_key(today))
                current = 0 if sandbox_conn_value is None else int(sandbox_conn_value)
                _set_meta_value(
                    sandbox_rollback_conn,
                    _parse_count_key(today),
                    str(max(0, current - 1)),
                )
                sandbox_rollback_conn.commit()
            finally:
                sandbox_rollback_conn.close()
            return False, "今日解析次数已用完,记录仍已保存"
    finally:
        global_conn.close()

    return True, None


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
    text: str,
    counterpart: str | None,
) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        conn = init_db(sandbox_db_path)
        try:
            _set_parse_status(conn, evidence_id, "failed")
            _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        return

    allowed, reason = _consume_parse_budget(sandbox_db_path, global_meta_db_path)
    if not allowed:
        conn = init_db(sandbox_db_path)
        try:
            _set_parse_status(conn, evidence_id, "failed")
            _set_parse_detail(conn, evidence_id, reason or "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        return

    context = get_settings(sandbox_db_path)
    if counterpart:
        context["counterpart"] = counterpart

    try:
        parsed = llm.extract_slots(text, _today_str(), context=context)
    except Exception:
        parsed = None
    parsed = llm.normalize_slots(parsed)
    if parsed is None:
        conn = init_db(sandbox_db_path)
        try:
            _set_parse_status(conn, evidence_id, "failed")
            _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
        finally:
            conn.close()
        return

    conn = init_db(sandbox_db_path)
    try:
        glossary = context.get("glossary", [])
        requester_id = llm.resolve_actor_with_glossary(conn, parsed.get("requester_name"), glossary)
        owner_id = llm.resolve_actor_with_glossary(conn, parsed.get("owner_name"), glossary)
        slot_due = llm.due_date_to_millis(parsed.get("due_date"))

        update_slots(
            conn,
            evidence_id,
            slot_requester=requester_id,
            slot_owner=owner_id,
            slot_deliverable=parsed.get("deliverable"),
            slot_due=slot_due,
            slot_due_raw=parsed.get("due_raw"),
            slot_direction=parsed.get("direction"),
            plain_summary=parsed.get("plain_summary"),
            caveats=parsed.get("caveats", []),
        )
        conn.execute(
            "UPDATE evidence SET kind = ? WHERE evidence_id = ?",
            (_final_kind(parsed, requester_id, owner_id, slot_due), evidence_id),
        )
        _set_parse_status(conn, evidence_id, "done")
        _set_parse_detail(conn, evidence_id, "")
        conn.commit()
    except Exception:
        conn.rollback()
        _set_parse_status(conn, evidence_id, "failed")
        _set_parse_detail(conn, evidence_id, "解析暂不可用,记录已完整保存")
    finally:
        conn.close()


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


def _prepare_recent_row(row: sqlite3.Row, sandbox: SandboxContext) -> dict[str, Any]:
    platform, scene = source_label(row["source_hint"])
    raw_text = row["raw_text"] or ""
    filename = _extract_filename(raw_text)
    return {
        "evidence_id": row["evidence_id"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": raw_text,
        "raw_text_preview": _truncate_text(raw_text),
        "is_long_text": len(raw_text) > 100,
        "parse_status": row["parse_status"],
        "parse_detail": row["parse_detail"],
        "plain_summary": row["plain_summary"],
        "deliverable": row["slot_deliverable"],
        "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d"),
        "caveats": _decode_json_array(row["caveats"]),
        "is_verified": bool(row["is_verified"]),
        "media_type": row["media_type"],
        "blob_url": f"/blob/{row['evidence_id']}" if row["media_type"] in {"image", "file"} else None,
        "filename": filename,
        "is_image": row["media_type"] == "image",
        "is_file": row["media_type"] == "file",
    }


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
    thread_rows = conn.execute(
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
    recent_rows = conn.execute(
        """
        SELECT evidence_id, occurred_at, source_hint, raw_text, media_type,
               plain_summary, slot_deliverable, slot_due, slot_due_raw, caveats
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
        ORDER BY seq DESC
        LIMIT 20
        """
    ).fetchall()
    parse_status_map = _get_parse_status_map(conn, [row["evidence_id"] for row in recent_rows])
    parse_detail_map = _get_parse_detail_map(conn, [row["evidence_id"] for row in recent_rows])
    verified_map = _get_verified_map(conn, [row["evidence_id"] for row in recent_rows])
    decorated_recent_rows = []
    for row in recent_rows:
        row_dict = dict(row)
        row_dict["parse_status"] = parse_status_map.get(row["evidence_id"], "failed")
        row_dict["parse_detail"] = parse_detail_map.get(row["evidence_id"])
        row_dict["is_verified"] = verified_map.get(row["evidence_id"], False)
        decorated_recent_rows.append(row_dict)
    return {
        "threads": [_prepare_thread_card(row) for row in thread_rows],
        "references": [_prepare_reference_row(row) for row in reference_rows],
        "recent_records": [_prepare_recent_row(row, sandbox) for row in decorated_recent_rows],
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

    platform, scene = source_label(row["source_hint"])
    filename = _extract_filename(row["raw_text"])
    return {
        "evidence_id": row["evidence_id"],
        "seq": row["seq"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "captured_at_text": _format_datetime(row["captured_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text": row["raw_text"] or "",
        "plain_summary": row["plain_summary"],
        "deliverable": row["slot_deliverable"],
        "due_date_value": _format_datetime(row["slot_due"], "%Y-%m-%d"),
        "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d"),
        "caveats": _decode_json_array(row["caveats"]),
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
    }


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
                    "model": "deepseek-chat",
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
            return {
                "parse_status": parse_status,
                "slots_filled": row["slots_filled"],
                "plain_summary": row["plain_summary"],
                "deliverable": row["slot_deliverable"],
                "due_text": row["slot_due_raw"] or _format_datetime(row["slot_due"], "%m-%d"),
                "caveats": _decode_json_array(row["caveats"]),
                "detail": _get_parse_detail(conn, evidence_id),
                "is_verified": _is_verified(conn, evidence_id),
                "media_type": row["media_type"],
            }
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
                "threads": context["threads"],
                "references": context["references"],
                "recent_records": context["recent_records"],
                "source_presets": SOURCE_PRESETS,
                "settings": _settings_payload(sandbox.db_path),
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
            parse_detail = _get_parse_detail(status_conn, evidence_id)
            is_verified = _is_verified(status_conn, evidence_id)
        finally:
            status_conn.close()

        response = TEMPLATES.TemplateResponse(
            request,
            "evidence.html",
            {
                "page_title": "记录详情",
                "evidence": context,
                "parse_status": parse_status,
                "parse_detail": parse_detail,
                "is_verified": is_verified,
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
        upload = payload["upload"]
        file_bytes = payload["file_bytes"]
        upload_media_type = payload["media_type"]
        if not text and file_bytes is None:
            raise HTTPException(status_code=400, detail="请输入内容或选择一个文件")
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail="内容过长,请控制在 20000 字以内")

        source = payload["source"]
        if not source:
            raise HTTPException(status_code=400, detail="请选择或填写来源")
        if source not in SOURCE_PRESETS and len(source) > 20:
            raise HTTPException(status_code=400, detail="自定义来源不能超过 20 个字")

        source_detail = payload["source_detail"]
        source_hint = source if not source_detail else f"{source}-{source_detail}"
        counterpart = payload["counterpart"] or None

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
            if today_count >= 50:
                raise HTTPException(status_code=429, detail="你今天已经存了 50 条,明天再来继续")

            now_ms = int(time.time() * 1000)
            media_type = "text"
            append_payload: bytes | str = text
            raw_text_override = None
            parse_status = "pending"
            parse_detail = ""
            if file_bytes is not None and upload_media_type is not None:
                _ensure_upload_budget(conn, sandbox.blobs_root, file_bytes)
                media_type = upload_media_type
                append_payload = file_bytes
                raw_text_override = _build_file_label(upload.filename if upload is not None else None)
                parse_status = "unsupported"
                parse_detail = _unsupported_detail(media_type)

            row = append_evidence(
                conn,
                blobs_root=sandbox.blobs_root,
                media_type=media_type,
                payload=append_payload,
                captured_at=now_ms,
                occurred_at=now_ms,
                source_hint=source_hint,
                kind="reference",
            )
            if raw_text_override is not None:
                conn.execute(
                    "UPDATE evidence SET raw_text = ?, plain_summary = ? WHERE evidence_id = ?",
                    (raw_text_override, text or None, row["evidence_id"]),
                )
                row = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (row["evidence_id"],)).fetchone()
                row = dict(row)
            _set_parse_status(conn, row["evidence_id"], parse_status)
            _set_parse_detail(conn, row["evidence_id"], parse_detail)
        finally:
            conn.close()

        if file_bytes is None:
            background_tasks.add_task(
                _run_parse_pipeline,
                sandbox.db_path,
                request.app.state.global_meta_db_path,
                row["evidence_id"],
                text,
                counterpart,
            )

        response = JSONResponse(
            {
                "evidence_id": row["evidence_id"],
                "seq": row["seq"],
                "occurred_at": row["occurred_at"],
                "parse_status": parse_status,
                "media_type": row["media_type"],
            }
        )
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
