from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app import llm
from app.labels import KIND, RISK, STATUS, source_badge_class, source_label, thread_headline
from app.labels import SOURCE_PRESETS
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
from evidence_core.db import init_db
from evidence_core.store import append_evidence, update_slots, verify_chain
from scripts.seed_demo import seed_demo_data


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


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


def _prepare_recent_row(row: sqlite3.Row) -> dict[str, Any]:
    platform, scene = source_label(row["source_hint"])
    raw_text = row["raw_text"] or ""
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
    }


def _settings_payload(db_path: Path) -> dict[str, Any]:
    settings = get_settings(db_path)
    return {
        "self_names": settings["self_names"],
        "glossary": settings["glossary"],
        "has_self_names": bool(settings["self_names"]),
    }


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


def _fetch_index_data(conn: sqlite3.Connection) -> dict[str, Any]:
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
        SELECT evidence_id, occurred_at, source_hint, raw_text,
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
        "recent_records": [_prepare_recent_row(row) for row in decorated_recent_rows],
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
            content_hash, slots_filled, kind, slot_direction
        FROM evidence
        WHERE evidence_id = ?
        """,
        (evidence_id,),
    ).fetchone()
    if row is None:
        return None

    platform, scene = source_label(row["source_hint"])
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
                       slot_due, slot_due_raw, caveats
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

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        sandbox: SandboxContext = Depends(get_sandbox),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_index_data(conn)
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
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/evidence")
    async def create_evidence(
        request: Request,
        background_tasks: BackgroundTasks,
        sandbox: SandboxContext = Depends(get_sandbox),
    ) -> JSONResponse:
        payload = await request.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="请输入要存证的内容")
        if len(text) > 20000:
            raise HTTPException(status_code=400, detail="内容过长,请控制在 20000 字以内")

        source = str(payload.get("source", "")).strip()
        if not source:
            raise HTTPException(status_code=400, detail="请选择或填写来源")
        if source not in SOURCE_PRESETS and len(source) > 20:
            raise HTTPException(status_code=400, detail="自定义来源不能超过 20 个字")

        source_detail_raw = payload.get("source_detail")
        source_detail = None if source_detail_raw is None else str(source_detail_raw).strip()
        source_hint = source if not source_detail else f"{source}-{source_detail}"
        counterpart_raw = payload.get("counterpart")
        counterpart = None if counterpart_raw is None else str(counterpart_raw).strip()
        counterpart = counterpart or None

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
            row = append_evidence(
                conn,
                blobs_root=sandbox.blobs_root,
                media_type="text",
                payload=text,
                captured_at=now_ms,
                occurred_at=now_ms,
                source_hint=source_hint,
                kind="reference",
            )
            _set_parse_status(conn, row["evidence_id"], "pending")
            _set_parse_detail(conn, row["evidence_id"], "")
        finally:
            conn.close()

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
                "parse_status": "pending",
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
