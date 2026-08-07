from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.labels import KIND, RISK, STATUS, source_badge_class, source_label, thread_headline
from app.labels import SOURCE_PRESETS
from app.sandbox import SandboxContext, apply_sandbox_cookie, cleanup_expired, get_sandbox
from evidence_core.db import init_db
from evidence_core.store import append_evidence
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


def _safe_diag_detail(message: str, api_key: str) -> str:
    detail = message.strip() or "unknown error"
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    return detail


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
    return {
        "evidence_id": row["evidence_id"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "platform": platform,
        "platform_class": source_badge_class(platform),
        "scene": scene,
        "raw_text_preview": _truncate_text(row["raw_text"] or ""),
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
        SELECT evidence_id, occurred_at, source_hint, raw_text
        FROM evidence
        WHERE evidence_id NOT LIKE 'ev_demo_%'
        ORDER BY seq DESC
        LIMIT 20
        """
    ).fetchall()
    return {
        "threads": [_prepare_thread_card(row) for row in thread_rows],
        "references": [_prepare_reference_row(row) for row in reference_rows],
        "recent_records": [_prepare_recent_row(row) for row in recent_rows],
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


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"starting workchain, port={os.environ.get('PORT')}, cwd={os.getcwd()}")
        demo_dir = _get_demo_dir()
        sandbox_root = _get_sandbox_root()
        db_path = demo_dir / "workchain.db"
        try:
            if not demo_dir.exists() or not db_path.exists():
                seed_demo_data(demo_dir)

            cleanup_expired(sandbox_root)
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
            },
        )
        apply_sandbox_cookie(response, sandbox)
        return response

    @app.post("/api/evidence")
    async def create_evidence(
        request: Request,
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
        finally:
            conn.close()

        response = JSONResponse(
            {
                "evidence_id": row["evidence_id"],
                "seq": row["seq"],
                "occurred_at": row["occurred_at"],
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
