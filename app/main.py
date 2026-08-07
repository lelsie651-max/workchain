from __future__ import annotations

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.labels import KIND, RISK, STATUS
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


def get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    conn = _open_readonly_connection(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def _prepare_thread_card(row: sqlite3.Row) -> dict[str, Any]:
    risk_flags = _decode_json_array(row["risk_flags"])
    return {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "status_label": STATUS[row["status"]],
        "version": row["version"],
        "risk_labels": [RISK.get(flag, flag) for flag in risk_flags],
        "evidence_count": row["evidence_count"],
        "last_activity_text": _format_datetime(row["last_activity_at"], "%m-%d %H:%M"),
    }


def _prepare_reference_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "seq": row["seq"],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "source_hint": row["source_hint"],
        "raw_text": row["raw_text"],
    }


def _prepare_timeline_row(row: sqlite3.Row) -> dict[str, Any]:
    caveats = _decode_json_array(row["caveats"])
    return {
        "seq": row["seq"],
        "kind": row["kind"],
        "kind_label": KIND[row["kind"]],
        "occurred_at_text": _format_datetime(row["occurred_at"], "%m-%d %H:%M"),
        "source_hint": row["source_hint"],
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
            COUNT(e.evidence_id) AS evidence_count
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
    return {
        "threads": [_prepare_thread_card(row) for row in thread_rows],
        "references": [_prepare_reference_row(row) for row in reference_rows],
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
    return {
        "thread": {
            "thread_id": thread_row["thread_id"],
            "title": thread_row["title"],
            "status_label": STATUS[thread_row["status"]],
            "version": thread_row["version"],
            "risk_labels": [RISK.get(flag, flag) for flag in _decode_json_array(thread_row["risk_flags"])],
            "current_deliverable": thread_row["current_deliverable"],
            "current_due_text": _format_datetime(thread_row["current_due"], "%m-%d %H:%M"),
            "last_activity_text": _format_datetime(thread_row["last_activity_at"], "%m-%d %H:%M"),
            "evidence_count": len(evidence_rows),
        },
        "entries": [_prepare_timeline_row(row) for row in evidence_rows],
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        demo_dir = _get_demo_dir()
        db_path = demo_dir / "workchain.db"
        if not demo_dir.exists() or not db_path.exists():
            seed_demo_data(demo_dir)

        app.state.demo_dir = demo_dir
        app.state.db_path = db_path
        yield

    app = FastAPI(title="WorkChain", lifespan=lifespan)

    @app.get("/healthz")
    def healthz(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        evidence_count = conn.execute("SELECT COUNT(*) AS count FROM evidence").fetchone()["count"]
        return {"status": "ok", "evidence_count": evidence_count}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, conn: sqlite3.Connection = Depends(get_conn)) -> HTMLResponse:
        context = _fetch_index_data(conn)
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "page_title": "WorkChain",
                "threads": context["threads"],
                "references": context["references"],
            },
        )

    @app.get("/thread/{thread_id}", response_class=HTMLResponse)
    def thread_detail(
        request: Request,
        thread_id: str,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> HTMLResponse:
        context = _fetch_thread_detail(conn, thread_id)
        if context is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return TEMPLATES.TemplateResponse(
            request,
            "thread.html",
            {
                "page_title": context["thread"]["title"],
                "thread": context["thread"],
                "entries": context["entries"],
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
