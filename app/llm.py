from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx


ALLOWED_DIRECTIONS = {"i_owe", "owed_to_me", "none"}
ALLOWED_KINDS = {"request", "confirm", "change", "deliver", "dispute", "reference"}

SYSTEM_PROMPT = """你从一段职场聊天记录中抽取“谁要求谁做什么、什么时候要”。
只输出 JSON,不要解释,不要 markdown 代码块。
字段:
- requester_name: 提出方姓名或称呼,无法判断填 null
- owner_name: 被要求方,若是“我”则填 "我",无法判断填 null
- deliverable: 交付物,一句话,无法判断填 null
- due_raw: 原文中表示时限的词,如“下周五”,无则 null
- due_date: 推算出的日期 YYYY-MM-DD,无法推算填 null
- direction: i_owe / owed_to_me / none
- kind: request / confirm / change / deliver / dispute / reference
- plain_summary: 一句话大白话:这段在要求你做什么
- caveats: 字符串数组,记录歧义处,无则空数组
若这段话不构成任何请求或承诺(闲聊、通知、八卦),除 kind="reference" 与 plain_summary 外其余字段一律 null。
today 参数为当前日期,用于推算相对时间。"""


def _default_slots() -> dict[str, Any]:
    return {
        "requester_name": None,
        "owner_name": None,
        "deliverable": None,
        "due_raw": None,
        "due_date": None,
        "direction": "none",
        "kind": "reference",
        "plain_summary": None,
        "caveats": [],
    }


def _coerce_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_due_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def _normalize_payload_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result = _default_slots()
    result["requester_name"] = _coerce_text(payload.get("requester_name"))
    result["owner_name"] = _coerce_text(payload.get("owner_name"))
    result["deliverable"] = _coerce_text(payload.get("deliverable"))
    result["due_raw"] = _coerce_text(payload.get("due_raw"))
    result["due_date"] = _coerce_due_date(payload.get("due_date"))

    direction = payload.get("direction")
    if isinstance(direction, str) and direction in ALLOWED_DIRECTIONS:
        result["direction"] = direction

    kind = payload.get("kind")
    if isinstance(kind, str) and kind in ALLOWED_KINDS:
        result["kind"] = kind

    result["plain_summary"] = _coerce_text(payload.get("plain_summary"))

    caveats = payload.get("caveats")
    if isinstance(caveats, list):
        result["caveats"] = [item for item in caveats if isinstance(item, str)]

    return result


def normalize_slots(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return _normalize_payload_dict(payload)


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None

    candidates = [raw.strip()]
    stripped = raw.strip()

    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(stripped[start : end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return normalize_slots(parsed)
    return None


def extract_slots(text: str, today: str) -> dict[str, Any] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        response = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"today={today}\ntext={text}",
                    },
                ],
            },
            timeout=20.0,
        )
        if response.status_code != 200:
            return None

        payload = response.json()
        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not isinstance(content, str):
            return None
        return parse_llm_json(content)
    except Exception:
        return None


def resolve_actor(conn: sqlite3.Connection, name: str | None) -> str | None:
    if name is None:
        return None

    if name == "我":
        row = conn.execute("SELECT actor_id FROM actors WHERE is_self = 1 LIMIT 1").fetchone()
        return None if row is None else row["actor_id"]

    rows = conn.execute(
        "SELECT actor_id, canonical_name, aliases FROM actors ORDER BY created_at ASC"
    ).fetchall()
    for row in rows:
        if row["canonical_name"] == name:
            return row["actor_id"]
        aliases = json.loads(row["aliases"]) if row["aliases"] else []
        if name in aliases:
            return row["actor_id"]

    actor_id = f"act_{uuid.uuid4().hex[:12]}"
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            actor_id,
            name,
            json.dumps([], ensure_ascii=False, separators=(",", ":")),
            None,
            None,
            0,
            0.5,
            int(time.time() * 1000),
        ),
    )
    return actor_id


def due_date_to_millis(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)
