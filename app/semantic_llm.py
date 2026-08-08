from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import httpx

from app.llm import get_deepseek_model


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
FACT_TYPES = {
    "request",
    "commitment",
    "confirmation",
    "scope_change",
    "responsibility_change",
    "deadline_change",
    "delivery",
    "cancellation",
    "denial",
    "statement",
    "reference",
}
INTERPRETATION_KINDS = {"explanation", "term", "action_hint", "uncertainty"}

SYSTEM_PROMPT = """你是 WorkChain 的 Semantic Parser V2。你的任务是从一段原始文本中抽取中立、可验证的语义事实。

硬性规则:
1. 只输出 JSON 对象,不要解释,不要 markdown 代码块,不要额外前后话。
2. 中立记录“谁对谁表达了什么”“谁声称了什么”“发生了什么变化”,不要默认上传者是参与者,不要站任何一方立场。
3. 不要求出现“我”;群聊、围观、第三方转述、会议记录都要正常解析。
4. 一段内容可以拆成多个原子 Fact。不同要求、责任变化、范围变化、截止变化,不要揉成一条总结。
5. 未经证明的指控、传闻、猜测,只能保留为 statement / denial / reference 等中立表达,不要改写成客观真相。
6. 非工作内容不要硬套 requester/owner/deliverable 语义;八卦、声明、借款、闲聊也可作为 statement / reference / denial。
7. Fact 与 Interpretation 严格分离。解释、术语释义、行动建议、不确定性不得写进 Fact content。
8. 遇到黑话、模糊要求、口语化表达,优先生成 explanation / term / uncertainty,帮助不熟悉语境的人理解。
9. 缺上下文时承认不知道,不要补造人物、事件、关系或日期。
10. glossary 仅供参考,原文语境优先。不要使用上传者身份标签来推断事实。
11. 只有 anchor_date 明确可靠时,才允许把“明天/下周五/下下周五”等相对时间换算为具体 due_date,并同时回填 due_anchor_date。
12. 没有可靠 anchor_date 时,保留 due_raw,并把 due_date / due_anchor_date 设为 null。绝不能使用服务器今天或模型自选日期脑补。

输出契约:
{
  "facts": [
    {
      "fact_type": "request|commitment|confirmation|scope_change|responsibility_change|deadline_change|delivery|cancellation|denial|statement|reference",
      "content": "中立、原子化的一句话事实表达",
      "confidence": 0.0,
      "actors": [{"name": "人物称呼", "role": "该人物在本条 fact 中的角色"}],
      "occurred_date": null,
      "due_raw": null,
      "due_date": null,
      "due_anchor_date": null
    }
  ],
  "interpretations": [
    {
      "fact_index": 0,
      "kind": "explanation|term|action_hint|uncertainty",
      "content": "解释层内容",
      "confidence": 0.0
    }
  ],
  "ambiguities": ["仍然缺失或无法确认的点"]
}

日期要求:
- 所有日期必须是 YYYY-MM-DD 或 null。
- 无法保证合法性时输出 null。
- occurred_date 是文本中明确指向的发生日期;没有明确日期就用 null。
"""

GLOSSARY_SUFFIX = "glossary 仅作语义参考,原文语境优先,不要机械套用。"


def _default_result() -> dict[str, Any]:
    return {
        "facts": [],
        "interpretations": [],
        "ambiguities": [],
    }


def _coerce_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _coerce_date(value: Any) -> str | None:
    value = _coerce_text(value)
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def _coerce_confidence(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    return default


def _normalize_actor(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    name = _coerce_text(value.get("name"))
    role = _coerce_text(value.get("role"))
    if name is None or role is None:
        return None
    return {"name": name, "role": role}


def _normalize_fact(value: Any, *, anchor_date: str | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    fact_type = _coerce_text(value.get("fact_type"))
    content = _coerce_text(value.get("content"))
    if fact_type not in FACT_TYPES or content is None:
        return None

    actors_raw = value.get("actors")
    actors: list[dict[str, str]] = []
    if isinstance(actors_raw, list):
        for item in actors_raw:
            normalized_actor = _normalize_actor(item)
            if normalized_actor is not None:
                actors.append(normalized_actor)

    due_raw = _coerce_text(value.get("due_raw"))
    due_date = _coerce_date(value.get("due_date"))
    occurred_date = _coerce_date(value.get("occurred_date"))
    due_anchor_date = _coerce_date(value.get("due_anchor_date"))
    if anchor_date is None:
        due_date = None
        due_anchor_date = None
    elif due_date is not None:
        due_anchor_date = anchor_date
    else:
        due_anchor_date = None

    return {
        "fact_type": fact_type,
        "content": content,
        "confidence": _coerce_confidence(value.get("confidence")),
        "actors": actors,
        "occurred_date": occurred_date,
        "due_raw": due_raw,
        "due_date": due_date,
        "due_anchor_date": due_anchor_date,
    }


def _normalize_interpretation(value: Any, *, fact_count: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_index = value.get("fact_index")
    if not isinstance(raw_index, int) or isinstance(raw_index, bool):
        return None
    if raw_index < 0 or raw_index >= fact_count:
        return None
    kind = _coerce_text(value.get("kind"))
    content = _coerce_text(value.get("content"))
    if kind not in INTERPRETATION_KINDS or content is None:
        return None
    return {
        "fact_index": raw_index,
        "kind": kind,
        "content": content,
        "confidence": _coerce_confidence(value.get("confidence")),
    }


def normalize_semantics(payload: Any, *, anchor_date: str | None = None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    result = _default_result()

    facts_raw = payload.get("facts")
    if isinstance(facts_raw, list):
        for item in facts_raw:
            normalized_fact = _normalize_fact(item, anchor_date=anchor_date)
            if normalized_fact is not None:
                result["facts"].append(normalized_fact)

    interpretations_raw = payload.get("interpretations")
    if isinstance(interpretations_raw, list):
        for item in interpretations_raw:
            normalized_interpretation = _normalize_interpretation(
                item,
                fact_count=len(result["facts"]),
            )
            if normalized_interpretation is not None:
                result["interpretations"].append(normalized_interpretation)

    ambiguities_raw = payload.get("ambiguities")
    if isinstance(ambiguities_raw, list):
        result["ambiguities"] = [
            item.strip()
            for item in ambiguities_raw
            if isinstance(item, str) and item.strip()
        ]

    return result


def parse_semantic_json(raw: Any, *, anchor_date: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None

    stripped = raw.strip()
    candidates = [stripped]

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
        normalized = normalize_semantics(parsed, anchor_date=anchor_date)
        if normalized is not None:
            return normalized
    return None


def _normalize_glossary(glossary: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in glossary or []:
        if not isinstance(item, dict):
            continue
        term = _coerce_text(item.get("term"))
        meaning = _coerce_text(item.get("meaning"))
        kind = _coerce_text(item.get("kind"))
        if term is None or meaning is None or kind is None:
            continue
        normalized.append({"term": term, "meaning": meaning, "kind": kind})
    return normalized


def build_semantic_context_block(
    *,
    anchor_date: str | None,
    glossary: list[dict[str, Any]] | None,
    source_hint: str | None,
) -> str:
    lines = ["【解析上下文】"]
    if anchor_date is None:
        lines.append("anchor_date=null (没有可靠时间锚点,不得把相对日期换算成具体日期)")
    else:
        lines.append(f"anchor_date={anchor_date}")

    source_hint = _coerce_text(source_hint)
    if source_hint is not None:
        lines.append(f"source_hint={source_hint}")

    glossary_items = _normalize_glossary(glossary)
    if glossary_items:
        lines.append("glossary:")
        for item in glossary_items:
            lines.append(f"- {item['term']} -> {item['meaning']} ({item['kind']})")
        lines.append(GLOSSARY_SUFFIX)
    return "\n".join(lines)


def build_semantic_messages(
    text: str,
    *,
    anchor_date: str | None,
    glossary: list[dict[str, Any]] | None,
    source_hint: str | None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "system",
            "content": build_semantic_context_block(
                anchor_date=anchor_date,
                glossary=glossary,
                source_hint=source_hint,
            ),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"text={text}",
        }
    )
    return messages


def extract_semantics(
    text: str,
    *,
    anchor_date: str | None = None,
    glossary: list[dict[str, Any]] | None = None,
    source_hint: str | None = None,
) -> dict[str, Any] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    anchor_date = _coerce_date(anchor_date)
    messages = build_semantic_messages(
        text,
        anchor_date=anchor_date,
        glossary=glossary,
        source_hint=source_hint,
    )

    try:
        response = httpx.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_deepseek_model(),
                "temperature": 0,
                "messages": messages,
            },
            timeout=20.0,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        return parse_semantic_json(content, anchor_date=anchor_date)
    except Exception:
        return None
