from __future__ import annotations

import json
from typing import Any

from app import ai_provider


ASSEMBLER_VERSION = "1.0"
ASSEMBLER_PROVIDER = "deepseek"
AUTO_ACCEPT_CONFIDENCE_THRESHOLD = 0.85

SYSTEM_PROMPT = """你是 WorkChain 的 Context Assembler V1。你的任务只有一件事：判断多份图片证据应该如何分组阅读，以及每组内部的阅读顺序。

硬性规则:
1. 只输出 JSON 对象。
2. 只做 Context Group，不做 Fact、Event、标题、责任判断、真假判断。
3. 不要求 anchor_date，不做日期换算。
4. 每个 evidence 至少属于一个 group。
5. 同一 evidence 可以同时出现在多个 group。
6. analysis_order 只表示 AI 阅读顺序，不等于用户上传顺序。
7. 如果无法可靠判断，请把不确定性写入 ambiguities，不要硬编关联。

输出契约:
{
  "groups": [
    {
      "group_key": "group_1",
      "evidence_ids": ["ev-1", "ev-2"],
      "analysis_order": ["ev-1", "ev-2"],
      "relation": "continuation|related_context|standalone",
      "confidence": 0.0
    }
  ],
  "ambiguities": ["..."]
}
"""


def _coerce_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    return 0.0


def build_context_assembly_messages(
    items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = _coerce_text(item.get("evidence_id"))
        projection_text = _coerce_text(item.get("projection_text"))
        if evidence_id is None or projection_text is None:
            continue
        payload_items.append(
            {
                "evidence_id": evidence_id,
                "submission_position": item.get("submission_position"),
                "projection_text": projection_text,
            }
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "inputs": payload_items,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def fallback_singleton_groups(
    items: list[dict[str, Any]],
    *,
    warning: str | None = None,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    ambiguities: list[str] = []
    if warning:
        ambiguities.append(warning)
    for item in items:
        evidence_id = _coerce_text(item.get("evidence_id"))
        if evidence_id is None:
            continue
        groups.append(
            {
                "group_key": f"singleton_{evidence_id}",
                "evidence_ids": [evidence_id],
                "analysis_order": [evidence_id],
                "relation": "standalone",
                "confidence": 0.0,
            }
        )
    return {
        "groups": groups,
        "ambiguities": ambiguities,
    }


def normalize_context_assembly_result(
    payload: Any,
    *,
    evidence_ids: list[str],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw_groups = payload.get("groups")
    raw_ambiguities = payload.get("ambiguities")
    if not isinstance(raw_groups, list):
        return None
    normalized_groups: list[dict[str, Any]] = []
    covered_evidence_ids: set[str] = set()
    valid_relations = {"continuation", "related_context", "standalone"}
    allowed_evidence_ids = set(evidence_ids)
    for index, item in enumerate(raw_groups, start=1):
        if not isinstance(item, dict):
            continue
        group_evidence_ids = [
            evidence_id
            for evidence_id in item.get("evidence_ids", [])
            if isinstance(evidence_id, str) and evidence_id in allowed_evidence_ids
        ]
        analysis_order = [
            evidence_id
            for evidence_id in item.get("analysis_order", [])
            if isinstance(evidence_id, str) and evidence_id in group_evidence_ids
        ]
        if not group_evidence_ids:
            continue
        if not analysis_order:
            analysis_order = list(group_evidence_ids)
        relation = _coerce_text(item.get("relation")) or "standalone"
        if relation not in valid_relations:
            relation = "standalone"
        normalized = {
            "group_key": _coerce_text(item.get("group_key")) or f"group_{index}",
            "evidence_ids": group_evidence_ids,
            "analysis_order": analysis_order,
            "relation": relation,
            "confidence": _coerce_confidence(item.get("confidence")),
        }
        covered_evidence_ids.update(group_evidence_ids)
        normalized_groups.append(normalized)

    for evidence_id in evidence_ids:
        if evidence_id not in covered_evidence_ids:
            normalized_groups.append(
                {
                    "group_key": f"singleton_{evidence_id}",
                    "evidence_ids": [evidence_id],
                    "analysis_order": [evidence_id],
                    "relation": "standalone",
                    "confidence": 0.0,
                }
            )

    ambiguities = [
        item.strip()
        for item in (raw_ambiguities if isinstance(raw_ambiguities, list) else [])
        if isinstance(item, str) and item.strip()
    ]
    return {
        "groups": normalized_groups,
        "ambiguities": ambiguities,
    }


def assemble_context_groups(
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    evidence_ids = [
        evidence_id
        for evidence_id in (_coerce_text(item.get("evidence_id")) for item in items)
        if evidence_id is not None
    ]
    messages = build_context_assembly_messages(items)
    provider_result = ai_provider.chat_semantic_json_diagnostic_result(
        messages,
        max_tokens=2048,
        temperature=0,
    )
    diagnostic = provider_result.get("diagnostic")
    content = provider_result.get("content")
    if not content:
        return (
            fallback_singleton_groups(items, warning="context_assembly_failed_fallback_to_singletons"),
            diagnostic,
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return (
            fallback_singleton_groups(items, warning="context_assembly_invalid_json_fallback_to_singletons"),
            diagnostic,
        )
    normalized = normalize_context_assembly_result(parsed, evidence_ids=evidence_ids)
    if normalized is None:
        return (
            fallback_singleton_groups(items, warning="context_assembly_invalid_contract_fallback_to_singletons"),
            diagnostic,
        )
    return normalized, diagnostic


def groups_require_user_review(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return True
    ambiguities = result.get("ambiguities")
    if isinstance(ambiguities, list) and any(isinstance(item, str) and item.strip() for item in ambiguities):
        return True
    for group in result.get("groups", []):
        if not isinstance(group, dict):
            return True
        if _coerce_confidence(group.get("confidence")) < AUTO_ACCEPT_CONFIDENCE_THRESHOLD:
            return True
    return False
