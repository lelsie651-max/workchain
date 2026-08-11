from __future__ import annotations

import json
from typing import Any

from evidence_core.extraction_contract import normalize_observations


def _coerce_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _coerce_message_index(value: Any, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return fallback
    return value


def _normalize_quote_like(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    speaker_display_name = _coerce_text(value.get("speaker_display_name"))
    text = _coerce_text(value.get("text"))
    if speaker_display_name is None and text is None:
        return None
    return {
        "speaker_display_name": speaker_display_name,
        "text": text,
    }


def _normalize_reactions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        emoji = _coerce_text(item.get("emoji"))
        actor_display_name = _coerce_text(item.get("actor_display_name")) or "unknown"
        if emoji is None:
            continue
        normalized.append(
            {
                "emoji": emoji,
                "actor_display_name": actor_display_name,
            }
        )
    return normalized


def _normalize_side(value: Any) -> str:
    normalized = (_coerce_text(value) or "").lower()
    if normalized in {"left", "left_side", "lhs"}:
        return "left"
    if normalized in {"right", "right_side", "rhs"}:
        return "right"
    return "unknown"


def _trusted_header_display_name(payload: dict[str, Any], speaker_ref: str) -> str | None:
    if speaker_ref != "left_account":
        return None
    if _coerce_text(payload.get("conversation_type")) != "direct_chat":
        return None
    if _coerce_text(payload.get("observed_platform")) != "微信":
        return None
    return _coerce_text(payload.get("chat_header"))


def _message_identity_key(item: dict[str, Any]) -> str | None:
    for key in ("speaker_ref", "avatar_ref", "visible_sender_label"):
        value = _coerce_text(item.get(key))
        if value is not None:
            return f"{key}:{value}"
    return None


def _normalize_projection_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        text = _coerce_text(item.get("text"))
        quote = _normalize_quote_like(item.get("quote"))
        reply = _normalize_quote_like(item.get("reply"))
        reactions = _normalize_reactions(item.get("reactions"))
        if text is None and quote is None and reply is None and not reactions:
            continue
        normalized.append(
            {
                "index": _coerce_message_index(item.get("index"), fallback_index),
                "speaker_ref": _coerce_text(item.get("speaker_ref")) or "unknown",
                "raw_speaker_ref": _coerce_text(item.get("speaker_ref")) or "unknown",
                "side": _normalize_side(item.get("side")),
                "visible_sender_label": _coerce_text(item.get("visible_sender_label")),
                "avatar_ref": _coerce_text(item.get("avatar_ref")),
                "display_name": _coerce_text(item.get("display_name")),
                "text": text,
                "quote": quote,
                "reply": reply,
                "reactions": reactions,
            }
        )
    normalized.sort(key=lambda item: (item["index"], item["speaker_ref"], item.get("text") or ""))
    return normalized


def _normalize_projection_participants(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker_ref = _coerce_text(item.get("speaker_ref"))
        if speaker_ref is None:
            continue
        normalized.append(
            {
                "speaker_ref": speaker_ref,
                "display_name": _coerce_text(item.get("display_name")),
                "side": _coerce_text(item.get("side")),
            }
        )
    return normalized


def _timestamp_markers_from_observations(observations: Any) -> list[str]:
    markers: list[str] = []
    for item in normalize_observations(observations):
        if item.get("kind") != "timestamp":
            continue
        content = _coerce_text(item.get("content"))
        if content is not None and content not in markers:
            markers.append(content)
    return markers


def build_semantic_projection(
    *,
    transcript: str | None,
    observations: Any = None,
    structured_payload: Any = None,
) -> dict[str, Any] | None:
    payload = structured_payload if isinstance(structured_payload, dict) else None
    if payload is None:
        normalized_transcript = _coerce_text(transcript)
        if normalized_transcript is None:
            return None
        return {
            "projection_version": "1.0",
            "source_kind": "plain_text",
            "conversation_type": "unknown",
            "participants": [],
            "time_markers": _timestamp_markers_from_observations(observations),
            "messages": [
                {
                    "index": 1,
                    "speaker_ref": "unknown",
                    "display_name": None,
                    "text": normalized_transcript,
                    "quote": None,
                    "reply": None,
                    "reactions": [],
                }
            ],
            "missing_speaker_refs": [],
        }

    participants = _normalize_projection_participants(payload.get("participants"))
    participant_name_map = {
        item["speaker_ref"]: item.get("display_name")
        for item in participants
        if item.get("speaker_ref")
    }
    raw_messages = _normalize_projection_messages(payload.get("messages"))

    left_identities = {
        identity
        for item in raw_messages
        if item.get("side") == "left"
        for identity in [_message_identity_key(item)]
        if identity is not None
    }
    right_identities = {
        identity
        for item in raw_messages
        if item.get("side") == "right"
        for identity in [_message_identity_key(item)]
        if identity is not None
    }
    alias_map: dict[str, str] = {}
    if len(left_identities) == 1:
        alias_map[next(iter(left_identities))] = "left_account"
    if len(right_identities) == 1:
        alias_map[next(iter(right_identities))] = "right_account"

    visible_sender_name_map: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        identity_key = _message_identity_key(item)
        aliased_ref = alias_map.get(identity_key, item["speaker_ref"])
        visible_sender_label = _coerce_text(item.get("visible_sender_label"))
        if visible_sender_label is not None and aliased_ref in {"left_account", "right_account"}:
            visible_sender_name_map.setdefault(aliased_ref, visible_sender_label)
        trusted_header_name = _trusted_header_display_name(payload, aliased_ref)
        display_name = (
            visible_sender_label
            or trusted_header_name
            or (
                participant_name_map.get(item["speaker_ref"])
                if aliased_ref not in {"left_account", "right_account"}
                else None
            )
        )
        messages.append(
            {
                "index": item["index"],
                "speaker_ref": aliased_ref,
                "display_name": display_name,
                "text": item.get("text"),
                "quote": item.get("quote"),
                "reply": item.get("reply"),
                "reactions": item.get("reactions") or [],
            }
        )

    time_markers = _timestamp_markers_from_observations(observations)
    conversation_type = _coerce_text(payload.get("conversation_type")) or "unknown"
    missing_speaker_refs: list[str] = []
    known_alias_names = {
        speaker_ref: (
            visible_sender_name_map.get(speaker_ref)
            or _trusted_header_display_name(payload, speaker_ref)
        )
        for speaker_ref in ("left_account", "right_account")
    }
    for speaker_ref in ("left_account", "right_account"):
        if any(message["speaker_ref"] == speaker_ref for message in messages):
            if _coerce_text(known_alias_names.get(speaker_ref)) is None:
                missing_speaker_refs.append(speaker_ref)

    if not messages and not time_markers:
        return None

    return {
        "projection_version": "1.0",
        "source_kind": "structured_conversation",
        "conversation_type": conversation_type,
        "participants": participants,
        "time_markers": time_markers,
        "messages": messages,
        "missing_speaker_refs": missing_speaker_refs,
        "speaker_topology": {
            "left_identity_count": len(left_identities),
            "right_identity_count": len(right_identities),
            "supports_group_labels": bool(messages) and len(left_identities) <= 1 and len(right_identities) <= 1,
        },
    }


def serialize_semantic_projection(
    projection: dict[str, Any] | None,
    *,
    evidence_id: str | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> str | None:
    if not isinstance(projection, dict):
        return None
    normalized_labels = {
        key: value
        for key, value in (speaker_labels or {}).items()
        if _coerce_text(key) is not None and _coerce_text(value) is not None
    }

    lines: list[str] = []
    if evidence_id is not None:
        lines.append(f"[evidence {evidence_id}]")
    lines.append(
        f"[semantic_projection v={projection.get('projection_version') or '1.0'}]"
    )

    for marker in projection.get("time_markers", []):
        text = _coerce_text(marker)
        if text is not None:
            lines.append(f"[time_marker] {text}")

    for item in projection.get("messages", []):
        if not isinstance(item, dict):
            continue
        speaker_ref = _coerce_text(item.get("speaker_ref")) or "unknown"
        display_name = (
            _coerce_text(normalized_labels.get(speaker_ref))
            or _coerce_text(item.get("display_name"))
        )
        prefix = f"[message {item.get('index') or 0}][{speaker_ref}]"
        if display_name is not None:
            prefix += f"[display_name={json.dumps(display_name, ensure_ascii=False)}]"
        quote = _normalize_quote_like(item.get("quote"))
        if quote is not None:
            prefix += (
                f"[quote speaker={json.dumps(quote.get('speaker_display_name') or 'unknown', ensure_ascii=False)}"
                f" text={json.dumps(quote.get('text') or 'unknown', ensure_ascii=False)}]"
            )
        reply = _normalize_quote_like(item.get("reply"))
        if reply is not None:
            prefix += (
                f"[reply speaker={json.dumps(reply.get('speaker_display_name') or 'unknown', ensure_ascii=False)}"
                f" text={json.dumps(reply.get('text') or 'unknown', ensure_ascii=False)}]"
            )
        for reaction in _normalize_reactions(item.get("reactions")):
            prefix += (
                f"[reaction emoji={json.dumps(reaction['emoji'], ensure_ascii=False)}"
                f" actor={json.dumps(reaction['actor_display_name'], ensure_ascii=False)}]"
            )
        text = _coerce_text(item.get("text")) or ""
        lines.append(f"{prefix} {text}".rstrip())

    rendered = "\n".join(line for line in lines if isinstance(line, str) and line.strip()).strip()
    return rendered or None
