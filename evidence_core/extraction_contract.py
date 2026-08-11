from __future__ import annotations

from typing import Any


def coerce_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def coerce_optional_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    return None


def normalize_warnings(warnings: Any) -> list[str]:
    normalized: list[str] = []
    if not isinstance(warnings, list):
        return normalized
    for item in warnings:
        warning = coerce_optional_text(item)
        if warning is not None:
            normalized.append(warning)
    return normalized


def normalize_observations(observations: Any) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        kind = coerce_optional_text(item.get("kind"))
        content = coerce_optional_text(item.get("content"))
        if kind is None or content is None:
            continue
        normalized.append(
            {
                "kind": kind,
                "content": content,
                "confidence": coerce_optional_confidence(item.get("confidence")),
            }
        )
    return normalized


def normalize_structured_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return payload


def build_extraction_result(
    *,
    transcript: str | None,
    observations: Any,
    provider: str,
    model: str | None,
    warnings: Any = None,
    structured_payload: Any = None,
) -> dict[str, Any]:
    return {
        "transcript": coerce_optional_text(transcript),
        "observations": normalize_observations(observations),
        "provider": coerce_optional_text(provider) or "unknown",
        "model": coerce_optional_text(model),
        "warnings": normalize_warnings(warnings),
        "structured_payload": normalize_structured_payload(structured_payload),
    }
