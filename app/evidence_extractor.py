from __future__ import annotations

from typing import Any

from app import ocr


def _coerce_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _coerce_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    return None


def normalize_observations(observations: Any) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        kind = _coerce_text(item.get("kind"))
        content = _coerce_text(item.get("content"))
        if kind is None or content is None:
            continue
        normalized.append(
            {
                "kind": kind,
                "content": content,
                "confidence": _coerce_confidence(item.get("confidence")),
            }
        )
    return normalized


def build_extraction_result(
    *,
    transcript: str | None,
    observations: Any,
    provider: str,
    model: str | None,
    warnings: Any = None,
) -> dict[str, Any]:
    normalized_warnings: list[str] = []
    if isinstance(warnings, list):
        for item in warnings:
            warning = _coerce_text(item)
            if warning is not None:
                normalized_warnings.append(warning)

    return {
        "transcript": _coerce_text(transcript),
        "observations": normalize_observations(observations),
        "provider": _coerce_text(provider) or "unknown",
        "model": _coerce_text(model),
        "warnings": normalized_warnings,
    }


def extract_image_evidence(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    transcript, note = ocr.image_to_text(image_bytes, mime_type)
    warnings = [] if not note else [note]
    return build_extraction_result(
        transcript=transcript,
        observations=[],
        provider="dashscope",
        model=ocr.OCR_MODEL,
        warnings=warnings,
    )
