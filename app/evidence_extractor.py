from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from app import ocr, vision_provider
from evidence_core.extraction_contract import (
    build_extraction_result,
    normalize_observations,
)


DEFAULT_IMAGE_EXTRACTION_PROVIDER = "ocr"
ARK_VISION_EXTRACTION_PROVIDER = "ark_vision"
IMAGE_EXTRACTION_PROVIDER_ENV = "WORKCHAIN_IMAGE_EXTRACTION_PROVIDER"
ARK_FALLBACK_WARNING = "ark_vision_failed_fallback_to_ocr"
IMAGE_PROVIDER_LABELS = {
    DEFAULT_IMAGE_EXTRACTION_PROVIDER: "DashScope OCR",
    ARK_VISION_EXTRACTION_PROVIDER: "Doubao Ark Vision",
}


def get_image_extraction_provider() -> str:
    return os.getenv(IMAGE_EXTRACTION_PROVIDER_ENV, "").strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER


def get_image_extraction_provider_label(provider: str | None = None) -> str:
    selected_provider = (provider or get_image_extraction_provider()).strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER
    return IMAGE_PROVIDER_LABELS.get(selected_provider, selected_provider)


def _extract_with_ocr(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    transcript, note = ocr.image_to_text(image_bytes, mime_type)
    warnings = [] if not note else [note]
    return build_extraction_result(
        transcript=transcript,
        observations=[],
        provider="dashscope",
        model=ocr.OCR_MODEL,
        warnings=warnings,
    )


def _provider_model(provider: str) -> str | None:
    if provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER:
        return ocr.OCR_MODEL
    if provider == ARK_VISION_EXTRACTION_PROVIDER:
        return vision_provider.get_ark_vision_model()
    return None


def _is_provider_configured(provider: str) -> bool:
    if provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER:
        return ocr.is_configured()
    if provider == ARK_VISION_EXTRACTION_PROVIDER:
        return bool(vision_provider.get_ark_api_key())
    return False


def _provider_not_configured_detail(provider: str) -> str:
    if provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER:
        return "图片识别未配置(DASHSCOPE_API_KEY 未设置)"
    if provider == ARK_VISION_EXTRACTION_PROVIDER:
        return "Ark Vision 未配置(ARK_API_KEY 未设置)"
    return f"图片提取 provider 配置无效({provider})"


def _has_extraction_content(extraction: dict[str, Any] | None) -> bool:
    if extraction is None:
        return False
    if extraction.get("transcript") is not None:
        return True
    observations = extraction.get("observations")
    return isinstance(observations, list) and bool(observations)


def _prepend_warning(extraction: dict[str, Any], warning: str) -> dict[str, Any]:
    warnings = [warning]
    for item in extraction.get("warnings", []):
        if item != warning:
            warnings.append(item)
    return build_extraction_result(
        transcript=extraction.get("transcript"),
        observations=extraction.get("observations"),
        provider=extraction.get("provider") or "unknown",
        model=extraction.get("model"),
        warnings=warnings,
        structured_payload=extraction.get("structured_payload"),
    )


def get_image_extraction_startup(provider: str | None = None) -> dict[str, Any]:
    selected_provider = (provider or get_image_extraction_provider()).strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER
    supported = selected_provider in {DEFAULT_IMAGE_EXTRACTION_PROVIDER, ARK_VISION_EXTRACTION_PROVIDER}
    configured = supported and _is_provider_configured(selected_provider)
    return {
        "configured_provider": selected_provider,
        "configured_provider_label": get_image_extraction_provider_label(selected_provider),
        "configured_model": _provider_model(selected_provider),
        "supported": supported,
        "configured": configured,
        "requires_ocr_budget_on_start": selected_provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER,
        "detail": None if configured else _provider_not_configured_detail(selected_provider),
    }


def run_production_image_extraction(
    image_bytes: bytes,
    mime_type: str,
    *,
    provider: str | None = None,
    source_hint: str | None = None,
    allow_ocr_fallback: bool = False,
    consume_ocr_fallback_budget: Callable[[], tuple[bool, str | None]] | None = None,
    ocr_fallback_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    selected_provider = (provider or get_image_extraction_provider()).strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER
    extraction = extract_image_evidence(
        image_bytes,
        mime_type,
        provider=selected_provider,
        source_hint=source_hint,
    )
    if _has_extraction_content(extraction):
        return {
            "configured_provider": selected_provider,
            "configured_provider_label": get_image_extraction_provider_label(selected_provider),
            "configured_model": _provider_model(selected_provider),
            "extraction": extraction,
            "fallback_used": False,
            "detail": None,
        }

    if selected_provider != ARK_VISION_EXTRACTION_PROVIDER:
        warnings = [] if extraction is None else extraction.get("warnings", [])
        detail = warnings[0] if warnings else "图片提取暂不可用,原件已完整保存"
        return {
            "configured_provider": selected_provider,
            "configured_provider_label": get_image_extraction_provider_label(selected_provider),
            "configured_model": _provider_model(selected_provider),
            "extraction": None,
            "fallback_used": False,
            "detail": detail,
        }

    fallback_unavailable_reason = ocr_fallback_unavailable_reason
    if allow_ocr_fallback and not fallback_unavailable_reason and not ocr.is_configured():
        fallback_unavailable_reason = "图片识别未配置(DASHSCOPE_API_KEY 未设置)"

    if allow_ocr_fallback and not fallback_unavailable_reason and consume_ocr_fallback_budget is not None:
        allowed, reason = consume_ocr_fallback_budget()
        if not allowed:
            fallback_unavailable_reason = reason or ocr_fallback_unavailable_reason

    if allow_ocr_fallback and not fallback_unavailable_reason:
        fallback_extraction = extract_image_evidence(
            image_bytes,
            mime_type,
            provider=DEFAULT_IMAGE_EXTRACTION_PROVIDER,
            source_hint=source_hint,
        )
        if _has_extraction_content(fallback_extraction):
            return {
                "configured_provider": selected_provider,
                "configured_provider_label": get_image_extraction_provider_label(selected_provider),
                "configured_model": _provider_model(selected_provider),
                "extraction": _prepend_warning(fallback_extraction, ARK_FALLBACK_WARNING),
                "fallback_used": True,
                "detail": None,
            }

        fallback_warnings = [] if fallback_extraction is None else fallback_extraction.get("warnings", [])
        detail = fallback_warnings[0] if fallback_warnings else "图片提取暂不可用,原件已完整保存"
        return {
            "configured_provider": selected_provider,
            "configured_provider_label": get_image_extraction_provider_label(selected_provider),
            "configured_model": _provider_model(selected_provider),
            "extraction": None,
            "fallback_used": False,
            "detail": f"Ark Vision 提取失败,且 OCR fallback 未成功({detail})。原件已完整保存",
        }

    fallback_reason = fallback_unavailable_reason
    if not fallback_reason:
        fallback_reason = "OCR fallback 暂不可用"
    return {
        "configured_provider": selected_provider,
        "configured_provider_label": get_image_extraction_provider_label(selected_provider),
        "configured_model": _provider_model(selected_provider),
        "extraction": None,
        "fallback_used": False,
        "detail": f"Ark Vision 提取失败,且 {fallback_reason}。原件已完整保存",
    }


def extract_image_evidence(
    image_bytes: bytes,
    mime_type: str,
    *,
    provider: str | None = None,
    source_hint: str | None = None,
) -> dict[str, Any] | None:
    selected_provider = (provider or get_image_extraction_provider()).strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER

    if selected_provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER:
        return _extract_with_ocr(image_bytes, mime_type)
    if selected_provider == ARK_VISION_EXTRACTION_PROVIDER:
        if source_hint is None:
            return vision_provider.extract_visual_evidence(image_bytes, mime_type)
        return vision_provider.extract_visual_evidence(
            image_bytes,
            mime_type,
            source_hint=source_hint,
        )
    raise ValueError(f"unsupported image extraction provider: {selected_provider}")


__all__ = [
    "ARK_VISION_EXTRACTION_PROVIDER",
    "ARK_FALLBACK_WARNING",
    "DEFAULT_IMAGE_EXTRACTION_PROVIDER",
    "IMAGE_EXTRACTION_PROVIDER_ENV",
    "build_extraction_result",
    "extract_image_evidence",
    "get_image_extraction_provider_label",
    "get_image_extraction_startup",
    "get_image_extraction_provider",
    "normalize_observations",
    "run_production_image_extraction",
]
