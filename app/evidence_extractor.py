from __future__ import annotations

import os
from typing import Any

from app import ocr, vision_provider
from evidence_core.extraction_contract import (
    build_extraction_result,
    normalize_observations,
)


DEFAULT_IMAGE_EXTRACTION_PROVIDER = "ocr"
ARK_VISION_EXTRACTION_PROVIDER = "ark_vision"
IMAGE_EXTRACTION_PROVIDER_ENV = "WORKCHAIN_IMAGE_EXTRACTION_PROVIDER"


def get_image_extraction_provider() -> str:
    return os.getenv(IMAGE_EXTRACTION_PROVIDER_ENV, "").strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER


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


def extract_image_evidence(
    image_bytes: bytes,
    mime_type: str,
    *,
    provider: str | None = None,
) -> dict[str, Any] | None:
    selected_provider = (provider or get_image_extraction_provider()).strip() or DEFAULT_IMAGE_EXTRACTION_PROVIDER

    if selected_provider == DEFAULT_IMAGE_EXTRACTION_PROVIDER:
        return _extract_with_ocr(image_bytes, mime_type)
    if selected_provider == ARK_VISION_EXTRACTION_PROVIDER:
        return vision_provider.extract_visual_evidence(image_bytes, mime_type)
    raise ValueError(f"unsupported image extraction provider: {selected_provider}")


__all__ = [
    "ARK_VISION_EXTRACTION_PROVIDER",
    "DEFAULT_IMAGE_EXTRACTION_PROVIDER",
    "IMAGE_EXTRACTION_PROVIDER_ENV",
    "build_extraction_result",
    "extract_image_evidence",
    "get_image_extraction_provider",
    "normalize_observations",
]
