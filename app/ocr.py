from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

from openai import OpenAI
from PIL import Image


OCR_MAX_SIDE = 2000
OCR_TIMEOUT_SECONDS = 30.0
OCR_MODEL = "vanchin/deepseek-ocr"
OCR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OCR_PROMPT = (
    "Read all the text in the image. Preserve the original line breaks "
    "and reading order. Output only the text content, no explanation."
)


def is_configured() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())


def _prepare_image_for_ocr(image_bytes: bytes) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        working = image.convert("RGB")
        max_side = max(working.size)
        if max_side > OCR_MAX_SIDE:
            scale = OCR_MAX_SIDE / max_side
            target_size = (
                max(1, int(round(working.size[0] * scale))),
                max(1, int(round(working.size[1] * scale))),
            )
            working = working.resize(target_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        working.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), "image/jpeg"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part).strip()
    return ""


def image_to_text(image_bytes: bytes, mime_type: str) -> tuple[str | None, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return None, "图片识别未配置"

    try:
        prepared_bytes, prepared_mime = _prepare_image_for_ocr(image_bytes)
        data_url = f"data:{prepared_mime};base64,{base64.b64encode(prepared_bytes).decode('ascii')}"
        client = OpenAI(api_key=api_key, base_url=OCR_BASE_URL)
        response = client.chat.completions.create(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                }
            ],
            timeout=OCR_TIMEOUT_SECONDS,
        )
        text = _content_to_text(response.choices[0].message.content).strip()
        if len(text) < 5:
            return None, "这张图里没有识别到文字,原件已完整保存"
        return text, ""
    except Exception as exc:
        return None, f"图片识别暂不可用({type(exc).__name__})"
