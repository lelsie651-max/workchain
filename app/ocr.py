from __future__ import annotations

import base64
import os
import sys
import time
from io import BytesIO
from typing import Any

import httpx
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
OCR_DIAG_DETAIL = "this probe consumed a very small number of tokens"


def is_configured() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())


def _current_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "").strip()


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


def _sanitize_text(value: Any, api_key: str) -> str:
    text = " ".join(str(value).split())
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:160]


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, httpx.TimeoutException)) or type(exc).__name__ == "APITimeoutError"


def _is_connection_exception(exc: Exception) -> bool:
    return isinstance(exc, httpx.RequestError) or type(exc).__name__ == "APIConnectionError"


def _classify_exception(exc: Exception, api_key: str) -> str:
    status_code = _status_code_from_exception(exc)
    exc_name = type(exc).__name__
    if status_code in {401, 403}:
        return f"图片识别鉴权失败(状态码 {status_code})"
    if status_code == 404:
        return "图片识别模型不可用(状态码 404,可能未开通该模型)"
    if status_code == 429:
        return "图片识别调用过于频繁"
    if _is_timeout_exception(exc):
        return "图片识别超时"
    if _is_connection_exception(exc):
        return f"无法连接图片识别服务:{exc_name}"

    detail = str(status_code) if status_code is not None else _sanitize_text(exc, api_key)
    if not detail:
        detail = exc_name
    return f"图片识别失败:{exc_name}({detail})"


def _log_failure(exc: Exception, api_key: str) -> None:
    brief = _sanitize_text(exc, api_key) or type(exc).__name__
    print(f"[ocr] failed: {type(exc).__name__}: {brief}", file=sys.stderr)


def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    prepared_bytes, prepared_mime = _prepare_image_for_ocr(image_bytes)
    return f"data:{prepared_mime};base64,{base64.b64encode(prepared_bytes).decode('ascii')}"


def _create_completion(image_bytes: bytes, mime_type: str, api_key: str) -> Any:
    data_url = _build_data_url(image_bytes, mime_type)
    client = OpenAI(api_key=api_key, base_url=OCR_BASE_URL)
    return client.chat.completions.create(
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


def _build_diag_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), (240, 240, 240))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_text(image_bytes: bytes, mime_type: str) -> tuple[str | None, str]:
    api_key = _current_api_key()
    if not api_key:
        return None, "图片识别未配置(DASHSCOPE_API_KEY 未设置)"

    try:
        response = _create_completion(image_bytes, mime_type, api_key)
        text = _content_to_text(response.choices[0].message.content).strip()
        if len(text) < 5:
            return None, "这张图里没有识别到文字,原件已完整保存"
        return text, ""
    except Exception as exc:
        _log_failure(exc, api_key)
        return None, _classify_exception(exc, api_key)


def diagnose_ocr() -> dict[str, Any]:
    api_key = _current_api_key()
    if not api_key:
        return {
            "configured": False,
            "reachable": None,
            "detail": "DASHSCOPE_API_KEY not set",
        }

    start = time.perf_counter()
    try:
        response = _create_completion(_build_diag_png_bytes(), "image/png", api_key)
        _ = _content_to_text(response.choices[0].message.content)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "configured": True,
            "reachable": True,
            "status_code": 200,
            "latency_ms": latency_ms,
            "detail": f"OCR API reachable; {OCR_DIAG_DETAIL}",
            "model": OCR_MODEL,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_failure(exc, api_key)
        return {
            "configured": True,
            "reachable": False,
            "status_code": _status_code_from_exception(exc),
            "latency_ms": latency_ms,
            "detail": f"{_classify_exception(exc, api_key)}; {OCR_DIAG_DETAIL}",
            "model": OCR_MODEL,
        }
