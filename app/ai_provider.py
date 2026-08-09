from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx


DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS = 60.0
DEEPSEEK_TEXT_TIMEOUT_ENV = "DEEPSEEK_TEXT_TIMEOUT_SECONDS"
THINKING_DISABLED = {"type": "disabled"}
_REQUEST_ID_HEADER_NAMES = ("x-request-id", "request-id")
_SAFE_MESSAGE_LIMIT = 300


class AIProviderError(ValueError):
    """Raised when provider configuration or provider response is unsupported."""


def get_text_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "").strip() or DEFAULT_DEEPSEEK_MODEL


def get_text_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def get_text_timeout_seconds() -> float:
    raw = os.getenv(DEEPSEEK_TEXT_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS
    if value <= 0:
        return DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS
    return value


def _trim_safe_message(message: str) -> str:
    normalized = re.sub(r"\s+", " ", message).strip() or "unknown error"
    return normalized[:_SAFE_MESSAGE_LIMIT]


def _safe_message(message: str, *, api_key: str) -> str:
    sanitized = str(message or "")
    if api_key:
        sanitized = sanitized.replace(api_key, "[redacted]")
        sanitized = sanitized.replace(f"Bearer {api_key}", "Bearer [redacted]")
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)
    return _trim_safe_message(sanitized)


def _request_id_from_response(response: Any, payload: Any = None) -> str | None:
    headers = getattr(response, "headers", None)
    if headers:
        for name in _REQUEST_ID_HEADER_NAMES:
            value = headers.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(payload, dict):
        value = payload.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _diagnostic(
    *,
    success: bool,
    stage: str,
    timeout_seconds: float,
    model: str,
    status_code: int | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
    safe_message: str | None = None,
    request_id: str | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return {
        "success": success,
        "stage": stage,
        "status_code": status_code,
        "error_code": error_code,
        "error_type": error_type,
        "safe_message": safe_message,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "timeout_seconds": timeout_seconds,
        "thinking_mode": "disabled",
        "model": model,
    }


def build_text_config_diagnostic() -> dict[str, Any]:
    timeout_seconds = get_text_timeout_seconds()
    return _diagnostic(
        success=False,
        stage="config",
        status_code=None,
        error_code="not_configured",
        error_type="missing_api_key",
        safe_message="DEEPSEEK_API_KEY not configured",
        request_id=None,
        latency_ms=0,
        timeout_seconds=timeout_seconds,
        model=get_text_model(),
    )


def _post_chat_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
    thinking: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    api_key = get_text_api_key()
    model = get_text_model()
    timeout_seconds = get_text_timeout_seconds() if timeout_seconds is None else timeout_seconds
    if not api_key:
        return {
            "content": None,
            "diagnostic": build_text_config_diagnostic(),
        }

    start = time.perf_counter()
    try:
        response = httpx.post(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": messages,
                **({"thinking": thinking} if thinking is not None else {}),
            },
            timeout=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="timeout",
                status_code=None,
                error_code="timeout",
                error_type=type(exc).__name__,
                safe_message=_safe_message(
                    f"DeepSeek request timed out after {timeout_seconds:g} seconds",
                    api_key=api_key,
                ),
                request_id=None,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }
    except httpx.NetworkError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="network",
                status_code=None,
                error_code="network_error",
                error_type=type(exc).__name__,
                safe_message=_safe_message(
                    f"Network error while calling DeepSeek API: {exc}",
                    api_key=api_key,
                ),
                request_id=None,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="network",
                status_code=None,
                error_code="request_error",
                error_type=type(exc).__name__,
                safe_message=_safe_message(
                    f"Request error while calling DeepSeek API: {exc}",
                    api_key=api_key,
                ),
                request_id=None,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }

    latency_ms = int((time.perf_counter() - start) * 1000)
    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        response_text = getattr(response, "text", "")
        request_id = _request_id_from_response(response)
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="http",
                status_code=status_code,
                error_code=f"http_{status_code}" if isinstance(status_code, int) else "http_error",
                error_type="http_error",
                safe_message=_safe_message(
                    f"DeepSeek API returned HTTP {status_code}: {response_text}",
                    api_key=api_key,
                ),
                request_id=request_id,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        request_id = _request_id_from_response(response)
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="response_json",
                status_code=200,
                error_code="invalid_response_json",
                error_type=type(exc).__name__,
                safe_message=_safe_message(
                    f"DeepSeek returned HTTP 200 but response JSON was invalid: {exc}",
                    api_key=api_key,
                ),
                request_id=request_id,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }

    request_id = _request_id_from_response(response, payload)
    content = payload.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="output_text",
                status_code=200,
                error_code="missing_output_text",
                error_type="invalid_response_shape",
                safe_message="DeepSeek returned HTTP 200 but choices.message.content was missing",
                request_id=request_id,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }

    if not content.strip():
        return {
            "content": None,
            "diagnostic": _diagnostic(
                success=False,
                stage="empty_content",
                status_code=200,
                error_code="empty_content",
                error_type="empty_content",
                safe_message="DeepSeek returned HTTP 200 but model content was empty",
                request_id=request_id,
                latency_ms=latency_ms,
                timeout_seconds=timeout_seconds,
                model=model,
            ),
        }

    return {
        "content": content,
        "diagnostic": _diagnostic(
            success=True,
            stage="success",
            status_code=200,
            error_code=None,
            error_type=None,
            safe_message=None,
            request_id=request_id,
            latency_ms=latency_ms,
            timeout_seconds=timeout_seconds,
            model=model,
        ),
    }


def chat_json_diagnostic_result(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
) -> dict[str, Any]:
    return _post_chat_json(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=None,
    )


def chat_semantic_json_diagnostic_result(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
) -> dict[str, Any]:
    return _post_chat_json(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=THINKING_DISABLED,
        timeout_seconds=get_text_timeout_seconds(),
    )


def diagnose_deepseek_text_preflight() -> dict[str, Any]:
    result = chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": 'Return a compact JSON object: {"pong":true}'}],
        max_tokens=32,
        temperature=0,
    )
    diagnostic = dict(result["diagnostic"])
    content = result["content"]
    if content is None:
        return diagnostic
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        diagnostic.update(
            {
                "success": False,
                "stage": "model_json",
                "error_code": "invalid_model_json",
                "error_type": type(exc).__name__,
                "safe_message": "DeepSeek returned content, but the JSON ping payload was invalid",
            }
        )
        return diagnostic
    return diagnostic


def chat_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
) -> str | None:
    return chat_json_diagnostic_result(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )["content"]
