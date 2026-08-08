from __future__ import annotations

import os
from typing import Any

import httpx


DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_TEXT_TIMEOUT_SECONDS = 20.0


class AIProviderError(ValueError):
    """Raised when provider configuration or provider response is unsupported."""


def get_text_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "").strip() or DEFAULT_DEEPSEEK_MODEL


def chat_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
) -> str | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        response = httpx.post(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_text_model(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": messages,
            },
            timeout=DEFAULT_TEXT_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None

        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str):
            return None
        return content
    except Exception:
        return None
