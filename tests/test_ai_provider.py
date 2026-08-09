from __future__ import annotations

import json

import httpx
import pytest

from app import ai_provider


class _FakeResponse:
    def __init__(self, *, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_chat_json_returns_none_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=128) is None


def test_semantic_request_uses_thinking_disabled_and_default_timeout(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv(ai_provider.DEEPSEEK_TEXT_TIMEOUT_ENV, raising=False)
    captured = {}

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return _FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": '{"ok":true}'}}]},
            headers={"x-request-id": "req-success"},
        )

    monkeypatch.setattr(ai_provider.httpx, "post", fake_post)

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "system", "content": "rule"}],
        max_tokens=4096,
        temperature=0,
    )

    assert result["content"] == '{"ok":true}'
    assert captured["url"] == ai_provider.DEEPSEEK_CHAT_COMPLETIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["temperature"] == 0
    assert captured["timeout"] == ai_provider.DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS
    assert result["diagnostic"]["success"] is True
    assert result["diagnostic"]["stage"] == "success"
    assert result["diagnostic"]["thinking_mode"] == "disabled"
    assert result["diagnostic"]["timeout_seconds"] == ai_provider.DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS
    assert result["diagnostic"]["request_id"] == "req-success"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("75", 75.0),
        ("0", ai_provider.DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS),
        ("-3", ai_provider.DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS),
        ("bad", ai_provider.DEFAULT_DEEPSEEK_TEXT_TIMEOUT_SECONDS),
    ],
)
def test_text_timeout_env_override_and_fallback(monkeypatch, env_value: str, expected: float):
    monkeypatch.setenv(ai_provider.DEEPSEEK_TEXT_TIMEOUT_ENV, env_value)

    assert ai_provider.get_text_timeout_seconds() == expected


def test_chat_json_allows_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-custom")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return _FakeResponse(status_code=200, payload={"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr(ai_provider.httpx, "post", fake_post)

    ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64)

    assert captured["model"] == "deepseek-v4-custom"


def test_timeout_is_classified(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv(ai_provider.DEEPSEEK_TEXT_TIMEOUT_ENV, "61")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")),
    )

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    assert result["content"] is None
    assert result["diagnostic"]["stage"] == "timeout"
    assert result["diagnostic"]["error_code"] == "timeout"
    assert result["diagnostic"]["timeout_seconds"] == 61.0


@pytest.mark.parametrize("status_code", [400, 401, 402, 422, 429, 500])
def test_http_statuses_are_classified(monkeypatch, status_code: int):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=status_code,
            payload={},
            headers={"request-id": f"req-{status_code}"},
            text=f"upstream failed with sk-test and http {status_code}",
        ),
    )

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    assert result["content"] is None
    assert result["diagnostic"]["stage"] == "http"
    assert result["diagnostic"]["status_code"] == status_code
    assert result["diagnostic"]["error_code"] == f"http_{status_code}"
    assert result["diagnostic"]["request_id"] == f"req-{status_code}"
    assert "sk-test" not in result["diagnostic"]["safe_message"]
    assert "[redacted]" in result["diagnostic"]["safe_message"]


def test_invalid_response_json_is_classified(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=200,
            payload=json.JSONDecodeError("bad", "{}", 0),
        ),
    )

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    assert result["content"] is None
    assert result["diagnostic"]["stage"] == "response_json"
    assert result["diagnostic"]["error_code"] == "invalid_response_json"


def test_missing_output_text_is_classified(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(status_code=200, payload={"choices": [{"message": {}}]}),
    )

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    assert result["content"] is None
    assert result["diagnostic"]["stage"] == "output_text"
    assert result["diagnostic"]["error_code"] == "missing_output_text"


def test_empty_content_is_classified(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": "   "}}]},
        ),
    )

    result = ai_provider.chat_semantic_json_diagnostic_result(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    assert result["content"] is None
    assert result["diagnostic"]["stage"] == "empty_content"
    assert result["diagnostic"]["error_code"] == "empty_content"


def test_preflight_reports_invalid_model_json(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": "{not-json}"}}]},
        ),
    )

    diagnostic = ai_provider.diagnose_deepseek_text_preflight()

    assert diagnostic["success"] is False
    assert diagnostic["stage"] == "model_json"
    assert diagnostic["error_code"] == "invalid_model_json"
    assert diagnostic["thinking_mode"] == "disabled"


def test_successful_preflight_returns_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=200,
            payload={"id": "chatcmpl-123", "choices": [{"message": {"content": '{"pong":true}'}}]},
        ),
    )

    diagnostic = ai_provider.diagnose_deepseek_text_preflight()

    assert diagnostic["success"] is True
    assert diagnostic["stage"] == "success"
    assert diagnostic["request_id"] == "chatcmpl-123"
