from __future__ import annotations

import json

import httpx

from app import ai_provider


def test_chat_json_returns_none_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=128) is None


def test_chat_json_uses_default_model_and_json_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    captured = {}

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": '{"ok":true}'}}]
                },
            },
        )()

    monkeypatch.setattr(ai_provider.httpx, "post", fake_post)

    content = ai_provider.chat_json(
        [{"role": "system", "content": "rule"}],
        max_tokens=4096,
        temperature=0,
    )

    assert content == '{"ok":true}'
    assert captured["url"] == ai_provider.DEEPSEEK_CHAT_COMPLETIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["max_tokens"] == 4096
    assert captured["json"]["temperature"] == 0
    assert captured["timeout"] == ai_provider.DEFAULT_TEXT_TIMEOUT_SECONDS


def test_chat_json_allows_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-custom")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": '{"ok":true}'}}]
                },
            },
        )()

    monkeypatch.setattr(ai_provider.httpx, "post", fake_post)

    ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64)

    assert captured["model"] == "deepseek-v4-custom"


def test_chat_json_returns_none_on_timeout_non_200_and_bad_response(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")),
    )
    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64) is None

    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: type("Resp", (), {"status_code": 502, "json": lambda self: {}})(),
    )
    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64) is None

    monkeypatch.setattr(
        ai_provider.httpx,
        "post",
        lambda *args, **kwargs: type(
            "Resp",
            (),
            {"status_code": 200, "json": lambda self: {"choices": [{"message": {"content": 123}}]}},
        )(),
    )
    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64) is None

    class BadJsonResponse:
        status_code = 200

        def json(self):
            raise json.JSONDecodeError("bad", "x", 0)

    monkeypatch.setattr(ai_provider.httpx, "post", lambda *args, **kwargs: BadJsonResponse())
    assert ai_provider.chat_json([{"role": "user", "content": "hi"}], max_tokens=64) is None
