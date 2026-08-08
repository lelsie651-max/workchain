from __future__ import annotations

import json

import httpx

from app import vision_provider


def test_extract_visual_evidence_returns_none_without_ark_key(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    assert vision_provider.extract_visual_evidence(b"fake-image", "image/png") is None


def test_extract_visual_evidence_calls_ark_responses_api_and_normalizes_result(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.delenv("ARK_VISION_MODEL", raising=False)
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
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
                    "output_text": json.dumps(
                        {
                            "transcript": "请周五前补齐渠道复盘数据",
                            "observations": [
                                {
                                    "kind": "reaction",
                                    "content": "有人对该消息显示👍反应",
                                    "confidence": 0.74,
                                    "actor_name": "不应保留",
                                }
                            ],
                            "warnings": ["画面上半部分存在遮挡"],
                        },
                        ensure_ascii=False,
                    )
                },
            },
        )()

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    result = vision_provider.extract_visual_evidence(b"fake-image", "image/png")

    assert result == {
        "transcript": "请周五前补齐渠道复盘数据",
        "observations": [
            {
                "kind": "reaction",
                "content": "有人对该消息显示👍反应",
                "confidence": 0.74,
            }
        ],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": ["画面上半部分存在遮挡"],
    }
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/responses"
    assert captured["headers"]["Authorization"] == "Bearer ark-test-key"
    assert captured["timeout"] == vision_provider.ARK_TIMEOUT_SECONDS
    assert captured["json"]["model"] == "doubao-seed-2-0-lite-260215"
    assert captured["json"]["input"][0]["role"] == "system"
    assert "reaction 存在" in captured["json"]["input"][0]["content"][0]["text"]
    assert captured["json"]["input"][1]["content"][0]["type"] == "input_text"
    assert captured["json"]["input"][1]["content"][1]["type"] == "input_image"
    assert captured["json"]["input"][1]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_extract_visual_evidence_keeps_reaction_identity_unknown_when_not_visible(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "output_text": json.dumps(
                        {
                            "transcript": "后面对话里有人说可能是小王点的赞",
                            "observations": [
                                {
                                    "kind": "reaction",
                                    "content": "有人对该消息显示👍反应,身份在画面中不可见",
                                    "confidence": 0.68,
                                }
                            ],
                            "warnings": [],
                        },
                        ensure_ascii=False,
                    )
                },
            },
        )()

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    result = vision_provider.extract_visual_evidence(b"fake-image", "image/png")

    assert result["observations"] == [
        {
            "kind": "reaction",
            "content": "有人对该消息显示👍反应,身份在画面中不可见",
            "confidence": 0.68,
        }
    ]
    assert "不得猜是谁点的" in captured["json"]["input"][0]["content"][0]["text"]


def test_extract_visual_evidence_returns_none_on_bad_response(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")),
    )
    assert vision_provider.extract_visual_evidence(b"fake-image", "image/png") is None

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: type("Resp", (), {"status_code": 502, "json": lambda self: {}})(),
    )
    assert vision_provider.extract_visual_evidence(b"fake-image", "image/png") is None

    monkeypatch.setattr(
        vision_provider.httpx,
        "post",
        lambda *args, **kwargs: type(
            "Resp",
            (),
            {"status_code": 200, "json": lambda self: {"output_text": "not valid json"}},
        )(),
    )
    assert vision_provider.extract_visual_evidence(b"fake-image", "image/png") is None


def test_extract_visual_evidence_allows_model_override(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.setenv("ARK_VISION_MODEL", "doubao-custom-vision")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {"output_text": json.dumps({"transcript": None, "observations": [], "warnings": ["无"]})},
            },
        )()

    monkeypatch.setattr(vision_provider.httpx, "post", fake_post)

    result = vision_provider.extract_visual_evidence(b"fake-image", "image/png")

    assert captured["model"] == "doubao-custom-vision"
    assert result["model"] == "doubao-custom-vision"
