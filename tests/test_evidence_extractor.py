from __future__ import annotations

import pytest

from app import evidence_extractor


def test_extract_image_evidence_defaults_to_current_ocr(monkeypatch):
    monkeypatch.delenv(evidence_extractor.IMAGE_EXTRACTION_PROVIDER_ENV, raising=False)
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: ("审批通过,周五前交付渠道复盘数据", ""),
    )
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: pytest.fail("默认不应调用实验视觉 provider"),
    )

    result = evidence_extractor.extract_image_evidence(b"fake-image", "image/png")

    assert result == {
        "transcript": "审批通过,周五前交付渠道复盘数据",
        "observations": [],
        "provider": "dashscope",
        "model": "vanchin/deepseek-ocr",
        "warnings": [],
    }


def test_extract_image_evidence_keeps_text_only_ocr_behavior(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: (None, "这张图里没有识别到文字,原件已完整保存"),
    )

    result = evidence_extractor.extract_image_evidence(b"fake-image", "image/png")

    assert result == {
        "transcript": None,
        "observations": [],
        "provider": "dashscope",
        "model": "vanchin/deepseek-ocr",
        "warnings": ["这张图里没有识别到文字,原件已完整保存"],
    }


def test_extract_image_evidence_can_use_experimental_ark_vision_provider(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: {
            "transcript": (
                "[chat_title] 微信单聊\n"
                "[message 1][right_account] 计划6月16日上午搬走\n"
                "[message 2][left_contact] 好的\n"
                "[message 3][right_account] 8月16日,打错了\n"
                "[message 4][left_contact] 这个月16号哈"
            ),
            "observations": [
                {
                    "kind": "chat_context",
                    "content": "画面为微信单聊截图,顶部可见聊天标题。",
                    "confidence": 0.74,
                },
                {
                    "kind": "participant_layout",
                    "content": "右侧第1/3条消息属于同一视觉发送方,左侧第2/4条消息属于另一视觉发送方。",
                    "confidence": 0.8,
                }
            ],
            "provider": "doubao-ark",
            "model": "doubao-seed-2-0-lite-260215",
            "warnings": [],
        },
    )

    result = evidence_extractor.extract_image_evidence(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
    )

    assert result == {
        "transcript": (
            "[chat_title] 微信单聊\n"
            "[message 1][right_account] 计划6月16日上午搬走\n"
            "[message 2][left_contact] 好的\n"
            "[message 3][right_account] 8月16日,打错了\n"
            "[message 4][left_contact] 这个月16号哈"
        ),
        "observations": [
            {
                "kind": "chat_context",
                "content": "画面为微信单聊截图,顶部可见聊天标题。",
                "confidence": 0.74,
            },
            {
                "kind": "participant_layout",
                "content": "右侧第1/3条消息属于同一视觉发送方,左侧第2/4条消息属于另一视觉发送方。",
                "confidence": 0.8,
            }
        ],
        "provider": "doubao-ark",
        "model": "doubao-seed-2-0-lite-260215",
        "warnings": [],
    }


def test_extract_image_evidence_ocr_text_only_path_does_not_fabricate_speaker_refs(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: ("计划6月16日上午搬走\n好的\n8月16日,打错了", ""),
    )

    result = evidence_extractor.extract_image_evidence(b"fake-image", "image/png")

    assert result["provider"] == "dashscope"
    assert result["transcript"] == "计划6月16日上午搬走\n好的\n8月16日,打错了"
    assert "[message 1]" not in result["transcript"]
    assert "[right_account]" not in result["transcript"]


def test_extract_image_evidence_can_switch_to_experimental_provider_via_env(monkeypatch):
    monkeypatch.setenv(
        evidence_extractor.IMAGE_EXTRACTION_PROVIDER_ENV,
        evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
    )
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: {
            "transcript": None,
            "observations": [{"kind": "reaction", "content": "有人对该消息显示👍反应", "confidence": None}],
            "provider": "doubao-ark",
            "model": "doubao-seed-2-0-lite-260215",
            "warnings": ["画面局部遮挡"],
        },
    )

    result = evidence_extractor.extract_image_evidence(b"fake-image", "image/png")

    assert result["provider"] == "doubao-ark"
    assert result["warnings"] == ["画面局部遮挡"]


def test_extract_image_evidence_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="unsupported image extraction provider"):
        evidence_extractor.extract_image_evidence(
            b"fake-image",
            "image/png",
            provider="unknown-provider",
        )


def test_image_extraction_startup_defaults_to_ocr(monkeypatch):
    monkeypatch.delenv(evidence_extractor.IMAGE_EXTRACTION_PROVIDER_ENV, raising=False)
    monkeypatch.setattr("app.evidence_extractor.ocr.is_configured", lambda: True)

    startup = evidence_extractor.get_image_extraction_startup()

    assert startup == {
        "configured_provider": "ocr",
        "configured_provider_label": "DashScope OCR",
        "configured_model": "vanchin/deepseek-ocr",
        "supported": True,
        "configured": True,
        "requires_ocr_budget_on_start": True,
        "detail": None,
    }


def test_image_extraction_startup_for_ark_does_not_require_dashscope(monkeypatch):
    monkeypatch.setenv(
        evidence_extractor.IMAGE_EXTRACTION_PROVIDER_ENV,
        evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("app.evidence_extractor.vision_provider.get_ark_api_key", lambda: "ark-test-key")

    startup = evidence_extractor.get_image_extraction_startup()

    assert startup["configured_provider"] == "ark_vision"
    assert startup["configured"] is True
    assert startup["requires_ocr_budget_on_start"] is False


def test_run_production_image_extraction_uses_ark_only_when_successful(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: {
            "transcript": "Ark transcript",
            "observations": [{"kind": "reaction", "content": "有人点了赞", "confidence": 0.6}],
            "provider": "doubao-ark",
            "model": "ark-model",
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: pytest.fail("Ark 成功时不应调用 OCR"),
    )

    result = evidence_extractor.run_production_image_extraction(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        allow_ocr_fallback=True,
    )

    assert result["configured_provider"] == "ark_vision"
    assert result["fallback_used"] is False
    assert result["detail"] is None
    assert result["extraction"]["provider"] == "doubao-ark"
    assert result["extraction"]["observations"] == [{"kind": "reaction", "content": "有人点了赞", "confidence": 0.6}]


def test_run_production_image_extraction_falls_back_to_ocr_with_warning(monkeypatch):
    budget_calls = []
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: None,
    )
    monkeypatch.setattr("app.evidence_extractor.ocr.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: ("OCR transcript", ""),
    )

    result = evidence_extractor.run_production_image_extraction(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        allow_ocr_fallback=True,
        consume_ocr_fallback_budget=lambda: (budget_calls.append("used") or True, None),
    )

    assert budget_calls == ["used"]
    assert result["fallback_used"] is True
    assert result["detail"] is None
    assert result["extraction"]["provider"] == "dashscope"
    assert result["extraction"]["warnings"] == [evidence_extractor.ARK_FALLBACK_WARNING]


def test_run_production_image_extraction_returns_safe_failure_when_fallback_unavailable(monkeypatch):
    budget_calls = []
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type: None,
    )
    monkeypatch.setattr("app.evidence_extractor.ocr.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: pytest.fail("预算失败时不应调用 OCR fallback"),
    )

    result = evidence_extractor.run_production_image_extraction(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        allow_ocr_fallback=True,
        consume_ocr_fallback_budget=lambda: (budget_calls.append("checked") or False, "今日图片识别次数已用完,原件已完整保存"),
    )

    assert budget_calls == ["checked"]
    assert result["extraction"] is None
    assert result["fallback_used"] is False
    assert "今日图片识别次数已用完" in result["detail"]
