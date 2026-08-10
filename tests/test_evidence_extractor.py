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
        lambda image_bytes, mime_type, source_hint=None: pytest.fail("默认不应调用实验视觉 provider"),
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


def test_extract_image_evidence_passes_source_hint_to_ark_provider(monkeypatch):
    captured: dict[str, object] = {}

    def fake_extract_visual_evidence(image_bytes, mime_type, source_hint=None):
        captured["source_hint"] = source_hint
        return {
            "transcript": "Ark transcript",
            "observations": [],
            "provider": "doubao-ark",
            "model": "doubao-seed-2-0-lite-260215",
            "warnings": [],
        }

    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        fake_extract_visual_evidence,
    )

    result = evidence_extractor.extract_image_evidence(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        source_hint="微信-单聊",
    )

    assert captured["source_hint"] == "微信-单聊"
    assert result["transcript"] == "Ark transcript"


def test_extract_image_evidence_can_use_experimental_ark_vision_provider(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type, source_hint=None: {
            "transcript": (
                "[scene] platform=微信; conversation_type=direct_chat\n"
                "[chat_header] 戴雯\n"
                "[participant][left_account] display_name=戴雯\n"
                "[participant][right_account] display_name=unknown\n"
                "[message 1][left_account] 饭之\n"
                "[message 2][left_account] 刚开始给我13呢\n"
                "[message 3][left_account] 我脸都绿了\n"
                "[message 4][right_account] 笑死我了\n"
                "[message 5][left_account] 我说我理想中是18\n"
                '[message 6][left_account][quote speaker="unknown" text="戴雯: 刚开始给我13呢"] 感觉被侮辱了\n'
                "[message 7][left_account] 不是很开心\n"
                "[message 8][right_account] 冷静，收集其他同事情况，不动声色！\n"
                "[message 9][right_account] 先看看有没有周栋准备不带着去上海的\n"
                "[message 10][right_account] 不要太高调免得让其他人可能没被带走的失落之类的\n"
                "[message 11][left_account] 肯定是有的\n"
                "[message 12][left_account] David还跟我强调说会有人不被带走\n"
                "[message 13][left_account] 我自己都没啥心情了"
            ),
            "observations": [
                {
                    "kind": "platform_detection",
                    "content": '{"declared_platform": "微信", "observed_platform": "微信", "source_consistency": "match", "platform_confidence": 0.97}',
                    "confidence": 0.97,
                },
                {
                    "kind": "chat_context",
                    "content": "platform=微信; conversation_type=direct_chat",
                    "confidence": None,
                },
                {
                    "kind": "participant_layout",
                    "content": "左侧消息与右侧消息为两个稳定发送方。",
                    "confidence": 0.8,
                },
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
        source_hint="微信-单聊",
    )

    assert result["transcript"].startswith("[scene] platform=微信; conversation_type=direct_chat")
    assert "[participant][left_account] display_name=戴雯" in result["transcript"]
    assert "[participant][right_account] display_name=unknown" in result["transcript"]
    assert "[message 1][left_account] 饭之" in result["transcript"]
    assert "[message 13][left_account] 我自己都没啥心情了" in result["transcript"]
    assert '[message 6][left_account][quote speaker="unknown" text="戴雯: 刚开始给我13呢"] 感觉被侮辱了' in result["transcript"]
    assert "[left_饭之]" not in result["transcript"]


def test_extract_image_evidence_preserves_platform_mismatch_provenance(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type, source_hint=None: {
            "transcript": (
                "[scene] platform=微信; declared_platform=飞书; source_consistency=mismatch; "
                "conversation_type=direct_chat\n"
                "[chat_header] 戴雯\n"
                "[participant][left_account] display_name=戴雯\n"
                "[participant][right_account] display_name=unknown\n"
                "[message 1][left_account] 饭之"
            ),
            "observations": [
                {
                    "kind": "chat_context",
                    "content": "platform=微信; declared_platform=飞书; source_consistency=mismatch; conversation_type=direct_chat",
                    "confidence": None,
                }
            ],
            "provider": "doubao-ark",
            "model": "doubao-seed-2-0-lite-260215",
            "warnings": ["source_platform_mismatch:declared=飞书;observed=微信"],
        },
    )

    result = evidence_extractor.extract_image_evidence(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        source_hint="飞书-单聊",
    )

    assert "declared_platform=飞书" in result["transcript"]
    assert "source_consistency=mismatch" in result["transcript"]
    assert result["warnings"] == ["source_platform_mismatch:declared=飞书;observed=微信"]


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
    assert "[participant]" not in result["transcript"]


def test_extract_image_evidence_can_switch_to_experimental_provider_via_env(monkeypatch):
    monkeypatch.setenv(
        evidence_extractor.IMAGE_EXTRACTION_PROVIDER_ENV,
        evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
    )
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type, source_hint=None: {
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
    captured: dict[str, object] = {}

    def fake_extract_visual_evidence(image_bytes, mime_type, source_hint=None):
        captured["source_hint"] = source_hint
        return {
            "transcript": "Ark transcript",
            "observations": [{"kind": "reaction", "content": "有人点了赞", "confidence": 0.6}],
            "provider": "doubao-ark",
            "model": "ark-model",
            "warnings": [],
        }

    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        fake_extract_visual_evidence,
    )
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: pytest.fail("Ark 成功时不应调用 OCR"),
    )

    result = evidence_extractor.run_production_image_extraction(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
        source_hint="微信-单聊",
        allow_ocr_fallback=True,
    )

    assert captured["source_hint"] == "微信-单聊"
    assert result["configured_provider"] == "ark_vision"
    assert result["fallback_used"] is False
    assert result["detail"] is None
    assert result["extraction"]["provider"] == "doubao-ark"
    assert result["extraction"]["observations"] == [{"kind": "reaction", "content": "有人点了赞", "confidence": 0.6}]


def test_run_production_image_extraction_falls_back_to_ocr_with_warning(monkeypatch):
    budget_calls = []
    monkeypatch.setattr(
        "app.evidence_extractor.vision_provider.extract_visual_evidence",
        lambda image_bytes, mime_type, source_hint=None: None,
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
        lambda image_bytes, mime_type, source_hint=None: None,
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
