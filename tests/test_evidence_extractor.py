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
            "warnings": [],
        },
    )

    result = evidence_extractor.extract_image_evidence(
        b"fake-image",
        "image/png",
        provider=evidence_extractor.ARK_VISION_EXTRACTION_PROVIDER,
    )

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
        "warnings": [],
    }


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
