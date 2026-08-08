from __future__ import annotations

from app import evidence_extractor


def test_extract_image_evidence_wraps_current_ocr_result(monkeypatch):
    monkeypatch.setattr(
        "app.evidence_extractor.ocr.image_to_text",
        lambda image_bytes, mime_type: ("审批通过,周五前交付渠道复盘数据", ""),
    )

    result = evidence_extractor.extract_image_evidence(b"fake-image", "image/png")

    assert result == {
        "transcript": "审批通过,周五前交付渠道复盘数据",
        "observations": [],
        "provider": "dashscope",
        "model": "vanchin/deepseek-ocr",
        "warnings": [],
    }


def test_extract_image_evidence_keeps_text_only_behavior_without_fake_visual_observations(monkeypatch):
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


def test_build_extraction_result_normalizes_observations():
    result = evidence_extractor.build_extraction_result(
        transcript="  已识别文字  ",
        observations=[
            {"kind": "reaction", "content": " 小王账号对该消息显示👍反应 ", "confidence": 0.8},
            {"kind": "", "content": "bad", "confidence": 0.1},
            {"kind": "read_status", "content": "", "confidence": 0.1},
        ],
        provider=" custom-provider ",
        model=" model-1 ",
        warnings=[" 可能有遮挡 ", "", None],
    )

    assert result == {
        "transcript": "已识别文字",
        "observations": [
            {
                "kind": "reaction",
                "content": "小王账号对该消息显示👍反应",
                "confidence": 0.8,
            }
        ],
        "provider": "custom-provider",
        "model": "model-1",
        "warnings": ["可能有遮挡"],
    }
