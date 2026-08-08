from __future__ import annotations

import ast
from pathlib import Path

from evidence_core.extraction_contract import build_extraction_result, normalize_observations


def test_build_extraction_result_normalizes_pure_data_fields():
    result = build_extraction_result(
        transcript="  已识别文字  ",
        observations=[
            {"kind": "reaction", "content": " 有人对该消息显示👍反应 ", "confidence": 0.7, "actor": "不应保留"},
            {"kind": "", "content": "bad", "confidence": 0.2},
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
                "content": "有人对该消息显示👍反应",
                "confidence": 0.7,
            }
        ],
        "provider": "custom-provider",
        "model": "model-1",
        "warnings": ["可能有遮挡"],
    }


def test_normalize_observations_keeps_only_contract_fields():
    observations = normalize_observations(
        [
            {
                "kind": "reaction",
                "content": "有人对该消息显示👍反应",
                "confidence": 0.61,
                "actor_name": "小王",
                "source": "后续对话猜测",
            }
        ]
    )

    assert observations == [
        {
            "kind": "reaction",
            "content": "有人对该消息显示👍反应",
            "confidence": 0.61,
        }
    ]
    assert "actor_name" not in observations[0]


def test_evidence_core_modules_do_not_import_app_package():
    evidence_core_dir = Path(__file__).resolve().parents[1] / "evidence_core"

    for path in evidence_core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app"):
                raise AssertionError(f"{path.name} must not import app.*")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app"):
                        raise AssertionError(f"{path.name} must not import app.*")
