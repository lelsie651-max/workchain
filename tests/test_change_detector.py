import json
from unittest.mock import patch

from app import change_detector


def _fact(
    fact_id: str,
    evidence_id: str,
    content: str,
    *,
    fact_type: str = "statement",
    occurred_date: str | None = None,
    due_date: str | None = None,
    due_raw: str | None = None,
    actors: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "fact_id": fact_id,
        "fact_type": fact_type,
        "content": content,
        "occurred_date": occurred_date,
        "due_date": due_date,
        "due_raw": due_raw,
        "evidence_id": evidence_id,
        "actors": actors or [],
    }


def _provider_result(payload: dict[str, object]) -> dict[str, object]:
    return {
        "content": json.dumps(payload, ensure_ascii=False),
        "diagnostic": {"success": True, "stage": "success", "model": "deepseek-v4-flash"},
    }


def test_change_detector_case_a_returns_contradiction():
    facts = [
        _fact("fact-1", "ev-1", "张三要求采用方案A。", fact_type="request", occurred_date="2026-05-01"),
        _fact("fact-2", "ev-2", "张三表示“我之前要求的是方案B。”", fact_type="statement", occurred_date="2026-05-06"),
    ]

    with patch(
        "app.change_detector.chat_json_diagnostic_result",
        return_value=_provider_result(
            {
                "changes": [
                    {
                        "change_type": "contradiction",
                        "earlier_fact_index": 0,
                        "later_fact_index": 1,
                        "summary": "较早记录明确是方案A，较新记录回述为方案B。",
                        "confidence": 0.92,
                    }
                ]
            }
        ),
    ):
        result = change_detector.detect_changes("evt-1", facts)

    assert result == {
        "changes": [
            {
                "change_type": "contradiction",
                "earlier_fact_id": "fact-1",
                "later_fact_id": "fact-2",
                "summary": "较早记录明确是方案A，较新记录回述为方案B。",
                "confidence": 0.92,
            }
        ]
    }


def test_change_detector_case_b_returns_deadline_change():
    facts = [
        _fact("fact-1", "ev-1", "周五交。", fact_type="deadline_change", occurred_date="2026-05-01", due_date="2026-05-08"),
        _fact("fact-2", "ev-2", "改成周三交。", fact_type="deadline_change", occurred_date="2026-05-03", due_date="2026-05-06"),
    ]

    with patch(
        "app.change_detector.chat_json_diagnostic_result",
        return_value=_provider_result(
            {
                "changes": [
                    {
                        "change_type": "deadline_change",
                        "earlier_fact_index": 0,
                        "later_fact_index": 1,
                        "summary": "截止时间从周五调整到了周三。",
                        "confidence": 0.88,
                    }
                ]
            }
        ),
    ):
        result = change_detector.detect_changes("evt-1", facts)

    assert result["changes"][0]["change_type"] == "deadline_change"
    assert result["changes"][0]["earlier_fact_id"] == "fact-1"
    assert result["changes"][0]["later_fact_id"] == "fact-2"


def test_change_detector_case_c_returns_responsibility_change():
    facts = [
        _fact(
            "fact-1",
            "ev-1",
            "以前小王负责。",
            fact_type="responsibility_change",
            actors=[{"name": "小王", "role": "owner"}],
        ),
        _fact(
            "fact-2",
            "ev-2",
            "后来改为小李负责。",
            fact_type="responsibility_change",
            actors=[{"name": "小李", "role": "owner"}],
        ),
    ]

    with patch(
        "app.change_detector.chat_json_diagnostic_result",
        return_value=_provider_result(
            {
                "changes": [
                    {
                        "change_type": "responsibility_change",
                        "earlier_fact_index": 0,
                        "later_fact_index": 1,
                        "summary": "负责人从小王变为小李。",
                        "confidence": 0.9,
                    }
                ]
            }
        ),
    ):
        result = change_detector.detect_changes("evt-1", facts)

    assert result["changes"][0]["change_type"] == "responsibility_change"


def test_change_detector_case_d_confirmation_is_not_change():
    facts = [
        _fact("fact-1", "ev-1", "做A。", fact_type="request"),
        _fact("fact-2", "ev-2", "好的。", fact_type="confirmation"),
    ]

    with patch(
        "app.change_detector.chat_json_diagnostic_result",
        return_value=_provider_result({"changes": []}),
    ):
        result = change_detector.detect_changes("evt-1", facts)

    assert result == {"changes": []}


def test_change_detector_case_e_progress_update_is_not_change():
    facts = [
        _fact("fact-1", "ev-1", "做A。", fact_type="request"),
        _fact("fact-2", "ev-2", "A已经完成。", fact_type="delivery"),
    ]

    with patch(
        "app.change_detector.chat_json_diagnostic_result",
        return_value=_provider_result({"changes": []}),
    ):
        result = change_detector.detect_changes("evt-1", facts)

    assert result == {"changes": []}


def test_change_detector_short_circuits_single_evidence_without_model_call():
    facts = [
        _fact("fact-1", "ev-1", "做A。"),
        _fact("fact-2", "ev-1", "改成做B。"),
    ]

    with patch("app.change_detector.chat_json_diagnostic_result") as mock_provider:
        result = change_detector.detect_changes_diagnostic_result("evt-1", facts)

    assert result["result"] == {"changes": []}
    mock_provider.assert_not_called()
