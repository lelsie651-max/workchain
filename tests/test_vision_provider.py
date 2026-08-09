from app.vision_provider import VISION_SYSTEM_PROMPT


def test_vision_prompt_requires_timestamp_for_full_date_and_forbids_time_only_guessing():
    assert 'kind 必须是 "timestamp"' in VISION_SYSTEM_PROMPT
    assert "完整年月日,或完整日期+时间" in VISION_SYSTEM_PROMPT
    assert '只有 "19:21" 这类时分' in VISION_SYSTEM_PROMPT
    assert "不得补出年月日" in VISION_SYSTEM_PROMPT
    assert "不得使用上传时间、保存时间或任何画面外时间去推断聊天日期" in VISION_SYSTEM_PROMPT
