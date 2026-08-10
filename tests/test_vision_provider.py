from __future__ import annotations

import json

from app import vision_provider
from app.vision_provider import VISION_SYSTEM_PROMPT, VISION_USER_PROMPT


def _daiwen_payload(*, observed_platform: str = "微信") -> dict:
    return {
        "observed_platform": observed_platform,
        "platform_confidence": 0.97 if observed_platform != "unknown" else None,
        "conversation_type": "direct_chat",
        "chat_header": "戴雯",
        "participants": [
            {"speaker_ref": "left_戴雯", "side": "left", "display_name": None},
            {"speaker_ref": "right_用户", "side": "right", "display_name": None},
        ],
        "messages": [
            {"index": 1, "speaker_ref": "left_饭之", "side": "left", "text": "饭之"},
            {"index": 2, "speaker_ref": "left_戴雯", "side": "left", "text": "刚开始给我13呢"},
            {"index": 3, "speaker_ref": "left_戴雯", "side": "left", "text": "我脸都绿了"},
            {"index": 4, "speaker_ref": "right_用户", "side": "right", "text": "笑死我了"},
            {"index": 5, "speaker_ref": "left_戴雯", "side": "left", "text": "我说我理想中是18"},
            {
                "index": 6,
                "speaker_ref": "left_戴雯",
                "side": "left",
                "text": "感觉被侮辱了",
                "quote": {"speaker_display_name": None, "text": "戴雯: 刚开始给我13呢"},
            },
            {"index": 7, "speaker_ref": "left_戴雯", "side": "left", "text": "不是很开心"},
            {"index": 8, "speaker_ref": "right_用户", "side": "right", "text": "冷静，收集其他同事情况，不动声色！"},
            {"index": 9, "speaker_ref": "right_用户", "side": "right", "text": "先看看有没有周栋准备不带着去上海的"},
            {
                "index": 10,
                "speaker_ref": "right_用户",
                "side": "right",
                "text": "不要太高调免得让其他人可能没被带走的失落之类的",
            },
            {"index": 11, "speaker_ref": "left_戴雯", "side": "left", "text": "肯定是有的"},
            {"index": 12, "speaker_ref": "left_戴雯", "side": "left", "text": "David还跟我强调说会有人不被带走"},
            {"index": 13, "speaker_ref": "left_戴雯", "side": "left", "text": "我自己都没啥心情了"},
        ],
        "observations": [
            {"kind": "participant_layout", "content": "左侧和右侧气泡分离可见。", "confidence": 0.83}
        ],
        "warnings": [],
    }


def test_vision_prompt_requires_timestamp_for_full_date_and_forbids_time_only_guessing():
    assert 'kind 必须是 "timestamp"' in VISION_SYSTEM_PROMPT
    assert "完整年月日,或完整日期+时间" in VISION_SYSTEM_PROMPT
    assert '只有 "19:21" 这类时分' in VISION_SYSTEM_PROMPT
    assert "不得补出年月日" in VISION_SYSTEM_PROMPT
    assert "不得使用上传时间、保存时间或任何画面外时间去推断聊天日期" in VISION_SYSTEM_PROMPT


def test_vision_prompt_requires_declared_vs_observed_platform_and_safe_unknown_side():
    assert "用户填写的 source / platform metadata 只是 declared source" in VISION_SYSTEM_PROMPT
    assert "你必须先根据截图 UI 独立判断 observed_platform" in VISION_SYSTEM_PROMPT
    assert "declared source 只帮助阅读" in VISION_SYSTEM_PROMPT
    assert "不确定时 observed_platform=unknown" in VISION_SYSTEM_PROMPT
    assert '"conversation_type": "direct_chat | group_chat | unknown"' in VISION_SYSTEM_PROMPT
    assert "unknown_account" in VISION_SYSTEM_PROMPT
    assert "绝对禁止 left_戴雯、left_饭之、right_用户" in VISION_SYSTEM_PROMPT
    assert "chat_header 只表示顶部直接可见 UI 文本" in VISION_SYSTEM_PROMPT
    assert "不得把这种规则跨平台套用" in VISION_SYSTEM_PROMPT
    assert "direct_chat 中如果某条 message 的 side 无法确定" in VISION_SYSTEM_PROMPT
    assert "不得只因为 chat_header 含有" in VISION_SYSTEM_PROMPT
    assert "structured conversation" in VISION_USER_PROMPT


def test_build_vision_user_prompt_uses_declared_metadata_not_trusted_platform():
    prompt = vision_provider._build_vision_user_prompt("微信-项目群")

    assert "declared_platform=微信" in prompt
    assert "这只是用户声明,可能正确也可能错误" in prompt
    assert "请先根据截图 UI 独立判断 observed_platform" in prompt
    assert "用户补充的场景提示: 项目群" in prompt
    assert "请按该平台场景阅读界面,不要重新猜平台" not in prompt


def test_build_vision_user_prompt_handles_unknown_platform_without_hard_guess():
    prompt = vision_provider._build_vision_user_prompt("其他-截图来源不明")

    assert "不要硬猜" in prompt
    assert "截图来源不明" in prompt
    assert "declared_platform=微信" not in prompt


def test_diagnose_visual_evidence_uses_responses_json_schema(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {
                "output_text": '{"transcript":"看到聊天截图","observed_platform":"unknown","platform_confidence":null,'
                '"conversation_type":"unknown","chat_header":null,"participants":[],"messages":[],"observations":[],'
                '"warnings":[]}'
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("ARK_API_KEY", "ark-test-key")
    monkeypatch.setattr("app.vision_provider.httpx.post", fake_post)

    diagnostic = vision_provider.diagnose_visual_evidence(b"fake-image", "image/png", source_hint="微信-单聊")

    assert diagnostic["success"] is True
    request_json = captured["json"]
    assert request_json["text"]["format"]["type"] == "json_schema"
    assert request_json["text"]["format"]["name"] == "workchain_visual_extraction"
    assert request_json["text"]["format"]["strict"] is True
    assert request_json["thinking"] == {"type": "disabled"}


def test_normalize_visual_result_gold_case_wechat_direct_chat():
    result = vision_provider._normalize_visual_result(_daiwen_payload(), source_hint="微信-单聊")

    expected_transcript = (
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
    )

    assert result["transcript"] == expected_transcript
    assert result["provider"] == "doubao-ark"
    assert result["warnings"] == [
        "normalized_direct_chat_speaker_ref:left_戴雯",
        "normalized_direct_chat_speaker_ref:right_用户",
        "normalized_direct_chat_speaker_ref:left_饭之",
    ]
    assert result["observations"][0]["kind"] == "platform_detection"
    assert json.loads(result["observations"][0]["content"]) == {
        "declared_platform": "微信",
        "observed_platform": "微信",
        "source_consistency": "match",
        "platform_confidence": 0.97,
    }
    assert result["observations"][1] == {
        "kind": "chat_context",
        "content": "platform=微信; conversation_type=direct_chat",
        "confidence": None,
    }
    assert "[participant][left_account] display_name=饭之" not in result["transcript"]
    assert "[left_饭之]" not in result["transcript"]


def test_normalize_visual_result_preserves_declared_vs_observed_platform_mismatch():
    result = vision_provider._normalize_visual_result(_daiwen_payload(), source_hint="飞书-单聊")

    assert "[scene] platform=微信; declared_platform=飞书; source_consistency=mismatch; conversation_type=direct_chat" in result["transcript"]
    assert "source_platform_mismatch:declared=飞书;observed=微信" in result["warnings"]
    assert json.loads(result["observations"][0]["content"]) == {
        "declared_platform": "飞书",
        "observed_platform": "微信",
        "source_consistency": "mismatch",
        "platform_confidence": 0.97,
    }
    assert "[scene] platform=飞书;" not in result["transcript"]


def test_normalize_visual_result_observed_unknown_is_not_forced_by_declared_source():
    payload = _daiwen_payload(observed_platform="unknown")

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-单聊")

    assert "[scene] platform=unknown; declared_platform=微信; source_consistency=unknown; conversation_type=direct_chat" in result["transcript"]
    assert "[participant][left_account] display_name=unknown" in result["transcript"]
    assert json.loads(result["observations"][0]["content"]) == {
        "declared_platform": "微信",
        "observed_platform": "unknown",
        "source_consistency": "unknown",
        "platform_confidence": None,
    }
    assert "source_platform_mismatch:declared=微信;observed=unknown" not in result["warnings"]


def test_normalize_visual_result_does_not_apply_cross_platform_header_mapping():
    payload = {
        "observed_platform": "飞书",
        "platform_confidence": 0.91,
        "conversation_type": "direct_chat",
        "chat_header": "戴雯",
        "participants": [],
        "messages": [
            {"index": 1, "speaker_ref": "left_user", "side": "left", "text": "收到"},
            {"index": 2, "speaker_ref": "right_user", "side": "right", "text": "好"},
        ],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="飞书-单聊")

    assert "[participant][left_account] display_name=unknown" in result["transcript"]
    assert "[participant][right_account] display_name=unknown" in result["transcript"]


def test_normalize_visual_result_wechat_direct_chat_keeps_baomaqun_as_header_not_group():
    payload = {
        "observed_platform": "微信",
        "platform_confidence": 0.94,
        "conversation_type": "direct_chat",
        "chat_header": "宝妈群",
        "participants": [],
        "messages": [
            {"index": 1, "speaker_ref": "left_user", "side": "left", "text": "今天先这样"},
            {"index": 2, "speaker_ref": "right_user", "side": "right", "text": "收到"},
        ],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-单聊")

    assert "[scene] platform=微信; conversation_type=direct_chat" in result["transcript"]
    assert "[chat_header] 宝妈群" in result["transcript"]
    assert "[participant][left_account] display_name=宝妈群" in result["transcript"]
    assert "[participant][participant_1]" not in result["transcript"]
    assert "[participant][participant_2]" not in result["transcript"]


def test_normalize_visual_result_project_yanfaqun_recall_becomes_system_event_without_participant_2():
    payload = {
        "observed_platform": "微信",
        "platform_confidence": 0.95,
        "conversation_type": "group_chat",
        "chat_header": "项目研发群",
        "participants": [],
        "messages": [
            {"index": 1, "speaker_ref": "left_user", "side": "left", "text": "今天先这样"},
            {"index": 2, "speaker_ref": "right_user", "side": "right", "text": "收到"},
        ],
        "system_events": [
            {
                "type": "message_recalled",
                "visible_text": "冉冉孤生竹🎋撤回了一条消息",
                "actor_display_name": "冉冉孤生竹🎋",
            }
        ],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-单聊")

    assert "[scene] platform=微信; conversation_type=direct_chat" in result["transcript"]
    assert "[chat_header] 项目研发群" in result["transcript"]
    assert '[system_event][message_recalled actor="冉冉孤生竹🎋"] 撤回了一条消息' in result["transcript"]
    assert "[participant][participant_2]" not in result["transcript"]
    assert "group_chat_downgraded_to_direct_without_structural_evidence" in result["warnings"]


def test_normalize_visual_result_requires_structural_evidence_for_group_chat():
    payload = {
        "observed_platform": "微信",
        "platform_confidence": 0.9,
        "conversation_type": "group_chat",
        "chat_header": "项目研发群",
        "participants": [],
        "messages": [
            {"index": 1, "speaker_ref": "left_user", "side": "left", "text": "收到"},
            {"index": 2, "speaker_ref": "right_user", "side": "right", "text": "好"},
        ],
        "system_events": [{"type": "system_notice", "visible_text": "凡加入了群聊", "actor_display_name": "凡"}],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-单聊")

    assert "[scene] platform=微信; conversation_type=direct_chat" in result["transcript"]
    assert "group_chat_downgraded_to_direct_without_structural_evidence" in result["warnings"]


def test_normalize_visual_result_real_group_fixture_reuses_two_left_participants():
    payload = {
        "observed_platform": "微信",
        "platform_confidence": 0.93,
        "conversation_type": "group_chat",
        "chat_header": "项目研发群",
        "participants": [
            {"speaker_ref": "left_a", "side": "left", "display_name": "凡", "layout_identity": "avatar-a"},
            {"speaker_ref": "left_b", "side": "left", "display_name": "念文雯", "layout_identity": "avatar-b"},
        ],
        "messages": [
            {"index": 1, "speaker_ref": "left_a", "side": "left", "visible_sender_label": "凡", "avatar_ref": "avatar-a", "text": "我先看下"},
            {"index": 2, "speaker_ref": "left_b", "side": "left", "visible_sender_label": "念文雯", "avatar_ref": "avatar-b", "text": "我补截图"},
            {"index": 3, "speaker_ref": "left_b", "side": "left", "visible_sender_label": "念文雯", "avatar_ref": "avatar-b", "text": "已经发群里了"},
            {"index": 4, "speaker_ref": "right_me", "side": "right", "text": "收到"},
        ],
        "system_events": [],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-项目群")

    assert "[scene] platform=微信; conversation_type=group_chat" in result["transcript"]
    assert "[participant][participant_1] side=left display_name=凡" in result["transcript"]
    assert "[participant][participant_2] side=left display_name=念文雯" in result["transcript"]
    assert "[message 2][participant_2] 我补截图" in result["transcript"]
    assert "[message 3][participant_2] 已经发群里了" in result["transcript"]
    assert "[message 4][right_account] 收到" in result["transcript"]


def test_normalize_visual_result_keeps_exact_emoji_or_unknown():
    payload = {
        "observed_platform": "飞书",
        "platform_confidence": 0.9,
        "conversation_type": "group_chat",
        "chat_header": "A大冲小强",
        "participants": [
            {"speaker_ref": "A大", "side": "left", "display_name": "A大", "layout_identity": "avatar-a"},
            {"speaker_ref": "冲小强", "side": "left", "display_name": "冲小强", "layout_identity": "avatar-b"},
        ],
        "messages": [
            {
                "index": 1,
                "speaker_ref": "冲小强",
                "side": "left",
                "text": "我先改文案😄",
                "reactions": [
                    {"emoji": "👍", "actor_display_name": "unknown"},
                    {"emoji": "uncertain emoji", "actor_display_name": "unknown"},
                ],
            },
            {"index": 2, "speaker_ref": "right_me", "side": "right", "text": "收到"},
        ],
        "system_events": [],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="飞书-项目群")

    assert '[reaction emoji="👍" actor="unknown"]' in result["transcript"]
    assert '[reaction emoji="[emoji_unknown]" actor="unknown"]' in result["transcript"]
    assert "emoji_uncertain_normalized_to_unknown" in result["warnings"]


def test_normalize_visual_result_unknown_side_does_not_default_to_left():
    payload = {
        "observed_platform": "微信",
        "platform_confidence": 0.66,
        "conversation_type": "direct_chat",
        "chat_header": "戴雯",
        "participants": [],
        "messages": [
            {"index": 1, "speaker_ref": "mystery_ref", "side": "unknown", "text": "看到了"},
        ],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="微信-单聊")

    assert "[participant][unknown_account] side=unknown display_name=unknown" in result["transcript"]
    assert "[message 1][unknown_account] 看到了" in result["transcript"]
    assert "[message 1][left_account] 看到了" not in result["transcript"]
    assert "missing_direct_chat_side:message_1" in result["warnings"]


def test_normalize_visual_result_a_da_chong_xiaoqiang_group_chat_regression():
    payload = {
        "observed_platform": "飞书",
        "platform_confidence": 0.92,
        "conversation_type": "group_chat",
        "chat_header": "A大冲小强",
        "participants": [
            {"speaker_ref": "A大", "side": "left", "display_name": "A大", "layout_identity": "avatar-a"},
            {"speaker_ref": "冲小强", "side": "left", "display_name": "冲小强", "layout_identity": "avatar-b"},
        ],
        "messages": [
            {"index": 1, "speaker_ref": "A大", "side": "left", "text": "今天先不要发"},
            {
                "index": 2,
                "speaker_ref": "冲小强",
                "side": "left",
                "text": "收到",
                "reply": {"speaker_display_name": "A大", "text": "今天先不要发"},
            },
            {
                "index": 3,
                "speaker_ref": "冲小强",
                "side": "left",
                "text": "我先改文案",
                "reactions": [{"emoji": "👍", "actor_display_name": "unknown"}],
            },
            {"index": 4, "speaker_ref": "right_me", "side": "right", "text": "我来跟进"},
        ],
        "observations": [],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="飞书-项目群")

    assert "[scene] platform=飞书; conversation_type=group_chat" in result["transcript"]
    assert "[participant][participant_1] side=left display_name=A大" in result["transcript"]
    assert "[participant][participant_2] side=left display_name=冲小强" in result["transcript"]
    assert '[message 2][participant_2][reply speaker="A大" text="今天先不要发"] 收到' in result["transcript"]
    assert '[message 3][participant_2][reaction emoji="👍" actor="unknown"] 我先改文案' in result["transcript"]
    assert "[message 4][right_account] 我来跟进" in result["transcript"]


def test_normalize_visual_result_non_chat_keeps_plain_transcript():
    payload = {
        "transcript": "审批通过,周五前交付渠道复盘数据",
        "observed_platform": "unknown",
        "platform_confidence": None,
        "conversation_type": "unknown",
        "chat_header": None,
        "participants": [],
        "messages": [],
        "observations": [{"kind": "timestamp", "content": "2026-08-09 19:21", "confidence": 0.91}],
        "warnings": [],
    }

    result = vision_provider._normalize_visual_result(payload, source_hint="其他-未知")

    assert result == {
        "transcript": "审批通过,周五前交付渠道复盘数据",
        "observations": [
            {
                "kind": "platform_detection",
                "content": '{"declared_platform": null, "observed_platform": "unknown", "source_consistency": "unknown", "platform_confidence": null}',
                "confidence": None,
            },
            {"kind": "timestamp", "content": "2026-08-09 19:21", "confidence": 0.91},
        ],
        "provider": "doubao-ark",
        "model": vision_provider.get_ark_vision_model(),
        "warnings": [],
    }
