from __future__ import annotations

import json

import httpx

from app import semantic_llm


def _response_with_content(content):
    return type(
        "Resp",
        (),
        {
            "status_code": 200,
            "json": lambda self: {
                "choices": [{"message": {"content": content}}]
            },
        },
    )()


def test_extract_semantics_splits_atomic_facts_from_one_message(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "scope_change",
                            "content": "老板要求新增渠道复盘中的竞品对比部分",
                            "confidence": 0.92,
                            "actors": [{"name": "老板", "role": "requester"}],
                            "occurred_date": None,
                            "due_raw": None,
                            "due_date": None,
                            "due_anchor_date": None,
                        },
                        {
                            "fact_type": "responsibility_change",
                            "content": "负责人从小李变更为小王",
                            "confidence": 0.88,
                            "actors": [
                                {"name": "小李", "role": "previous_owner"},
                                {"name": "小王", "role": "new_owner"},
                            ],
                            "occurred_date": None,
                            "due_raw": None,
                            "due_date": None,
                            "due_anchor_date": None,
                        },
                        {
                            "fact_type": "deadline_change",
                            "content": "截止时间被提前到下周三",
                            "confidence": 0.9,
                            "actors": [],
                            "occurred_date": None,
                            "due_raw": "下周三",
                            "due_date": "2026-08-12",
                            "due_anchor_date": "2026-08-07",
                        },
                    ],
                    "interpretations": [],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics(
        "老板说竞品对比也加进去，这块以后小王负责，截止提前到下周三。",
        anchor_date="2026-08-07",
    )

    assert [fact["fact_type"] for fact in result["facts"]] == [
        "scope_change",
        "responsibility_change",
        "deadline_change",
    ]
    assert result["facts"][2]["due_date"] == "2026-08-12"


def test_group_chat_without_user_speaking_still_parses(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "statement",
                            "content": "群里有人说明天下午开评审会",
                            "confidence": 0.77,
                            "actors": [{"name": "王经理", "role": "speaker"}],
                            "occurred_date": None,
                            "due_raw": "明天下午",
                            "due_date": None,
                            "due_anchor_date": None,
                        }
                    ],
                    "interpretations": [],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics("群里王经理说，明天下午统一评审。")

    assert result["facts"][0]["fact_type"] == "statement"
    assert result["facts"][0]["actors"] == [{"name": "王经理", "role": "speaker"}]


def test_prompt_keeps_unproven_accusation_as_neutral_statement_rule(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return _response_with_content(json.dumps({"facts": [], "interpretations": [], "ambiguities": []}))

    monkeypatch.setattr(semantic_llm.httpx, "post", fake_post)

    semantic_llm.extract_semantics("某人声称张总挪用20万。")

    joined = "\n".join(message["content"] for message in captured["messages"])
    assert "未经证明的指控、传闻、猜测" in joined
    assert "只能保留为 statement / denial / reference" in joined


def test_non_work_content_is_not_forced_into_work_fields(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "statement",
                            "content": "他们在聊谁跟谁分手了",
                            "confidence": 0.64,
                            "actors": [],
                            "occurred_date": None,
                            "due_raw": None,
                            "due_date": None,
                            "due_anchor_date": None,
                            "deliverable": "不应存在",
                        }
                    ],
                    "interpretations": [],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics("听说他们俩分手了。")

    assert result["facts"][0] == {
        "fact_type": "statement",
        "content": "他们在聊谁跟谁分手了",
        "confidence": 0.64,
        "actors": [],
        "occurred_date": None,
        "due_raw": None,
        "due_date": None,
        "due_anchor_date": None,
    }


def test_slang_generates_interpretation_and_not_fact_content(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "request",
                            "content": "对方要求先补齐一版材料",
                            "confidence": 0.83,
                            "actors": [{"name": "师父", "role": "requester"}],
                            "occurred_date": None,
                            "due_raw": None,
                            "due_date": None,
                            "due_anchor_date": None,
                        }
                    ],
                    "interpretations": [
                        {
                            "fact_index": 0,
                            "kind": "term",
                            "content": "“拉齐一下”通常指先统一口径或补齐必要信息",
                            "confidence": 0.79,
                        }
                    ],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics("师父说先把这事拉齐一下。")

    assert "拉齐一下" not in result["facts"][0]["content"]
    assert result["interpretations"][0]["kind"] == "term"
    assert "拉齐一下" in result["interpretations"][0]["content"]


def test_anchor_date_allows_due_date_resolution_and_keeps_raw_and_anchor(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "deadline_change",
                            "content": "截止时间改为下下周五",
                            "confidence": 0.85,
                            "actors": [],
                            "occurred_date": None,
                            "due_raw": "下下周五",
                            "due_date": "2026-08-21",
                            "due_anchor_date": "1999-01-01",
                        }
                    ],
                    "interpretations": [],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics("改到下下周五交。", anchor_date="2026-08-07")

    assert result["facts"][0]["due_raw"] == "下下周五"
    assert result["facts"][0]["due_date"] == "2026-08-21"
    assert result["facts"][0]["due_anchor_date"] == "2026-08-07"


def test_without_anchor_due_date_must_be_null(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content(
            json.dumps(
                {
                    "facts": [
                        {
                            "fact_type": "deadline_change",
                            "content": "截止时间改为下下周五",
                            "confidence": 0.85,
                            "actors": [],
                            "occurred_date": None,
                            "due_raw": "下下周五",
                            "due_date": "2026-08-21",
                            "due_anchor_date": "2026-08-07",
                        }
                    ],
                    "interpretations": [],
                    "ambiguities": [],
                }
            )
        ),
    )

    result = semantic_llm.extract_semantics("改到下下周五交。", anchor_date=None)

    assert result["facts"][0]["due_raw"] == "下下周五"
    assert result["facts"][0]["due_date"] is None
    assert result["facts"][0]["due_anchor_date"] is None


def test_glossary_enters_request_context_and_self_names_do_not(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return _response_with_content(json.dumps({"facts": [], "interpretations": [], "ambiguities": []}))

    monkeypatch.setattr(semantic_llm.httpx, "post", fake_post)

    semantic_llm.extract_semantics(
        "活爹说这事先拉齐。",
        glossary=[{"term": "活爹", "kind": "person", "meaning": "张伟"}],
        source_hint="飞书群-项目A",
    )

    joined = "\n".join(message["content"] for message in captured["messages"])
    assert "glossary:" in joined
    assert "活爹 -> 张伟 (person)" in joined
    assert "source_hint=飞书群-项目A" in joined
    assert "self_names" not in joined


def test_malformed_json_timeout_non_200_and_invalid_fields_are_safe(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: _response_with_content("```json\nnot valid\n```"),
    )
    assert semantic_llm.extract_semantics("留档") is None

    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")),
    )
    assert semantic_llm.extract_semantics("留档") is None

    monkeypatch.setattr(
        semantic_llm.httpx,
        "post",
        lambda *args, **kwargs: type(
            "Resp",
            (),
            {"status_code": 502, "json": lambda self: {}},
        )(),
    )
    assert semantic_llm.extract_semantics("留档") is None

    normalized = semantic_llm.parse_semantic_json(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_type": "not-a-fact",
                        "content": "应该被丢弃",
                        "confidence": 3,
                        "actors": "bad",
                        "occurred_date": "2026-13-40",
                        "due_raw": "下周五",
                        "due_date": "2026-99-99",
                        "due_anchor_date": "2026-08-07",
                    },
                    {
                        "fact_type": "statement",
                        "content": "保留这一条",
                        "confidence": 2,
                        "actors": [{"name": "王总", "role": "speaker"}, {"name": "", "role": "x"}],
                        "occurred_date": "2026-13-40",
                        "due_raw": "下周五",
                        "due_date": "2026-08-15",
                        "due_anchor_date": "bad",
                    },
                ],
                "interpretations": [
                    {"fact_index": 5, "kind": "explanation", "content": "越界应丢弃", "confidence": 0.5},
                    {"fact_index": 0, "kind": "bad", "content": "非法种类", "confidence": 0.5},
                    {"fact_index": 0, "kind": "uncertainty", "content": "这一条保留", "confidence": 5},
                ],
                "ambiguities": [" 还有歧义 ", 123],
            }
        ),
        anchor_date="2026-08-07",
    )

    assert normalized == {
        "facts": [
            {
                "fact_type": "statement",
                "content": "保留这一条",
                "confidence": 0.0,
                "actors": [{"name": "王总", "role": "speaker"}],
                "occurred_date": None,
                "due_raw": "下周五",
                "due_date": "2026-08-15",
                "due_anchor_date": "2026-08-07",
            }
        ],
        "interpretations": [
            {
                "fact_index": 0,
                "kind": "uncertainty",
                "content": "这一条保留",
                "confidence": 0.0,
            }
        ],
        "ambiguities": ["还有歧义"],
    }


def test_default_model_is_deepseek_v4_flash(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return _response_with_content(json.dumps({"facts": [], "interpretations": [], "ambiguities": []}))

    monkeypatch.setattr(semantic_llm.httpx, "post", fake_post)

    semantic_llm.extract_semantics("留档")

    assert captured["model"] == "deepseek-v4-flash"


def test_deepseek_model_env_can_override_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-custom")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return _response_with_content(json.dumps({"facts": [], "interpretations": [], "ambiguities": []}))

    monkeypatch.setattr(semantic_llm.httpx, "post", fake_post)

    semantic_llm.extract_semantics("留档")

    assert captured["model"] == "deepseek-v4-custom"
