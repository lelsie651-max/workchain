from __future__ import annotations

import json

import httpx

from app import llm
from evidence_core.db import init_db
from scripts.seed_demo import seed_demo_data


def test_parse_llm_json_handles_multiple_wrappers():
    raw_json = (
        '{"requester_name":"张总","owner_name":"我","deliverable":"复盘",'
        '"due_raw":"下周五","due_date":"2026-08-08","direction":"i_owe",'
        '"kind":"request","plain_summary":"张总让你做复盘","caveats":[]}'
    )
    fenced = f"```json\n{raw_json}\n```"
    explained = f"下面是结果:\n{raw_json}\n请查收"

    assert llm.parse_llm_json(raw_json)["deliverable"] == "复盘"
    assert llm.parse_llm_json(fenced)["deliverable"] == "复盘"
    assert llm.parse_llm_json(explained)["deliverable"] == "复盘"
    assert llm.parse_llm_json("not json") is None


def test_extract_slots_returns_none_on_timeout(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm.httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.TimeoutException("boom")))

    assert llm.extract_slots("张总:下周五给我复盘", "2026-08-01") is None


def test_resolve_actor_maps_self_and_alias(tmp_path):
    demo_dir = tmp_path / "demo"
    seed_demo_data(demo_dir)
    conn = init_db(demo_dir / "workchain.db")
    try:
        self_actor = llm.resolve_actor(conn, "我")
        alias_actor = llm.resolve_actor(conn, "张总")
        assert self_actor == "act_self"
        assert alias_actor == "act_zhang"
    finally:
        conn.close()


def test_build_context_block_includes_known_info_and_suffix():
    context = {
        "self_names": ["热心市民小李", "小李"],
        "counterpart": "冯云生(师父)",
        "glossary": [
            {"term": "活爹", "kind": "person", "meaning": "我的上级"},
            {"term": "老地方", "kind": "phrase", "meaning": "公司楼下咖啡店"},
        ],
    }

    block = llm.build_context_block(context)

    assert "【已知信息】" in block
    assert "热心市民小李、小李" in block
    assert "活爹 → 我的上级(指人)" in block
    assert "老地方 → 公司楼下咖啡店(说法)" in block
    assert "以原文语境为准" in block


def test_build_context_block_omits_known_info_when_empty():
    assert llm.build_context_block(None) == ""
    assert llm.build_context_block({}) == ""
    assert llm.build_context_block({"self_names": [""], "counterpart": "", "glossary": []}) == ""


def test_extract_slots_includes_context_when_present(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": json.dumps({"kind": "reference", "plain_summary": "留档"})}}]
                },
            },
        )()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    llm.extract_slots(
        "行,师父我下下周过去",
        "2026-08-07",
        context={
            "self_names": ["热心市民小李"],
            "counterpart": "冯云生(师父)",
            "glossary": [{"term": "活爹", "kind": "person", "meaning": "张伟"}],
        },
    )

    joined = "\n".join(message["content"] for message in captured["messages"])
    assert "【已知信息】" in joined
    assert "活爹" in joined
    assert "以原文语境为准" in joined


def test_extract_slots_omits_context_when_absent(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["messages"] = kwargs["json"]["messages"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": json.dumps({"kind": "reference", "plain_summary": "留档"})}}]
                },
            },
        )()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    llm.extract_slots("留档", "2026-08-07", context=None)

    joined = "\n".join(message["content"] for message in captured["messages"])
    assert "【已知信息】" not in joined
    assert "以原文语境为准" not in joined


def test_extract_slots_uses_default_deepseek_v4_flash_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": json.dumps({"kind": "reference", "plain_summary": "留档"})}}]
                },
            },
        )()

    monkeypatch.setattr(llm.httpx, "post", fake_post)

    llm.extract_slots("留档", "2026-08-07")

    assert captured["model"] == "deepseek-v4-flash"


def test_extract_slots_allows_deepseek_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-test")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["model"] = kwargs["json"]["model"]
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": json.dumps({"kind": "reference", "plain_summary": "留档"})}}]
                },
            },
        )()

    monkeypatch.setattr(llm.httpx, "post", fake_post)

    llm.extract_slots("留档", "2026-08-07")

    assert captured["model"] == "deepseek-v4-test"


def test_resolve_actor_with_glossary_backfills_alias(tmp_path):
    demo_dir = tmp_path / "demo"
    seed_demo_data(demo_dir)
    conn = init_db(demo_dir / "workchain.db")
    try:
        actor_id = llm.resolve_actor_with_glossary(
            conn,
            "活爹",
            [{"term": "活爹", "kind": "person", "meaning": "张伟"}],
        )
        row = conn.execute(
            "SELECT aliases FROM actors WHERE actor_id = ?",
            (actor_id,),
        ).fetchone()
        aliases = json.loads(row["aliases"])
        assert actor_id == "act_zhang"
        assert "活爹" in aliases
    finally:
        conn.close()
