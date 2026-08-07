from __future__ import annotations

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
