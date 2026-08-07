from __future__ import annotations

"""
Generate a deterministic demo dataset for WorkChain.

Thread rows and evidence.thread_id assignments in this script are manual demo
orchestration data for presentation purposes. They are not produced by the
future thread-merging algorithm.
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from evidence_core.db import init_db
from evidence_core.store import append_evidence, update_slots, verify_chain


def _ts(iso_value: str) -> int:
    return int(datetime.fromisoformat(iso_value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _json_text(value: list[str]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _actors() -> list[dict]:
    created_at = _ts("2026-03-05T09:00:00")
    return [
        {
            "actor_id": "act_self",
            "canonical_name": "我",
            "aliases": [],
            "org": "增长运营部",
            "role_hint": "本人",
            "is_self": 1,
            "confidence": None,
            "created_at": created_at,
        },
        {
            "actor_id": "act_zhang",
            "canonical_name": "张伟",
            "aliases": ["张总", "老张"],
            "org": "增长运营部",
            "role_hint": "上级",
            "is_self": 0,
            "confidence": None,
            "created_at": created_at,
        },
        {
            "actor_id": "act_li",
            "canonical_name": "李娜",
            "aliases": [],
            "org": "商业分析组",
            "role_hint": "同级",
            "is_self": 0,
            "confidence": None,
            "created_at": created_at,
        },
        {
            "actor_id": "act_wang",
            "canonical_name": "王强",
            "aliases": [],
            "org": "平台研发组",
            "role_hint": "下游",
            "is_self": 0,
            "confidence": None,
            "created_at": created_at,
        },
    ]


def _threads() -> list[dict]:
    return [
        {
            "thread_id": "thr_channel",
            "title": "渠道复盘数据",
            "status": "disputed",
            "owner_actor_id": "act_self",
            "requester_actor_id": "act_zhang",
            "current_deliverable": "渠道复盘数据最终版（范围收窄，含竞品对比）",
            "current_due": _ts("2026-04-08T18:00:00"),
            "version": 4,
            "risk_flags": ["changed_3x", "unconfirmed", "due_advanced"],
            "first_seen_at": _ts("2026-03-05T09:10:00"),
            "last_activity_at": _ts("2026-04-10T18:30:00"),
        },
        {
            "thread_id": "thr_userlist",
            "title": "用户明细导出",
            "status": "open",
            "owner_actor_id": "act_self",
            "requester_actor_id": "act_li",
            "current_deliverable": "用户明细导出",
            "current_due": _ts("2026-03-28T18:00:00"),
            "version": 1,
            "risk_flags": ["overdue"],
            "first_seen_at": _ts("2026-03-06T10:00:00"),
            "last_activity_at": _ts("2026-03-26T10:22:00"),
        },
        {
            "thread_id": "thr_apidoc",
            "title": "接口文档补充",
            "status": "open",
            "owner_actor_id": "act_self",
            "requester_actor_id": "act_wang",
            "current_deliverable": "接口文档",
            "current_due": _ts("2026-03-13T18:00:00"),
            "version": 1,
            "risk_flags": ["due_missing"],
            "first_seen_at": _ts("2026-03-12T15:30:00"),
            "last_activity_at": _ts("2026-03-12T15:30:00"),
        },
    ]


def _story_records() -> list[dict]:
    return [
        {
            "evidence_id": "ev_demo_01",
            "kind": "request",
            "occurred_at": _ts("2026-03-05T09:10:00"),
            "captured_at": _ts("2026-03-05T09:11:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "张总:小陈，这周内把渠道复盘数据出一下，我周会要用，先把各渠道表现拉齐。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "渠道复盘数据",
                "slot_due": _ts("2026-03-06T18:00:00"),
                "slot_due_raw": "这周内",
                "slot_direction": "i_owe",
                "plain_summary": "张总要求你在本周五前提供渠道复盘数据。",
                "caveats": ["未指定具体日期,默认按周五下班前"],
            },
        },
        {
            "evidence_id": "ev_demo_02",
            "kind": "confirm",
            "occurred_at": _ts("2026-03-05T09:18:00"),
            "captured_at": _ts("2026-03-05T09:19:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "我:收到，我按周五下班前给你一版渠道复盘，先按周维度整理。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "渠道复盘数据",
                "slot_due": _ts("2026-03-06T18:00:00"),
                "slot_due_raw": "周五下班前",
                "slot_direction": "i_owe",
                "plain_summary": "你确认会在周五下班前交付渠道复盘数据。",
            },
        },
        {
            "evidence_id": "ev_demo_03",
            "kind": "reference",
            "occurred_at": _ts("2026-03-05T12:30:00"),
            "captured_at": _ts("2026-03-05T12:31:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": None,
            "payload": "李娜:昨天楼下咖啡店又涨价了，行政是不是又换供应商了？",
            "slots": None,
        },
        {
            "evidence_id": "ev_demo_04",
            "kind": "request",
            "occurred_at": _ts("2026-03-06T10:00:00"),
            "captured_at": _ts("2026-03-06T10:01:00"),
            "source_hint": "飞书-私聊-李娜",
            "thread_id": "thr_userlist",
            "payload": "李娜:你这边顺手帮我导一份用户明细吧，我下午要给销售核对名单。",
            "slots": {
                "slot_requester": "act_li",
                "slot_owner": "act_self",
                "slot_deliverable": "用户明细导出",
                "slot_due": _ts("2026-03-06T17:30:00"),
                "slot_due_raw": "下午",
                "slot_direction": "i_owe",
                "plain_summary": "李娜要求你今天下午前提供一份用户明细导出。",
                "caveats": ["仅说明下午要用,默认按当日17:30处理"],
            },
        },
        {
            "evidence_id": "ev_demo_05",
            "kind": "deliver",
            "occurred_at": _ts("2026-03-06T17:40:00"),
            "captured_at": _ts("2026-03-06T17:41:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "我:渠道复盘 v1 我先发你，先按周维度拆了渠道成本、转化和留存，附件表我也放共享盘了。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "渠道复盘数据 v1",
                "slot_due": _ts("2026-03-06T18:00:00"),
                "slot_due_raw": "周五下班前",
                "slot_direction": "i_owe",
                "plain_summary": "你交付了按周维度整理的渠道复盘 v1。",
            },
        },
        {
            "evidence_id": "ev_demo_06",
            "kind": "change",
            "occurred_at": _ts("2026-03-09T09:20:00"),
            "captured_at": _ts("2026-03-09T09:21:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "张总:我看了下，别按周了，改成按季度拆分，最好把一季度每个月也顺带补出来。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "按季度拆分的渠道复盘数据",
                "slot_due": _ts("2026-03-11T18:00:00"),
                "slot_due_raw": "尽快",
                "slot_direction": "i_owe",
                "plain_summary": "张总把渠道复盘需求改为按季度拆分，并希望尽快补出结果。",
                "caveats": ["未明确是否覆盖原周维度版本"],
            },
        },
        {
            "evidence_id": "ev_demo_07",
            "kind": "reference",
            "occurred_at": _ts("2026-03-10T11:00:00"),
            "captured_at": _ts("2026-03-10T11:01:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": None,
            "payload": "行政:本周五下午公司统一放假半天，晚上的团建请大家自行安排返程。",
            "slots": None,
        },
        {
            "evidence_id": "ev_demo_08",
            "kind": "confirm",
            "occurred_at": _ts("2026-03-10T11:20:00"),
            "captured_at": _ts("2026-03-10T11:21:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "我:那渠道复盘这块我是不是顺延到周三前给？如果还是要今天，我得先砍掉别的活。张总在吗？",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "按季度拆分的渠道复盘数据",
                "slot_direction": "i_owe",
                "plain_summary": "你询问渠道复盘是否顺延到周三前，但没有收到明确确认。",
                "caveats": ["对方未回复,交付时限处于未确认状态"],
            },
        },
        {
            "evidence_id": "ev_demo_09",
            "kind": "request",
            "occurred_at": _ts("2026-03-12T15:30:00"),
            "captured_at": _ts("2026-03-12T15:31:00"),
            "source_hint": "飞书-私聊-王强",
            "thread_id": "thr_apidoc",
            "payload": "王强:接口联调用的那份文档你也给我一份吧，最好今天下班前，我这边要给前端同步字段说明。",
            "slots": {
                "slot_requester": "act_wang",
                "slot_owner": "act_self",
                "slot_deliverable": "接口文档",
                "slot_due": _ts("2026-03-13T18:00:00"),
                "slot_due_raw": "今天下班前",
                "slot_direction": "i_owe",
                "plain_summary": "王强要求你在今天下班前提供接口文档。",
            },
        },
        {
            "evidence_id": "ev_demo_10",
            "kind": "change",
            "occurred_at": _ts("2026-03-16T09:50:00"),
            "captured_at": _ts("2026-03-16T09:51:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "张总:再加一个竞品对比页，至少把三家同行的投放结构放进去，周三中午前给我。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "含竞品对比的渠道复盘数据",
                "slot_due": _ts("2026-03-18T12:00:00"),
                "slot_due_raw": "周三中午前",
                "slot_direction": "i_owe",
                "plain_summary": "张总再次变更需求，要求渠道复盘增加竞品对比并提前到周三中午前。",
                "caveats": ["竞品范围未明确,默认按三家同行处理"],
            },
        },
        {
            "evidence_id": "ev_demo_11",
            "kind": "dispute",
            "occurred_at": _ts("2026-03-18T17:30:00"),
            "captured_at": _ts("2026-03-18T17:31:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "张总:这不是我要的，我要的是能直接拿去会上讲的版本，不是你现在这个分析底稿。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "可直接用于周会的渠道复盘材料",
                "slot_direction": "i_owe",
                "plain_summary": "张总否认当前交付符合预期，要求改成可直接上会的渠道复盘材料。",
            },
        },
        {
            "evidence_id": "ev_demo_12",
            "kind": "reference",
            "occurred_at": _ts("2026-03-19T12:00:00"),
            "captured_at": _ts("2026-03-19T12:01:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": None,
            "payload": "王强:中午点什么外卖？昨天那家黄焖鸡还行，就是米饭太硬了。",
            "slots": None,
        },
        {
            "evidence_id": "ev_demo_13",
            "kind": "deliver",
            "occurred_at": _ts("2026-03-24T20:10:00"),
            "captured_at": _ts("2026-03-24T20:11:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "我:渠道复盘 v2 我重做了，里面补了季度拆分和竞品页，今晚先发群里，你明早看下还有没有要改的。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "渠道复盘数据 v2",
                "slot_due": _ts("2026-03-25T09:30:00"),
                "slot_due_raw": "明早看",
                "slot_direction": "i_owe",
                "plain_summary": "你交付了补齐季度拆分和竞品对比的渠道复盘 v2。",
            },
        },
        {
            "evidence_id": "ev_demo_14",
            "kind": "request",
            "occurred_at": _ts("2026-03-26T10:15:00"),
            "captured_at": _ts("2026-03-26T10:16:00"),
            "source_hint": "飞书-私聊-李娜",
            "thread_id": "thr_userlist",
            "payload": "李娜:上次那份用户明细你还没给我呢，今天能不能先把近三个月新增用户导出来？",
            "slots": {
                "slot_requester": "act_li",
                "slot_owner": "act_self",
                "slot_deliverable": "近三个月新增用户明细",
                "slot_due": _ts("2026-03-28T18:00:00"),
                "slot_due_raw": "今天能不能先给",
                "slot_direction": "i_owe",
                "plain_summary": "李娜再次催要近三个月新增用户明细。",
                "caveats": ["原始消息问句化表达,默认按本周内处理"],
            },
        },
        {
            "evidence_id": "ev_demo_15",
            "kind": "confirm",
            "occurred_at": _ts("2026-03-26T10:22:00"),
            "captured_at": _ts("2026-03-26T10:23:00"),
            "source_hint": "飞书-私聊-李娜",
            "thread_id": "thr_userlist",
            "payload": "我:收到，这周我给你补上，先把渠道复盘这边收个尾，最晚周五前给你。",
            "slots": {
                "slot_requester": "act_li",
                "slot_owner": "act_self",
                "slot_deliverable": "近三个月新增用户明细",
                "slot_due": _ts("2026-03-27T18:00:00"),
                "slot_due_raw": "这周,最晚周五前",
                "slot_direction": "i_owe",
                "plain_summary": "你确认会在本周五前处理并交付用户明细。",
            },
        },
        {
            "evidence_id": "ev_demo_16",
            "kind": "change",
            "occurred_at": _ts("2026-04-02T14:05:00"),
            "captured_at": _ts("2026-04-02T14:06:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "张总:先别做全渠道了，范围收窄到信息流和搜索两块，重点把异常波动原因讲清楚，下周三给我。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "聚焦信息流和搜索的渠道复盘数据",
                "slot_due": _ts("2026-04-08T18:00:00"),
                "slot_due_raw": "下周三",
                "slot_direction": "i_owe",
                "plain_summary": "张总第三次变更需求，把渠道复盘范围收窄到信息流和搜索。",
            },
        },
        {
            "evidence_id": "ev_demo_17",
            "kind": "reference",
            "occurred_at": _ts("2026-04-03T09:00:00"),
            "captured_at": _ts("2026-04-03T09:01:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": None,
            "payload": "HR:下周一开始工位调整，大家中午前把桌面收一下，电脑外设别落在原位。",
            "slots": None,
        },
        {
            "evidence_id": "ev_demo_18",
            "kind": "deliver",
            "occurred_at": _ts("2026-04-10T18:30:00"),
            "captured_at": _ts("2026-04-10T18:31:00"),
            "source_hint": "飞书-项目复盘群",
            "thread_id": "thr_channel",
            "payload": "我:最终版我已经发你邮箱和群文件了，这次只保留信息流和搜索两块，也补了竞品对比和异常原因页。",
            "slots": {
                "slot_requester": "act_zhang",
                "slot_owner": "act_self",
                "slot_deliverable": "渠道复盘数据最终版",
                "slot_due": _ts("2026-04-10T18:30:00"),
                "slot_due_raw": "最终版",
                "slot_direction": "i_owe",
                "plain_summary": "你交付了收窄范围后的渠道复盘最终版。",
            },
        },
    ]


def _insert_actors(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                actor["actor_id"],
                actor["canonical_name"],
                _json_text(actor["aliases"]),
                actor["org"],
                actor["role_hint"],
                actor["is_self"],
                actor["confidence"],
                actor["created_at"],
            )
            for actor in _actors()
        ],
    )
    conn.commit()


def _insert_threads(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT INTO threads (
            thread_id, title, status, owner_actor_id, requester_actor_id,
            current_deliverable, current_due, version, risk_flags,
            first_seen_at, last_activity_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                thread["thread_id"],
                thread["title"],
                thread["status"],
                thread["owner_actor_id"],
                thread["requester_actor_id"],
                thread["current_deliverable"],
                thread["current_due"],
                thread["version"],
                _json_text(thread["risk_flags"]),
                thread["first_seen_at"],
                thread["last_activity_at"],
            )
            for thread in _threads()
        ],
    )
    conn.commit()


def seed_demo_data(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "workchain.db"
    blobs_root = out_dir / "blobs"
    blobs_root.mkdir(parents=True, exist_ok=True)

    conn = init_db(db_path)
    try:
        _insert_actors(conn)
        _insert_threads(conn)

        for record in _story_records():
            appended = append_evidence(
                conn,
                blobs_root=blobs_root,
                media_type="text",
                payload=record["payload"],
                captured_at=record["captured_at"],
                occurred_at=record["occurred_at"],
                source_hint=record["source_hint"],
                kind=record["kind"],
                evidence_id=record["evidence_id"],
            )
            if record["slots"] is not None:
                update_slots(conn, appended["evidence_id"], **record["slots"])

            if record["thread_id"] is not None:
                conn.execute(
                    "UPDATE evidence SET thread_id = ? WHERE evidence_id = ?",
                    (record["thread_id"], appended["evidence_id"]),
                )
                conn.commit()

        verify_result = verify_chain(conn, blobs_root=blobs_root)
        if verify_result != (True, None, None):
            return {
                "db_path": db_path,
                "blobs_root": blobs_root,
                "verify_result": verify_result,
            }

        evidence_total = conn.execute("SELECT COUNT(*) AS count FROM evidence").fetchone()["count"]
        thread_counts = {
            row["thread_id"]: row["count"]
            for row in conn.execute(
                "SELECT thread_id, COUNT(*) AS count FROM evidence GROUP BY thread_id ORDER BY thread_id"
            ).fetchall()
        }
        slots_summary = conn.execute(
            """
            SELECT
                SUM(CASE WHEN slots_filled = 0 THEN 1 ELSE 0 END) AS zero_slots,
                SUM(CASE WHEN slots_filled >= 3 THEN 1 ELSE 0 END) AS rich_slots
            FROM evidence
            """
        ).fetchone()
        return {
            "db_path": db_path,
            "blobs_root": blobs_root,
            "verify_result": verify_result,
            "evidence_total": evidence_total,
            "thread_counts": thread_counts,
            "zero_slots": slots_summary["zero_slots"],
            "rich_slots": slots_summary["rich_slots"],
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result = seed_demo_data(Path(args.out))
    print(f"verify_chain: {result['verify_result']}")
    print(f"database: {result['db_path']}")
    print(f"blobs: {result['blobs_root']}")

    if result["verify_result"] != (True, None, None):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
