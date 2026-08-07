from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from evidence_core.db import init_db
from evidence_core.store import verify_chain


def _make_client(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))
    client = TestClient(create_app())
    return client, demo_dir, sandbox_root


def _sandbox_db_path(client: TestClient, sandbox_root: Path) -> Path:
    sandbox_id = client.cookies.get("wc_sid")
    assert sandbox_id is not None
    return sandbox_root / sandbox_id / "workchain.db"


def test_healthz_returns_ok_and_evidence_count_18(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "evidence_count": 18}


def test_index_contains_three_thread_titles(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "这不是我要的" in html
    assert '<h1 class="text-4xl' not in html
    assert "你手上的事" in html
    assert "事项线" not in html
    assert "自动留证" in html
    assert "渠道复盘数据" in html
    assert "用户明细导出" in html
    assert "接口文档补充" in html
    assert "需求改了 3 次,你一次都没等到确认" in html
    assert "飞书" in html
    assert "企业微信" in html
    assert "存证" in html


def test_index_contains_reference_section_and_reference_texts(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    html = response.text
    assert "参考信息" in html
    assert "已识别为参考信息,未计入待办" in html
    assert "昨天楼下咖啡店又涨价了" in html
    assert "公司统一放假半天" in html
    assert "中午点什么外卖" in html
    assert "下周一开始工位调整" in html


def test_thread_channel_page_contains_10_evidence_cards(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_channel")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 10
    assert "⚠️ 需求在这里发生了变更" in response.text
    assert "10 条记录" in response.text
    assert 'href="/help"' in response.text


def test_thread_userlist_page_contains_3_evidence_cards(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_userlist")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 3


def test_nonexistent_thread_returns_404(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_nonexistent")

    assert response.status_code == 404


def test_index_does_not_expose_hash_fields(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    html = response.text
    assert "chain_hash" not in html
    assert "content_hash" not in html
    assert "prev_hash" not in html
    assert 'href="/help"' in html


def test_help_page_returns_200_and_contains_review_notes(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/help")

    assert response.status_code == 200
    html = response.text
    assert "给评审的说明" in html
    assert "即将开放" in html
    assert "https://github.com/lelsie651-max/workchain" in html


def test_startup_auto_generates_demo_data_when_directory_missing(tmp_path, monkeypatch):
    demo_dir = tmp_path / "missing_demo_data"
    sandbox_root = tmp_path / "missing_sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    assert not demo_dir.exists()

    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert demo_dir.exists()
    assert (demo_dir / "workchain.db").exists()

    conn = sqlite3.connect(demo_dir / "workchain.db")
    try:
        count = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    finally:
        conn.close()

    assert count == 18


def test_concurrent_requests_all_return_200(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)
    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(5):
                index_response = client.get("/")
                thread_response = client.get("/thread/thr_channel")
                assert index_response.status_code == 200
                assert thread_response.status_code == 200
        except Exception as exc:  # pragma: no cover - only used on failure
            with lock:
                errors.append(str(exc))

    with client:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []


def test_healthz_is_stable_across_20_requests(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        for _ in range(20):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "evidence_count": 18}


def test_post_evidence_appends_reference_record(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.post(
            "/api/evidence",
            json={"text": "张总刚在群里又说先把原始明细留一下。", "source": "飞书", "source_detail": "项目复盘群"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["seq"] == 19
    assert data["evidence_id"].startswith("ev_")
    assert data["occurred_at"] is not None
    assert _sandbox_db_path(client, sandbox_root).exists()


def test_index_shows_recent_user_records_after_post(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        client.post(
            "/api/evidence",
            json={"text": "李娜刚补了一句，说先不用发销售。", "source": "企业微信", "source_detail": "私聊-李娜"},
        )
        response = client.get("/")

    assert response.status_code == 200
    assert "我刚存的" in response.text
    assert "这些记录只属于你" in response.text
    assert "李娜刚补了一句" in response.text


def test_post_evidence_rejects_empty_text_and_too_long_text(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        empty_response = client.post(
            "/api/evidence",
            json={"text": "   ", "source": "飞书", "source_detail": "项目复盘群"},
        )
        long_response = client.post(
            "/api/evidence",
            json={"text": "a" * 20001, "source": "飞书", "source_detail": "项目复盘群"},
        )

    assert empty_response.status_code == 400
    assert long_response.status_code == 400


def test_two_clients_do_not_see_each_other_records(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    monkeypatch.setenv("WORKCHAIN_SANDBOX_ROOT", str(sandbox_root))

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    with client_a, client_b:
        client_a.post(
            "/api/evidence",
            json={"text": "只属于 A 的内容。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        client_b.post(
            "/api/evidence",
            json={"text": "只属于 B 的内容。", "source": "Jira", "source_detail": "WORK-999"},
        )

        home_a = client_a.get("/")
        home_b = client_b.get("/")

    assert "只属于 A 的内容" in home_a.text
    assert "只属于 B 的内容" not in home_a.text
    assert "只属于 B 的内容" in home_b.text
    assert "只属于 A 的内容" not in home_b.text


def test_verify_chain_still_passes_after_posting_to_sandbox(tmp_path, monkeypatch):
    client, _, sandbox_root = _make_client(tmp_path, monkeypatch)

    with client:
        client.post(
            "/api/evidence",
            json={"text": "这是新存的一条参考记录。", "source": "Slack", "source_detail": "growth-sync"},
        )

    db_path = _sandbox_db_path(client, sandbox_root)
    conn = init_db(db_path)
    try:
        assert verify_chain(conn, blobs_root=db_path.parent / "blobs") == (True, None, None)
    finally:
        conn.close()


def test_demo_database_mtime_does_not_change_after_post(tmp_path, monkeypatch):
    client, demo_dir, _ = _make_client(tmp_path, monkeypatch)

    with client:
        client.get("/")
        demo_db_path = demo_dir / "workchain.db"
        before = demo_db_path.stat().st_mtime_ns
        time.sleep(0.01)
        response = client.post(
            "/api/evidence",
            json={"text": "确认一下，演示库不能被写。", "source": "飞书", "source_detail": "项目复盘群"},
        )
        after = demo_db_path.stat().st_mtime_ns

    assert response.status_code == 200
    assert before == after


def test_same_sandbox_hits_daily_rate_limit_on_51st_post(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path, monkeypatch)

    with client:
        last_response = None
        for idx in range(51):
            last_response = client.post(
                "/api/evidence",
                json={"text": f"第 {idx + 1} 条临时记录", "source": "飞书", "source_detail": "项目复盘群"},
            )

    assert last_response is not None
    assert last_response.status_code == 429
