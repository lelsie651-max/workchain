from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def _make_client(tmp_path, monkeypatch):
    demo_dir = tmp_path / "web_demo_data"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))
    client = TestClient(create_app())
    return client, demo_dir


def test_healthz_returns_ok_and_evidence_count_18(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "evidence_count": 18}


def test_index_contains_three_thread_titles(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "这不是我要的" in html
    assert "你手上的事" in html
    assert "事项线" not in html
    assert "渠道复盘数据" in html
    assert "用户明细导出" in html
    assert "接口文档补充" in html
    assert "需求改了 3 次,你一次都没等到确认" in html
    assert "即将开放" in html
    assert "飞书" in html
    assert "企业微信" in html


def test_index_contains_reference_section_and_reference_texts(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

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
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_channel")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 10
    assert "⚠️ 需求在这里发生了变更" in response.text
    assert "10 条记录" in response.text


def test_thread_userlist_page_contains_3_evidence_cards(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_userlist")

    assert response.status_code == 200
    assert response.text.count('data-testid="timeline-card"') == 3


def test_nonexistent_thread_returns_404(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/thread/thr_nonexistent")

    assert response.status_code == 404


def test_index_does_not_expose_hash_fields(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        response = client.get("/")

    html = response.text
    assert "chain_hash" not in html
    assert "content_hash" not in html
    assert "prev_hash" not in html


def test_startup_auto_generates_demo_data_when_directory_missing(tmp_path, monkeypatch):
    demo_dir = tmp_path / "missing_demo_data"
    monkeypatch.setenv("WORKCHAIN_DEMO_DIR", str(demo_dir))

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
    client, _ = _make_client(tmp_path, monkeypatch)
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
    client, _ = _make_client(tmp_path, monkeypatch)

    with client:
        for _ in range(20):
            response = client.get("/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "evidence_count": 18}
