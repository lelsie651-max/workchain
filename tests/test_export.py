from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from evidence_core.chain import compute_record_digest
from evidence_core.db import init_db
from evidence_core.export import export_evidence_package
from evidence_core.store import append_evidence, update_slots


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "workchain.db"


@pytest.fixture
def blobs_root(tmp_path):
    path = tmp_path / "blobs_src"
    path.mkdir()
    return path


@pytest.fixture
def export_dir(tmp_path):
    return tmp_path / "export"


def _append_text(conn, blobs_root: Path, *, text: str, captured_at: int, **kwargs):
    return append_evidence(
        conn,
        blobs_root=blobs_root,
        media_type="text",
        payload=text,
        captured_at=captured_at,
        **kwargs,
    )


def _load_manifest(export_dir: Path) -> dict:
    return json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_verify_module() -> object:
    verify_path = Path(__file__).resolve().parents[1] / "verify.py"
    spec = importlib.util.spec_from_file_location("standalone_verify", verify_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _insert_actor(conn, actor_id: str) -> None:
    conn.execute(
        """
        INSERT INTO actors (
            actor_id, canonical_name, aliases, org, role_hint,
            is_self, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (actor_id, actor_id, "[]", None, None, 0, None, 1723000000),
    )
    conn.commit()


def test_export_creates_manifest_blobs_and_verify_py(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="hello", captured_at=1723000000)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)

    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "blobs").exists()
    assert (export_dir / "verify.py").exists()


def test_manifest_record_keys_match_exactly_required_set(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="hello", captured_at=1723000000)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest = _load_manifest(export_dir)

    assert set(manifest["records"][0].keys()) == {
        "evidence_id",
        "seq",
        "content_hash",
        "occurred_at",
        "captured_at",
        "media_type",
        "source_hint",
        "record_digest",
        "prev_hash",
        "chain_hash",
    }


def test_manifest_does_not_include_ai_interpretation_fields(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _insert_actor(conn, "act-1")
    _insert_actor(conn, "act-2")
    row = _append_text(conn, blobs_root, text="hello", captured_at=1723000000, source_hint="中文源")
    update_slots(
        conn,
        row["evidence_id"],
        slot_requester="act-1",
        slot_owner="act-2",
        slot_deliverable="slides",
        slot_due=1724000000,
        slot_due_raw="下周五",
        slot_direction="i_owe",
        plain_summary="整理并提交",
        caveats=["备注"],
    )

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest_text = (export_dir / "manifest.json").read_text(encoding="utf-8")

    assert "slot_" not in manifest_text
    assert "plain_summary" not in manifest_text
    assert "caveats" not in manifest_text
    assert "raw_text" not in manifest_text


def test_verify_module_record_digest_matches_project_implementation_for_20_records(
    db_file, blobs_root, export_dir
):
    conn = init_db(db_file)
    random_gen = random.Random(0)
    source_hints = [
        None,
        "飞书群-项目A",
        "mailbox+tag@example.com",
        "emoji-ish <> [] {} !@#$%^&*()",
        "e\u0301-source",
        "多行\nsource",
    ]
    payloads = ["中文", "plain", "symbols !@#$", "e\u0301", "line1\nline2"]

    for index in range(20):
        _append_text(
            conn,
            blobs_root,
            text=random_gen.choice(payloads) + f"-{index}",
            captured_at=1723000000 + index,
            occurred_at=None if index % 3 == 0 else 1722900000 + index,
            source_hint=random_gen.choice(source_hints),
        )

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest = _load_manifest(export_dir)
    verify_module = _load_verify_module()
    db_rows = conn.execute("SELECT * FROM evidence ORDER BY seq ASC").fetchall()

    for manifest_record, db_row in zip(manifest["records"][:20], db_rows[:20], strict=True):
        assert verify_module.compute_record_digest(manifest_record) == compute_record_digest(dict(db_row))


def test_clean_export_passes_standalone_verify(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    for index in range(5):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_tampered_export_blob_fails_with_correct_seq(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="one", captured_at=1723000000)
    target = _append_text(conn, blobs_root, text="two", captured_at=1723000001)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    blob_path = export_dir / "blobs" / Path(target["content_hash"][:2]) / f"{target['content_hash']}.bin"
    blob_path.write_bytes(b"tampered")

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "seq=2" in result.stdout


def test_tampered_manifest_record_field_fails_verification(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="one", captured_at=1723000000, occurred_at=1722990000)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest = _load_manifest(export_dir)
    manifest["records"][0]["occurred_at"] = 123
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1


def test_missing_middle_record_in_manifest_reports_seq_gap(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    for index in range(5):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest = _load_manifest(export_dir)
    del manifest["records"][2]
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "seq gap" in result.stdout


def test_missing_last_record_with_checkpoint_fails(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    for index in range(100):
        _append_text(conn, blobs_root, text=f"payload-{index}", captured_at=1723000000 + index)

    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    manifest = _load_manifest(export_dir)
    manifest["records"].pop()
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1


def test_invalid_manifest_json_exits_with_code_2(db_file, blobs_root, export_dir):
    conn = init_db(db_file)
    _append_text(conn, blobs_root, text="one", captured_at=1723000000)
    export_evidence_package(conn, blobs_root=blobs_root, out_dir=export_dir)
    (export_dir / "manifest.json").write_text("{invalid", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(export_dir / "verify.py"), "--dir", str(export_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2


def test_verify_source_does_not_reference_project_package():
    source = (Path(__file__).resolve().parents[1] / "verify.py").read_text(encoding="utf-8")

    assert "evidence_core" not in source
    assert "from evidence_core" not in source
