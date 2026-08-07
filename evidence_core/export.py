from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from evidence_core import chain


MANIFEST_RECORD_FIELDS = (
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
)


def _blob_relative_path(content_hash: str) -> Path:
    return Path(content_hash[:2]) / f"{content_hash}.bin"


def _repo_verify_path() -> Path:
    return Path(__file__).resolve().parents[1] / "verify.py"


def _resolve_export_rows(conn, *, thread_id: str | None, evidence_ids: list[str] | None):
    if thread_id and evidence_ids:
        raise ValueError("thread_id and evidence_ids are mutually exclusive")

    if thread_id:
        return conn.execute(
            "SELECT * FROM evidence WHERE thread_id = ? ORDER BY seq ASC",
            (thread_id,),
        ).fetchall()

    if evidence_ids is not None:
        if not evidence_ids:
            return []
        placeholders = ",".join("?" for _ in evidence_ids)
        rows = conn.execute(
            f"SELECT * FROM evidence WHERE evidence_id IN ({placeholders}) ORDER BY seq ASC",
            evidence_ids,
        ).fetchall()
        order_map = {evidence_id: index for index, evidence_id in enumerate(evidence_ids)}
        return sorted(rows, key=lambda row: (order_map[row["evidence_id"]], row["seq"]))

    return conn.execute("SELECT * FROM evidence ORDER BY seq ASC").fetchall()


def export_evidence_package(
    conn,
    *,
    blobs_root: Path,
    out_dir: Path,
    thread_id: str | None = None,
    evidence_ids: list[str] | None = None,
) -> Path:
    blobs_root = Path(blobs_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blobs").mkdir(parents=True, exist_ok=True)

    rows = _resolve_export_rows(conn, thread_id=thread_id, evidence_ids=evidence_ids)
    if not rows:
        raise ValueError("no evidence selected for export")
    exported_seqs = {row["seq"] for row in rows}
    checkpoints = [
        dict(checkpoint)
        for checkpoint in conn.execute(
            "SELECT at_seq, chain_hash, created_at FROM checkpoints ORDER BY at_seq ASC, created_at ASC"
        ).fetchall()
        if checkpoint["at_seq"] in exported_seqs
    ]

    records = []
    seen_hashes: set[str] = set()
    for row in rows:
        record = dict(row)
        content_hash = record["content_hash"]
        record_export = {
            "evidence_id": record["evidence_id"],
            "seq": record["seq"],
            "content_hash": content_hash,
            "occurred_at": record["occurred_at"],
            "captured_at": record["captured_at"],
            "media_type": record["media_type"],
            "source_hint": record["source_hint"],
            "record_digest": chain.compute_record_digest(record),
            "prev_hash": record["prev_hash"],
            "chain_hash": record["chain_hash"],
        }
        records.append(record_export)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            relative_blob_path = _blob_relative_path(content_hash)
            source_blob_path = blobs_root / relative_blob_path
            target_blob_path = out_dir / "blobs" / relative_blob_path
            target_blob_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_blob_path, target_blob_path)

    manifest = {
        "version": 1,
        "generated_at": int(time.time() * 1000),
        "records": records,
        "checkpoints": checkpoints,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    repo_verify_path = _repo_verify_path()
    if not repo_verify_path.exists():
        raise FileNotFoundError(f"verify.py not found at {repo_verify_path}")
    shutil.copyfile(repo_verify_path, out_dir / "verify.py")

    return out_dir
