from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request, Response


COOKIE_NAME = "wc_sid"
COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class SandboxContext:
    sandbox_id: str
    sandbox_dir: Path
    db_path: Path
    blobs_root: Path
    should_set_cookie: bool


def _sandbox_paths(root: Path, sandbox_id: str) -> SandboxContext:
    sandbox_dir = root / sandbox_id
    return SandboxContext(
        sandbox_id=sandbox_id,
        sandbox_dir=sandbox_dir,
        db_path=sandbox_dir / "workchain.db",
        blobs_root=sandbox_dir / "blobs",
        should_set_cookie=False,
    )


def _is_valid_sandbox_id(value: str | None) -> bool:
    if value is None or len(value) != 32:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _copy_demo_data(demo_dir: Path, sandbox: SandboxContext) -> None:
    sandbox.sandbox_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo_dir / "workchain.db", sandbox.db_path)
    if sandbox.blobs_root.exists():
        shutil.rmtree(sandbox.blobs_root)
    shutil.copytree(demo_dir / "blobs", sandbox.blobs_root)


def get_sandbox(request: Request, response: Response) -> SandboxContext:
    del response  # Cookie is applied on the concrete response object inside routes.

    sandbox_root = request.app.state.sandbox_root
    sandbox_root.mkdir(parents=True, exist_ok=True)

    cookie_value = request.cookies.get(COOKIE_NAME)
    should_set_cookie = not _is_valid_sandbox_id(cookie_value)
    sandbox_id = cookie_value if not should_set_cookie else uuid.uuid4().hex
    sandbox = _sandbox_paths(sandbox_root, sandbox_id)

    if not sandbox.db_path.exists() or not sandbox.blobs_root.exists():
        _copy_demo_data(request.app.state.demo_dir, sandbox)

    return SandboxContext(
        sandbox_id=sandbox.sandbox_id,
        sandbox_dir=sandbox.sandbox_dir,
        db_path=sandbox.db_path,
        blobs_root=sandbox.blobs_root,
        should_set_cookie=should_set_cookie,
    )


def apply_sandbox_cookie(response, sandbox: SandboxContext) -> None:
    if not sandbox.should_set_cookie:
        return
    response.set_cookie(
        COOKIE_NAME,
        sandbox.sandbox_id,
        max_age=COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )


def cleanup_expired(sandbox_root: Path | None = None, max_age_hours: int = 24) -> None:
    sandbox_root = Path(sandbox_root or os.getenv("WORKCHAIN_SANDBOX_ROOT", "sandboxes"))
    if not sandbox_root.exists():
        return

    cutoff = time.time() - (max_age_hours * 60 * 60)
    for child in sandbox_root.iterdir():
        if not child.is_dir():
            continue
        if child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
