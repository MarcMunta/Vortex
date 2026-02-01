from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .utils import diff_paths


DEFAULT_ALLOWED = [
    "src/",
    "tests/",
]
DEFAULT_FORBIDDEN = [
    ".env",
    ".env.",
    ".git/",
    "data/",
    "data/db",
    "data/keys",
    "keys",
    "secrets",
]


def _normalize(path: Path) -> str:
    return str(path).replace("\\", "/")


def _is_allowed(path: Path, allowed: Iterable[str], forbidden: Iterable[str]) -> bool:
    rel = _normalize(path)
    for forb in forbidden:
        if rel.startswith(forb):
            return False
    for allow in allowed:
        if rel.startswith(allow) or rel == allow:
            return True
    return False


def apply_patch(
    base_dir: Path,
    patch_id: str,
    settings: dict | None = None,
    *,
    human_approved: bool = False,
) -> dict:
    queue_dir = base_dir / "data" / "self_patch" / "queue" / patch_id
    patch_path = queue_dir / "patch.diff"
    sandbox_path = queue_dir / "sandbox.json"
    if not human_approved:
        return {"ok": False, "error": "human approval required"}
    if not patch_path.exists():
        return {"ok": False, "error": "patch.diff not found"}
    if not sandbox_path.exists():
        return {"ok": False, "error": "sandbox.json not found"}
    sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    if not sandbox.get("ok"):
        return {"ok": False, "error": "sandbox not ok"}

    diff_text = patch_path.read_text(encoding="utf-8")
    paths = diff_paths(diff_text)
    cfg = settings.get("self_patch", {}) if settings else {}
    allowed = cfg.get("allowed_paths", DEFAULT_ALLOWED)
    forbidden = cfg.get("forbidden_paths", DEFAULT_FORBIDDEN)
    for rel in paths:
        if not _is_allowed(rel, allowed, forbidden):
            return {"ok": False, "error": f"forbidden path: {rel}"}

    try:
        subprocess.run(["git", "apply", str(patch_path)], cwd=str(base_dir), check=True)
    except Exception as exc:
        return {"ok": False, "error": f"apply failed: {exc}"}
    meta_path = queue_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "applied"
            meta["applied_ts"] = time.time()
            meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True, "patch_id": patch_id}
