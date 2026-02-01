from __future__ import annotations

import fnmatch
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import diff_paths


DEFAULT_ALLOWED = [
    "src/",
    "tests/",
]
DEFAULT_FORBIDDEN_GLOBS = [
    ".env",
    ".env.*",
    "data/**",
    "*.key",
    "*.pem",
    "*.p12",
    "*.sqlite",
    "*.db",
    "keys/**",
    "secrets/**",
    "src/c3rnt2/self_patch/**",
    "src/c3rnt2/selfimprove/**",
]


@dataclass
class PatchApplyResult:
    ok: bool
    patch_id: str
    error: str | None = None


def _normalize(path: Path) -> str:
    return str(path).replace("\\", "/")


def _resolve_queue_root(base_dir: Path, settings: dict | None) -> Path:
    cfg = settings.get("self_patch", {}) if settings else {}
    queue_dir = cfg.get("queue_dir")
    if queue_dir:
        queue_path = Path(queue_dir)
        if not queue_path.is_absolute():
            queue_path = base_dir / queue_path
        return queue_path
    return base_dir / "data" / "self_patch" / "queue"


def _is_forbidden(rel: str, forbidden_globs: Iterable[str]) -> bool:
    rel = rel.replace("\\", "/")
    for pattern in forbidden_globs:
        pat = str(pattern).replace("\\", "/")
        if pat.endswith("/") and rel.startswith(pat):
            return True
        if fnmatch.fnmatch(rel, pat):
            return True
    return False


def _is_allowed(rel: str, allowed_paths: Iterable[str], forbidden_globs: Iterable[str]) -> bool:
    rel = rel.replace("\\", "/")
    if _is_forbidden(rel, forbidden_globs):
        return False
    for allow in allowed_paths:
        allow_norm = str(allow).replace("\\", "/")
        if rel.startswith(allow_norm) or rel == allow_norm.rstrip("/"):
            return True
    return False


def apply_patch(patch_id: str, base_dir: Path, settings: dict | None = None) -> PatchApplyResult:
    base_dir = Path(base_dir)
    queue_root = _resolve_queue_root(base_dir, settings)
    queue_dir = queue_root / patch_id
    patch_path = queue_dir / "patch.diff"
    sandbox_path = queue_dir / "sandbox.json"
    meta_path = queue_dir / "meta.json"

    if not patch_path.exists():
        return PatchApplyResult(ok=False, patch_id=patch_id, error="patch.diff not found")
    if not sandbox_path.exists():
        return PatchApplyResult(ok=False, patch_id=patch_id, error="sandbox.json not found")
    sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    if not sandbox.get("ok"):
        return PatchApplyResult(ok=False, patch_id=patch_id, error="sandbox not ok")
    if not meta_path.exists():
        return PatchApplyResult(ok=False, patch_id=patch_id, error="meta.json not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not (meta.get("ready_for_review") or meta.get("status") == "ready_for_review"):
        return PatchApplyResult(ok=False, patch_id=patch_id, error="patch not approved")

    diff_text = patch_path.read_text(encoding="utf-8")
    cfg = settings.get("self_patch", {}) if settings else {}
    allowed = cfg.get("allowed_paths", DEFAULT_ALLOWED)
    forbidden = cfg.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS)
    max_kb = int(cfg.get("max_patch_kb", 128)) if cfg else 128
    run_tests = bool(cfg.get("run_tests_on_apply", True))
    if max_kb > 0 and len(diff_text.encode("utf-8")) > max_kb * 1024:
        return PatchApplyResult(ok=False, patch_id=patch_id, error="patch exceeds max_patch_kb")
    for rel in diff_paths(diff_text):
        if not _is_allowed(_normalize(rel), allowed, forbidden):
            return PatchApplyResult(ok=False, patch_id=patch_id, error=f"forbidden path: {rel}")

    try:
        subprocess.run(["git", "apply", str(patch_path)], cwd=str(base_dir), check=True)
    except Exception as exc:
        return PatchApplyResult(ok=False, patch_id=patch_id, error=f"apply failed: {exc}")

    if run_tests:
        try:
            result = subprocess.run(["pytest", "-q"], cwd=str(base_dir), capture_output=True, text=True)
            if result.returncode not in (0, 5):
                try:
                    subprocess.run(["git", "apply", "-R", str(patch_path)], cwd=str(base_dir), check=False)
                except Exception:
                    pass
                try:
                    meta["status"] = "rolled_back"
                    meta["tests_output"] = (result.stdout + result.stderr)[-2000:]
                    meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
                except Exception:
                    pass
                return PatchApplyResult(ok=False, patch_id=patch_id, error="tests_failed_rolled_back")
        except Exception as exc:
            return PatchApplyResult(ok=False, patch_id=patch_id, error=f"tests_failed: {exc}")

    try:
        meta["status"] = "applied"
        meta["applied_ts"] = time.time()
        meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass
    return PatchApplyResult(ok=True, patch_id=patch_id)
