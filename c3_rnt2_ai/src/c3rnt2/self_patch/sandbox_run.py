from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

from .apply_patch import DEFAULT_ALLOWED, DEFAULT_FORBIDDEN, _is_allowed
from .utils import diff_paths


def _run_cmd(cmd: List[str], cwd: Path, env: dict) -> dict:
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
        return {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except Exception as exc:
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": str(exc)}


def _build_env() -> dict:
    env = os.environ.copy()
    # best-effort network disable
    env["C3RNT2_NO_NET"] = "1"
    env["NO_PROXY"] = "*"
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["http_proxy"] = ""
    env["https_proxy"] = ""
    return env


def _check_paths(diff_text: str, allowed: Iterable[str], forbidden: Iterable[str]) -> tuple[bool, str | None]:
    paths = diff_paths(diff_text)
    if not paths:
        return False, "no paths in diff"
    for rel in paths:
        if not _is_allowed(rel, allowed, forbidden):
            return False, f"forbidden path: {rel}"
    return True, None


def sandbox_run(base_dir: Path, patch_id: str, bench: bool = True, settings: dict | None = None) -> dict:
    queue_dir = base_dir / "data" / "self_patch" / "queue" / patch_id
    patch_path = queue_dir / "patch.diff"
    if not patch_path.exists():
        return {"ok": False, "error": "patch.diff not found", "patch_id": patch_id}

    diff_text = patch_path.read_text(encoding="utf-8")
    cfg = settings.get("self_patch", {}) if settings else {}
    allowed = cfg.get("allowed_paths", DEFAULT_ALLOWED)
    forbidden = cfg.get("forbidden_paths", DEFAULT_FORBIDDEN)
    ok_paths, path_error = _check_paths(diff_text, allowed, forbidden)
    if not ok_paths:
        result = {"ok": False, "stage": "validate_paths", "error": path_error, "patch_id": patch_id}
        (queue_dir / "sandbox.json").write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
        meta_path = queue_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["status"] = "blocked"
                meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
            except Exception:
                pass
        return result

    sandbox_root = base_dir / "data" / "self_patch" / "sandbox" / patch_id
    if sandbox_root.exists():
        shutil.rmtree(sandbox_root)
    shutil.copytree(base_dir, sandbox_root, ignore=shutil.ignore_patterns(".git", "data", "__pycache__"))
    env = _build_env()

    apply_result = _run_cmd(["git", "apply", str(patch_path)], cwd=sandbox_root, env=env)
    if apply_result["returncode"] != 0:
        result = {"ok": False, "stage": "apply", "apply": apply_result}
        (queue_dir / "sandbox.json").write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
        return result

    checks = []
    if shutil.which("ruff"):
        checks.append(_run_cmd(["ruff", "check", "src"], cwd=sandbox_root, env=env))
    checks.append(_run_cmd(["pytest", "-q"], cwd=sandbox_root, env=env))
    if bench:
        bench_script = sandbox_root / "scripts" / "bench_generate.py"
        if bench_script.exists():
            checks.append(_run_cmd(["python", str(bench_script), "--profile", "dev_small", "--max-new-tokens", "16"], cwd=sandbox_root, env=env))

    ok = all(item.get("returncode", 1) == 0 for item in checks)
    result = {
        "ok": ok,
        "patch_id": patch_id,
        "checks": checks,
        "ts": time.time(),
    }
    (queue_dir / "sandbox.json").write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    meta_path = queue_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "ready_for_review" if ok else "sandbox_failed"
            meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
        except Exception:
            pass
    return result


def run_sandbox(base_dir: Path, patch_id: str, bench: bool = True, settings: dict | None = None) -> dict:
    return sandbox_run(base_dir, patch_id, bench=bench, settings=settings)
