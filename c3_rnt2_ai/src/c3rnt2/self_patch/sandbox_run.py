from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

from .apply_patch import DEFAULT_ALLOWED, DEFAULT_FORBIDDEN_GLOBS, _is_allowed, _resolve_queue_root
from .policy import command_allowed, policy_from_settings
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
    env["C3RNT2_NO_NET"] = "1"
    env["NO_PROXY"] = "*"
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["http_proxy"] = ""
    env["https_proxy"] = ""
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _check_paths(
    diff_text: str,
    allowed: Iterable[str],
    forbidden: Iterable[str],
    *,
    allow_empty: bool = False,
) -> tuple[bool, str | None]:
    paths = diff_paths(diff_text)
    if not paths:
        return (True, None) if allow_empty else (False, "no paths in diff")
    for rel in paths:
        if not _is_allowed(str(rel).replace("\\", "/"), allowed, forbidden):
            return False, f"forbidden path: {rel}"
    return True, None


def sandbox_run(
    base_dir: Path,
    patch_id: str,
    bench: bool = True,
    settings: dict | None = None,
    *,
    allow_empty: bool = False,
) -> dict:
    queue_root = _resolve_queue_root(base_dir, settings)
    queue_dir = queue_root / patch_id
    patch_path = queue_dir / "patch.diff"
    if not patch_path.exists():
        return {"ok": False, "error": "patch.diff not found", "patch_id": patch_id}

    diff_text = patch_path.read_text(encoding="utf-8")
    cfg = settings.get("self_patch", {}) if settings else {}
    allowed = cfg.get("allowed_paths", DEFAULT_ALLOWED)
    forbidden = cfg.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS)
    max_kb = int(cfg.get("max_patch_kb", 128)) if cfg else 128
    if max_kb > 0 and len(diff_text.encode("utf-8")) > max_kb * 1024:
        result = {"ok": False, "stage": "validate_size", "error": "patch exceeds max_patch_kb", "patch_id": patch_id}
        (queue_dir / "sandbox.json").write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
        return result

    ok_paths, path_error = _check_paths(diff_text, allowed, forbidden, allow_empty=allow_empty)
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

    sandbox_root = Path(cfg.get("sandbox_dir", base_dir / "data" / "self_patch" / "sandbox")) / patch_id
    if sandbox_root.exists():
        shutil.rmtree(sandbox_root)
    shutil.copytree(
        base_dir,
        sandbox_root,
        ignore=shutil.ignore_patterns(
            ".git",
            "data",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
        ),
    )
    env = _build_env()

    apply_result = _run_cmd(["git", "apply", str(patch_path)], cwd=sandbox_root, env=env)
    if apply_result["returncode"] != 0:
        result = {"ok": False, "stage": "apply", "apply": apply_result}
        (queue_dir / "sandbox.json").write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
        return result

    policy = policy_from_settings(settings)
    checks = []
    if shutil.which("ruff"):
        cmd = ["ruff", "check", "src"]
        if command_allowed(cmd, policy):
            checks.append(_run_cmd(cmd, cwd=sandbox_root, env=env))
    cmd = ["pytest", "-q"]
    if command_allowed(cmd, policy):
        checks.append(_run_cmd(cmd, cwd=sandbox_root, env=env))
    else:
        checks.append({"cmd": "pytest -q", "returncode": -1, "stdout": "", "stderr": "command_not_allowed"})
    if bench:
        bench_script = sandbox_root / "scripts" / "bench_generate.py"
        if bench_script.exists():
            cmd = ["python", str(bench_script), "--profile", "dev_small", "--max-new-tokens", "16"]
            if command_allowed(cmd, policy):
                checks.append(_run_cmd(cmd, cwd=sandbox_root, env=env))

    ok = True
    for item in checks:
        rc = item.get("returncode", 1)
        cmd = str(item.get("cmd", ""))
        if cmd.startswith("pytest") and rc in (0, 5):
            continue
        if rc != 0:
            ok = False
            break
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
