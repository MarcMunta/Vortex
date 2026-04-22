from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable


DEFAULT_IGNORE = (
    ".git",
    ".venv",
    "data",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
)


def _sanitize_env() -> dict:
    safe = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            continue
        safe[key] = value
    safe["C3RNT2_NO_NET"] = "1"
    return safe


def _copy_repo(repo_root: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        shutil.copytree(
            repo_root,
            dst,
            ignore=shutil.ignore_patterns(*DEFAULT_IGNORE),
        )
    )


def run_sandbox_command(
    repo_root: Path,
    cmd: Iterable[str],
    sandbox_root: Path,
    timeout_s: int = 120,
) -> dict:
    ts = time.strftime("%Y%m%d_%H%M%S")
    nonce = str(time.time_ns())[-6:]
    sandbox_root.mkdir(parents=True, exist_ok=True)
    sandbox_dir = sandbox_root / f"run_{ts}_{nonce}"
    repo_copy = _copy_repo(repo_root, sandbox_dir / "repo")
    try:
        result = subprocess.run(
            list(cmd),
            cwd=str(repo_copy),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_sanitize_env(),
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "sandbox": str(repo_copy),
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "sandbox": str(repo_copy),
        }
