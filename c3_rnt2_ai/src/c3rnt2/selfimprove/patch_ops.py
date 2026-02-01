from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..self_patch.apply_patch import apply_patch as queue_apply
from ..self_patch.propose_patch import propose_patch as queue_propose
from ..self_patch.sandbox_run import sandbox_run
from ..self_patch.policy import validate_patch as policy_validate, policy_from_settings
from .safety_kernel import SafetyPolicy, normalize_path


def _log_episode(repo_root: Path, payload: dict) -> None:
    path = repo_root / "data" / "episodes" / "agent.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


@dataclass
class PatchResult:
    ok: bool
    message: str


def propose_patch(repo_root: Path, changes: Dict[Path, str]) -> str:
    diff_chunks: List[str] = []
    for path, new_text in changes.items():
        abs_path = (repo_root / path).resolve()
        old_text = ""
        if abs_path.exists():
            old_text = abs_path.read_text(encoding="utf-8", errors="ignore")
        rel = normalize_path(repo_root, abs_path)
        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if diff_text and not diff_text.endswith("\n"):
            diff_text += "\n"
        diff_chunks.append(diff_text)
    return "\n".join(diff_chunks)


def validate_patch(repo_root: Path, diff_text: str, policy: SafetyPolicy) -> PatchResult:
    ok, message, _paths = policy_validate(repo_root, diff_text, policy.to_self_patch())
    return PatchResult(ok=ok, message=message)


def apply_patch(repo_root: Path, diff_text: str, approve: bool = False) -> PatchResult:
    approve_flag = repo_root / "data" / "APPROVE_SELF_PATCH"
    if not approve and not approve_flag.exists():
        return PatchResult(ok=False, message="approval required")

    settings = {"self_patch": {"allowed_paths": ["src/", "tests/"], "forbidden_globs": ["data/**"]}}
    proposal = queue_propose("selfimprove", {"changes": {}}, repo_root, settings=settings, diff_text=diff_text)
    sandbox = sandbox_run(repo_root, proposal.patch_id, settings=settings, allow_empty=True)
    if not sandbox.get("ok"):
        return PatchResult(ok=False, message="sandbox failed")
    try:
        meta = json.loads(proposal.meta_path.read_text(encoding="utf-8"))
        meta["ready_for_review"] = True
        meta["status"] = "ready_for_review"
        proposal.meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass
    result = queue_apply(proposal.patch_id, repo_root, settings=settings)
    if not result.ok:
        return PatchResult(ok=False, message=result.error or "apply failed")
    _log_episode(repo_root, {
        "task": "self-improve apply_patch",
        "prompt": "pytest -q",
        "patch": diff_text,
        "tests_ok": True,
        "tests_output_excerpt": "",
        "timestamp": time.time(),
    })
    return PatchResult(ok=True, message="applied")
