from __future__ import annotations

import subprocess
from pathlib import Path

from c3rnt2.selfimprove.patch_ops import apply_patch, propose_patch, validate_patch
from c3rnt2.selfimprove.safety_kernel import SafetyPolicy


def test_selfimprove_requires_approval(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    target = repo / "src" / "c3rnt2" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")

    diff = propose_patch(repo, {Path("src/c3rnt2/file.txt"): "new\n"})
    result = apply_patch(repo, diff, approve=False)
    assert not result.ok
    assert target.read_text(encoding="utf-8") == "old\n"

    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "APPROVE_SELF_PATCH").write_text("ok", encoding="utf-8")
    result = apply_patch(repo, diff, approve=False)
    assert result.ok
    assert target.read_text(encoding="utf-8") == "new\n"


def test_patch_ops_blocks_forbidden_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    diff = propose_patch(repo, {Path("src/c3rnt2/selfimprove/safety_kernel.py"): "blocked"})
    result = validate_patch(repo, diff, SafetyPolicy())
    assert not result.ok
