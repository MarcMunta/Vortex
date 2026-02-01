from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import difflib


@dataclass
class PatchMeta:
    patch_id: str
    goal: str
    context: str | None
    created_ts: float
    status: str


def _normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _paths_from_diff(diff_text: str) -> List[Path]:
    paths: List[Path] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:].strip()
            paths.append(Path(rel))
    return paths


def generate_diff(repo_root: Path, changes: Dict[Path, str]) -> str:
    chunks: List[str] = []
    for path, new_text in changes.items():
        abs_path = (repo_root / path).resolve()
        old_text = ""
        if abs_path.exists():
            old_text = abs_path.read_text(encoding="utf-8", errors="ignore")
        rel = _normalize_path(abs_path.relative_to(repo_root))
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
        chunks.append(diff_text)
    return "\n".join(chunks)


def diff_paths(diff_text: str) -> List[Path]:
    return _paths_from_diff(diff_text)
