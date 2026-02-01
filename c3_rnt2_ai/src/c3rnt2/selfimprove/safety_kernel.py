from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ..self_patch.policy import (
    DEFAULT_ALLOWED_COMMANDS,
    DEFAULT_ALLOWED_PATHS,
    DEFAULT_FORBIDDEN_GLOBS,
    DEFAULT_MAX_PATCH_KB,
    SelfPatchPolicy,
    normalize_path,
    is_forbidden as _is_forbidden,
)


@dataclass(frozen=True)
class SafetyPolicy:
    max_patch_kb: int = DEFAULT_MAX_PATCH_KB
    allow_commands: Tuple[str, ...] = DEFAULT_ALLOWED_COMMANDS
    allowed_paths: Tuple[str, ...] = DEFAULT_ALLOWED_PATHS
    forbidden_globs: Tuple[str, ...] = DEFAULT_FORBIDDEN_GLOBS

    def to_self_patch(self) -> SelfPatchPolicy:
        return SelfPatchPolicy(
            allowed_paths=self.allowed_paths,
            forbidden_globs=self.forbidden_globs,
            max_patch_kb=self.max_patch_kb,
            allowed_commands=self.allow_commands,
        )


def is_forbidden(repo_root: Path, path: Path) -> bool:
    rel = normalize_path(repo_root, path)
    return _is_forbidden(rel, DEFAULT_FORBIDDEN_GLOBS)
