from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _normalize_level(raw: Any, default: str = "none") -> str:
    value = str(raw or default).strip().lower()
    if value not in {"none", "read", "edit", "full"}:
        return default
    return value


def _normalize_action_mode(raw: Any, default: str = "safe") -> str:
    value = str(raw or default).strip().lower()
    if value not in {"safe", "full"}:
        return default
    return value


def _looks_windows_absolute(raw_path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(raw_path or "").strip()))


def _looks_posix_absolute(raw_path: str) -> bool:
    return str(raw_path or "").strip().startswith("/")


def _repo_name_fallback(base_dir: Path, raw_path: str) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    name = PureWindowsPath(raw).name if _looks_windows_absolute(raw) else Path(raw).name
    repo_name = str(os.getenv("C3RNT2_HOST_WORKSPACE_REPO_NAME") or "").strip()
    base = base_dir.resolve()
    if name and name in {base.name, repo_name}:
        return base
    return None


def _resolve_workspace_base(base_dir: Path, workspace_root: str) -> Path:
    base_dir = base_dir.resolve()
    if not workspace_root:
        return base_dir
    repo_fallback = _repo_name_fallback(base_dir, workspace_root)
    if repo_fallback is not None:
        return repo_fallback
    shared_mount = str(os.getenv("C3RNT2_HOST_WORKSPACE_MOUNT") or "").strip()
    shared_mount_path = Path(shared_mount).resolve() if shared_mount else None
    host_workspace_root = str(os.getenv("C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT") or "").strip()
    raw_base = Path(workspace_root)
    if raw_base.is_absolute():
        candidate = raw_base.resolve()
        if candidate.exists():
            return candidate
        if shared_mount_path and shared_mount_path.exists():
            if host_workspace_root:
                try:
                    host_pure = PureWindowsPath(host_workspace_root)
                    raw_pure = PureWindowsPath(workspace_root)
                    rel = raw_pure.relative_to(host_pure)
                    nested = (shared_mount_path / Path(*rel.parts)).resolve()
                    if nested.exists():
                        return nested
                except Exception:
                    pass
            nested = (shared_mount_path / raw_base.name).resolve()
            if nested.exists():
                return nested
            return shared_mount_path
        return base_dir
    if _looks_windows_absolute(workspace_root) or _looks_posix_absolute(workspace_root):
        if shared_mount_path and shared_mount_path.exists():
            if host_workspace_root:
                try:
                    host_pure = PureWindowsPath(host_workspace_root)
                    raw_pure = PureWindowsPath(workspace_root)
                    rel = raw_pure.relative_to(host_pure)
                    nested = (shared_mount_path / Path(*rel.parts)).resolve()
                    if nested.exists():
                        return nested
                except Exception:
                    pass
            nested = (shared_mount_path / Path(workspace_root).name).resolve()
            if nested.exists():
                return nested
            return shared_mount_path
        return base_dir
    if shared_mount_path and shared_mount_path.exists():
        nested = (shared_mount_path / raw_base).resolve()
        if nested.exists():
            return nested
    return (base_dir / raw_base).resolve()


def _absolute_project_relative(workspace_root: str, project_path: str) -> str | None:
    if not project_path:
        return None
    if _looks_windows_absolute(project_path):
        project_pure = PureWindowsPath(project_path)
        workspace_pure = (
            PureWindowsPath(workspace_root)
            if _looks_windows_absolute(workspace_root)
            else None
        )
    elif _looks_posix_absolute(project_path):
        project_pure = PurePosixPath(project_path)
        workspace_pure = (
            PurePosixPath(workspace_root)
            if _looks_posix_absolute(workspace_root)
            else None
        )
    else:
        return project_path
    if workspace_pure is not None:
        try:
            rel = project_pure.relative_to(workspace_pure)
            return str(Path(*rel.parts))
        except Exception:
            pass
    return project_pure.name or None


def _resolve_scope(base_dir: Path, workspace_root: str, project_path: str) -> Path:
    base_scope = _resolve_workspace_base(base_dir, workspace_root)
    scope_root = base_scope
    if project_path:
        repo_fallback = _repo_name_fallback(base_dir, project_path)
        if repo_fallback is not None:
            return repo_fallback
        project_rel = _absolute_project_relative(workspace_root, project_path)
        raw_project = Path(project_rel or project_path)
        candidate = (
            raw_project.resolve()
            if raw_project.is_absolute()
            else (base_scope / raw_project).resolve()
        )
        if workspace_root:
            try:
                candidate.relative_to(base_scope)
            except Exception:
                candidate = base_scope
        scope_root = candidate
    return scope_root


@dataclass(frozen=True)
class AgentPermissions:
    level: str
    action_mode: str
    workspace_root: str
    project_path: str
    scope_root: Path

    @classmethod
    def default_full(cls, base_dir: Path) -> "AgentPermissions":
        scope_root = base_dir.resolve()
        return cls(
            level="full",
            action_mode="full",
            workspace_root=str(scope_root),
            project_path="",
            scope_root=scope_root,
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        base_dir: Path,
        *,
        default_level: str = "none",
        default_action_mode: str = "safe",
    ) -> "AgentPermissions":
        raw = dict(payload or {})
        level = _normalize_level(raw.get("level"), default_level)
        action_mode = _normalize_action_mode(
            raw.get("action_mode", raw.get("actionMode")),
            default_action_mode if level == "full" else "safe",
        )
        workspace_root = str(
            raw.get("workspace_root", raw.get("workspaceRoot", "")) or ""
        ).strip()
        project_path = str(
            raw.get("project_path", raw.get("projectPath", "")) or ""
        ).strip()
        scope_root = _resolve_scope(base_dir, workspace_root, project_path)
        return cls(
            level=level,
            action_mode=action_mode,
            workspace_root=workspace_root,
            project_path=project_path,
            scope_root=scope_root,
        )

    @property
    def can_read(self) -> bool:
        return self.level in {"read", "edit", "full"}

    @property
    def can_write(self) -> bool:
        return self.level in {"edit", "full"} and self.action_mode == "full"

    @property
    def can_run_commands(self) -> bool:
        return self.level == "full" and self.action_mode == "full"

    @property
    def can_open_browser(self) -> bool:
        return self.level == "full" and self.action_mode == "full"

    @property
    def requires_tool_runner(self) -> bool:
        return self.can_read or self.can_write or self.can_run_commands or self.can_open_browser

    @property
    def scope_label(self) -> str:
        return str(self.scope_root)

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "action_mode": self.action_mode,
            "workspace_root": self.workspace_root,
            "project_path": self.project_path,
            "scope_root": str(self.scope_root),
            "can_read": self.can_read,
            "can_write": self.can_write,
            "can_run_commands": self.can_run_commands,
            "can_open_browser": self.can_open_browser,
        }


def permissions_from_request(
    payload: dict[str, Any] | None,
    base_dir: Path,
) -> AgentPermissions | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("permissions")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raw = {}
    return AgentPermissions.from_payload(raw, base_dir)


def build_agent_permission_context(permissions: AgentPermissions | None) -> str:
    if permissions is None:
        return (
            "Permisos locales: modo legacy. Trabaja dentro del repositorio actual con "
            "lectura, escritura, comandos y navegador disponibles."
        )
    if not permissions.can_read:
        return (
            "Permisos locales: sin acceso real al proyecto. No leas archivos locales, "
            "no escribas, no ejecutes comandos y no abras navegador."
        )
    parts = [
        f"Permisos locales: scope autorizado {permissions.scope_label}.",
        "Lectura local permitida.",
    ]
    if permissions.can_write:
        parts.append("Escritura local permitida.")
    else:
        parts.append("Escritura local bloqueada.")
    if permissions.can_run_commands:
        parts.append("Comandos del proyecto permitidos.")
    else:
        parts.append("Comandos del proyecto bloqueados.")
    if permissions.can_open_browser:
        parts.append("Apertura de navegador permitida.")
    else:
        parts.append("Apertura de navegador bloqueada.")
    return " ".join(parts)
