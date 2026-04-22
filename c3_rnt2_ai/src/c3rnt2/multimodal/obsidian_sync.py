from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any


NOTE_FOLDERS = {
    "architecture": "Projects/Vortex/Architecture",
    "session": "Projects/Vortex/Sessions",
    "decision": "Projects/Vortex/Decisions",
    "prompt": "Projects/Vortex/Prompts",
    "bug": "Projects/Vortex/Bugs",
    "experiment": "Projects/Vortex/Experiments",
}

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return text or f"note-{int(time.time())}"


def _looks_windows_absolute(raw: str) -> bool:
    text = str(raw or "").strip()
    return bool(_WINDOWS_ABS_RE.match(text) or text.startswith("\\\\"))


class ObsidianSyncService:
    def __init__(self, *, settings: dict[str, Any], base_dir: Path) -> None:
        self.settings = settings
        self.base_dir = Path(base_dir)
        self.state_dir = (self.base_dir / "data" / "multimodal").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.override_path = self.state_dir / "obsidian_override.json"
        if not self.override_path.exists():
            self.override_path.write_text("{}", encoding="utf-8")

    def _load_override(self) -> dict[str, Any]:
        try:
            import json

            payload = json.loads(self.override_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _write_override(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json

        self.override_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return payload

    def _host_mount_root(self) -> Path | None:
        raw = str(os.getenv("C3RNT2_HOST_WORKSPACE_MOUNT") or "").strip()
        if not raw:
            return None
        try:
            return Path(raw).resolve()
        except Exception:
            return Path(raw)

    def _host_windows_root(self) -> str:
        return str(os.getenv("C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT") or "").strip()

    def _repo_name(self) -> str:
        return str(os.getenv("C3RNT2_HOST_WORKSPACE_REPO_NAME") or "Vortex").strip() or "Vortex"

    def _translate_host_workspace_path(self, raw: str) -> Path | None:
        text = str(raw or "").strip()
        mount_root = self._host_mount_root()
        if mount_root is None or not _looks_windows_absolute(text):
            return None
        normalized_raw = text.replace("/", "\\")
        host_windows_root = self._host_windows_root().replace("/", "\\").rstrip("\\")
        if host_windows_root:
            lowered_raw = normalized_raw.lower()
            lowered_root = host_windows_root.lower()
            if lowered_raw == lowered_root:
                return mount_root.resolve()
            if lowered_raw.startswith(lowered_root + "\\"):
                rel = normalized_raw[len(host_windows_root):].lstrip("\\")
                return (mount_root / Path(rel.replace("\\", "/"))).resolve()
        repo_name = self._repo_name().lower()
        lowered_raw = normalized_raw.lower()
        marker = f"\\{repo_name}\\"
        if lowered_raw.endswith(f"\\{repo_name}"):
            return mount_root.resolve()
        if marker in lowered_raw:
            index = lowered_raw.index(marker)
            rel = normalized_raw[index + len(marker):].lstrip("\\")
            return (mount_root / Path(rel.replace("\\", "/"))).resolve()
        return None

    def _display_path(self, path: Path) -> str:
        mount_root = self._host_mount_root()
        host_windows_root = self._host_windows_root().replace("/", "\\").rstrip("\\")
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if mount_root is not None and host_windows_root:
            try:
                relative = resolved.relative_to(mount_root.resolve())
                tail = "\\".join(relative.parts)
                return host_windows_root if not tail else f"{host_windows_root}\\{tail}"
            except Exception:
                pass
        return str(resolved)

    def current_config(self) -> dict[str, Any]:
        cfg = dict(self.settings.get("obsidian", {}) or {})
        cfg.update(self._load_override())
        cfg.setdefault("enabled", False)
        cfg.setdefault("vault_path", "data/obsidian_vault")
        cfg.setdefault("folder_map", dict(NOTE_FOLDERS))
        return cfg

    def set_config(self, patch: dict[str, Any] | None) -> dict[str, Any]:
        current = self._load_override()
        current.update(dict(patch or {}))
        self._write_override(current)
        return self.current_config()

    def resolve_vault_path(self) -> Path:
        cfg = self.current_config()
        raw = str(cfg.get("vault_path") or "").strip()
        if not raw:
            return (self.base_dir / "data" / "obsidian_vault").resolve()
        translated = self._translate_host_workspace_path(raw)
        if translated is not None:
            return translated
        expanded = Path(os.path.expanduser(raw))
        if expanded.is_absolute():
            return expanded.resolve()
        return (self.base_dir / expanded).resolve()

    def status(self) -> dict[str, Any]:
        cfg = self.current_config()
        configured_vault = str(cfg.get("vault_path") or "").strip()
        vault_path = self.resolve_vault_path()
        available = vault_path.exists() and vault_path.is_dir()
        folder_map = dict(cfg.get("folder_map") or NOTE_FOLDERS)
        last_note = str(self._load_override().get("last_saved_note") or "").strip() or None
        return {
            "ok": True,
            "enabled": bool(cfg.get("enabled", False)),
            "vault_path": configured_vault or self._display_path(vault_path),
            "resolved_vault_path": str(vault_path),
            "available": available,
            "validated": available,
            "folders": folder_map,
            "last_saved_note": last_note,
        }

    def _ensure_vault(self) -> Path:
        vault_path = self.resolve_vault_path()
        vault_path.mkdir(parents=True, exist_ok=True)
        return vault_path

    def save_note(
        self,
        *,
        note_type: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        cfg = self.current_config()
        vault_path = self._ensure_vault()
        folder_map = dict(cfg.get("folder_map") or NOTE_FOLDERS)
        resolved_type = str(note_type or "session").strip().lower()
        folder_name = str(folder_map.get(resolved_type) or folder_map.get("session") or NOTE_FOLDERS["session"])
        target_dir = (vault_path / folder_name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        file_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(title)}.md"
        target_path = target_dir / file_name
        note_tags = [tag for tag in (tags or []) if str(tag).strip()]
        frontmatter = [
            "---",
            f"type: {resolved_type}",
            f"title: {title}",
            f"created_at: {ts}",
            f"tags: [{', '.join(note_tags)}]",
        ]
        for key, value in (metadata or {}).items():
            frontmatter.append(f"{key}: {value}")
        frontmatter.append("---")
        body = "\n".join(frontmatter) + "\n\n" + str(content or "").strip() + "\n"
        target_path.write_text(body, encoding="utf-8")
        override = self._load_override()
        override["last_saved_note"] = self._display_path(target_path)
        override["last_saved_type"] = resolved_type
        override["last_saved_at"] = float(time.time())
        self._write_override(override)
        return {
            "ok": True,
            "path": self._display_path(target_path),
            "resolved_path": str(target_path),
            "vault_path": str(cfg.get("vault_path") or "").strip() or self._display_path(vault_path),
            "resolved_vault_path": str(vault_path),
            "note_type": resolved_type,
        }

    def iter_recent_notes(self, *, limit: int = 24) -> list[dict[str, Any]]:
        status = self.status()
        if not bool(status.get("available")):
            return []
        vault = Path(str(status.get("resolved_vault_path") or self.resolve_vault_path()))
        items: list[dict[str, Any]] = []
        for path in vault.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            items.append(
                {
                    "path": self._display_path(path),
                    "resolved_path": str(path),
                    "title": path.stem,
                    "text": text,
                    "mtime": mtime,
                }
            )
        items.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
        return items[:limit]
