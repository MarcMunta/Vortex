from __future__ import annotations

import os
import re
import json
import hashlib
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
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BACKLINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return text or f"note-{int(time.time())}"


def _looks_windows_absolute(raw: str) -> bool:
    text = str(raw or "").strip()
    return bool(_WINDOWS_ABS_RE.match(text) or text.startswith("\\\\"))


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _trim_tokens(text: str, max_tokens: int) -> str:
    max_chars = max(0, int(max_tokens) * 4)
    value = str(text or "")
    return value if len(value) <= max_chars else value[:max_chars].rstrip()


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    result: dict[str, str] = {}
    for raw in text[3:end].splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        result[key.strip().lower()] = value.strip().strip('"')
    return result


class ObsidianSyncService:
    def __init__(self, *, settings: dict[str, Any], base_dir: Path) -> None:
        self.settings = settings
        self.base_dir = Path(base_dir)
        self.state_dir = (self.base_dir / "data" / "multimodal").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.override_path = self.state_dir / "obsidian_override.json"
        self.index_path = self.state_dir / "obsidian_index.json"
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
            "message": None if available else "Obsidian no configurado",
            "index_path": str(self.index_path),
            "folders": folder_map,
            "last_saved_note": last_note,
        }

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"notes": []}
        except Exception:
            return {"notes": []}

    def _write_index(self, notes: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "ok": True,
            "version": 1,
            "updated_at": time.time(),
            "notes": notes,
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return payload

    def _iter_markdown_paths(self, vault: Path) -> list[Path]:
        ignored_names = {".obsidian", ".git", ".trash", "__pycache__", "node_modules", ".cache"}
        paths: list[Path] = []
        for path in vault.rglob("*.md"):
            parts = set(path.relative_to(vault).parts)
            if any(part.startswith(".") for part in parts):
                continue
            if parts & ignored_names:
                continue
            if path.is_file():
                paths.append(path)
        return paths

    def _index_note(self, vault: Path, path: Path) -> dict[str, Any] | None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            stat = path.stat()
        except Exception:
            return None
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        meta = _frontmatter(text)
        headings = [match.group(2).strip() for match in _HEADING_RE.finditer(text)][:24]
        tags = sorted(set(_TAG_RE.findall(text)))
        raw_tags = str(meta.get("tags") or "").strip().strip("[]")
        if raw_tags:
            tags.extend([item.strip().strip('"').lstrip("#") for item in raw_tags.split(",") if item.strip()])
        title = meta.get("title") or (headings[0] if headings else path.stem)
        rel = path.relative_to(vault).as_posix()
        snippet = re.sub(r"\s+", " ", text).strip()
        return {
            "path": self._display_path(path),
            "resolved_path": str(path),
            "relative_path": rel,
            "title": title,
            "tags": sorted(set(tag for tag in tags if tag)),
            "date": meta.get("date") or meta.get("created_at") or meta.get("updated_at"),
            "headings": headings,
            "backlinks": sorted(set(_BACKLINK_RE.findall(text)))[:50],
            "hash": digest,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "text": text,
            "snippet": snippet[:1600],
        }

    def reindex(self) -> dict[str, Any]:
        status = self.status()
        if not bool(status.get("enabled")):
            return {"ok": True, "enabled": False, "available": False, "message": "Obsidian no configurado", "notes": 0}
        if not bool(status.get("available")):
            self._write_index([])
            return {"ok": True, "enabled": True, "available": False, "message": "Obsidian no configurado", "notes": 0}
        vault = Path(str(status.get("resolved_vault_path") or self.resolve_vault_path()))
        existing = {
            str(item.get("resolved_path")): item
            for item in (self._load_index().get("notes") or [])
            if isinstance(item, dict)
        }
        notes: list[dict[str, Any]] = []
        for path in self._iter_markdown_paths(vault):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                continue
            cached = existing.get(str(path))
            if cached and cached.get("hash") == digest:
                notes.append(cached)
                continue
            indexed = self._index_note(vault, path)
            if indexed:
                notes.append(indexed)
        notes.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
        self._write_index(notes)
        return {"ok": True, "enabled": True, "available": True, "notes": len(notes), "index_path": str(self.index_path)}

    def _score_note(self, note: dict[str, Any], query_terms: set[str]) -> float:
        haystack = " ".join(
            [
                str(note.get("title") or ""),
                str(note.get("relative_path") or ""),
                " ".join(note.get("tags") or []),
                " ".join(note.get("headings") or []),
                str(note.get("snippet") or ""),
            ]
        ).lower()
        if not query_terms:
            return float(note.get("mtime") or 0.0) / 10_000_000_000
        score = 0.0
        for term in query_terms:
            if not term:
                continue
            if term in str(note.get("title") or "").lower():
                score += 5.0
            if term in str(note.get("relative_path") or "").lower():
                score += 2.5
            if term in " ".join(note.get("tags") or []).lower():
                score += 3.0
            score += min(4.0, haystack.count(term) * 0.7)
        return score + min(1.0, float(note.get("mtime") or 0.0) / max(time.time(), 1.0))

    def search(self, query: str, *, top_k: int = 6, max_tokens: int = 5000) -> dict[str, Any]:
        status = self.status()
        if not bool(status.get("enabled")) or not bool(status.get("available")):
            return {"ok": True, "enabled": bool(status.get("enabled")), "available": False, "message": "Obsidian no configurado", "notes": []}
        index = self._load_index()
        notes = [item for item in (index.get("notes") or []) if isinstance(item, dict)]
        if not notes:
            self.reindex()
            index = self._load_index()
            notes = [item for item in (index.get("notes") or []) if isinstance(item, dict)]
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_/-]{3,}", str(query or ""))}
        ranked = sorted(notes, key=lambda item: self._score_note(item, terms), reverse=True)
        selected: list[dict[str, Any]] = []
        used = 0
        seen_hashes: set[str] = set()
        for note in ranked:
            score = self._score_note(note, terms)
            if terms and score <= 0:
                continue
            digest = str(note.get("hash") or "")
            if digest in seen_hashes:
                continue
            text = str(note.get("text") or note.get("snippet") or "")
            remaining = int(max_tokens) - used
            if remaining <= 0:
                break
            clipped = _trim_tokens(text, min(900, remaining))
            used += _estimate_tokens(clipped)
            seen_hashes.add(digest)
            selected.append(
                {
                    "path": note.get("path"),
                    "relative_path": note.get("relative_path"),
                    "title": note.get("title"),
                    "tags": note.get("tags") or [],
                    "headings": note.get("headings") or [],
                    "score": round(score, 3),
                    "text": clipped,
                    "hash": digest,
                }
            )
            if len(selected) >= int(top_k):
                break
        return {"ok": True, "enabled": True, "available": True, "notes": selected, "tokens_estimate": used}

    def build_context(self, query: str, *, top_k: int = 6, max_tokens: int = 5000) -> dict[str, Any]:
        result = self.search(query, top_k=top_k, max_tokens=max_tokens)
        notes = result.get("notes") or []
        if not notes:
            return {**result, "text": ""}
        parts = ["Curated Obsidian memory. Use only when relevant; note paths are traceability:"]
        for note in notes:
            parts.append(
                "\n".join(
                    [
                        f"- Path: {note.get('relative_path') or note.get('path')}",
                        f"  Title: {note.get('title')}",
                        f"  Tags: {', '.join(note.get('tags') or [])}",
                        f"  Excerpt:\n{note.get('text')}",
                    ]
                )
            )
        return {**result, "text": _trim_tokens("\n\n".join(parts), max_tokens)}

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
