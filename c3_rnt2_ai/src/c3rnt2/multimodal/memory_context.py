from __future__ import annotations

from typing import Any

from .obsidian_sync import ObsidianSyncService


class MemoryContextBuilder:
    def __init__(
        self,
        *,
        settings: dict[str, Any],
        obsidian_sync: ObsidianSyncService,
    ) -> None:
        self.settings = settings
        self.obsidian_sync = obsidian_sync

    def _cfg(self) -> dict[str, Any]:
        return self.settings.get("multimodal_memory", {}) or {}

    def _score(self, text: str, terms: list[str]) -> int:
        lowered = text.lower()
        score = 0
        for term in terms:
            token = str(term or "").strip().lower()
            if not token:
                continue
            score += lowered.count(token)
        return score

    def collect(self, *, terms: list[str], limit: int | None = None) -> dict[str, Any]:
        cfg = self._cfg()
        max_notes = int(limit or cfg.get("max_notes", 4))
        max_chars = int(cfg.get("max_chars", 1200))
        recent = self.obsidian_sync.iter_recent_notes(limit=max_notes * 6)
        ranked: list[dict[str, Any]] = []
        for item in recent:
            score = self._score(str(item.get("title") or "") + "\n" + str(item.get("text") or ""), terms)
            if score <= 0 and terms:
                continue
            ranked.append({**item, "score": score})
        ranked.sort(key=lambda item: (int(item.get("score") or 0), float(item.get("mtime") or 0.0)), reverse=True)
        selected = ranked[:max_notes] if ranked else recent[:max_notes]
        refs: list[dict[str, Any]] = []
        chunks: list[str] = []
        used_chars = 0
        for item in selected:
            text = str(item.get("text") or "").strip()
            excerpt = text[: min(320, max_chars - used_chars)]
            if not excerpt:
                continue
            refs.append({"kind": "obsidian", "ref": str(item.get("path") or ""), "title": str(item.get("title") or "")})
            chunks.append(f"- {item.get('title')}: {excerpt}")
            used_chars += len(excerpt)
            if used_chars >= max_chars:
                break
        return {"text": "\n".join(chunks).strip(), "refs": refs}
