from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from .models import default_panel, normalize_panel, normalize_region, normalize_transform


class PanelRegistry:
    def build_panel(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = payload if isinstance(payload, dict) else {}
        panel = default_panel(
            str(data.get("type") or "note"),
            title=str(data.get("title") or "").strip() or None,
        )
        panel["content"] = str(data.get("content") or "").strip()
        panel["source"] = dict(data.get("source") or {}) if isinstance(data.get("source"), dict) else {}
        panel["transform"] = normalize_transform(data.get("transform"))
        panel["page_index"] = max(0, int(data.get("page_index", data.get("pageIndex")) or 0))
        panel["page_count"] = max(
            1,
            int(
                data.get("page_count")
                or data.get("pageCount")
                or len((panel["source"].get("pages") or []))
                or 1
            ),
        )
        region = normalize_region(data.get("region"))
        if region is not None:
            panel["transform"].update(
                {
                    "x": region["x"],
                    "y": region["y"],
                    "width": region["width"],
                    "height": region["height"],
                }
            )
        panel["selected"] = bool(data.get("selected", False))
        panel["locked"] = bool(data.get("locked", False))
        panel["updated_at"] = float(time.time())
        return panel

    def merge_panel(self, existing: dict[str, Any], patch: dict[str, Any] | None) -> dict[str, Any]:
        current = normalize_panel(existing)
        data = patch if isinstance(patch, dict) else {}
        if "title" in data:
            current["title"] = str(data.get("title") or current["title"]).strip() or current["title"]
        if "content" in data:
            current["content"] = str(data.get("content") or "").strip()
        if "selected" in data:
            current["selected"] = bool(data.get("selected"))
        if "locked" in data:
            current["locked"] = bool(data.get("locked"))
        if isinstance(data.get("source"), dict):
            next_source = deepcopy(current.get("source") or {})
            next_source.update(dict(data.get("source") or {}))
            current["source"] = next_source
        if "page_index" in data or "pageIndex" in data:
            current["page_index"] = max(0, int(data.get("page_index", data.get("pageIndex")) or 0))
        if "page_count" in data or "pageCount" in data:
            current["page_count"] = max(1, int(data.get("page_count", data.get("pageCount")) or 1))
        if isinstance(data.get("transform"), dict):
            merged_transform = deepcopy(current["transform"])
            merged_transform.update(dict(data.get("transform") or {}))
            current["transform"] = normalize_transform(merged_transform)
        region = normalize_region(data.get("region"))
        if region is not None:
            current["transform"].update(
                {
                    "x": region["x"],
                    "y": region["y"],
                    "width": region["width"],
                    "height": region["height"],
                }
            )
        current["updated_at"] = float(time.time())
        return current
