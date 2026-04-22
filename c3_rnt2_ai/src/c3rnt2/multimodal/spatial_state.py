from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .models import normalize_spatial_session
from .panel_registry import PanelRegistry


class SpatialStateStore:
    def __init__(
        self,
        *,
        settings: dict[str, Any],
        base_dir: Path,
        panel_registry: PanelRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.base_dir = Path(base_dir)
        self.panel_registry = panel_registry or PanelRegistry()
        cfg = settings.get("multimodal_memory", {}) or {}
        raw_state_path = cfg.get("state_path") or "data/multimodal/spatial_session.json"
        self.state_path = Path(raw_state_path)
        if not self.state_path.is_absolute():
            self.state_path = (self.base_dir / self.state_path).resolve()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.state_path.exists():
            self._write(normalize_spatial_session({}))

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except Exception:
            return normalize_spatial_session({})
        try:
            import json

            return normalize_spatial_session(json.loads(raw))
        except Exception:
            return normalize_spatial_session({})

    def _write(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json

        normalized = normalize_spatial_session(payload)
        normalized["updated_at"] = float(time.time())
        self.state_path.write_text(
            json.dumps(normalized, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return normalized

    def get_session(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def replace_session(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            return self._write(payload or {})

    def update_session(self, patch: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            current = self._read()
            data = patch if isinstance(patch, dict) else {}
            next_payload = dict(current)
            next_payload.update(data)
            if isinstance(data.get("panels"), list):
                next_payload["panels"] = data.get("panels")
            return self._write(next_payload)

    def _upsert_panel(self, panel: dict[str, Any], *, select: bool = False) -> dict[str, Any]:
        current = self._read()
        panels = current.get("panels") or []
        next_panels: list[dict[str, Any]] = []
        replaced = False
        for existing in panels:
            if str(existing.get("id") or "") == str(panel.get("id") or ""):
                next_panels.append(panel)
                replaced = True
            else:
                next_panels.append(existing)
        if not replaced:
            next_panels.append(panel)
        current["panels"] = next_panels
        current["active_panel_ids"] = [str(item.get("id") or "") for item in next_panels if str(item.get("id") or "")]
        if select:
            current["selected_object_id"] = panel["id"]
            current["focused_item"] = {"kind": "panel", "panel_id": panel["id"], "title": panel.get("title")}
        current["active_presentation_id"] = panel["id"] if panel.get("type") == "presentation" else current.get("active_presentation_id")
        current["active_page_index"] = int(panel.get("page_index") or 0)
        current["recent_multimodal_summary"] = self.describe_session(current)
        return self._write(current)

    def open_panel(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            panel = self.panel_registry.build_panel(payload)
            session = self._upsert_panel(panel, select=True)
            return {"ok": True, "panel": panel, "session": session}

    def update_panel(self, panel_id: str, patch: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            current = self._read()
            for existing in current.get("panels") or []:
                if str(existing.get("id") or "") != str(panel_id or ""):
                    continue
                panel = self.panel_registry.merge_panel(existing, patch)
                session = self._upsert_panel(panel, select=bool(panel.get("selected")))
                return {"ok": True, "panel": panel, "session": session}
        return {"ok": False, "error": "panel_not_found", "panel_id": str(panel_id or "")}

    def navigate_panel(
        self,
        panel_id: str,
        *,
        delta: int = 1,
        index: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._read()
            for existing in current.get("panels") or []:
                if str(existing.get("id") or "") != str(panel_id or ""):
                    continue
                panel = dict(existing)
                page_count = max(1, int(panel.get("page_count") or 1))
                if index is None:
                    page_index = max(0, min(page_count - 1, int(panel.get("page_index") or 0) + int(delta)))
                else:
                    page_index = max(0, min(page_count - 1, int(index)))
                panel["page_index"] = page_index
                session = self._upsert_panel(panel, select=True)
                return {"ok": True, "panel": panel, "session": session}
        return {"ok": False, "error": "panel_not_found", "panel_id": str(panel_id or "")}

    def apply_event(self, event: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            current = self._read()
            data = dict(event or {})
            kind = str(data.get("kind") or data.get("type") or "event").strip().lower()
            current["interaction_mode"] = str(data.get("mode") or current.get("interaction_mode") or "idle").strip() or "idle"
            if kind in {"gesture", "pointer", "focus", "selection"}:
                current["last_gesture_event"] = data
                current["gesture_state"] = {
                    "gesture": data.get("gesture") or data.get("kind"),
                    "confidence": data.get("confidence"),
                    "tracker_mode": data.get("trackerMode") or data.get("tracker_mode"),
                    "ts": data.get("ts") or float(time.time()),
                    "enabled": data.get("gesture_enabled"),
                }
            if kind == "voice":
                current["last_voice_command"] = str(data.get("transcript") or data.get("command") or "").strip() or None
            if kind in {"camera", "camera_status"}:
                current["camera_state"] = {
                    "enabled": data.get("enabled"),
                    "ready": data.get("ready"),
                    "tracker_mode": data.get("trackerMode") or data.get("tracker_mode"),
                    "error": data.get("error"),
                    "ts": data.get("ts") or float(time.time()),
                }
            if isinstance(data.get("camera_state"), dict):
                current["camera_state"] = dict(data.get("camera_state") or {})
            if isinstance(data.get("gesture_state"), dict):
                current["gesture_state"] = dict(data.get("gesture_state") or {})
            panel_id = str(data.get("panel_id") or data.get("panelId") or "").strip() or None
            if panel_id:
                current["selected_object_id"] = panel_id
                current["focused_item"] = {
                    "kind": "panel",
                    "panel_id": panel_id,
                    "gesture": data.get("gesture"),
                }
            region = data.get("region") or data.get("selected_region") or data.get("selectedRegion")
            if region:
                from .models import normalize_region

                current["selected_region"] = normalize_region(region)
            transform_patch = data.get("transform_patch") or data.get("transformPatch")
            if panel_id and isinstance(transform_patch, dict):
                return self.update_panel(panel_id, {"transform": transform_patch, "selected": True})
            current["recent_multimodal_summary"] = self.describe_session(current)
            return self._write(current)

    def describe_session(self, payload: dict[str, Any] | None = None) -> str:
        session = normalize_spatial_session(payload or self.get_session())
        panel_count = len(session.get("panels") or [])
        selected = str(session.get("selected_object_id") or "").strip() or "none"
        mode = str(session.get("interaction_mode") or "idle")
        voice = str(session.get("last_voice_command") or "").strip() or "none"
        camera = session.get("camera_state") if isinstance(session.get("camera_state"), dict) else {}
        gesture = session.get("gesture_state") if isinstance(session.get("gesture_state"), dict) else {}
        tracker = str(camera.get("tracker_mode") or "unknown")
        last_gesture = str(gesture.get("gesture") or "none")
        return (
            f"spatial panels={panel_count} selected={selected} "
            f"mode={mode} voice={voice} tracker={tracker} gesture={last_gesture}"
        )
