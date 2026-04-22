from __future__ import annotations

from typing import Any

from .memory_context import MemoryContextBuilder
from .spatial_state import SpatialStateStore


class MultimodalSessionFusion:
    def __init__(
        self,
        *,
        settings: dict[str, Any],
        spatial_store: SpatialStateStore,
        memory_context: MemoryContextBuilder,
    ) -> None:
        self.settings = settings
        self.spatial_store = spatial_store
        self.memory_context = memory_context

    def _cfg(self) -> dict[str, Any]:
        return self.settings.get("multimodal_context", {}) or {}

    def build_context(
        self,
        *,
        messages: list[dict[str, Any]],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self._cfg()
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return {"enabled": False, "text": "", "refs": [], "summary": None}
        session = self.spatial_store.get_session()
        request_payload = dict(payload or {})
        selected_panel_id = str(session.get("selected_object_id") or "").strip() or None
        panels = session.get("panels") or []
        selected_panel = next(
            (panel for panel in panels if str(panel.get("id") or "") == selected_panel_id),
            None,
        )
        latest_user = ""
        for item in reversed(messages):
            if str(item.get("role") or "") == "user":
                latest_user = str(item.get("content") or "").strip()
                if latest_user:
                    break
        focus_terms = [
            latest_user,
            str(session.get("last_voice_command") or ""),
            str((selected_panel or {}).get("title") or ""),
            str((selected_panel or {}).get("content") or "")[:80],
        ]
        memory = self.memory_context.collect(terms=focus_terms)
        parts: list[str] = []
        if selected_panel is not None:
            transform = selected_panel.get("transform") or {}
            parts.append(
                "Focused panel: "
                f"id={selected_panel.get('id')} type={selected_panel.get('type')} "
                f"title={selected_panel.get('title')} page={selected_panel.get('page_index', 0) + 1}/{selected_panel.get('page_count', 1)} "
                f"x={transform.get('x')} y={transform.get('y')} scale={transform.get('scale')} "
                f"rotation={transform.get('rotation')} tilt_x={transform.get('tilt_x')} tilt_y={transform.get('tilt_y')}"
            )
        if session.get("selected_region"):
            region = session.get("selected_region") or {}
            parts.append(
                "Selected region: "
                f"x={region.get('x')} y={region.get('y')} width={region.get('width')} height={region.get('height')}"
            )
        if session.get("last_gesture_event"):
            event = session.get("last_gesture_event") or {}
            parts.append(
                "Latest gesture: "
                f"{event.get('gesture') or event.get('kind')} confidence={event.get('confidence')}"
            )
        if session.get("camera_state"):
            camera = session.get("camera_state") or {}
            parts.append(
                "Camera status: "
                f"enabled={camera.get('enabled')} ready={camera.get('ready')} "
                f"tracker={camera.get('tracker_mode')} error={camera.get('error')}"
            )
        if session.get("gesture_state"):
            gesture = session.get("gesture_state") or {}
            parts.append(
                "Gesture state: "
                f"gesture={gesture.get('gesture')} confidence={gesture.get('confidence')} "
                f"tracker={gesture.get('tracker_mode')}"
            )
        if session.get("last_voice_command"):
            parts.append(f"Latest voice command: {session.get('last_voice_command')}")
        if panels:
            summaries = [
                f"{panel.get('id')}:{panel.get('type')}:{panel.get('title')}"
                for panel in panels[:8]
            ]
            parts.append("Active panels: " + ", ".join(summaries))
        if memory.get("text"):
            parts.append("Curated Obsidian memory:\n" + str(memory.get("text")))
        text = "\n".join(part for part in parts if part).strip()
        max_chars = int(cfg.get("max_chars", 1800))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        summary = self.spatial_store.describe_session(session)
        return {
            "enabled": True,
            "text": text,
            "refs": list(memory.get("refs") or []),
            "summary": summary,
            "session": session,
            "request": request_payload,
        }
