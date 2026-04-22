from __future__ import annotations

import time
import uuid
from typing import Any


def _now() -> float:
    return float(time.time())


def _float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def normalize_region(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    width = max(0.0, _float(raw.get("width"), 0.0))
    height = max(0.0, _float(raw.get("height"), 0.0))
    if width <= 0.0 or height <= 0.0:
        return None
    return {
        "x": _float(raw.get("x"), 0.0),
        "y": _float(raw.get("y"), 0.0),
        "width": width,
        "height": height,
    }


def normalize_transform(raw: Any) -> dict[str, float]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "x": _float(data.get("x"), 80.0),
        "y": _float(data.get("y"), 72.0),
        "z": _float(data.get("z"), 0.0),
        "scale": max(0.1, _float(data.get("scale"), 1.0)),
        "rotation": _float(data.get("rotation"), 0.0),
        "skew_x": _float(data.get("skew_x", data.get("skewX")), 0.0),
        "skew_y": _float(data.get("skew_y", data.get("skewY")), 0.0),
        "tilt_x": _float(data.get("tilt_x", data.get("tiltX")), 0.0),
        "tilt_y": _float(data.get("tilt_y", data.get("tiltY")), 0.0),
        "perspective": max(300.0, _float(data.get("perspective"), 1100.0)),
        "width": max(140.0, _float(data.get("width"), 360.0)),
        "height": max(120.0, _float(data.get("height"), 220.0)),
    }


def default_panel(panel_type: str = "note", *, title: str | None = None) -> dict[str, Any]:
    now = _now()
    resolved_type = str(panel_type or "note").strip().lower() or "note"
    return {
        "id": f"panel-{uuid.uuid4().hex[:8]}",
        "type": resolved_type,
        "title": title or resolved_type.replace("_", " ").title(),
        "content": "",
        "source": {},
        "transform": normalize_transform(None),
        "page_index": 0,
        "page_count": 1,
        "selected": False,
        "locked": False,
        "created_at": now,
        "updated_at": now,
    }


def normalize_panel(raw: Any) -> dict[str, Any]:
    base = default_panel("note")
    data = raw if isinstance(raw, dict) else {}
    base["id"] = str(data.get("id") or base["id"]).strip() or base["id"]
    base["type"] = str(data.get("type") or base["type"]).strip().lower() or base["type"]
    base["title"] = str(data.get("title") or base["title"]).strip() or base["title"]
    base["content"] = str(data.get("content") or "").strip()
    base["source"] = dict(data.get("source") or {}) if isinstance(data.get("source"), dict) else {}
    base["transform"] = normalize_transform(data.get("transform"))
    base["page_index"] = max(0, _int(data.get("page_index", data.get("pageIndex")), 0))
    base["page_count"] = max(1, _int(data.get("page_count", data.get("pageCount")), 1))
    base["selected"] = bool(data.get("selected", False))
    base["locked"] = bool(data.get("locked", False))
    base["created_at"] = _float(data.get("created_at", data.get("createdAt")), base["created_at"])
    base["updated_at"] = _float(data.get("updated_at", data.get("updatedAt")), base["updated_at"])
    return base


def default_spatial_session(session_id: str | None = None) -> dict[str, Any]:
    now = _now()
    return {
        "session_id": session_id or f"spatial-{uuid.uuid4().hex[:8]}",
        "selected_object_id": None,
        "selected_region": None,
        "active_panel_ids": [],
        "active_presentation_id": None,
        "active_page_index": 0,
        "interaction_mode": "idle",
        "last_voice_command": None,
        "last_gesture_event": None,
        "camera_state": None,
        "gesture_state": None,
        "focused_item": None,
        "recent_multimodal_summary": None,
        "panels": [],
        "updated_at": now,
        "created_at": now,
    }


def normalize_spatial_session(raw: Any) -> dict[str, Any]:
    base = default_spatial_session()
    data = raw if isinstance(raw, dict) else {}
    base["session_id"] = str(data.get("session_id", data.get("sessionId")) or base["session_id"]).strip() or base["session_id"]
    base["selected_object_id"] = str(data.get("selected_object_id", data.get("selectedObjectId")) or "").strip() or None
    base["selected_region"] = normalize_region(data.get("selected_region", data.get("selectedRegion")))
    raw_panels = data.get("panels") if isinstance(data.get("panels"), list) else []
    panels = [normalize_panel(item) for item in raw_panels]
    base["panels"] = panels
    base["active_panel_ids"] = [
        str(item).strip()
        for item in (data.get("active_panel_ids", data.get("activePanelIds")) or [])
        if str(item).strip()
    ]
    if not base["active_panel_ids"]:
        base["active_panel_ids"] = [panel["id"] for panel in panels]
    base["active_presentation_id"] = str(data.get("active_presentation_id", data.get("activePresentationId")) or "").strip() or None
    base["active_page_index"] = max(0, _int(data.get("active_page_index", data.get("activePageIndex")), 0))
    base["interaction_mode"] = str(data.get("interaction_mode", data.get("interactionMode")) or "idle").strip() or "idle"
    base["last_voice_command"] = str(data.get("last_voice_command", data.get("lastVoiceCommand")) or "").strip() or None
    last_gesture = data.get("last_gesture_event", data.get("lastGestureEvent"))
    base["last_gesture_event"] = dict(last_gesture) if isinstance(last_gesture, dict) else None
    camera_state = data.get("camera_state", data.get("cameraState"))
    base["camera_state"] = dict(camera_state) if isinstance(camera_state, dict) else None
    gesture_state = data.get("gesture_state", data.get("gestureState"))
    base["gesture_state"] = dict(gesture_state) if isinstance(gesture_state, dict) else None
    focused_item = data.get("focused_item", data.get("focusedItem"))
    base["focused_item"] = dict(focused_item) if isinstance(focused_item, dict) else None
    base["recent_multimodal_summary"] = str(
        data.get("recent_multimodal_summary", data.get("recentMultimodalSummary")) or ""
    ).strip() or None
    base["updated_at"] = _float(data.get("updated_at", data.get("updatedAt")), base["updated_at"])
    base["created_at"] = _float(data.get("created_at", data.get("createdAt")), base["created_at"])
    return base
