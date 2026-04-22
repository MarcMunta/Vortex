from __future__ import annotations

import re
from typing import Any


def normalize_transcript(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_voice_intent(
    transcript: str | None,
    *,
    session: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    text = normalize_transcript(transcript)
    if not text:
        return None
    lower = text.lower()
    selected_panel = str((session or {}).get("selected_object_id") or "").strip() or None

    if re.search(r"\b(open|abre)\b.*\b(presentation|presentacion|slides?)\b.*\b(here|aqui)\b", lower):
        return {
            "kind": "open_panel",
            "panel_type": "presentation",
            "target": "selected_region",
            "title": "Spatial presentation",
            "panel_id": selected_panel,
        }
    if re.search(r"\b(move|mueve)\b.*\b(left|izquierda)\b", lower):
        return {"kind": "transform_panel", "panel_id": selected_panel, "transform": {"x": -72.0}}
    if re.search(r"\b(move|mueve)\b.*\b(right|derecha)\b", lower):
        return {"kind": "transform_panel", "panel_id": selected_panel, "transform": {"x": 72.0}}
    if re.search(r"\b(move|mueve)\b.*\b(up|arriba)\b", lower):
        return {"kind": "transform_panel", "panel_id": selected_panel, "transform": {"y": -72.0}}
    if re.search(r"\b(move|mueve)\b.*\b(down|abajo)\b", lower):
        return {"kind": "transform_panel", "panel_id": selected_panel, "transform": {"y": 72.0}}
    if re.search(r"\b(tilt|inclina|perspective|perspectiva)\b", lower):
        return {
            "kind": "transform_panel",
            "panel_id": selected_panel,
            "transform": {"tilt_x": -10.0, "tilt_y": 14.0},
        }
    if re.fullmatch(r"(next|siguiente|next slide|next page)", lower):
        return {"kind": "navigate_panel", "panel_id": selected_panel, "delta": 1}
    if re.fullmatch(r"(previous|anterior|prev|back)", lower):
        return {"kind": "navigate_panel", "panel_id": selected_panel, "delta": -1}
    if re.search(r"\b(save|guarda)\b.*\b(obsidian)\b", lower):
        return {"kind": "save_obsidian", "panel_id": selected_panel}
    if re.search(r"\b(talk to me about this|habla conmigo sobre esto|help me implement this|ayudame a implementar esto)\b", lower):
        return {"kind": "chat_query", "panel_id": selected_panel, "query": text}
    return {"kind": "chat_query", "panel_id": selected_panel, "query": text}
