from __future__ import annotations

from typing import Any

from .model.core_transformer import CoreTransformer
from .hf_model import load_hf_model


def load_inference_model(settings: dict) -> Any:
    core = settings.get("core", {}) or {}
    backend = str(core.get("backend", "vortex")).lower()
    if backend == "hf":
        return load_hf_model(settings)
    return CoreTransformer.from_settings(settings)
