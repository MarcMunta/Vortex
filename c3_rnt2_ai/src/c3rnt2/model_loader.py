from __future__ import annotations

from typing import Any

from .model.core_transformer import CoreTransformer
from .hf_model import load_hf_model
from .tensorrt_backend import load_tensorrt_model


def load_inference_model(settings: dict, backend_override: str | None = None) -> Any:
    core = settings.get("core", {}) or {}
    backend = str(backend_override or core.get("backend", "vortex")).lower()
    if backend == "hf":
        try:
            return load_hf_model(settings)
        except Exception:
            fallback = core.get("backend_fallback") or core.get("hf_fallback")
            if fallback:
                local = dict(settings)
                core_local = dict(core)
                core_local["backend"] = str(fallback)
                local["core"] = core_local
                return CoreTransformer.from_settings(local)
            raise
    if backend == "tensorrt":
        return load_tensorrt_model(settings)
    return CoreTransformer.from_settings(settings)
