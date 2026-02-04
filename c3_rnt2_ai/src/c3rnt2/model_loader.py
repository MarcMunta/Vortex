from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .model.core_transformer import CoreTransformer
from .hf_model import load_hf_model
from .tensorrt_backend import load_tensorrt_model


def _fallback_backend(core: dict, current: str) -> str | None:
    cur = str(current or "").lower()
    fallback = None
    if cur == "hf" and core.get("hf_fallback") is not None:
        fallback = core.get("hf_fallback")
    if fallback is None:
        fallback = core.get("backend_fallback")
    if fallback is None and cur != "hf":
        fallback = core.get("hf_fallback")
    if fallback and str(fallback).lower() != cur:
        return str(fallback).lower()
    return None


def _llama_cpp_ready(core: dict, *, base_dir: Path | None = None) -> bool:
    model_path = core.get("llama_cpp_model_path")
    if not model_path:
        return False
    base_dir = Path(base_dir or ".").resolve()
    path = Path(str(model_path))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return False
    try:
        __import__("llama_cpp")
    except Exception:
        return False
    return True


def load_inference_model(settings: dict, backend_override: str | None = None) -> Any:
    core = settings.get("core", {}) or {}
    backend = str(backend_override or core.get("backend", "vortex")).lower()
    if backend_override is None and backend == "hf" and str(core.get("prefer_llama_cpp_if_available", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        if _llama_cpp_ready(core):
            backend = "llama_cpp"
    profile = str(settings.get("_profile") or "")
    if sys.platform.startswith("win") and profile == "rtx4080_16gb_120b_like" and backend == "hf":
        try:
            from .prepare import prepare_model_state

            state = prepare_model_state(settings, base_dir=Path(".").resolve())
        except Exception as exc:
            raise RuntimeError(f"prepare_model_check_failed:{exc}") from exc
        if str(state.get("backend_resolved") or "").lower() == "llama_cpp" and _llama_cpp_ready(core):
            backend = "llama_cpp"
        elif not bool(state.get("ok", False)):
            raise RuntimeError(
                "Unsafe HF config for rtx4080_16gb_120b_like on Windows. "
                "Run: python -m vortex prepare-model --profile rtx4080_16gb_120b_like"
            )
    if backend == "hf":
        try:
            return load_hf_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "llama_cpp":
        try:
            from .llama_cpp_backend import load_llama_cpp_model

            return load_llama_cpp_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "tensorrt":
        try:
            return load_tensorrt_model(settings)
        except Exception:
            fb = _fallback_backend(core, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            return CoreTransformer.from_settings(settings)
    return CoreTransformer.from_settings(settings)
