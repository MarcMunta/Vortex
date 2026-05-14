from __future__ import annotations

from pathlib import Path
from typing import Any

def _fallback_backend(settings: dict, current: str) -> str | None:
    core = settings.get("core", {}) or {}
    contract = settings.get("profile_contract", {}) or {}
    cur = str(current or "").lower()
    if bool(contract.get("disable_fallbacks", False)):
        return None
    fallback = None
    if cur == "hf" and core.get("hf_fallback") is not None:
        fallback = core.get("hf_fallback")
    if fallback is None:
        fallback = core.get("backend_fallback")
    if fallback is None and cur != "hf":
        if bool(core.get("allow_implicit_hf_fallback", True)):
            fallback = core.get("hf_fallback") or (
                "hf" if core.get("hf_model") else None
            )
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
    if backend in {"vllm"}:
        backend = "external"
    if backend_override is None and backend == "hf" and str(core.get("prefer_llama_cpp_if_available", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        if _llama_cpp_ready(core):
            backend = "llama_cpp"
    if backend == "external":
        try:
            from .external_engine import load_external_engine_model

            return load_external_engine_model(settings)
        except Exception:
            fb = _fallback_backend(settings, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "hf":
        try:
            from .hf_model import load_hf_model

            return load_hf_model(settings)
        except Exception:
            fb = _fallback_backend(settings, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "llama_cpp":
        try:
            from .llama_cpp_backend import load_llama_cpp_model

            return load_llama_cpp_model(settings)
        except Exception:
            fb = _fallback_backend(settings, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            raise
    if backend == "tensorrt":
        try:
            from .tensorrt_backend import load_tensorrt_model

            return load_tensorrt_model(settings)
        except Exception:
            fb = _fallback_backend(settings, backend)
            if fb:
                return load_inference_model(settings, backend_override=fb)
            from .model.core_transformer import CoreTransformer

            return CoreTransformer.from_settings(settings)
    from .model.core_transformer import CoreTransformer

    return CoreTransformer.from_settings(settings)
