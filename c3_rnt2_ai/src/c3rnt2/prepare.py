from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _truthy(val: object) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _resolve_path(base_dir: Path, raw: object | None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _llama_cpp_ready(core: dict, *, base_dir: Path) -> dict[str, Any]:
    gguf = _resolve_path(base_dir, core.get("llama_cpp_model_path"))
    if gguf is None:
        return {"ok": False, "error": "gguf_path_missing", "gguf_path": None}
    if not gguf.exists():
        return {"ok": False, "error": "gguf_missing", "gguf_path": str(gguf)}
    if importlib.util.find_spec("llama_cpp") is None:
        return {"ok": False, "error": "llama_cpp_python_missing", "gguf_path": str(gguf), "install": 'python -m pip install -e ".[llama_cpp]"'}
    return {"ok": True, "gguf_path": str(gguf)}


def _hf_quant_status(core: dict) -> dict[str, Any]:
    requested_4 = bool(core.get("hf_load_in_4bit"))
    requested_8 = bool(core.get("hf_load_in_8bit"))
    requested = bool(requested_4 or requested_8)
    bnb = importlib.util.find_spec("bitsandbytes") is not None
    mode = None
    if requested and bnb:
        mode = "4bit" if requested_4 else "8bit"
    return {"requested": requested, "bitsandbytes_available": bool(bnb), "quant_mode": mode}


def _hf_offload_status(core: dict, *, base_dir: Path) -> dict[str, Any]:
    device_map = core.get("hf_device_map")
    max_memory = core.get("hf_max_memory")
    offload_folder_raw = core.get("hf_offload_folder")
    offload_folder = _resolve_path(base_dir, offload_folder_raw)
    enabled = bool(device_map or max_memory or offload_folder_raw)
    ok = bool(device_map) and bool(max_memory) and bool(offload_folder_raw)
    return {
        "enabled": bool(enabled),
        "ok": bool(ok),
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": str(offload_folder) if offload_folder is not None else None,
    }


def prepare_model_state(settings: dict, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Offline (no downloads) validation of an inference configuration.

    Intended for Windows safety checks for the 120B-like profile (fail-closed).
    """
    base_dir = Path(base_dir or ".").resolve()
    profile = str(settings.get("_profile") or "").strip() or "unknown"
    core = settings.get("core", {}) or {}
    backend_requested = str(core.get("backend", "vortex")).strip().lower()
    prefer_llama = _truthy(core.get("prefer_llama_cpp_if_available"))

    thresholds = settings.get("bench_thresholds", {}) or {}
    required_ctx = thresholds.get("required_ctx")
    try:
        required_ctx_i = int(required_ctx) if required_ctx is not None else None
    except Exception:
        required_ctx_i = None

    llama = _llama_cpp_ready(core, base_dir=base_dir)
    hf_quant = _hf_quant_status(core)
    hf_offload = _hf_offload_status(core, base_dir=base_dir)
    hf_device = str(core.get("hf_device") or "").strip().lower() or None

    warnings: list[str] = []
    errors: list[str] = []

    if hf_quant.get("requested") and not hf_quant.get("bitsandbytes_available"):
        warnings.append("bitsandbytes_missing_for_quant")
    if hf_offload.get("enabled") and not hf_offload.get("ok"):
        warnings.append("hf_offload_incomplete")

    hf_safe = bool(hf_quant.get("quant_mode")) or bool(hf_offload.get("ok")) or hf_device == "cpu"

    backend_resolved = backend_requested
    is_windows = sys.platform.startswith("win")
    is_120b_like = profile == "rtx4080_16gb_120b_like"

    if backend_requested in {"hf", "transformers"}:
        backend_resolved = "hf"
        if prefer_llama and bool(llama.get("ok", False)):
            backend_resolved = "llama_cpp"
        elif is_windows and is_120b_like and not hf_safe and bool(llama.get("ok", False)):
            backend_resolved = "llama_cpp"
        elif is_windows and is_120b_like and not hf_safe:
            errors.append("unsafe_hf_config_windows_120b_like")
    elif backend_requested == "llama_cpp":
        backend_resolved = "llama_cpp"
        if not bool(llama.get("ok", False)):
            errors.append(str(llama.get("error") or "llama_cpp_not_ready"))

    # llama.cpp sanity (ctx/threads)
    llama_ctx = core.get("llama_cpp_ctx")
    llama_threads = core.get("llama_cpp_threads")
    try:
        llama_ctx_i = int(llama_ctx) if llama_ctx is not None else None
    except Exception:
        llama_ctx_i = None
    try:
        llama_threads_i = int(llama_threads) if llama_threads is not None else None
    except Exception:
        llama_threads_i = None

    if backend_resolved == "llama_cpp":
        if llama_ctx_i is not None and llama_ctx_i <= 0:
            errors.append("llama_cpp_ctx_invalid")
        if required_ctx_i is not None and llama_ctx_i is not None and llama_ctx_i < required_ctx_i:
            warnings.append("llama_cpp_ctx_below_required_ctx")
        if llama_threads_i is not None and llama_threads_i <= 0:
            errors.append("llama_cpp_threads_invalid")

    next_steps: list[str] = []
    if "unsafe_hf_config_windows_120b_like" in errors:
        next_steps.append("Configure HF quant (hf_load_in_4bit/8bit + bitsandbytes) or safe CPU offload (hf_device_map + hf_max_memory + hf_offload_folder).")
        next_steps.append("OR set core.llama_cpp_model_path to an existing .gguf and install the llama.cpp backend (.[llama_cpp]).")
        next_steps.append(f"Re-run: python -m vortex prepare-model --profile {profile}")

    state = {
        "ok": not errors,
        "ts": time.time(),
        "profile": profile,
        "cwd": str(base_dir),
        "os": os.name,
        "platform": sys.platform,
        "backend_requested": backend_requested,
        "backend_resolved": backend_resolved,
        "hf": {
            "device": hf_device,
            "quant": hf_quant,
            "offload": hf_offload,
            "safe": bool(hf_safe),
        },
        "llama_cpp": {
            "ready": bool(llama.get("ok", False)),
            "gguf_path": llama.get("gguf_path"),
            "ctx": llama_ctx_i,
            "threads": llama_threads_i,
            "required_ctx": required_ctx_i,
        },
        "quant_mode": hf_quant.get("quant_mode"),
        "offload_enabled": bool(hf_offload.get("enabled")),
        "gguf_path": llama.get("gguf_path"),
        "warnings": warnings or None,
        "errors": errors or None,
        "next_steps": next_steps or None,
    }
    return state


def write_prepared_state(state: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    base_dir = Path(base_dir or ".").resolve()
    profile = str(state.get("profile") or "unknown").strip() or "unknown"
    out_path = base_dir / "data" / "models" / f"prepared_{profile}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path

