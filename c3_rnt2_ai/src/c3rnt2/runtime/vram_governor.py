from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _is_cuda_device(device: object | None) -> bool:
    if device is None:
        return False
    if isinstance(device, str):
        return device.lower().startswith("cuda")
    try:
        if torch is not None and isinstance(device, torch.device):
            return device.type == "cuda"
    except Exception:
        return False
    return False


def _log_reduction(requested: int, decided: int, *, free_mb: float | None, peak_mb: float | None, device: object | None, dtype: object | None) -> None:
    payload = {
        "requested": requested,
        "decided": decided,
        "free_mb": None if free_mb is None else round(float(free_mb), 1),
        "peak_mb": None if peak_mb is None else round(float(peak_mb), 1),
        "device": str(device) if device is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
    }
    print(f"vram_governor {payload}")


def decide_max_new_tokens(requested_max_new: int, device: object | None, dtype: object | None, settings: dict) -> int:
    requested = int(requested_max_new or 0)
    if requested <= 0:
        return requested
    if torch is None or not _is_cuda_device(device):
        return requested
    if not torch.cuda.is_available():
        return requested

    core = settings.get("core", {}) or {}
    threshold_mb = float(core.get("vram_threshold_mb", 1024))
    floor_tokens = int(core.get("vram_floor_tokens", 32))
    ceil_tokens = int(core.get("vram_ceil_tokens", requested))
    safety_margin_mb = float(core.get("vram_safety_margin_mb", 512))

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024**2)
        total_mb = total_bytes / (1024**2)
    except Exception:
        return requested

    peak_mb = None
    try:
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        peak_mb = None

    effective_free_mb = free_mb
    if peak_mb is not None and total_mb > 0:
        recent_free = max(0.0, total_mb - peak_mb)
        effective_free_mb = min(effective_free_mb, recent_free)

    base_max = requested
    if ceil_tokens > 0:
        base_max = min(base_max, ceil_tokens)
    base_max = max(1, int(base_max))
    floor_tokens = max(1, int(floor_tokens))

    if effective_free_mb <= threshold_mb:
        decided = max(1, min(base_max, floor_tokens))
        if decided < requested:
            _log_reduction(requested, decided, free_mb=effective_free_mb, peak_mb=peak_mb, device=device, dtype=dtype)
        return decided

    if safety_margin_mb <= 0:
        return base_max

    ratio = (effective_free_mb - threshold_mb) / safety_margin_mb
    ratio = max(0.0, min(1.0, ratio))
    decided = int(round(floor_tokens + (base_max - floor_tokens) * ratio))
    decided = max(1, min(base_max, decided))
    if decided < requested:
        _log_reduction(requested, decided, free_mb=effective_free_mb, peak_mb=peak_mb, device=device, dtype=dtype)
    return decided
