from __future__ import annotations

from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def get_vram_free_mb() -> Optional[float]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return float(free) / (1024**2)


def get_vram_total_mb() -> Optional[float]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        _free, total = torch.cuda.mem_get_info()
    except Exception:
        return None
    return float(total) / (1024**2)


def should_reduce_decode(free_mb: Optional[float], threshold_mb: float) -> bool:
    if free_mb is None:
        return False
    try:
        return float(free_mb) < float(threshold_mb)
    except Exception:
        return False


def recommended_max_new_tokens(base_max: int, free_mb: Optional[float], floor: int, ceil: int) -> int:
    base = int(base_max)
    floor_val = max(1, int(floor))
    ceil_val = max(floor_val, int(ceil))
    base = max(floor_val, min(ceil_val, base))
    if free_mb is None:
        return base
    try:
        free = float(free_mb)
    except Exception:
        return base
    if free <= 0:
        return floor_val
    low = 1024.0
    high = 4096.0
    if free <= low:
        return floor_val
    if free >= high:
        return base
    ratio = (free - low) / (high - low)
    scaled = int(round(floor_val + ratio * (base - floor_val)))
    return max(floor_val, min(ceil_val, scaled))
