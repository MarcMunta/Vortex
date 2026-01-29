from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..compression.entropy_coder import decompress

try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

torch: Any = _torch


def _to_tensor(tile: np.ndarray, device: str = "cpu", pin_memory: bool = False, non_blocking: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch not available")
    if not tile.flags["C_CONTIGUOUS"]:
        tile = np.ascontiguousarray(tile)
    tensor = torch.from_numpy(tile)
    if pin_memory:
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=non_blocking)


def decompress_to_tensor(
    tile: Any,
    device: str = "cpu",
    codec: str | None = None,
    shape: Tuple[int, int] | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
):
    """Decompress tile payload if needed and move to device."""
    if isinstance(tile, np.ndarray):
        return _to_tensor(tile, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
    if isinstance(tile, (bytes, bytearray)):
        if codec is None or shape is None:
            raise ValueError("codec and shape required for compressed tiles")
        raw = decompress(bytes(tile), codec=codec)
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        return _to_tensor(arr, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
    raise TypeError("Unsupported tile type")
