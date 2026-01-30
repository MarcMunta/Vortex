from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..compression.entropy_coder import decompress

try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

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


if triton is not None:  # pragma: no cover
    @triton.jit
    def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)


    def _triton_copy(inp: "torch.Tensor") -> "torch.Tensor":
        out = torch.empty_like(inp)
        n = inp.numel()
        grid = (triton.cdiv(n, 1024),)
        _copy_kernel[grid](inp, out, n, BLOCK=1024)
        return out
else:
    def _triton_copy(inp: "torch.Tensor") -> "torch.Tensor":
        return inp


def decompress_to_tensor(
    tile: Any,
    device: str = "cpu",
    codec: str | None = None,
    shape: Tuple[int, int] | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    backend: str = "none",
):
    """Decompress tile payload if needed and move to device."""
    if isinstance(tile, np.ndarray):
        tensor = _to_tensor(tile, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
    elif isinstance(tile, (bytes, bytearray)):
        if codec is None or shape is None:
            raise ValueError("codec and shape required for compressed tiles")
        raw = decompress(bytes(tile), codec=codec)
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        tensor = _to_tensor(arr, device=device, pin_memory=pin_memory, non_blocking=non_blocking)
    else:
        raise TypeError("Unsupported tile type")
    if backend == "triton" and torch is not None and triton is not None:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            return _triton_copy(tensor)
    return tensor
