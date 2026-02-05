from __future__ import annotations

from dataclasses import dataclass
import time
from types import ModuleType
from typing import Any, Tuple, TYPE_CHECKING, cast

import numpy as np

from ..compression.entropy_coder import decompress

_torch: ModuleType | None
try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    triton = None
    tl = None

torch: ModuleType | None = _torch

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor as TorchTensor
else:
    TorchTensor = Any


@dataclass
class DecompressStats:
    bytes_decompressed: int = 0
    ms_cpu_decompress: float = 0.0
    ms_h2d: float = 0.0
    ms_triton_copy: float = 0.0


def _to_tensor(
    tile: np.ndarray,
    device: str = "cpu",
    pin_memory: bool = False,
    non_blocking: bool = False,
    stream: object | None = None,
) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch not available")
    torch_mod = cast(ModuleType, torch)
    if not tile.flags["C_CONTIGUOUS"]:
        tile = np.ascontiguousarray(tile)
    if not bool(tile.flags["WRITEABLE"]):
        tile = np.array(tile, copy=True)
    tensor = torch_mod.from_numpy(tile)
    if pin_memory:
        tensor = tensor.pin_memory()
    if device.startswith("cuda"):
        if stream is not None and hasattr(torch_mod.cuda, "stream"):
            with torch_mod.cuda.stream(stream):
                tensor = tensor.to(device, non_blocking=non_blocking)
        else:
            tensor = tensor.to(device, non_blocking=non_blocking)
    else:
        tensor = tensor.to(device, non_blocking=non_blocking)
    return tensor


if triton is not None:  # pragma: no cover

    @triton.jit
    def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)

    def _triton_copy(inp: TorchTensor) -> TorchTensor:
        torch_mod = cast(ModuleType, torch)
        out = torch_mod.empty_like(inp)
        n = inp.numel()
        grid = (triton.cdiv(n, 1024),)
        _copy_kernel[grid](inp, out, n, BLOCK=1024)
        return out

else:

    def _triton_copy(inp: TorchTensor) -> TorchTensor:
        return inp


def decompress_to_tensor(
    tile: Any,
    device: str = "cpu",
    codec: str | None = None,
    shape: Tuple[int, int] | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool | None = None,
    backend: str = "none",
    pinned: bool | None = None,
    stream: object | None = None,
) -> tuple[TorchTensor, DecompressStats]:
    """Decompress tile payload if needed and move to device."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    stats = DecompressStats()
    use_pin = pinned if pinned is not None else pin_memory
    if use_pin is None:
        use_pin = device.startswith("cuda")
    use_non_blocking = non_blocking if non_blocking is not None else device.startswith("cuda")

    if isinstance(tile, np.ndarray):
        arr = tile
    elif isinstance(tile, (bytes, bytearray)):
        if codec is None or shape is None:
            raise ValueError("codec and shape required for compressed tiles")
        start = time.perf_counter()
        raw = decompress(bytes(tile), codec=codec)
        stats.ms_cpu_decompress = (time.perf_counter() - start) * 1000.0
        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    else:
        raise TypeError("Unsupported tile type")

    stats.bytes_decompressed = int(arr.nbytes)
    tensor = _to_tensor(arr, device="cpu", pin_memory=bool(use_pin), non_blocking=False)

    if device != "cpu":
        start = time.perf_counter()
        torch_mod = cast(ModuleType, torch)
        if stream is not None and hasattr(torch_mod.cuda, "stream"):
            with torch_mod.cuda.stream(stream):
                tensor = tensor.to(device, non_blocking=bool(use_non_blocking))
        else:
            tensor = tensor.to(device, non_blocking=bool(use_non_blocking))
        stats.ms_h2d = (time.perf_counter() - start) * 1000.0

    if backend == "triton" and triton is not None and tensor.device.type == "cuda":
        start = time.perf_counter()
        tensor = _triton_copy(tensor)
        stats.ms_triton_copy = (time.perf_counter() - start) * 1000.0

    return tensor, stats
