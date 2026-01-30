from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class QuantizedKV:
    values: np.ndarray
    scale: float
    shape: Tuple[int, ...]
    bits: int


@dataclass
class KVHybridCache:
    window_size: int
    kv_quant_bits: int
    latent_slots: int
    exact_k: List[object] = field(default_factory=list)
    exact_v: List[object] = field(default_factory=list)
    quantized: List[QuantizedKV] = field(default_factory=list)
    latent: List[object] = field(default_factory=list)
    _latent_idx: int = 0

    def add(self, k, v):
        self.exact_k.append(k)
        self.exact_v.append(v)
        if len(self.exact_k) > self.window_size:
            old_k = self.exact_k.pop(0)
            old_v = self.exact_v.pop(0)
            self._quantize_and_store(old_k, old_v)

    def _quantize_int8(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        scale = float(np.max(np.abs(vec)) / 127.0) if vec.size else 1.0
        if scale == 0:
            scale = 1.0
        q = np.clip(vec / scale, -127, 127).astype(np.int8)
        return q, scale

    def _pack_2bit(self, q: np.ndarray) -> np.ndarray:
        # q expected in 0..3
        total = q.size
        padded = int(np.ceil(total / 4.0) * 4)
        if padded != total:
            q = np.pad(q, (0, padded - total), mode="constant", constant_values=0)
        q = q.reshape(-1, 4).astype(np.uint8)
        packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
        return packed

    def _unpack_2bit(self, packed: np.ndarray, total: int) -> np.ndarray:
        packed = packed.astype(np.uint8)
        q0 = packed & 0x3
        q1 = (packed >> 2) & 0x3
        q2 = (packed >> 4) & 0x3
        q3 = (packed >> 6) & 0x3
        q = np.stack([q0, q1, q2, q3], axis=1).reshape(-1)
        return q[:total]

    def _quantize_2bit(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        # Experimental 2-bit quantization. TODO: per-channel scaling and better packing.
        scale = float(np.max(np.abs(vec)) / 1.5) if vec.size else 1.0
        if scale == 0:
            scale = 1.0
        q = np.round(vec / scale).astype(np.int8)
        q = np.clip(q, -2, 1)
        q = (q + 2).astype(np.uint8)
        packed = self._pack_2bit(q)
        return packed, scale

    def _dequantize(self, quant: QuantizedKV) -> np.ndarray:
        if quant.bits == 8:
            return quant.values.astype(np.float32) * quant.scale
        if quant.bits == 2:
            total = int(np.prod(quant.shape))
            unpacked = self._unpack_2bit(quant.values, total).astype(np.int8) - 2
            return unpacked.astype(np.float32) * quant.scale
        return quant.values.astype(np.float32)

    def _quantize_and_store(self, k, v):
        if torch is None:
            return
        if self.kv_quant_bits <= 0:
            return
        vec = torch.cat([k.flatten(), v.flatten()]).detach().cpu().numpy()
        if self.kv_quant_bits == 8:
            q, scale = self._quantize_int8(vec)
        elif self.kv_quant_bits == 2:
            q, scale = self._quantize_2bit(vec)
        else:
            q, scale = self._quantize_int8(vec)
        self.quantized.append(QuantizedKV(values=q, scale=scale, shape=vec.shape, bits=self.kv_quant_bits))
        self._update_latent(vec)

    def _update_latent(self, vec: np.ndarray):
        if torch is None:
            return
        if self.latent_slots <= 0:
            return
        tensor = torch.tensor(vec, dtype=torch.float32)
        if len(self.latent) < self.latent_slots:
            self.latent.append(tensor)
        else:
            self.latent[self._latent_idx] = tensor
            self._latent_idx = (self._latent_idx + 1) % self.latent_slots

    def get_context(self) -> Tuple[List[object], List[object], List[object]]:
        return self.exact_k, self.exact_v, self.latent

    def dequantize_latest(self) -> np.ndarray | None:
        if not self.quantized:
            return None
        return self._dequantize(self.quantized[-1])
