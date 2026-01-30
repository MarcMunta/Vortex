from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from ..runtime.cache_manager import CacheManager
from ..runtime.paged_weights import PagedWeights
from ..runtime.weight_store import WeightStore


@dataclass
class PagedLinearStats:
    page_faults: int = 0
    bytes_h2d: int = 0
    prefetch_hits: int = 0
    bytes_compressed_read: int = 0


class PagedLinear(nn.Module):
    """Linear layer backed by paged weights (CPU storage + GPU cache)."""

    def __init__(
        self,
        store: WeightStore,
        bias: torch.Tensor | None,
        cache_budget_bytes: int,
        device: str = "cpu",
        prefetch_depth: int = 2,
        pin_memory: bool = True,
        accum_fp32: bool = False,
        gpu_decompress: str = "none",
    ) -> None:
        super().__init__()
        self.store = store
        self.out_features = store.out_features
        self.in_features = store.in_features
        self.tile_out = store.tile_out
        self.tile_in = store.tile_in
        self.device = device
        self.accum_fp32 = accum_fp32
        self.cache = CacheManager(capacity_bytes=int(cache_budget_bytes))
        self.paged = PagedWeights(
            tile_store=store.tiles,
            cache=self.cache,
            device=device,
            prefetch_depth=prefetch_depth,
            pin_memory=pin_memory,
            gpu_decompress=gpu_decompress,
        )
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        tile_out: int = 128,
        tile_in: int | None = None,
        cache_budget_bytes: int = 1024 * 1024 * 1024,
        compression: str | None = None,
        device: str = "cpu",
        prefetch_depth: int = 2,
        pin_memory: bool = True,
        accum_fp32: bool = False,
        gpu_decompress: str = "none",
    ) -> "PagedLinear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects nn.Linear")
        tile_in = tile_in or linear.weight.size(1)
        store = WeightStore.from_tensor(
            linear.weight,
            tile_out=tile_out,
            tile_in=tile_in,
            compression=compression,
        )
        return cls(
            store=store,
            bias=linear.bias,
            cache_budget_bytes=cache_budget_bytes,
            device=device,
            prefetch_depth=prefetch_depth,
            pin_memory=pin_memory,
            accum_fp32=accum_fp32,
            gpu_decompress=gpu_decompress,
        )


    def _store_compute_dtype(self) -> torch.dtype:
        dtype_str = str(self.store.dtype).lower()
        if "bfloat16" in dtype_str or "bf16" in dtype_str:
            return torch.bfloat16
        if "float16" in dtype_str or "half" in dtype_str or "fp16" in dtype_str:
            return torch.float16
        if "float32" in dtype_str or "fp32" in dtype_str:
            return torch.float32
        return torch.float32

    def forward_topk(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if k <= 0:
            raise ValueError("k must be > 0")
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, orig_shape[-1])
        else:
            x_flat = x
        if x_flat.size(-1) != self.in_features:
            raise ValueError("PagedLinear input feature mismatch")
        if k >= self.out_features:
            logits = self.forward(x)
            values, indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
            return values, indices

        compute_dtype = torch.float32 if self.accum_fp32 else self._store_compute_dtype()
        x_compute = x_flat if x_flat.dtype == compute_dtype else x_flat.to(dtype=compute_dtype)
        out_tiles = self.store.out_tiles
        in_tiles = self.store.in_tiles
        bias = self.bias
        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(dtype=compute_dtype)
        top_vals = None
        top_idx = None
        for out_idx in range(out_tiles):
            tile_ids = self.store.tile_ids_for_out(out_idx)
            if self.paged.prefetcher is not None:
                next_out = out_idx + 1
                if next_out < out_tiles:
                    self.paged.prefetch(self.store.tile_ids_for_out(next_out))
            tiles = self.paged.request_tiles(tile_ids)
            out_start = out_idx * self.tile_out
            out_end = min(self.out_features, out_start + self.tile_out)
            need_cast = hasattr(tiles[0], "dtype") and tiles[0].dtype != compute_dtype
            if need_cast:
                tiles = [tile.to(dtype=compute_dtype) for tile in tiles]
            if self.tile_in >= self.in_features or in_tiles == 1:
                weight = tiles[0]
                out_chunk = torch.matmul(x_compute, weight.transpose(0, 1))
            else:
                weight_full = torch.cat(tiles, dim=1)
                out_chunk = torch.matmul(x_compute, weight_full.transpose(0, 1))
            if bias is not None:
                out_chunk = out_chunk + bias[out_start:out_end]
            vals, idx = torch.topk(out_chunk, k=min(k, out_chunk.size(-1)), dim=-1)
            idx = idx + out_start
            if top_vals is None:
                top_vals = vals
                top_idx = idx
            else:
                cat_vals = torch.cat([top_vals, vals], dim=-1)
                cat_idx = torch.cat([top_idx, idx], dim=-1)
                top_vals, top_pos = torch.topk(cat_vals, k=k, dim=-1)
                top_idx = cat_idx.gather(1, top_pos)
        if top_vals is None or top_idx is None:
            raise RuntimeError("topk computation failed")
        return top_vals, top_idx

    def stats(self) -> Dict[str, float]:
        return {
            "page_faults": float(self.paged.stats.page_faults),
            "bytes_h2d": float(self.paged.stats.bytes_h2d),
            "prefetch_hits": float(self.paged.stats.prefetch_hits),
            "bytes_compressed_read": float(self.paged.stats.compressed_bytes),
            "bytes_decompressed": float(self.paged.stats.bytes_decompressed),
            "ms_cpu_decompress": float(self.paged.stats.ms_cpu_decompress),
            "ms_h2d": float(self.paged.stats.ms_h2d),
            "ms_triton_copy": float(self.paged.stats.ms_triton_copy),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, orig_shape[-1])
        else:
            x_flat = x
        if x_flat.size(-1) != self.in_features:
            raise ValueError("PagedLinear input feature mismatch")

        compute_dtype = torch.float32 if self.accum_fp32 else self._store_compute_dtype()
        x_compute = x_flat if x_flat.dtype == compute_dtype else x_flat.to(dtype=compute_dtype)
        out_tiles = self.store.out_tiles
        in_tiles = self.store.in_tiles
        bias = self.bias
        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(dtype=compute_dtype)
        out_chunks = []
        for out_idx in range(out_tiles):
            tile_ids = self.store.tile_ids_for_out(out_idx)
            if self.paged.prefetcher is not None:
                next_out = out_idx + 1
                if next_out < out_tiles:
                    self.paged.prefetch(self.store.tile_ids_for_out(next_out))
            tiles = self.paged.request_tiles(tile_ids)
            out_start = out_idx * self.tile_out
            out_end = min(self.out_features, out_start + self.tile_out)
            need_cast = hasattr(tiles[0], "dtype") and tiles[0].dtype != compute_dtype
            if need_cast:
                tiles = [tile.to(dtype=compute_dtype) for tile in tiles]
            if self.tile_in >= self.in_features or in_tiles == 1:
                weight = tiles[0]
                out_chunk = torch.matmul(x_compute, weight.transpose(0, 1))
            else:
                weight_full = torch.cat(tiles, dim=1)
                out_chunk = torch.matmul(x_compute, weight_full.transpose(0, 1))
            if bias is not None:
                out_chunk = out_chunk + bias[out_start:out_end]
            out_chunks.append(out_chunk)
        out = torch.cat(out_chunks, dim=1)
        if out.dtype != x_flat.dtype:
            out = out.to(dtype=x_flat.dtype)
        if x.dim() == 3:
            return out.view(orig_shape[0], orig_shape[1], self.out_features)
        return out
