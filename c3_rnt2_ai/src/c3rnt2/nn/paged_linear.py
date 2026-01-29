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
    ) -> None:
        super().__init__()
        self.store = store
        self.out_features = store.out_features
        self.in_features = store.in_features
        self.tile_out = store.tile_out
        self.tile_in = store.tile_in
        self.device = device
        self.cache = CacheManager(capacity_bytes=int(cache_budget_bytes))
        self.paged = PagedWeights(
            tile_store=store.tiles,
            cache=self.cache,
            device=device,
            prefetch_depth=prefetch_depth,
            pin_memory=pin_memory,
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
        )

    def stats(self) -> Dict[str, float]:
        return {
            "page_faults": float(self.paged.stats.page_faults),
            "bytes_h2d": float(self.paged.stats.bytes_h2d),
            "prefetch_hits": float(self.paged.stats.prefetch_hits),
            "bytes_compressed_read": float(self.paged.stats.compressed_bytes),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, orig_shape[-1])
        else:
            x_flat = x
        if x_flat.size(-1) != self.in_features:
            raise ValueError("PagedLinear input feature mismatch")

        out = x_flat.new_zeros(x_flat.size(0), self.out_features)
        out_tiles = self.store.out_tiles
        in_tiles = self.store.in_tiles
        bias = self.bias
        if bias is not None and bias.dtype != x_flat.dtype:
            bias = bias.to(dtype=x_flat.dtype)
        for out_idx in range(out_tiles):
            tile_ids = self.store.tile_ids_for_out(out_idx)
            if self.paged.prefetcher is not None:
                next_out = out_idx + 1
                if next_out < out_tiles:
                    self.paged.prefetch(self.store.tile_ids_for_out(next_out))
            tiles = self.paged.request_tiles(tile_ids)
            out_start = out_idx * self.tile_out
            out_end = min(self.out_features, out_start + self.tile_out)
            out_chunk = None
            for in_idx, tile in enumerate(tiles):
                in_start = in_idx * self.tile_in
                in_end = min(self.in_features, in_start + self.tile_in)
                x_slice = x_flat[:, in_start:in_end]
                if tile.dtype != x_flat.dtype:
                    tile = tile.to(dtype=x_flat.dtype)
                chunk = x_slice @ tile.transpose(0, 1)
                out_chunk = chunk if out_chunk is None else (out_chunk + chunk)
            if bias is not None:
                out_chunk = out_chunk + bias[out_start:out_end]
            out[:, out_start:out_end] = out_chunk
        if x.dim() == 3:
            return out.view(orig_shape[0], orig_shape[1], self.out_features)
        return out
