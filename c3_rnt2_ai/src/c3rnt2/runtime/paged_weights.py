from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np

from .cache_manager import CacheManager
from .gpu_decompress import decompress_to_tensor
from .prefetch import Prefetcher


@dataclass
class PagedWeightsStats:
    page_faults: int = 0
    bytes_transferred: int = 0
    compressed_bytes: int = 0
    decompressed_bytes: int = 0


class PagedWeights:
    """Tile-based weight manager with CPU storage and GPU cache (MVP)."""

    def __init__(
        self,
        tile_store: Dict[int, Any],
        cache: CacheManager,
        device: str = "cpu",
        prefetch_depth: int = 2,
        pin_memory: bool | None = None,
    ):
        self.tile_store = tile_store
        self.cache = cache
        self.device = device
        self.pin_memory = pin_memory if pin_memory is not None else device.startswith("cuda")
        self.non_blocking = device.startswith("cuda")
        self.stats = PagedWeightsStats()
        self.prefetcher = Prefetcher(
            self._load_tile,
            depth=prefetch_depth,
            device=device,
            pin_memory=self.pin_memory,
            async_mode=self.non_blocking,
        )

    def _load_tile(self, tile_id: int):
        tile = self.tile_store[tile_id]
        codec = None
        shape = None
        size_bytes = 0
        if isinstance(tile, dict):
            payload = tile.get("payload")
            codec = tile.get("codec")
            shape = tuple(tile.get("shape")) if tile.get("shape") else None
            size_bytes = int(tile.get("nbytes") or (len(payload) if payload is not None else 0))
            self.stats.bytes_transferred += size_bytes
            self.stats.compressed_bytes += size_bytes
            tensor = decompress_to_tensor(
                payload,
                device=self.device,
                codec=codec,
                shape=shape,
                pin_memory=self.pin_memory,
                non_blocking=self.non_blocking,
            )
        else:
            size_bytes = int(tile.nbytes)
            self.stats.bytes_transferred += size_bytes
            self.stats.compressed_bytes += size_bytes
            tensor = decompress_to_tensor(tile, device=self.device, pin_memory=self.pin_memory, non_blocking=self.non_blocking)
        if hasattr(tensor, "numel"):
            self.stats.decompressed_bytes += int(tensor.numel() * tensor.element_size())
        self.cache.put((tile_id,), tensor, size_bytes)
        return tensor

    def request_tiles(self, tile_ids: Iterable[int]) -> List[object]:
        result = []
        for tile_id in tile_ids:
            cached = self.cache.get((tile_id,))
            if cached is not None:
                result.append(cached)
            else:
                self.stats.page_faults += 1
                result.append(self._load_tile(tile_id))
        return result

    def prefetch(self, tile_ids: Iterable[int]) -> None:
        self.prefetcher.schedule(tile_ids)
        self.prefetcher.run()
