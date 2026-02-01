from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Iterable, List, Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .cache_manager import CacheManager
from .gpu_decompress import decompress_to_tensor, DecompressStats
from .prefetch import Prefetcher


@dataclass
class PagedWeightsStats:
    page_faults: int = 0
    bytes_transferred: int = 0
    compressed_bytes: int = 0
    decompressed_bytes: int = 0
    bytes_h2d: int = 0
    prefetch_hits: int = 0
    bytes_decompressed: int = 0
    ms_cpu_decompress: float = 0.0
    ms_h2d: float = 0.0
    ms_triton_copy: float = 0.0


class PagedWeights:
    """Tile-based weight manager with CPU storage and GPU cache (MVP)."""

    def __init__(
        self,
        tile_store: Dict[int, Any],
        cache: CacheManager,
        device: str = "cpu",
        prefetch_depth: int = 2,
        pin_memory: bool | None = None,
        gpu_decompress: str = "none",
    ):
        self.tile_store = tile_store
        self.cache = cache
        self.device = device
        self.pin_memory = pin_memory if pin_memory is not None else device.startswith("cuda")
        self.non_blocking = device.startswith("cuda")
        self.gpu_decompress = gpu_decompress
        self.stats = PagedWeightsStats()
        self._prefetched: set[int] = set()
        self._prefetch_events: Dict[int, object] = {}
        self.prefetcher = Prefetcher(
            self._load_tile_payload,
            depth=prefetch_depth,
            device=device,
            pin_memory=self.pin_memory,
            async_mode=self.non_blocking,
        )

    def _load_tile_payload(self, tile_id: int):
        tile = self.tile_store[tile_id]
        codec = None
        shape = None
        compressed_bytes = 0
        tensor = None
        dec_stats = DecompressStats()
        if isinstance(tile, dict):
            payload = tile.get("payload")
            codec = tile.get("codec")
            shape = tuple(tile.get("shape")) if tile.get("shape") else None
            if codec and payload is not None:
                compressed_bytes = int(len(payload)) if isinstance(payload, (bytes, bytearray)) else int(tile.get("nbytes") or 0)
            else:
                compressed_bytes = int(tile.get("nbytes") or (len(payload) if payload is not None else 0))
            tensor = decompress_to_tensor(
                payload,
                device="cpu" if self.device.startswith("cuda") else self.device,
                codec=codec,
                shape=shape,
                pin_memory=self.pin_memory,
                non_blocking=self.non_blocking,
                backend=self.gpu_decompress,
                stats=dec_stats,
            )
        else:
            compressed_bytes = int(tile.nbytes)
            tensor = decompress_to_tensor(
                tile,
                device="cpu" if self.device.startswith("cuda") else self.device,
                pin_memory=self.pin_memory,
                non_blocking=self.non_blocking,
                backend=self.gpu_decompress,
                stats=dec_stats,
            )
        size_bytes = int(tensor.numel() * tensor.element_size()) if hasattr(tensor, "numel") else int(compressed_bytes)
        return {
            "tile_id": tile_id,
            "tensor": tensor,
            "size_bytes": size_bytes,
            "compressed_bytes": compressed_bytes,
            "decompress_stats": dec_stats,
        }

    def _cache_payload(self, payload: dict) -> object:
        tile_id = payload["tile_id"]
        tensor = payload["tensor"]
        size_bytes = int(payload["size_bytes"])
        compressed_bytes = int(payload["compressed_bytes"])
        dec_stats = payload.get("decompress_stats")
        if dec_stats is not None:
            self.stats.bytes_decompressed += int(getattr(dec_stats, "bytes_decompressed", 0))
            self.stats.ms_cpu_decompress += float(getattr(dec_stats, "ms_cpu_decompress", 0.0))
            self.stats.ms_triton_copy += float(getattr(dec_stats, "ms_triton_copy", 0.0))
        if isinstance(payload.get("ms_h2d"), (int, float)):
            self.stats.ms_h2d += float(payload.get("ms_h2d"))
        self.stats.bytes_transferred += compressed_bytes
        self.stats.compressed_bytes += compressed_bytes
        if hasattr(tensor, "numel"):
            decompressed = int(tensor.numel() * tensor.element_size())
            self.stats.decompressed_bytes += decompressed
            if hasattr(tensor, "device") and tensor.device.type == "cuda":
                self.stats.bytes_h2d += decompressed
                self.cache.record_transfer(compressed_bytes, decompressed)
        self.cache.put((tile_id,), tensor, size_bytes)
        return tensor

    def request_tiles(self, tile_ids: Iterable[int]) -> List[object]:
        result = []
        for tile_id in tile_ids:
            cached = self.cache.get((tile_id,))
            if cached is not None:
                if tile_id in self._prefetched:
                    self.stats.prefetch_hits += 1
                    event = self._prefetch_events.pop(tile_id, None)
                    if event is not None and torch is not None and torch.cuda.is_available():
                        try:
                            event.wait(torch.cuda.current_stream())
                        except Exception:
                            pass
                    self._prefetched.discard(tile_id)
                result.append(cached)
            else:
                self.stats.page_faults += 1
                payload = self._load_tile_payload(tile_id)
                tensor = payload["tensor"]
                if self.device.startswith("cuda") and hasattr(tensor, "device") and tensor.device.type == "cpu":
                    start = time.perf_counter()
                    tensor = tensor.to(self.device, non_blocking=True)
                    payload["ms_h2d"] = (time.perf_counter() - start) * 1000.0
                    payload["tensor"] = tensor
                result.append(self._cache_payload(payload))
        return result

    def prefetch(self, tile_ids: Iterable[int]) -> None:
        self.prefetcher.schedule(tile_ids)
        loaded = self.prefetcher.run()
        for payload in loaded:
            if isinstance(payload, dict) and "tile_id" in payload:
                tile_id = int(payload["tile_id"])
                if self.cache.get((tile_id,)) is None:
                    self._cache_payload(payload)
                    self._prefetched.add(tile_id)
                    event = payload.get("event") if isinstance(payload, dict) else None
                    if event is None:
                        event = self.prefetcher.pop_event(tile_id)
                    if event is not None:
                        self._prefetch_events[tile_id] = event
