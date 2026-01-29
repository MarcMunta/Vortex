from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Hashable, Optional


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_in: int = 0
    bytes_h2d: int = 0
    bytes_compressed: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class CacheEntry:
    value: object
    size_bytes: int
    stability: float = 0.0
    version: int = 0


class CacheManager:
    """LRU cache with stability bias for C3 tiles."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self._entries: Dict[Hashable, CacheEntry] = {}
        self._heap: list[tuple[float, int, Hashable]] = []
        self._version = 0
        self.metrics = CacheMetrics()

    def _touch(self, key: Hashable, entry: CacheEntry) -> None:
        self._version += 1
        entry.version = self._version
        heapq.heappush(self._heap, (entry.stability, entry.version, key))

    def get(self, key: Hashable) -> Optional[object]:
        entry = self._entries.get(key)
        if entry is not None:
            entry.stability += 0.1
            self._touch(key, entry)
            self.metrics.hits += 1
            return entry.value
        self.metrics.misses += 1
        return None

    def put(self, key: Hashable, value: object, size_bytes: int, stability: float = 0.0) -> None:
        if key in self._entries:
            self.current_bytes -= self._entries[key].size_bytes
            self._entries.pop(key, None)
        entry = CacheEntry(value=value, size_bytes=size_bytes, stability=stability)
        self._entries[key] = entry
        self.current_bytes += size_bytes
        self.metrics.bytes_in += int(size_bytes)
        self._touch(key, entry)
        self._evict_if_needed()

    def record_transfer(self, bytes_compressed: int, bytes_h2d: int) -> None:
        self.metrics.bytes_compressed += int(bytes_compressed)
        self.metrics.bytes_h2d += int(bytes_h2d)

    def _evict_if_needed(self) -> None:
        while self.current_bytes > self.capacity_bytes and self._entries:
            while self._heap:
                _stability, version, key = heapq.heappop(self._heap)
                entry = self._entries.get(key)
                if entry is None or entry.version != version:
                    continue
                self._entries.pop(key, None)
                self.current_bytes -= entry.size_bytes
                self.metrics.evictions += 1
                break
            else:
                break

    def stats(self) -> Dict[str, float]:
        return {
            "capacity_bytes": float(self.capacity_bytes),
            "current_bytes": float(self.current_bytes),
            "hit_rate": self.metrics.hit_rate,
            "hits": float(self.metrics.hits),
            "misses": float(self.metrics.misses),
            "evictions": float(self.metrics.evictions),
            "bytes_h2d": float(self.metrics.bytes_h2d),
            "bytes_compressed": float(self.metrics.bytes_compressed),
        }
