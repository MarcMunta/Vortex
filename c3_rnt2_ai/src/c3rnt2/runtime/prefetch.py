from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class Prefetcher:
    """Simple CPU->GPU prefetch scheduler (sync MVP)."""

    def __init__(
        self,
        loader: Callable[[int], object],
        depth: int = 2,
        device: str = "cpu",
        pin_memory: Optional[bool] = None,
        async_mode: Optional[bool] = None,
    ):
        self.loader = loader
        self.depth = depth
        self.queue: Deque[int] = deque()
        self.device = device
        self.pin_memory = pin_memory if pin_memory is not None else device.startswith("cuda")
        self.async_mode = async_mode if async_mode is not None else device.startswith("cuda")
        self.stream = None
        self._events: Deque[object] = deque()
        if torch is not None and device.startswith("cuda"):
            if self.async_mode:
                self.stream = torch.cuda.Stream()

    def schedule(self, tile_ids: Iterable[int]) -> None:
        for tile_id in tile_ids:
            if len(self.queue) >= self.depth:
                break
            self.queue.append(tile_id)

    def run(self) -> List[object]:
        loaded = []
        while self.queue:
            tile_id = self.queue.popleft()
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    obj = self.loader(tile_id)
                    loaded.append(obj)
                    if torch is not None and isinstance(obj, torch.Tensor):
                        event = torch.cuda.Event()
                        event.record(self.stream)
                        self._events.append(event)
            else:
                loaded.append(self.loader(tile_id))
        return loaded

    def synchronize(self) -> None:
        if self.stream is None or not self._events:
            return
        last = self._events.pop()
        try:
            last.synchronize()
        finally:
            self._events.clear()
