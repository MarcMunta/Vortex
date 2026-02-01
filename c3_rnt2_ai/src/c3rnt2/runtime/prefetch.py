from __future__ import annotations

from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Iterable, List, Optional

try:
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None

torch: Any = _torch


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
        self.stream: object | None = None
        self._events: Deque[object] = deque()
        self._event_map: dict[int, object] = {}
        self._executor: ThreadPoolExecutor | None = None
        if torch is not None and device.startswith("cuda"):
            if self.async_mode:
                self.stream = torch.cuda.Stream()
                self._executor = ThreadPoolExecutor(max_workers=max(1, depth))

    def _maybe_to_device(self, obj: object) -> object:
        if torch is None or not self.device.startswith("cuda"):
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cpu":
                start = time.perf_counter()
                out = obj.to(self.device, non_blocking=True)
                ms = (time.perf_counter() - start) * 1000.0
                return out, ms
            return obj
        if isinstance(obj, dict) and "tensor" in obj:
            tensor = obj.get("tensor")
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                start = time.perf_counter()
                obj["tensor"] = tensor.to(self.device, non_blocking=True)
                obj["ms_h2d"] = (time.perf_counter() - start) * 1000.0
            return obj
        return obj

    def schedule(self, tile_ids: Iterable[int]) -> None:
        for tile_id in tile_ids:
            if len(self.queue) >= self.depth:
                break
            self.queue.append(tile_id)

    def run(self) -> List[object]:
        loaded = []
        if self._executor is not None:
            futures = [self._executor.submit(self.loader, tile_id) for tile_id in list(self.queue)]
            tile_list = list(self.queue)
            self.queue.clear()
            for tile_id, fut in zip(tile_list, futures):
                obj = fut.result()
                stream = self.stream
                if stream is not None:
                    with torch.cuda.stream(stream):
                        obj = self._maybe_to_device(obj)
                        if isinstance(obj, tuple):
                            obj, ms_h2d = obj
                        else:
                            ms_h2d = None
                        event = torch.cuda.Event()
                        event.record(stream)
                        self._events.append(event)
                        self._event_map[int(tile_id)] = event
                        if isinstance(obj, dict):
                            obj.setdefault("tile_id", tile_id)
                            obj["event"] = event
                            if ms_h2d is not None:
                                obj["ms_h2d"] = ms_h2d
                loaded.append(obj)
            return loaded
        while self.queue:
            tile_id = self.queue.popleft()
            stream = self.stream
            if stream is not None:
                with torch.cuda.stream(stream):
                    obj = self.loader(tile_id)
                    obj = self._maybe_to_device(obj)
                    if isinstance(obj, tuple):
                        obj, ms_h2d = obj
                    else:
                        ms_h2d = None
                    event = torch.cuda.Event()
                    event.record(stream)
                    self._events.append(event)
                    self._event_map[int(tile_id)] = event
                    if isinstance(obj, dict):
                        obj.setdefault("tile_id", tile_id)
                        obj["event"] = event
                        if ms_h2d is not None:
                            obj["ms_h2d"] = ms_h2d
                    loaded.append(obj)
            else:
                loaded.append(self.loader(tile_id))
        return loaded

    def pop_event(self, tile_id: int) -> object | None:
        return self._event_map.pop(int(tile_id), None)

    def synchronize(self) -> None:
        if self.stream is None or not self._events:
            return
        last = self._events.pop()
        try:
            last.synchronize()
        finally:
            self._events.clear()
            self._event_map.clear()
