from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional
    torch = None
    nn = None


@dataclass
class RNT2Codebook:
    block_size: int
    entries: List[bytes]
    _index: dict[bytes, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    @property
    def size(self) -> int:
        return len(self.entries)

    def lookup(self, code: int) -> bytes:
        return self.entries[code]

    def find(self, block: bytes) -> int | None:
        return self._index.get(block)

    def _rebuild_index(self) -> None:
        index: dict[bytes, int] = {}
        for idx, entry in enumerate(self.entries):
            if entry not in index:
                index[entry] = idx
        self._index = index

    @staticmethod
    def from_builtin(block_size: int = 64, size: int = 1024) -> "RNT2Codebook":
        # Simple built-in codebook of common ASCII patterns and whitespace blocks.
        entries: List[bytes] = []
        common = [
            b" " * block_size,
            (b"\n" * block_size)[:block_size],
            (b"0" * block_size)[:block_size],
            (b"1" * block_size)[:block_size],
            (b"a" * block_size)[:block_size],
            (b"{" + b" " * (block_size - 1))[:block_size],
            (b"}" + b" " * (block_size - 1))[:block_size],
        ]
        for item in common:
            if len(item) < block_size:
                item = item.ljust(block_size, b"\x00")
            entries.append(item)
        rng = np.random.default_rng(0)
        while len(entries) < size:
            block = rng.integers(0, 128, size=block_size, dtype=np.uint8).tobytes()
            entries.append(block)
        return RNT2Codebook(block_size=block_size, entries=entries[:size])

    @staticmethod
    def from_corpus(blocks: Iterable[bytes], block_size: int, size: int = 1024) -> "RNT2Codebook":
        from collections import Counter

        counter = Counter(blocks)
        most_common = [b for b, _ in counter.most_common(size)]
        if len(most_common) < size:
            # pad with deterministic random blocks
            rng = np.random.default_rng(0)
            while len(most_common) < size:
                most_common.append(rng.integers(0, 128, size=block_size, dtype=np.uint8).tobytes())
        return RNT2Codebook(block_size=block_size, entries=most_common[:size])


class RNT2Model:
    """Minimal RNT-2 model wrapper with optional neural components."""

    def __init__(self, codebook: RNT2Codebook):
        self.codebook = codebook
        self.encoder = None
        self.decoder = None

        if torch and nn:
            self.encoder = nn.Sequential(
                nn.Linear(codebook.block_size, 256),
                nn.ReLU(),
                nn.Linear(256, codebook.size),
            )
            self.decoder = nn.Sequential(
                nn.Embedding(codebook.size, 256),
                nn.Linear(256, codebook.block_size),
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "block_size": self.codebook.block_size,
            "entries": self.codebook.entries,
        }
        if torch:
            torch.save(payload, path)
        else:
            import pickle

            with path.open("wb") as f:
                pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path) -> "RNT2Model":
        path = Path(path)
        if torch:
            payload = torch.load(path, map_location="cpu")
        else:
            import pickle

            with path.open("rb") as f:
                payload = pickle.load(f)
        codebook = RNT2Codebook(block_size=payload["block_size"], entries=list(payload["entries"]))
        return RNT2Model(codebook=codebook)
