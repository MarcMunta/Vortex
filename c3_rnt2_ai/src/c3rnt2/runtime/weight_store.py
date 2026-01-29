from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..compression.entropy_coder import compress


@dataclass
class WeightStore:
    tiles: Dict[int, object]
    out_features: int
    in_features: int
    tile_out: int
    tile_in: int
    out_tiles: int
    in_tiles: int
    dtype: str

    @classmethod
    def from_tensor(
        cls,
        weight: "torch.Tensor",
        tile_out: int,
        tile_in: int,
        compression: str | None = None,
    ) -> "WeightStore":
        if torch is None:
            raise RuntimeError("PyTorch not available")
        weight_cpu = weight.detach().to("cpu")
        if weight_cpu.dtype not in (torch.float16, torch.bfloat16):
            weight_cpu = weight_cpu.float()
        if weight_cpu.dtype == torch.bfloat16:
            weight_cpu = weight_cpu.to(dtype=torch.float16)
        arr = weight_cpu.numpy()
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        out_features, in_features = arr.shape
        tile_out = max(1, min(int(tile_out), out_features))
        tile_in = max(1, min(int(tile_in), in_features))
        out_tiles = (out_features + tile_out - 1) // tile_out
        in_tiles = (in_features + tile_in - 1) // tile_in
        tiles: Dict[int, object] = {}
        for out_idx in range(out_tiles):
            o_start = out_idx * tile_out
            o_end = min(out_features, o_start + tile_out)
            for in_idx in range(in_tiles):
                i_start = in_idx * tile_in
                i_end = min(in_features, i_start + tile_in)
                tile = np.ascontiguousarray(arr[o_start:o_end, i_start:i_end])
                tile_id = out_idx * in_tiles + in_idx
                if compression and compression != "none":
                    try:
                        comp = compress(tile.tobytes(), codec=compression)
                        tiles[tile_id] = {
                            "payload": comp.payload,
                            "codec": comp.codec,
                            "shape": tile.shape,
                            "nbytes": int(tile.nbytes),
                        }
                    except Exception:
                        tiles[tile_id] = tile
                else:
                    tiles[tile_id] = tile
        return cls(
            tiles=tiles,
            out_features=out_features,
            in_features=in_features,
            tile_out=tile_out,
            tile_in=tile_in,
            out_tiles=out_tiles,
            in_tiles=in_tiles,
            dtype=str(weight_cpu.dtype),
        )

    def tile_ids_for_out(self, out_idx: int) -> List[int]:
        base = out_idx * self.in_tiles
        return [base + in_idx for in_idx in range(self.in_tiles)]
