from __future__ import annotations

import torch

from c3rnt2.nn.paged_linear import PagedLinear


def test_paged_linear_dtype_path() -> None:
    linear = torch.nn.Linear(4, 6, bias=True)
    linear.weight.data = linear.weight.data.to(dtype=torch.float16)
    if linear.bias is not None:
        linear.bias.data = linear.bias.data.to(dtype=torch.float16)
    paged = PagedLinear.from_linear(linear, tile_out=3, tile_in=4, cache_budget_bytes=1024 * 1024)
    x = torch.randn(2, 4, dtype=torch.float32)
    out = paged(x)
    assert out.shape == (2, 6)
    assert out.dtype == x.dtype
