from __future__ import annotations

import torch

from c3rnt2.model.lava_memory import LAVAMemory


def test_lava_memory_cache_shapes_and_finite():
    mem = LAVAMemory(hidden_size=32, latent_slots=8, top_k=2)
    x = torch.randn(4, 32)
    out = mem.read_step(x)
    assert out.shape == x.shape
    assert mem.addresses_norm.shape == mem.addresses.shape
    assert mem.addresses_norm_t.shape == (32, 8)
    assert torch.isfinite(out).all()
    mem.write_step(torch.randn(4, 32))
    out2 = mem.read_step(x)
    assert torch.isfinite(out2).all()


def test_lava_ivf_shapes_no_nan():
    mem = LAVAMemory(hidden_size=32, latent_slots=16, top_k=4, lava_clusters=4, lava_cluster_top=2, ann_mode="ivf")
    x = torch.randn(2, 3, 32)
    out = mem.read_block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
