import pytest
import numpy as np

torch = pytest.importorskip("torch")

from c3rnt2.model.kv_hybrid import KVHybridCache


def test_kv_quant_int8_roundtrip():
    cache = KVHybridCache(window_size=0, kv_quant_bits=8, latent_slots=0)
    k = torch.randn(2, 2)
    v = torch.randn(2, 2)
    cache.add(k, v)
    deq = cache.dequantize_latest()
    assert deq is not None
    orig = torch.cat([k.flatten(), v.flatten()]).numpy()
    assert np.allclose(deq, orig, atol=0.1)