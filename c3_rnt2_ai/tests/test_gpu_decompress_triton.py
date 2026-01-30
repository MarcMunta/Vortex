import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from c3rnt2.runtime import gpu_decompress


@pytest.mark.skipif(gpu_decompress.triton is None or torch is None or not torch.cuda.is_available(), reason="triton/cuda not available")
def test_triton_decompress_equivalence():
    arr = np.random.randn(16, 16).astype(np.float16)
    t_cpu, _ = gpu_decompress.decompress_to_tensor(arr, device="cuda", backend="none")
    t_tri, _ = gpu_decompress.decompress_to_tensor(arr, device="cuda", backend="triton")
    assert torch.allclose(t_cpu, t_tri, atol=1e-3)
