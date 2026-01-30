import pytest
np = pytest.importorskip("numpy")

from c3rnt2.runtime.gpu_decompress import decompress_to_tensor


def test_gpu_decompress_stats_cpu():
    arr = np.random.randn(8, 8).astype(np.float16)
    tensor, stats = decompress_to_tensor(arr, device="cpu", backend="none")
    assert tensor.shape == arr.shape
    assert stats.bytes_decompressed == arr.nbytes
    assert stats.ms_cpu_decompress >= 0.0
