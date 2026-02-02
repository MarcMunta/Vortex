from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_MISSING_NUMPY = importlib.util.find_spec("numpy") is None
_MISSING_TORCH = importlib.util.find_spec("torch") is None

_TORCH_TESTS = {
    "test_bad_decode.py",
    "test_bad_decode_mtp_shape.py",
    "test_bad_decode_no_unbound.py",
    "test_bootstrap_dataset_roundtrip.py",
    "test_bootstrap_guard.py",
    "test_bootstrap_teacher_missing_transformers.py",
    "test_bootstrap_teacher_quant.py",
    "test_checkpoint.py",
    "test_expert_training.py",
    "test_gpu_decompress_triton.py",
    "test_inference_stateful.py",
    "test_kv_quant_int8.py",
    "test_kv_quant_runtime.py",
    "test_lava_memory.py",
    "test_lora_targets.py",
    "test_paged_linear_dtype_path.py",
    "test_paged_lm_head.py",
    "test_router_training.py",
    "test_hf_weighted_sampling.py",
    "test_step_block.py",
    "test_trigger_skip.py",
}

_NUMPY_TESTS = {
    "test_gpu_decompress_stats.py",
    "test_gpu_decompress_triton.py",
    "test_kv_quant_int8.py",
    "test_tokenizer_exactness.py",
    "test_tokenizer_macro.py",
    "test_tokenizer_reversible.py",
}


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - env dependent
    path_str = str(collection_path).replace("\\", "/")
    if "/data/workspaces/" in path_str:
        return True
    name = getattr(collection_path, "name", None) or getattr(collection_path, "basename", "")
    if _MISSING_TORCH and name in _TORCH_TESTS:
        return True
    if _MISSING_NUMPY and name in _NUMPY_TESTS:
        return True
    return False
