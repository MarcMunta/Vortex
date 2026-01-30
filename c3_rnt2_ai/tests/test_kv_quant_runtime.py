from __future__ import annotations
import pytest

torch = pytest.importorskip("torch")

from c3rnt2.model.lava_memory import LAVAMemory
from c3rnt2.model.core_transformer import CoreTransformer


def _clone_lava_state(src: LAVAMemory, dst: LAVAMemory) -> None:
    with torch.no_grad():
        dst.addresses.copy_(src.addresses)
        dst.contents.copy_(src.contents)
        dst.age.copy_(src.age)
        dst.importance.copy_(src.importance)
    dst._refresh_address_cache()
    dst._refresh_quant_cache()


def test_kv_quant_int8_attention_path_close():
    mem = LAVAMemory(hidden_size=32, latent_slots=8, top_k=2, kv_quant="none")
    mem_q = LAVAMemory(hidden_size=32, latent_slots=8, top_k=2, kv_quant="int8")
    _clone_lava_state(mem, mem_q)
    x = torch.randn(4, 32)
    out_ref = mem.read_step(x)
    out_q = mem_q.read_step(x)
    assert out_q.shape == out_ref.shape
    assert torch.allclose(out_q, out_ref, atol=0.25)


def test_kv_quant_runtime_shapes_forward():
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
        "runtime": {"kv_quant": "int8"},
    }
    model = CoreTransformer.from_settings(settings)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model.forward(input_ids)
    assert logits.shape[-1] == model.config.vocab_size


def test_kv_quant_2bit_experimental_runs():
    mem_q = LAVAMemory(hidden_size=16, latent_slots=4, top_k=2, kv_quant="2bit")
    x = torch.randn(2, 16)
    out = mem_q.read_step(x)
    assert out.shape == x.shape
