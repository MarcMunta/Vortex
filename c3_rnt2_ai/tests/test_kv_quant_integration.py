import torch

from c3rnt2.config import load_settings
from c3rnt2.model.core_transformer import CoreTransformer
from c3rnt2.model.lava_memory import LAVAMemory


def test_kv_quant_integration_forward():
    settings = load_settings("dev_small")
    runtime = settings.get("runtime", {}) or {}
    runtime["kv_quant"] = "int8"
    settings["runtime"] = runtime
    model = CoreTransformer.from_settings(settings)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
    logits = model.forward(ids)
    assert logits.shape[-1] == model.config.vocab_size


def test_kv_quant_int8_dequant_path():
    mem = LAVAMemory(hidden_size=2, latent_slots=2, top_k=1, kv_quant_bits=8, dtype="fp32")
    with torch.no_grad():
        mem.addresses.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        mem.contents.copy_(torch.tensor([[5.0, 5.0], [9.0, 9.0]]))
        mem.addr_proj.weight.copy_(torch.eye(2))
        mem.read_proj.weight.copy_(torch.eye(2))
    mem._refresh_address_cache()
    mem._quantize_slot(0)
    with torch.no_grad():
        mem.contents[0].zero_()
    expected = mem._dequantize_contents(torch.tensor([[0]]))[0, 0]
    out = mem.read_block(torch.tensor([[[1.0, 0.0]]]))
    assert torch.allclose(out[0, 0], expected, atol=1e-2)
