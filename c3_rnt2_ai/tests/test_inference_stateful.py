from __future__ import annotations

import torch

from c3rnt2.model.core_transformer import CoreTransformer


def test_step_vs_forward():
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
    }
    model = CoreTransformer.from_settings(settings)
    ids = [1, 2, 3, 4]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits_full = model.forward(input_ids)
        last_full = logits_full[:, -1, :]
        last_logits, _state = model.init_state(prompt_ids=ids, return_logits=True, write_memory=False)
        assert last_logits is not None
        if last_full.dtype in (torch.float16, torch.bfloat16):
            atol = 5e-2
            rtol = 5e-2
        else:
            atol = 1e-4
            rtol = 1e-4
        assert torch.allclose(last_logits, last_full, atol=atol, rtol=rtol)


def test_cuda_inputs_on_cuda():
    if not torch.cuda.is_available():
        return
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
    }
    model = CoreTransformer.from_settings(settings)
    seen_devices = []

    original_forward = model.forward

    def wrapped_forward(input_ids, num_layers=None):
        seen_devices.append(input_ids.device.type)
        return original_forward(input_ids, num_layers=num_layers)

    model.forward = wrapped_forward  # type: ignore[method-assign]
    _ = model.full_next_logits([1, 2, 3])
    _ = model.draft_next_tokens([1, 2, 3], count=2)
    assert seen_devices
    assert all(dev == "cuda" for dev in seen_devices)
