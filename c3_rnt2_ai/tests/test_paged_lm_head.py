from __future__ import annotations

import torch

from c3rnt2.model.core_transformer import CoreTransformer


def test_paged_lm_head_smoke():
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
        "runtime": {"paged_lm_head": True, "paged_tile_out": 64, "cache_vram_budget_mb": 8},
    }
    model = CoreTransformer.from_settings(settings)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model.forward(input_ids)
    assert logits.shape[-1] == model.config.vocab_size
