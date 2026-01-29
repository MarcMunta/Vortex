from __future__ import annotations

import torch

from c3rnt2.model.bad_decode import bad_decode, _should_restrict_escape
from c3rnt2.model.core_transformer import CoreTransformer


def _build_settings():
    return {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
        "bad": {"block_size": 4, "entropy_threshold": -1.0},
        "decode": {"escape_restrict": True, "exact_copy_mode": True},
    }


def test_bad_decode_no_crash():
    model = CoreTransformer.from_settings(_build_settings())
    text, stats = bad_decode(
        model,
        prompt="hello",
        max_new_tokens=4,
        block_size=2,
        entropy_threshold=-1.0,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram=0,
        adaptive_granularity=True,
        exact_copy_mode=True,
        escape_restrict=True,
    )
    assert isinstance(text, str)
    assert stats.proposed >= 0


def test_escape_restrict_rules():
    assert _should_restrict_escape(True, False, "exact")
    assert _should_restrict_escape(True, False, "exact-copy")
    assert not _should_restrict_escape(True, False, "escape")
    assert _should_restrict_escape(False, True, "")
    assert not _should_restrict_escape(False, False, "exact")


def test_bad_decode_cuda_smoke():
    if not torch.cuda.is_available():
        return
    model = CoreTransformer.from_settings(_build_settings())
    _text, _stats = bad_decode(
        model,
        prompt="cuda",
        max_new_tokens=2,
        block_size=2,
        entropy_threshold=-1.0,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram=0,
        adaptive_granularity=True,
        exact_copy_mode=True,
        escape_restrict=True,
    )
