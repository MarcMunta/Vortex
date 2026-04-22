import pytest

from c3rnt2.continuous.lora import resolve_target_modules
from c3rnt2.training.hf_qlora import _resolve_lora_target_modules


def test_lora_target_modules_required() -> None:
    with pytest.raises(ValueError):
        resolve_target_modules({}, strict=True)
    targets = resolve_target_modules({}, strict=False)
    assert targets


def test_hf_qlora_resolves_supported_leaf_modules_from_wrapper_names() -> None:
    class _Wrapper:
        pass

    Linear = type("Linear", (), {})

    class _Model:
        def named_modules(self):
            return [
                ("language_model.layers.0.self_attn.q_proj", _Wrapper()),
                ("language_model.layers.0.self_attn.q_proj.linear", Linear()),
                ("language_model.layers.0.self_attn.k_proj", _Wrapper()),
                ("language_model.layers.0.self_attn.k_proj.linear", Linear()),
                ("language_model.layers.0.mlp.gate_proj", _Wrapper()),
                ("language_model.layers.0.mlp.gate_proj.linear", Linear()),
            ]

    targets = _resolve_lora_target_modules(
        _Model(),
        {
            "target_modules": ["q_proj", "k_proj", "gate_proj"],
            "target_module_scopes": ["language_model.layers."],
        },
    )

    assert "language_model.layers.0.self_attn.q_proj.linear" in targets
    assert "language_model.layers.0.self_attn.k_proj.linear" in targets
    assert "language_model.layers.0.mlp.gate_proj.linear" in targets
    assert "language_model.layers.0.self_attn.q_proj" not in targets
