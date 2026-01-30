from __future__ import annotations

import builtins
from pathlib import Path

from c3rnt2.continuous.bootstrap import run_bootstrap


def _minimal_settings(tmp_path: Path) -> dict:
    return {
        "tokenizer": {"vortex_tok_path": str(tmp_path / "tok.pt")},
        "core": {
            "hidden_size": 32,
            "layers": 2,
            "heads": 2,
            "vocab_size": 256,
        },
        "continuous": {
            "adapters": {"target_modules": ["lm_head"], "strict_target_modules": False},
            "eval": {"anchors_path": str(tmp_path / "anchors.jsonl")},
        },
    }


def test_bootstrap_teacher_missing_transformers(tmp_path: Path, monkeypatch) -> None:
    settings = _minimal_settings(tmp_path)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ModuleNotFoundError("transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = run_bootstrap(
        settings=settings,
        base_dir=tmp_path,
        teacher="Qwen/Qwen2.5-8B-Instruct",
        max_prompts=2,
        max_new_tokens=8,
        steps=1,
    )
    assert result.get("ok") is False
    assert "transformers" in str(result.get("error", "")).lower()
