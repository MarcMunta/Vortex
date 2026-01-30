from __future__ import annotations

import json
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


def test_bootstrap_dataset_roundtrip(tmp_path: Path) -> None:
    settings = _minimal_settings(tmp_path)
    dataset_dir = tmp_path / "data" / "registry" / "bootstrap"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "bootstrap_samples.jsonl"
    samples = [
        {"prompt": "Write a function to add two numbers", "response": "def add(a, b):\\n    return a + b"},
        {"prompt": "Explain unit testing", "response": "Unit tests verify small pieces of code."},
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    result = run_bootstrap(
        settings=settings,
        base_dir=tmp_path,
        reuse_dataset=True,
        steps=0,
        batch_tokens=64,
        grad_accum_steps=1,
    )
    assert result.get("ok") is True
    adapter_path = tmp_path / "data" / "registry" / "adapters" / "bootstrap.pt"
    assert adapter_path.exists()
    meta_path = tmp_path / "data" / "registry" / "bootstrap.json"
    assert meta_path.exists()
