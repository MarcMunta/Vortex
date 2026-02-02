from __future__ import annotations

import json
import sys
from pathlib import Path

from c3rnt2.training.hf_qlora import train_once


def test_train_once_accepts_training_events(tmp_path: Path, monkeypatch) -> None:
    episodes_dir = tmp_path / "data" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    training_path = episodes_dir / "training.jsonl"
    training_path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "prompt": "",
                "response": "ok",
                "source_kind": "chat_feedback",
                "quality": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _BlockedModule:
        def __getattr__(self, _name):  # pragma: no cover - used for import blocking
            raise ImportError("blocked")

    monkeypatch.setitem(sys.modules, "transformers", _BlockedModule())
    monkeypatch.setitem(sys.modules, "peft", _BlockedModule())

    settings = {"hf_train": {"model_name": "mock"}, "core": {"hf_model": "mock"}}
    result = train_once(settings, base_dir=tmp_path)
    assert result.error != "no_samples"
