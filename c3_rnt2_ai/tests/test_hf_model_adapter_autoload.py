from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def test_hf_model_loads_latest_adapter(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    from c3rnt2 import hf_model as hf_mod

    registry_dir = tmp_path / "data" / "registry" / "hf_train"
    registry_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = registry_dir / "run1" / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (registry_dir / "registry.json").write_text(json.dumps({"current_adapter": str(adapter_path)}), encoding="utf-8")

    class DummyModel:
        def __init__(self):
            self.model = object()
            self.cfg = types.SimpleNamespace(load_kwargs={})

    def _fake_try_load(_cfg):
        return DummyModel()

    class DummyPeft:
        @staticmethod
        def from_pretrained(model, path):
            return {"model": model, "adapter": path}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_mod, "_try_load", _fake_try_load)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=DummyPeft))

    settings = {"core": {"hf_model": "dummy", "hf_use_latest_adapter": True}}
    model = hf_mod.load_hf_model(settings)
    assert getattr(model, "adapter_path", None) == str(adapter_path)
