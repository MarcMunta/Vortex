from __future__ import annotations

from pathlib import Path

import pytest


def test_reload_adapter_endpoint(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        is_hf = True

        def __init__(self):
            self.tokenizer = None
            self.adapter_path = None

        def generate(self, _prompt: str, **_kwargs):
            return "ok"

        def load_adapter(self, path: str, merge: bool = False):
            _ = merge
            self.adapter_path = str(path)

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    adapter1 = tmp_path / "adapter1"
    adapter2 = tmp_path / "adapter2"
    adapter1.mkdir(parents=True, exist_ok=True)
    adapter2.mkdir(parents=True, exist_ok=True)
    idx = {"val": 0}

    def _fake_resolve(_base_dir, _settings):
        return adapter1 if idx["val"] == 0 else adapter2

    monkeypatch.setattr(server_mod, "_resolve_latest_adapter_path", _fake_resolve)

    settings = {"core": {"backend": "hf", "hf_use_latest_adapter": True}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    first = client.post("/v1/reload_adapter")
    assert first.status_code == 200
    assert dummy.adapter_path == str(adapter1)

    idx["val"] = 1
    second = client.post("/v1/reload_adapter")
    assert second.status_code == 200
    assert dummy.adapter_path == str(adapter2)
