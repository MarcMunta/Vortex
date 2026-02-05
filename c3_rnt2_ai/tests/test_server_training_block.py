from __future__ import annotations

from pathlib import Path

import pytest


def test_training_active_blocks_requests(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {"core": {"backend": "vortex"}, "server": {"block_during_training": True}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    app.state.training_active = True
    app.state.maintenance_until = 0.0

    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert resp.json().get("error", {}).get("code") == "training_active"
    assert "Retry-After" in resp.headers
