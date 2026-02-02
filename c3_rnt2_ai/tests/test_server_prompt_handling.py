from __future__ import annotations

from pathlib import Path

import pytest

from c3rnt2.continuous.knowledge_store import KnowledgeStore


def _setup_app(tmp_path: Path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.last_prompt = None

        def generate(self, prompt: str, **_kwargs):
            self.last_prompt = prompt
            return "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    knowledge_path = tmp_path / "data" / "continuous" / "knowledge.sqlite"
    store = KnowledgeStore(knowledge_path)
    store.ingest_text("web", "local", "RAGCTX", quality=0.9)
    settings = {
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": True, "top_k": 1, "max_chars": 100},
        "continuous": {"knowledge_path": str(knowledge_path)},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    return TestClient(app), dummy


def test_messages_ignore_prompt(monkeypatch, tmp_path: Path) -> None:
    client, model = _setup_app(tmp_path, monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "MSG_ONLY"}], "prompt": "SHOULD_NOT_APPEAR", "max_tokens": 4},
    )
    assert resp.status_code == 200
    assert model.last_prompt is not None
    assert "MSG_ONLY" in model.last_prompt
    assert "SHOULD_NOT_APPEAR" not in model.last_prompt
    assert "CONTEXT" in model.last_prompt
    assert "RAGCTX" in model.last_prompt


def test_prompt_only_injected(monkeypatch, tmp_path: Path) -> None:
    client, model = _setup_app(tmp_path, monkeypatch)
    resp = client.post("/v1/chat/completions", json={"prompt": "PROMPT_ONLY", "max_tokens": 4})
    assert resp.status_code == 200
    assert model.last_prompt is not None
    assert "PROMPT_ONLY" in model.last_prompt
    assert "CONTEXT" in model.last_prompt
    assert "RAGCTX" in model.last_prompt
