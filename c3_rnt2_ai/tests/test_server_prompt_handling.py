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


def test_hf_models_receive_messages_and_system(monkeypatch, tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyHFModel:
        is_hf = True

        def __init__(self) -> None:
            self.tokenizer = None
            self.last_prompt = None
            self.last_messages = None
            self.last_system = None

        def generate(self, prompt: str, **kwargs):
            self.last_prompt = prompt
            self.last_messages = kwargs.get("messages")
            self.last_system = kwargs.get("system")
            return "hf-ok"

    dummy = DummyHFModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    knowledge_path = tmp_path / "data" / "continuous" / "knowledge.sqlite"
    store = KnowledgeStore(knowledge_path)
    store.ingest_text("web", "local", "RAGCTX", quality=0.9)
    settings = {
        "core": {"backend": "hf", "hf_model": "google/gemma-3-12b-it", "hf_system_prompt": "SYS"},
        "rag": {"enabled": True, "top_k": 1, "max_chars": 100},
        "continuous": {"knowledge_path": str(knowledge_path)},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hola"}], "max_tokens": 4},
    )

    assert resp.status_code == 200
    assert dummy.last_system is not None
    assert dummy.last_system.startswith("SYS")
    assert "local_date:" in dummy.last_system
    assert dummy.last_messages == [{"role": "user", "content": "Hola"}]


def test_hf_models_receive_temporal_and_live_web_context(monkeypatch, tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyHFModel:
        is_hf = True

        def __init__(self) -> None:
            self.tokenizer = None
            self.last_prompt = None
            self.last_messages = None
            self.last_system = None

        def generate(self, prompt: str, **kwargs):
            self.last_prompt = prompt
            self.last_messages = kwargs.get("messages")
            self.last_system = kwargs.get("system")
            return "hf-ok"

    dummy = DummyHFModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)
    monkeypatch.setattr(
        server_mod,
        "_live_web_search_context",
        lambda *_args, **_kwargs: (
            "LIVE WEB RESULTS (use these as the freshest context for the current answer):\n[1] Example\nURL: https://example.com/match\nSnippet: Barca vs Atletico at 21:00.",
            [{"kind": "web", "ref": "https://example.com/match"}],
        ),
    )

    settings = {
        "core": {"backend": "hf", "hf_model": "google/gemma-4-E4B-it", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "A que hora juega hoy el Barca?"}],
            "max_tokens": 4,
            "web_ingest": True,
            "include_sources": True,
            "client_timezone": "Europe/Madrid",
            "client_now_iso": "2026-04-14T16:07:00+02:00",
        },
    )

    assert resp.status_code == 200
    assert dummy.last_system is not None
    assert "local_date: 2026-04-14" in dummy.last_system
    assert "timezone: Europe/Madrid" in dummy.last_system
    assert "LIVE WEB RESULTS" in dummy.last_system
    payload = resp.json()
    assert payload["sources"] == [{"kind": "web", "ref": "https://example.com/match"}]
