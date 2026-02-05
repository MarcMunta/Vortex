from __future__ import annotations

import json
from pathlib import Path

import pytest


def _setup_app(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return "ok"

        def stream_generate(self, _prompt: str, **_kwargs):
            yield "ok"

    dummy = DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    return TestClient(app)


def test_health_and_ready(tmp_path: Path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.text.strip() == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json().get("ok") is True


def test_models_list(tmp_path: Path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "list"
    models = data.get("data") or []
    assert any(m.get("id") == "core" for m in models if isinstance(m, dict))


def test_chat_completions_non_stream(tmp_path: Path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "core", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "chat.completion"
    assert data.get("choices")[0]["message"]["content"] == "ok"
    usage = data.get("usage") or {}
    assert usage.get("prompt_tokens") is not None
    assert usage.get("completion_tokens") is not None
    assert usage.get("total_tokens") is not None


def test_chat_completions_stream_sse(tmp_path: Path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "core", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")

    deltas: list[str] = []
    for raw in resp.text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        evt = json.loads(payload)
        choice0 = (evt.get("choices") or [{}])[0]
        delta = (choice0.get("delta") or {}).get("content")
        if isinstance(delta, str) and delta:
            deltas.append(delta)

    assert "".join(deltas) == "ok"
    assert "data: [DONE]" in resp.text


def test_metrics_endpoint(tmp_path: Path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "vortex_up 1" in text
