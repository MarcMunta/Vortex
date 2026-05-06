from __future__ import annotations

import json
from pathlib import Path

import pytest


def _client(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyHF:
        is_hf = True
        tokenizer = None

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, prompt: str, **kwargs):
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            return "```dart\nvoid main() {}\n```"

        def stream_generate(self, prompt: str, **kwargs):
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            yield "```dart\nvoid main() {}\n```"

    dummy = DummyHF()
    rag_calls: list[bool] = []
    web_calls: list[bool] = []

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda *_args, **_kwargs: dummy)
    monkeypatch.setattr(
        server_mod,
        "_inject_rag_context",
        lambda *args, **kwargs: (rag_calls.append(True) or (args[2], None, {"refs": [{"kind": "self_code", "ref": "src/local.py"}]})),
    )
    monkeypatch.setattr(
        server_mod,
        "_live_web_search_context",
        lambda *args, **kwargs: (web_calls.append(True) or ("WEB", [{"kind": "web", "ref": "https://duckduckgo.com/"}])),
    )
    app = server_mod.create_app(
        {
            "core": {"backend": "hf", "hf_model": "google/gemma-4-E4B-it", "hf_system_prompt": "SYS"},
            "decode": {"max_new_tokens": 512, "default_code_max_new_tokens": 1024, "hard_max_new_tokens": 2048},
            "generation": {"default_max_tokens": 512, "code_max_tokens": 1024, "hard_max_tokens": 2048},
            "rag": {"enabled": True},
            "multimodal_context": {"enabled": False},
            "chat_memory": {"enabled": False},
        },
        base_dir=tmp_path,
    )
    return TestClient(app), dummy, rag_calls, web_calls


def test_flutter_code_request_uses_code_budget_and_no_sources(tmp_path: Path, monkeypatch) -> None:
    client, dummy, rag_calls, web_calls = _client(tmp_path, monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "google/gemma-4-E4B-it",
            "stream": False,
            "include_sources": False,
            "web_ingest": False,
            "rag_mode": "off",
            "response_mode": "code",
            "code_language": "dart",
            "require_complete_code": True,
            "messages": [{"role": "user", "content": "Crea un login básico en Flutter"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "sources" not in data
    assert rag_calls == []
    assert web_calls == []
    assert dummy.calls[-1]["kwargs"]["max_new_tokens"] >= 1024
    assert data["max_tokens_effective"] >= 1024
    assert data["backend"] == "hf"
    assert data["active_model"] == "google/gemma-4-E4B-it"


def test_include_sources_false_suppresses_rag_refs_even_when_rag_enabled(tmp_path: Path, monkeypatch) -> None:
    client, _dummy, rag_calls, _web_calls = _client(tmp_path, monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Crea codigo Dart"}],
            "max_tokens": 4096,
            "include_sources": False,
            "web_ingest": False,
        },
    )

    assert resp.status_code == 200
    assert "sources" not in resp.json()
    assert rag_calls == []


def test_stream_finish_reason_length_is_propagated(tmp_path: Path, monkeypatch) -> None:
    client, _dummy, _rag_calls, _web_calls = _client(tmp_path, monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": True,
            "include_sources": False,
        },
    )

    done = None
    for raw in resp.text.splitlines():
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            break
        evt = json.loads(payload)
        if (evt.get("choices") or [{}])[0].get("finish_reason"):
            done = evt
    assert done is not None
    assert done["choices"][0]["finish_reason"] == "length"
