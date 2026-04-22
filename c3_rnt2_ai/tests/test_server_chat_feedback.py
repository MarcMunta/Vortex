from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_chat_episode_and_feedback(tmp_path: Path, monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi")
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

    settings = {"core": {"backend": "vortex", "hf_system_prompt": "You are a helpful assistant."}, "rag": {"enabled": False}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert resp.status_code == 200
    data = resp.json()
    request_id = data.get("request_id")
    assert request_id

    chat_path = tmp_path / "data" / "episodes" / "chat.jsonl"
    payload = json.loads(chat_path.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["request_id"] == request_id
    assert payload["prompt_text"]
    assert payload["response_text"] == "ok"

    fb = client.post("/v1/feedback", json={"request_id": request_id, "rating": "up", "ideal_response": "better"})
    assert fb.status_code == 200
    fb_data = fb.json()
    assert fb_data["ok"] is True
    assert fb_data["training_event"] is True
    assert fb_data["learning_queue_item"]["status"] == "queued"
    assert fb_data["learning_queue_item"]["source_kind"] == "chat_feedback"
    assert fb_data["learning_queue_depth"] == 1
    assert fb_data["quick_train_scheduled"] is False
    assert fb_data["queue_reason"] == "below_threshold"

    feedback_path = tmp_path / "data" / "episodes" / "feedback.jsonl"
    fb_payload = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[-1])
    assert fb_payload["request_id"] == request_id
    assert fb_payload["rating"] == "up"

    training_path = tmp_path / "data" / "episodes" / "training.jsonl"
    training_payload = json.loads(training_path.read_text(encoding="utf-8").splitlines()[-1])
    assert training_payload["request_id"] == request_id
    assert training_payload["response"] == "better"

    queue_path = tmp_path / "data" / "control" / "learning_queue.jsonl"
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8").splitlines()[-1])
    assert queue_payload["request_id"] == request_id
    assert queue_payload["source_kind"] == "chat_feedback"
    assert queue_payload["status"] == "queued"

    queue_state_path = tmp_path / "data" / "control" / "learning_queue_state.json"
    queue_state = json.loads(queue_state_path.read_text(encoding="utf-8"))
    assert queue_payload["id"] in queue_state["items"]
    assert queue_state["items"][queue_payload["id"]]["status"] == "queued"


def test_status_reports_chat_ready_when_stack_is_degraded(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        tokenizer = None

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda _settings, base_dir=None: {
            "offline_ready": False,
            "engine_ready": True,
            "model_ready": True,
            "training_ready": True,
            "web_disabled": False,
            "docker_ready": True,
            "degraded_reason": "maintenance_mode",
            "offline_reason": "offline_cache_missing",
            "engine_reason": None,
            "model_reason": None,
            "training_reason": None,
            "docker_reason": None,
            "engine_kind": "hf",
            "engine_base_url": "http://127.0.0.1:8000",
            "active_model": "Qwen2.5-7B-Instruct",
        },
    )

    settings = {"core": {"backend": "vortex"}, "rag": {"enabled": False}}
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/status")
    assert resp.status_code in {200, 503}
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["chat_ready"] is True
    assert payload["chat_mode"] == "fallback_degraded"
    assert payload["chat_block_reason"] is None


def test_chat_model_local_alias_uses_default_backend(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyTokenizer:
        def apply_chat_template(self, _messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return "<templated>"

    class DummyModel:
        def __init__(self, backend: str) -> None:
            self.backend = backend
            self.tokenizer = DummyTokenizer() if backend == "hf" else None

        def generate(self, prompt: str, **_kwargs):
            return f"{self.backend}:{prompt}"

    loads: list[str] = []

    def _fake_load_backend_model(_settings, _base_dir, backend):
        loads.append(str(backend))
        return DummyModel(str(backend))

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)

    settings = {
        "core": {"backend": "hf", "hf_system_prompt": "You are a helpful assistant."},
        "rag": {"enabled": False},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model"] == "hf"
    assert payload["choices"][0]["message"]["content"] == "hf:<templated>"
    assert loads == ["hf"]
