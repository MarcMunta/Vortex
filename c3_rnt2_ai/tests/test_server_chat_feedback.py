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

    feedback_path = tmp_path / "data" / "episodes" / "feedback.jsonl"
    fb_payload = json.loads(feedback_path.read_text(encoding="utf-8").splitlines()[-1])
    assert fb_payload["request_id"] == request_id
    assert fb_payload["rating"] == "up"

    training_path = tmp_path / "data" / "episodes" / "training.jsonl"
    training_payload = json.loads(training_path.read_text(encoding="utf-8").splitlines()[-1])
    assert training_payload["request_id"] == request_id
    assert training_payload["response"] == "better"
