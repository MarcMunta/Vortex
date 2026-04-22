from __future__ import annotations

from pathlib import Path

import pytest


def _make_client(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        tokenizer = None

        def generate(self, prompt: str, **_kwargs):
            return prompt

        def stream_generate(self, prompt: str, **_kwargs):
            yield prompt

    monkeypatch.setattr(
        server_mod,
        "_load_backend_model",
        lambda *_args, **_kwargs: DummyModel(),
    )

    settings = {
        "core": {"backend": "vortex", "hf_system_prompt": "You are a helpful assistant."},
        "rag": {"enabled": False},
        "chat_memory": {"enabled": True, "top_k": 6, "max_chars": 1800},
        "voice": {"enabled": True, "push_to_talk": True},
        "obsidian": {
            "enabled": True,
            "vault_path": str(tmp_path / "vault"),
            "folder_map": {
                "architecture": "Projects/Vortex/Architecture",
                "session": "Projects/Vortex/Sessions",
                "decision": "Projects/Vortex/Decisions",
                "prompt": "Projects/Vortex/Prompts",
                "bug": "Projects/Vortex/Bugs",
                "experiment": "Projects/Vortex/Experiments",
            },
        },
        "multimodal_memory": {
            "enabled": True,
            "state_path": str(tmp_path / "state" / "spatial.json"),
            "max_notes": 4,
            "max_chars": 1200,
        },
        "multimodal_context": {"enabled": True, "max_chars": 1800},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    return TestClient(app)


def test_voice_transcribe_opens_presentation_and_saves_obsidian(tmp_path: Path, monkeypatch) -> None:
    client = _make_client(tmp_path, monkeypatch)

    seeded = client.post(
        "/v1/spatial/session",
        json={
            "session_id": "spatial-smoke",
            "selected_region": {"x": 120, "y": 90, "width": 320, "height": 220},
            "selected_object_id": None,
            "active_panel_ids": [],
            "active_presentation_id": None,
            "active_page_index": 0,
            "interaction_mode": "inspect",
            "panels": [],
            "updated_at": 1,
            "created_at": 1,
        },
    )
    assert seeded.status_code == 200

    opened = client.post(
        "/v1/voice/transcribe",
        json={"text": "open this presentation here", "language": "en"},
    )
    assert opened.status_code == 200
    opened_payload = opened.json()
    assert opened_payload["intent"]["kind"] == "open_panel"
    assert opened_payload["action_result"]["ok"] is True
    opened_session = opened_payload["action_result"]["session"]
    assert len(opened_session["panels"]) == 1
    assert opened_session["panels"][0]["type"] == "presentation"
    assert opened_session["panels"][0]["transform"]["width"] == 320.0

    saved = client.post(
        "/v1/voice/transcribe",
        json={"text": "save this to obsidian", "language": "en"},
    )
    assert saved.status_code == 200
    saved_payload = saved.json()
    assert saved_payload["intent"]["kind"] == "save_obsidian"
    saved_path = Path(saved_payload["action_result"]["path"])
    assert saved_path.exists()
    assert saved_path.suffix == ".md"


def test_chat_completions_includes_multimodal_context(tmp_path: Path, monkeypatch) -> None:
    client = _make_client(tmp_path, monkeypatch)

    opened = client.post(
        "/v1/spatial/panels/open",
        json={
            "type": "note",
            "title": "Flutter plan",
            "content": "Implement Flutter and Dart workflow guidance.",
            "selected": True,
        },
    )
    assert opened.status_code == 200
    panel_id = opened.json()["panel"]["id"]

    event = client.post(
        "/v1/spatial/events",
        json={
            "kind": "voice",
            "panel_id": panel_id,
            "transcript": "talk to me about this",
        },
    )
    assert event.status_code == 200

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is my current workspace focus?"}],
            "max_tokens": 64,
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "Focused panel:" in content
    assert "Flutter plan" in content
    assert "Latest voice command: talk to me about this" in content
