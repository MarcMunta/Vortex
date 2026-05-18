from __future__ import annotations

from pathlib import Path

import pytest


def _make_client(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
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
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    return TestClient(app)


def test_chat_sessions_sync_roundtrip(tmp_path: Path, monkeypatch) -> None:
    client = _make_client(tmp_path, monkeypatch)

    first_payload = {
        "account_id": "acc-1",
        "replace": True,
        "sessions": [
            {
                "id": "session-1",
                "title": "Primera",
                "updatedAt": 10,
                "messages": [
                    {"id": "m-1", "role": "user", "content": "hola", "timestamp": 1},
                    {"id": "m-2", "role": "ai", "content": "respuesta", "timestamp": 2},
                ],
            }
        ],
    }
    resp = client.post("/v1/chat/sessions/sync", json=first_payload)
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    listed = client.get("/v1/chat/sessions", params={"account_id": "acc-1"})
    assert listed.status_code == 200
    sessions = listed.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == "session-1"
    assert sessions[0]["messages"][0]["content"] == "hola"

    second_payload = {
        "account_id": "acc-1",
        "replace": True,
        "sessions": [
            {
                "id": "session-2",
                "title": "Segunda",
                "updatedAt": 20,
                "messages": [
                    {"id": "m-3", "role": "user", "content": "nuevo", "timestamp": 3},
                ],
            }
        ],
    }
    replaced = client.post("/v1/chat/sessions/sync", json=second_payload)
    assert replaced.status_code == 200
    assert replaced.json()["count"] == 1

    listed_after = client.get("/v1/chat/sessions", params={"account_id": "acc-1"})
    assert listed_after.status_code == 200
    sessions_after = listed_after.json()["sessions"]
    assert [session["id"] for session in sessions_after] == ["session-2"]

    deleted = client.delete("/v1/chat/sessions", params={"account_id": "acc-1"})
    assert deleted.status_code == 200
    assert client.get("/v1/chat/sessions", params={"account_id": "acc-1"}).json()["sessions"] == []


def test_chat_completions_injects_persistent_memory(tmp_path: Path, monkeypatch) -> None:
    client = _make_client(tmp_path, monkeypatch)

    sync_payload = {
        "account_id": "acc-ctx",
        "replace": True,
        "sessions": [
            {
                "id": "session-current",
                "title": "Infra Docker",
                "updatedAt": 30,
                "messages": [
                    {
                        "id": "old-1",
                        "role": "user",
                        "content": "Recuerda que el puerto del control plane es 8765 para Vortex.",
                        "timestamp": 10,
                    },
                ],
            },
            {
                "id": "session-older",
                "title": "Memoria Docker",
                "updatedAt": 20,
                "messages": [
                    {
                        "id": "old-2",
                        "role": "ai",
                        "content": "Docker Desktop estaba limitado a 30GB antes del cambio.",
                        "timestamp": 9,
                    },
                ],
            },
        ],
    }
    assert client.post("/v1/chat/sessions/sync", json=sync_payload).status_code == 200

    resp = client.post(
        "/v1/chat/completions",
        json={
            "account_id": "acc-ctx",
            "session_id": "session-current",
            "context_message_ids": [],
            "messages": [
                {
                    "role": "user",
                    "content": "Que puerto usabamos para el control y cuanta memoria tenia Docker?",
                }
            ],
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "CONVERSATION MEMORY" in content
    assert "8765" in content
    assert "30GB" in content


def test_workspace_folder_browser_lists_mapped_downloads(tmp_path: Path, monkeypatch) -> None:
    downloads_mount = tmp_path / "host" / "downloads"
    (downloads_mount / "test_flutter").mkdir(parents=True)
    monkeypatch.setenv("C3RNT2_HOST_DOWNLOADS_WINDOWS_ROOT", r"C:\Users\marcm\Downloads")
    monkeypatch.setenv("C3RNT2_HOST_DOWNLOADS_MOUNT", str(downloads_mount))
    client = _make_client(tmp_path, monkeypatch)

    roots = client.post("/v1/workspace/folders/list", json={})
    assert roots.status_code == 200
    root_entries = roots.json()["entries"]
    downloads_root = next(entry for entry in root_entries if entry["path"] == r"C:\Users\marcm\Downloads")

    children = client.post("/v1/workspace/folders/list", json={"path": downloads_root["path"]})

    assert children.status_code == 200
    assert any(entry["path"] == r"C:\Users\marcm\Downloads\test_flutter" for entry in children.json()["entries"])
