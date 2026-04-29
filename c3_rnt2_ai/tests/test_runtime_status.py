from __future__ import annotations

from pathlib import Path

import pytest


def test_gemma_hf_status_does_not_report_sglang_port(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyHF:
        tokenizer = None
        is_hf = True

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda *_args, **_kwargs: DummyHF())
    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda _settings, base_dir=None: {
            "offline_ready": True,
            "engine_ready": True,
            "model_ready": True,
            "training_ready": True,
            "web_disabled": True,
            "docker_ready": True,
            "engine_kind": "hf",
            "engine_base_url": "http://127.0.0.1:30000",
            "active_model": "google/gemma-4-E4B-it",
        },
    )

    app = server_mod.create_app(
        {"core": {"backend": "hf", "hf_model": "google/gemma-4-E4B-it"}, "rag": {"enabled": False}},
        base_dir=tmp_path,
    )
    payload = TestClient(app).get("/v1/status").json()

    assert payload["engine_kind"] == "hf"
    assert payload["active_model"] == "google/gemma-4-E4B-it"
    assert "30000" not in str(payload.get("engine_base_url"))
    assert payload["status_source"] in {"runtime_real", "api_status"}


def test_legacy_sglang_status_can_report_30000(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyExternal:
        tokenizer = None
        is_external = True

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda *_args, **_kwargs: DummyExternal())
    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda _settings, base_dir=None: {
            "offline_ready": True,
            "engine_ready": True,
            "model_ready": True,
            "training_ready": True,
            "web_disabled": True,
            "docker_ready": True,
            "engine_kind": "sglang",
            "engine_base_url": "http://127.0.0.1:30000",
            "active_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        },
    )

    app = server_mod.create_app(
        {
            "core": {
                "backend": "external",
                "external_engine": "sglang",
                "external_base_url": "http://127.0.0.1:30000",
                "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            },
            "rag": {"enabled": False},
        },
        base_dir=tmp_path,
    )
    payload = TestClient(app).get("/v1/status").json()

    assert payload["engine_kind"] == "sglang"
    assert payload["engine_base_url"] == "http://127.0.0.1:30000"
