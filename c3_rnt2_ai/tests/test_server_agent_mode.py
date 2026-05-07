from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _setup_app(tmp_path: Path, monkeypatch, *, model=None, settings=None):
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

    dummy = model or DummyModel()

    def _fake_load_backend_model(_settings, _base_dir, _backend):
        return dummy

    monkeypatch.setattr(server_mod, "_load_backend_model", _fake_load_backend_model)
    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda settings, base_dir=None: {
            "ok": True,
            "offline_ready": True,
            "engine_ready": True,
            "engine_kind": "vortex",
            "engine_base_url": None,
            "model_ready": True,
            "active_backend": "core",
            "active_model": "core",
            "training_ready": True,
            "web_disabled": True,
            "docker_ready": True,
            "degraded_reason": None,
            "offline_reason": "offline_ready",
            "engine_reason": "engine_ready",
            "model_reason": "model_ready",
            "training_reason": "training_ready",
            "docker_reason": "docker_not_required",
            "wsl_ready": True,
            "wsl_reason": "wsl_not_required",
            "ollama_ready": None,
            "ollama_reason": None,
        },
    )

    settings = settings or {
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
        "agent": {"max_iters": 4},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    return TestClient(app), dummy, server_mod


def test_chat_completions_agent_mode_non_stream(tmp_path: Path, monkeypatch) -> None:
    client, dummy, server_mod = _setup_app(tmp_path, monkeypatch)
    seen: dict[str, object] = {}
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)

    def _fake_run_agent(task, settings, base_dir, **kwargs):
        seen["task"] = task
        seen["settings"] = settings
        seen["base_dir"] = base_dir
        seen["model"] = kwargs.get("model")
        seen["model_lock"] = kwargs.get("model_lock")
        seen["permissions"] = kwargs.get("permissions")
        seen["workspace_root"] = kwargs.get("workspace_root")
        return {
            "ok": True,
            "summary": "agent-ok",
            "patch_id": "patch-1",
            "patch": "--- a/lib/main.dart\n+++ b/lib/main.dart\n@@\n-old\n+new\n",
            "file_changes": [
                {
                    "path": "lib/main.dart",
                    "diff": "--- a/lib/main.dart\n+++ b/lib/main.dart\n@@\n-old\n+new\n",
                }
            ],
            "tests_ok": True,
            "browser_actions": [{"target": "http://localhost:4173", "opened": False}],
        }

    monkeypatch.setattr(server_mod, "run_agent", _fake_run_agent)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "core",
            "messages": [{"role": "user", "content": "arregla el modo agente"}],
            "agent_mode": True,
            "stream": False,
            "include_perf": True,
            "include_sources": True,
            "permissions": {
                "level": "full",
                "workspace_root": str(tmp_path),
                "project_path": "repo",
                "action_mode": "full",
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    assert data["model"] == "core"
    assert "agent-ok" in content
    assert "Tests: ok" in content
    assert "Patch: patch-1" in content
    assert "Archivos: lib/main.dart" in content
    assert "```file:lib/main.dart" not in content
    assert data["sources"] == []
    assert data["perf"]["agent_mode"] is True
    assert data["perf"]["agent_strategy"] == "tool_runner"
    assert data["perf"]["tests_ok"] is True
    assert data["perf"]["file_changes"][0]["path"] == "lib/main.dart"
    assert data["perf"]["browser_actions"][0]["target"] == "http://localhost:4173"
    assert seen["model"] is dummy
    assert callable(seen["model_lock"])
    assert getattr(seen["permissions"], "can_write", False) is True
    assert str(seen["workspace_root"]) == str((tmp_path / "repo").resolve())
    assert "Objetivo principal" in str(seen["task"])


def test_chat_completions_agent_mode_stream(tmp_path: Path, monkeypatch) -> None:
    client, _dummy, server_mod = _setup_app(tmp_path, monkeypatch)
    monkeypatch.setattr(
        server_mod,
        "run_agent",
        lambda *args, **kwargs: {
            "ok": True,
            "summary": "agent-stream",
            "patch_id": None,
            "patch": "",
            "file_changes": [
                {
                    "path": "app.py",
                    "diff": "--- /dev/null\n+++ b/app.py\n@@\n+print('ok')\n",
                }
            ],
            "tests_ok": False,
        },
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "core",
            "messages": [{"role": "user", "content": "haz streaming del modo agente"}],
            "agent_mode": True,
            "stream": True,
            "include_perf": True,
        },
    )

    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")
    assert "Agente iniciado" in resp.text
    assert "agent-stream" in resp.text
    assert "file_changes" in resp.text
    assert "Archivos: app.py" in resp.text
    assert "```file:app.py" not in resp.text
    assert "data: [DONE]" in resp.text


def test_chat_completions_agent_mode_does_not_503_when_model_failed(tmp_path: Path, monkeypatch) -> None:
    client, _dummy, server_mod = _setup_app(tmp_path, monkeypatch)
    seen: dict[str, object] = {}
    client.app.state.models = {}
    client.app.state.model = None
    client.app.state.model_loading = False
    client.app.state.model_load_error = "missing_gguf"

    def _fake_run_agent(task, settings, base_dir, **kwargs):
        seen["model"] = kwargs.get("model")
        seen["allow_model_load"] = kwargs.get("allow_model_load")
        return {
            "ok": False,
            "summary": "agent_model_unavailable: missing_gguf",
            "patch_id": None,
            "patch": "",
            "tests_ok": False,
            "tools_ok": False,
            "browser_actions": [],
            "tool_calls": [],
        }

    monkeypatch.setattr(server_mod, "run_agent", _fake_run_agent)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "core",
            "messages": [{"role": "user", "content": "arregla el modo agente"}],
            "agent_mode": True,
            "stream": False,
            "include_perf": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "agent_model_unavailable" in data["choices"][0]["message"]["content"]
    assert data["perf"]["agent_mode"] is True
    assert data["perf"]["agent_model_unavailable_reason"] == "model_load_failed:missing_gguf"
    assert seen["model"] is None
    assert seen["allow_model_load"] is False


def test_chat_completions_agent_mode_external_runtime_uses_direct_generate(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}

    class DummyExternalModel:
        is_external = True

        def __init__(self):
            self.tokenizer = None
            self.model_id = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
            self.cfg = SimpleNamespace(model=self.model_id)

        def generate(self, prompt: str, **kwargs):
            seen["prompt"] = prompt
            seen["messages"] = kwargs.get("messages")
            return "Plan limpio"

        def runtime_stats(self):
            return {"requests_total": 1, "retries_total": 0}

    settings = {
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://sglang-runtime:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "hf_system_prompt": "SYS",
        },
        "rag": {"enabled": False},
        "agent": {"max_iters": 4},
    }
    client, _dummy, server_mod = _setup_app(
        tmp_path,
        monkeypatch,
        model=DummyExternalModel(),
        settings=settings,
    )

    def _unexpected_run_agent(*_args, **_kwargs):
        raise AssertionError("run_agent should not be used for external agent mode")

    monkeypatch.setattr(server_mod, "run_agent", _unexpected_run_agent)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user",
                    "content": "En 2 bullets, plan minimo para app Flutter limpia con Dart.",
                }
            ],
            "agent_mode": True,
            "stream": False,
            "include_perf": True,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "external"
    assert data["choices"][0]["message"]["content"] == "Plan limpio"
    assert data["perf"]["agent_mode"] is True
    assert data["perf"]["agent_strategy"] == "external_chat"
    assert isinstance(seen.get("messages"), list)
    assert "No generes diffs" in str(seen["messages"][0]["content"])


def test_chat_completions_agent_mode_external_runtime_with_permissions_uses_runner(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)

    class DummyExternalModel:
        is_external = True

        def __init__(self):
            self.tokenizer = None
            self.model_id = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
            self.cfg = SimpleNamespace(model=self.model_id)

        def generate(self, prompt: str, **kwargs):
            raise AssertionError("direct external generate should not run with full permissions")

    settings = {
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://sglang-runtime:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "hf_system_prompt": "SYS",
        },
        "rag": {"enabled": False},
        "agent": {"max_iters": 4},
    }
    client, _dummy, server_mod = _setup_app(
        tmp_path,
        monkeypatch,
        model=DummyExternalModel(),
        settings=settings,
    )

    def _fake_run_agent(task, settings, base_dir, **kwargs):
        seen["task"] = task
        seen["permissions"] = kwargs.get("permissions")
        seen["workspace_root"] = kwargs.get("workspace_root")
        return {
            "ok": True,
            "summary": "runner-ok",
            "patch_id": None,
            "patch": "",
            "tests_ok": False,
        }

    monkeypatch.setattr(server_mod, "run_agent", _fake_run_agent)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
            "messages": [
                {
                    "role": "user",
                    "content": "Arranca el proyecto y comprueba la app.",
                }
            ],
            "agent_mode": True,
            "stream": False,
            "permissions": {
                "level": "full",
                "workspace_root": str(tmp_path),
                "project_path": "workspace",
                "action_mode": "full",
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "runner-ok" in data["choices"][0]["message"]["content"]
    assert getattr(seen["permissions"], "can_run_commands", False) is True
    assert str(seen["workspace_root"]) == str((tmp_path / "workspace").resolve())


def test_operational_status_treats_external_runtime_as_ready(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import server as server_mod

    monkeypatch.setattr(
        server_mod,
        "prepare_model_state",
        lambda settings, base_dir=None: {
            "ok": False,
            "offline_ready": True,
            "engine_ready": True,
            "engine_kind": "vortex",
            "engine_base_url": None,
            "model_ready": False,
            "model_reason": "model_not_required",
            "engine_reason": "engine_not_required",
            "active_model": None,
            "training_ready": False,
            "web_disabled": True,
            "docker_ready": True,
            "degraded_reason": "model_not_required",
            "offline_reason": "web_disabled",
            "training_reason": "hf_train_disabled",
            "docker_reason": "docker_not_required",
            "ollama_ready": True,
            "ollama_reason": "ollama_not_required",
            "wsl_ready": True,
            "wsl_reason": "wsl_not_required",
        },
    )

    model_id = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
    settings = {
        "_profile": "rtx4080_16gb_programming_runtime_docker",
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://sglang-runtime:30000",
            "external_model": model_id,
        },
    }
    app_state = SimpleNamespace(
        models={"external": SimpleNamespace(model_id=model_id)},
        model=None,
        model_loading=False,
        model_load_error=None,
        model_load_started_at=None,
        model_loaded_at=None,
        active_profile="rtx4080_16gb_programming_runtime_docker",
        boot_fallback=None,
        instructions=None,
    )

    payload = server_mod._build_operational_status(app_state, settings, tmp_path)

    assert payload["ok"] is True
    assert payload["chat_ready"] is True
    assert payload["engine_kind"] == "sglang"
    assert payload["engine_base_url"] == "http://sglang-runtime:30000"
    assert payload["active_model"] == model_id


def test_requested_external_model_alias_routes_to_external_even_without_external_default() -> None:
    from c3rnt2 import server as server_mod

    settings = {
        "core": {
            "backend": "vortex",
            "external_engine": "sglang",
            "external_base_url": "http://sglang-runtime:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        }
    }

    backend, use_router = server_mod._resolve_requested_backend(
        "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        settings,
        "core",
    )

    assert backend == "external"
    assert use_router is False


def test_requested_external_model_alias_routes_to_loaded_external_runtime() -> None:
    from c3rnt2 import server as server_mod

    external_model = SimpleNamespace(
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        cfg=SimpleNamespace(model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"),
    )

    backend, use_router = server_mod._resolve_requested_backend(
        "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        {"core": {"backend": "vortex"}},
        "core",
        external_model=external_model,
    )

    assert backend == "external"
    assert use_router is False
