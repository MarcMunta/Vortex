from __future__ import annotations

import json
import os
from pathlib import Path

from c3rnt2.agent.permissions import AgentPermissions
from c3rnt2.agent.runner import run_agent, Action
from c3rnt2.agent.tools import AgentTools
from c3rnt2.lab_guard import evaluate_lab_request


def test_agent_runner_dry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    settings = {"tools": {"web": {"enabled": False, "allow_domains": []}}, "agent": {"web_allowlist": []}}

    calls = {"count": 0}

    def provider(_messages):
        if calls["count"] == 0:
            calls["count"] += 1
            return Action(type="run_tests", args={})
        return Action(type="finish", args={"summary": "done"})

    report = run_agent("Run tests", settings, tmp_path, max_iters=2, action_provider=provider)
    assert report["ok"]
    episodes = tmp_path / "data" / "episodes" / "agent.jsonl"
    assert episodes.exists()


def test_agent_runner_blocks_public_security_target(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": []},
        "local_lab": {
            "guardrails_enabled": True,
            "lab_confirmation_token": "LAB_CONFIRMED",
        },
    }

    report = run_agent(
        "Exploit https://example.com with a payload.",
        settings,
        tmp_path,
        max_iters=1,
        action_provider=lambda _messages: Action(type="finish", args={"summary": "should_not_run"}),
    )
    assert report["ok"] is False
    assert report["blocked"] is True
    assert "public" in report["summary"].lower() or "third-party" in report["summary"].lower()


def test_lab_guard_allows_programming_login_with_file_context() -> None:
    settings = {
        "local_lab": {
            "guardrails_enabled": True,
            "lab_confirmation_token": "LAB_CONFIRMED",
        },
    }
    task = (
        "Resuelve la peticion del usuario como operador tecnico.\n\n"
        "Objetivo principal:\nCrea un login basico en Flutter.\n\n"
        "Contexto reciente:\n[assistant]\nArchivo escrito: lib/main.dart\n"
        "Continua desde el ultimo punto y cierra bloques de codigo."
    )

    result = evaluate_lab_request([{"role": "user", "content": task}], settings)

    assert result["action"] == "allow"


def test_agent_runner_uses_fresh_model_lock_context_per_generation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {"tools": {"web": {"enabled": False, "allow_domains": []}}, "agent": {"web_allowlist": []}}

    class OneShotContext:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            if self.entered:
                raise AssertionError("context_reused")
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return "not-json"
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    model = DummyModel()
    lock_calls = {"count": 0}

    def _model_lock():
        lock_calls["count"] += 1
        return OneShotContext()

    report = run_agent(
        "Fix context manager reuse",
        settings,
        tmp_path,
        max_iters=1,
        model=model,
        model_lock=_model_lock,
    )

    assert report["ok"] is True
    assert report["summary"] == "done"
    assert model.calls == 2
    assert lock_calls["count"] == 2


def test_agent_runner_compacts_and_continues_after_invalid_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "max_iters": 2,
            "json_repair_retries": 1,
            "max_context_compactions": 2,
        },
    }

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls <= 2:
                return "not-json"
            return json.dumps({"type": "finish", "args": {"summary": "continued"}})

    model = DummyModel()
    report = run_agent("Continue after bad model output", settings, tmp_path, model=model)

    assert report["ok"] is True
    assert report["summary"] == "continued"
    assert model.calls == 3
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["invalid_json_count"] == 1
    assert episode["context_compactions_done"] >= 1


def test_agent_runner_continues_after_iteration_window_compaction(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "max_context_compactions": 1},
    }
    calls = {"count": 0}

    def provider(_messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return Action(type="read_file", args={"path": "missing.txt"})
        return Action(type="finish", args={"summary": "done_after_compaction"})

    report = run_agent("Keep going past first window", settings, tmp_path, max_iters=1, action_provider=provider)

    assert report["ok"] is True
    assert report["summary"] == "done_after_compaction"
    assert calls["count"] == 2
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["context_compactions_done"] == 1


def test_agent_runner_reads_generation_limits_from_settings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "max_iters": 3,
            "action_max_new_tokens": 777,
            "final_summary_max_new_tokens": 888,
        },
    }

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.kwargs = []

        def generate(self, _prompt: str, **kwargs):
            self.kwargs.append(dict(kwargs))
            if len(self.kwargs) == 1:
                return json.dumps({"type": "read_file", "args": {"path": "missing.txt"}})
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    model = DummyModel()
    report = run_agent("Use configured limits", settings, tmp_path, model=model)

    assert report["ok"] is True
    assert report["summary"] == "done"
    assert len(report["tool_calls"]) == 1
    assert [item["max_new_tokens"] for item in model.kwargs] == [777, 777]

    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["max_iters"] == 3
    assert episode["action_max_new_tokens"] == 777
    assert episode["final_summary_max_new_tokens"] == 888


def test_agent_runner_stops_on_wall_time_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "max_iters": 5, "max_wall_time_s": 0.001},
    }

    def provider(_messages):
        return Action(type="read_file", args={"path": "missing.txt"})

    report = run_agent("Stop by time", settings, tmp_path, action_provider=provider)

    assert report["ok"] is True
    assert len(report["tool_calls"]) < 5


def test_agent_tools_safe_mode_blocks_commands_and_writes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hola", encoding="utf-8")
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "safe"},
        tmp_path,
    )
    tools = AgentTools(
        allowlist=[],
        web_cfg={"enabled": False, "allow_domains": []},
        repo_root=tmp_path,
        permissions=permissions,
    )

    read_result = tools.read_file("README.md")
    command_result = tools.run_command("python --version")
    apply_result = tools.apply_patch(tmp_path, "patch-1", approve=True)
    write_result = tools.write_file("README.md", "cambio")
    delete_result = tools.delete_file("README.md")

    assert read_result.ok is True
    assert command_result.ok is False
    assert "permission_denied:command" in command_result.output
    assert apply_result.ok is False
    assert "permission_denied:write" in apply_result.output
    assert write_result.ok is False
    assert "permission_denied:write" in write_result.output
    assert delete_result.ok is False
    assert "permission_denied:write" in delete_result.output


def test_agent_tools_full_mode_allows_dev_command(tmp_path: Path) -> None:
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    tools = AgentTools(
        allowlist=[],
        web_cfg={"enabled": False, "allow_domains": []},
        repo_root=tmp_path,
        permissions=permissions,
    )

    result = tools.run_command("python --version")

    assert result.ok is True


def test_agent_tools_full_mode_writes_files_and_records_browser_actions(
    tmp_path: Path, monkeypatch
) -> None:
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    tools = AgentTools(
        allowlist=[],
        web_cfg={"enabled": False, "allow_domains": []},
        repo_root=tmp_path,
        permissions=permissions,
    )
    monkeypatch.setattr("c3rnt2.agent.tools.webbrowser.open", lambda *_args, **_kwargs: False)

    write_result = tools.write_file("lib/main.dart", "void main() {}\n")
    delete_result = tools.delete_file("lib/main.dart")
    browser_result = tools.open_browser("http://localhost:4173")

    assert write_result.ok is True
    assert write_result.meta["path"] == "lib/main.dart"
    assert "+void main() {}" in write_result.meta["diff"]
    assert delete_result.ok is True
    assert delete_result.meta["path"] == "lib/main.dart"
    assert "-void main() {}" in delete_result.meta["diff"]
    assert not (tmp_path / "lib" / "main.dart").exists()
    assert browser_result.ok is True
    assert tools.browser_actions == [
        {"target": "http://localhost:4173", "opened": True, "backend_opened": False}
    ]


def test_agent_runner_direct_file_create_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "delete_file"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    create_report = run_agent(
        "Crea el archivo notes/demo.txt con texto hola-agente. No ejecutes tests.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )
    assert create_report["ok"] is True
    assert create_report["tools_ok"] is True
    assert (tmp_path / "notes" / "demo.txt").read_text(encoding="utf-8") == "hola-agente"
    assert create_report["file_changes"][0]["path"] == "notes/demo.txt"
    assert "+hola-agente" in create_report["file_changes"][0]["diff"]

    delete_report = run_agent(
        "Borra el archivo notes/demo.txt.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )
    assert delete_report["ok"] is True
    assert delete_report["tools_ok"] is True
    assert not (tmp_path / "notes" / "demo.txt").exists()
    assert delete_report["file_changes"][0]["path"] == "notes/demo.txt"
    assert "-hola-agente" in delete_report["file_changes"][0]["diff"]


def test_agent_runner_writes_flutter_code_when_model_returns_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "json_repair_retries": 0},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class MarkdownCodeModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return (
                "```dart\n"
                "import 'package:flutter/material.dart';\n"
                "void main() => runApp(const MaterialApp(home: LoginPage()));\n"
                "class LoginPage extends StatelessWidget {\n"
                "  const LoginPage({super.key});\n"
                "  @override\n"
                "  Widget build(BuildContext context) => const Scaffold(body: Text('Login'));\n"
                "}\n"
                "```"
            )

    report = run_agent(
        "Crea un login basico en Flutter.",
        settings,
        tmp_path,
        max_iters=3,
        model=MarkdownCodeModel(),
        permissions=permissions,
    )

    main_path = tmp_path / "lib" / "main.dart"
    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert main_path.exists()
    assert "LoginPage" in main_path.read_text(encoding="utf-8")
    assert str(main_path) in report["summary"]
    assert report["tool_calls"][0]["action"] == "write_file"
    assert report["file_changes"][0]["path"] == "lib/main.dart"
    assert "+class LoginPage" in report["file_changes"][0]["diff"]


def test_agent_runner_infers_exists_summary_from_workspace(tmp_path: Path) -> None:
    (tmp_path / "vortex-chat").mkdir(parents=True, exist_ok=True)
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": []},
    }

    report = run_agent(
        "Dime si existe la carpeta vortex-chat.",
        settings,
        tmp_path,
        max_iters=1,
        action_provider=lambda _messages: Action(type="finish", args={"summary": "finished"}),
    )

    assert report["ok"] is True
    assert report["summary"] == "Sí, existe la carpeta vortex-chat en el proyecto."
