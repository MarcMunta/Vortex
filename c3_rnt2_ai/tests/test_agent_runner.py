from __future__ import annotations

import json
import os
from pathlib import Path

from c3rnt2.agent.permissions import AgentPermissions
from c3rnt2.agent.runner import run_agent, Action
from c3rnt2.agent.tools import AgentTools, ToolResult
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


def test_agent_runner_executes_known_node_project_directly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "node -e \"console.log('build ok')\""}}),
        encoding="utf-8",
    )
    settings = {"tools": {"web": {"enabled": False, "allow_domains": []}}, "agent": {"web_allowlist": []}}
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class FinishingModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps({"type": "finish", "args": {"summary": "finished"}})

    report = run_agent(
        "Ejecuta el proyecto.",
        settings,
        tmp_path,
        max_iters=1,
        model=FinishingModel(),
        permissions=permissions,
    )

    assert any(call["action"] == "run_command" for call in report["tool_calls"])
    assert report["blocked"] is False


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


def test_agent_tools_background_command_reports_immediate_failure(tmp_path: Path) -> None:
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    tools = AgentTools(
        allowlist=[],
        web_cfg={"enabled": False, "allow_domains": []},
        repo_root=tmp_path,
        permissions=permissions,
        agent_cfg={"background_probe_s": 3},
    )

    result = tools.run_command(
        "python -m vortex_missing_background_module_12345",
        background=True,
    )

    assert result.ok is False
    assert result.meta["background"] is True
    assert result.meta["returncode"] != 0
    assert "vortex_missing_background_module_12345" in result.output


def test_agent_tools_command_env_adds_flutter_and_android_tool_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flutter_bin = tmp_path / ".puro" / "envs" / "stable" / "flutter" / "bin"
    sdk_root = tmp_path / "AppData" / "Local" / "Android" / "Sdk"
    emulator_bin = sdk_root / "emulator"
    platform_tools = sdk_root / "platform-tools"
    for path in (flutter_bin, emulator_bin, platform_tools):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
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

    path_value = tools._command_env()["PATH"]

    assert str(flutter_bin.resolve()) in path_value
    assert str(emulator_bin.resolve()) in path_value
    assert str(platform_tools.resolve()) in path_value


def test_agent_tools_resolves_bat_executable_from_augmented_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flutter_bin = tmp_path / ".puro" / "envs" / "stable" / "flutter" / "bin"
    flutter_bin.mkdir(parents=True, exist_ok=True)
    (flutter_bin / "flutter.bat").write_text("@echo off\r\n", encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
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

    args, error = tools._validate_command("flutter --version", tmp_path)

    assert error is None
    assert args is not None
    assert Path(args[0]).name.lower() == "flutter.bat"


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


def test_agent_runner_direct_readme_natural_edit_and_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    (tmp_path / "README.md").write_text("old readme\n", encoding="utf-8")

    edit_report = run_agent(
        'Edita el readme para que ponga "hola mundo".',
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert edit_report["ok"] is True
    assert edit_report["tools_ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hola mundo"
    assert edit_report["tool_calls"][0]["action"] == "write_file"
    assert edit_report["file_changes"][0]["path"] == "README.md"
    assert "-old readme" in edit_report["file_changes"][0]["diff"]
    assert "+hola mundo" in edit_report["file_changes"][0]["diff"]

    delete_report = run_agent(
        "Borra el readme.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert delete_report["ok"] is True
    assert delete_report["tools_ok"] is True
    assert not (tmp_path / "README.md").exists()
    assert delete_report["tool_calls"][0]["action"] == "delete_file"
    assert delete_report["file_changes"][0]["path"] == "README.md"
    assert "-hola mundo" in delete_report["file_changes"][0]["diff"]


def test_agent_runner_direct_readme_edit_accepts_punctuation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    (tmp_path / "README.md").write_text("old readme\n", encoding="utf-8")

    report = run_agent(
        'Modifica el readme para que ponga, "hola marc hola pol hola joan"',
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hola marc hola pol hola joan"
    assert report["file_changes"][0]["path"] == "README.md"
    assert "+hola marc hola pol hola joan" in report["file_changes"][0]["diff"]


def test_agent_runner_direct_edit_missing_file_reports_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    edit_report = run_agent(
        'Edita missing.md para que ponga "hola mundo".',
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert edit_report["ok"] is True
    assert edit_report["tools_ok"] is False
    assert not (tmp_path / "missing.md").exists()
    assert "No encuentro el archivo `missing.md`" in edit_report["summary"]

    delete_report = run_agent(
        "Borra missing.md.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert delete_report["ok"] is True
    assert delete_report["tools_ok"] is False
    assert "No encuentro el archivo `missing.md`" in delete_report["summary"]


def test_agent_runner_direct_create_modify_command_and_delete_procedure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "delete_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        (
            "Crea el archivo notes/demo.txt con texto uno; "
            "modifica el archivo notes/demo.txt con texto dos; "
            "ejecuta el comando \"python --version\"; "
            "borra el archivo notes/demo.txt."
        ),
        settings,
        tmp_path,
        max_iters=5,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert [call["action"] for call in report["tool_calls"]] == [
        "write_file",
        "write_file",
        "run_command",
        "delete_file",
    ]
    assert not (tmp_path / "notes" / "demo.txt").exists()
    assert {change["path"] for change in report["file_changes"]} == {"notes/demo.txt"}
    assert any("+uno" in change["diff"] for change in report["file_changes"])
    assert any("-uno" in change["diff"] and "+dos" in change["diff"] for change in report["file_changes"])
    assert any("-dos" in change["diff"] for change in report["file_changes"])
    assert report["summary"] == "He creado `notes/demo.txt`. He actualizado `notes/demo.txt`. He borrado `notes/demo.txt`."


def test_agent_runner_direct_create_file_accepts_common_spanish_article(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Crea un archivo qa_agent_ui_check.txt con el texto agente-ui-ok",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert (tmp_path / "qa_agent_ui_check.txt").read_text(encoding="utf-8") == "agente-ui-ok"
    assert report["tool_calls"][0]["action"] == "write_file"
    assert report["file_changes"][0]["path"] == "qa_agent_ui_check.txt"
    assert "+agente-ui-ok" in report["file_changes"][0]["diff"]


def test_agent_runner_direct_file_actions_ignore_stale_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    (tmp_path / "qa_agent_ui_check.txt").write_text("agente-ui-ok", encoding="utf-8")

    report = run_agent(
        (
            "Objetivo principal:\n"
            "Edita no_existe_qa.txt para que ponga hola\n\n"
            "Contexto reciente:\n"
            "[user]\n"
            "Crea un archivo qa_agent_ui_check.txt con el texto agente-ui-ok\n"
            "[assistant]\n"
            "He creado `qa_agent_ui_check.txt`."
        ),
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is False
    assert (tmp_path / "qa_agent_ui_check.txt").read_text(encoding="utf-8") == "agente-ui-ok"
    assert not (tmp_path / "no_existe_qa.txt").exists()
    assert "No encuentro el archivo `no_existe_qa.txt`" in report["summary"]
    assert report["tool_calls"][0]["args"]["path"] == "no_existe_qa.txt"


def test_agent_runner_adds_forgot_password_button_to_flutter_login(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    (tmp_path / "lib" / "main.dart").write_text(
        """import 'package:flutter/material.dart';

void main() {
  runApp(const VortexLoginApp());
}

class VortexLoginApp extends StatelessWidget {
  const VortexLoginApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: const LoginPage());
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  void _submit() {}

  Widget _buildLoginCard(BuildContext context) {
    return Column(
      children: [
              FilledButton(
                onPressed: _submit,
                child: const Text('Sign in'),
              ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) => _buildLoginCard(context);
}
""",
        encoding="utf-8",
    )
    (tmp_path / "test" / "widget_test.dart").write_text(
        """import 'package:flutter_test/flutter_test.dart';
import 'package:existing_flutter/main.dart' as app;

void main() {
  testWidgets('app starts', (WidgetTester tester) async {
    app.main();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
""",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Añade un boton sencillo por si el usuario no se acuerda de la contrasena.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
        allow_model_load=False,
    )

    main_text = (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    test_text = (tmp_path / "test" / "widget_test.dart").read_text(encoding="utf-8")
    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert report["tests_ok"] is True
    assert "Olvidaste la contrasena?" in main_text
    assert "Olvidaste la contrasena?" in test_text
    assert commands == ["flutter test"]
    assert {change["path"] for change in report["file_changes"]} == {
        "lib/main.dart",
        "test/widget_test.dart",
    }


def test_agent_runner_adds_animated_dark_mode_button_to_flutter_login(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    (tmp_path / "lib" / "main.dart").write_text(
        """import 'package:flutter/material.dart';

void main() {
  runApp(const VortexLoginApp());
}

class VortexLoginApp extends StatelessWidget {
  const VortexLoginApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vortex Login',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginPage(),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  Widget _buildLoginCard(BuildContext context) {
    return Column(
      children: [
              Text('Login', style: Theme.of(context).textTheme.headlineMedium),
      ],
    );
  }

  @override
  Widget build(BuildContext context) => _buildLoginCard(context);
}
""",
        encoding="utf-8",
    )
    (tmp_path / "test" / "widget_test.dart").write_text(
        """import 'package:flutter_test/flutter_test.dart';
import 'package:existing_flutter/main.dart' as app;

void main() {
  testWidgets('app starts', (WidgetTester tester) async {
    app.main();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
""",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Añade un boton para el modo oscuro con animacion basica y transicion del color blanco al negro.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
        allow_model_load=False,
    )

    main_text = (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    test_text = (tmp_path / "test" / "widget_test.dart").read_text(encoding="utf-8")
    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert report["tests_ok"] is True
    assert "themeMode:" in main_text
    assert "AnimatedSwitcher" in main_text
    assert "Modo oscuro" in main_text
    assert "find.byTooltip('Modo oscuro')" in test_text
    assert commands == ["flutter test"]
    assert {change["path"] for change in report["file_changes"]} == {
        "lib/main.dart",
        "test/widget_test.dart",
    }


def test_agent_runner_adds_dark_mode_without_magic_words_when_model_loaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "test").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    (tmp_path / "lib" / "main.dart").write_text(
        """import 'package:flutter/material.dart';

void main() {
  runApp(const VortexLoginApp());
}

class VortexLoginApp extends StatelessWidget {
  const VortexLoginApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vortex Login',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginPage(),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  Widget _buildLoginCard(BuildContext context) {
    return Column(
      children: [
              Text('Login', style: Theme.of(context).textTheme.headlineMedium),
      ],
    );
  }

  @override
  Widget build(BuildContext context) => _buildLoginCard(context);
}
""",
        encoding="utf-8",
    )
    (tmp_path / "test" / "widget_test.dart").write_text(
        """import 'package:flutter_test/flutter_test.dart';
import 'package:existing_flutter/main.dart' as app;

void main() {
  testWidgets('app starts', (WidgetTester tester) async {
    app.main();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
""",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file", "run_command"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(ok=True, output=f"ran:{command}")

    class BadPlanningModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps({"type": "finish", "args": {"summary": "finished"}})

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Anade un modo oscuro a la aplicacion modificando las carpetas y archivos necesarios",
        settings,
        tmp_path,
        max_iters=3,
        model=BadPlanningModel(),
        permissions=permissions,
    )

    main_text = (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert "themeMode:" in main_text
    assert "AnimatedSwitcher" in main_text
    assert commands == ["flutter test"]


def test_agent_runner_adds_dark_mode_to_generic_flutter_material_app(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pubspec.yaml").write_text(
        "name: generic_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    (tmp_path / "lib" / "main.dart").write_text(
        """import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Generic App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const Placeholder(),
    );
  }
}
""",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file", "run_command"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        return ToolResult(ok=True, output=f"ran:{command}")

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Anade un modo oscuro a la aplicacion modificando lo necesario",
        settings,
        tmp_path,
        max_iters=2,
        permissions=permissions,
        allow_model_load=False,
    )

    main_text = (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert report["tools_ok"] is True
    assert "themeMode: ThemeMode.system" in main_text
    assert "darkTheme: ThemeData" in main_text
    assert "brightness: Brightness.dark" in main_text


def test_agent_runner_understands_ejecutame_flutter_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["run_command"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(ok=True, output=f"ran:{command}")

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Ejecutame el proyecto flutter",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
    )

    assert report["tools_ok"] is True
    assert commands[:6] == [
        "flutter --version",
        "flutter create --platforms=web --no-overwrite .",
        "flutter devices",
        "flutter pub get",
        "flutter test",
        "flutter build web",
    ]
    assert commands[-1] == "flutter run -d web-server --web-hostname 0.0.0.0 --web-port 19090"


def test_agent_runner_read_file_answers_pubspec_project_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["read_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "safe"},
        tmp_path,
    )

    report = run_agent(
        "Lee el archivo pubspec.yaml y dime que proyecto es",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["tools_ok"] is True
    assert "El proyecto es `existing_flutter`" in report["summary"]


def test_agent_runner_scaffolds_flutter_login_project_without_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Hazme un proyecto Flutter que se pueda ejecutar con un login basico.",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert (tmp_path / "pubspec.yaml").exists()
    assert (tmp_path / "lib" / "main.dart").exists()
    assert (tmp_path / "test" / "widget_test.dart").exists()
    assert "VortexLoginApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert "pubspec.yaml" in report["summary"]
    assert {change["path"] for change in report["file_changes"]} >= {
        "pubspec.yaml",
        "lib/main.dart",
        "test/widget_test.dart",
        "README.md",
    }


def test_agent_runner_scaffolds_basic_flutter_project_without_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Haz un codigo basico de Flutter en mi proyecto, que se pueda ejecutar.",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert "VortexFlutterApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert "vortex_flutter_app" in (tmp_path / "pubspec.yaml").read_text(encoding="utf-8")
    assert {change["path"] for change in report["file_changes"]} >= {
        "pubspec.yaml",
        "lib/main.dart",
        "test/widget_test.dart",
        "README.md",
    }


def test_agent_runner_flutter_project_uses_terminal_and_emulator_without_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib" / "main.dart").write_text("old-main\n", encoding="utf-8")
    commands: list[tuple[str, bool]] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append((command, background))
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        (
            "Crea un proyecto basico de Flutter en este workspace, modificalo si ya existe, "
            "usa terminal para validar, que se pueda ejecutar y ejecutalo en emulador."
        ),
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert (tmp_path / "pubspec.yaml").exists()
    assert "VortexFlutterApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert [item[0] for item in commands] == [
        "flutter --version",
        "flutter create --platforms=android --no-overwrite .",
        "flutter devices",
        "flutter pub get",
        "flutter test",
        "flutter run -d emulator-5554 --debug",
    ]
    assert commands[-1][1] is True
    assert [call["action"] for call in report["tool_calls"]].count("run_command") == len(commands)
    assert any(
        change["path"] == "lib/main.dart" and "-old-main" in change["diff"] and "+class VortexFlutterApp" in change["diff"]
        for change in report["file_changes"]
    )


def test_agent_runner_flutter_terminal_request_uses_tools_without_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Crea un proyecto basico de Flutter, usa terminal y ejecutalo en emulador.",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert "flutter pub get" in commands
    assert "flutter run -d emulator-5554 --debug" in commands
    assert (tmp_path / "lib" / "main.dart").exists()


def test_agent_runner_infers_flutter_workspace_for_login_and_emulator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        "Ahora crea un login basico e inicia el emulador.",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert report["tests_ok"] is True
    assert "VortexLoginApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert "flutter devices" in commands
    assert "flutter run -d emulator-5554 --debug" in commands
    assert "He actualizado" in report["summary"] or "He creado" in report["summary"]


def test_agent_runner_runs_existing_flutter_project_without_repeating_previous_login_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        (
            "Resuelve la peticion del usuario como operador tecnico dentro del repositorio actual.\n\n"
            "Objetivo principal:\nQuiero que ejecutes el proyecto\n\n"
            "Contexto reciente:\n[user]\nHaz un login basico para el proyecto de flutter"
        ),
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert report["tests_ok"] is True
    assert [call["action"] for call in report["tool_calls"]] == ["run_command"] * 7
    assert commands == [
        "flutter --version",
        "flutter create --platforms=web --no-overwrite .",
        "flutter devices",
        "flutter pub get",
        "flutter test",
        "flutter build web",
        "flutter run -d web-server --web-hostname 0.0.0.0 --web-port 19090",
    ]
    assert report["file_changes"] == []
    assert "pubspec.yaml" not in report["summary"]
    assert "He iniciado la app web local" in report["summary"]


def test_agent_runner_reports_emulator_unavailable_without_repeating_stale_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    (tmp_path / "pubspec.yaml").write_text(
        "name: existing_flutter\n\ndependencies:\n  flutter:\n    sdk: flutter\n",
        encoding="utf-8",
    )
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["write_file", "run_command"],
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    commands: list[str] = []

    def _fake_run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int = 120,
        background: bool = False,
    ) -> ToolResult:
        commands.append(command)
        if command.startswith("flutter run"):
            return ToolResult(
                ok=False,
                output="No supported devices connected.",
                meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 1},
            )
        return ToolResult(
            ok=True,
            output=f"ran:{command}",
            meta={"cwd": str(self.repo_root), "command": command.split(), "returncode": 0},
        )

    monkeypatch.setattr(AgentTools, "run_command", _fake_run_command)

    report = run_agent(
        (
            "Resuelve la peticion del usuario como operador tecnico dentro del repositorio actual.\n\n"
            "Objetivo principal:\nQuiero que ejecutes el proyecto\n\n"
            "Contexto reciente:\n[user]\nHaz un login basico para el proyecto de flutter"
        ),
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        allow_model_load=False,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is False
    assert report["tests_ok"] is True
    assert commands == [
        "flutter --version",
        "flutter create --platforms=web --no-overwrite .",
        "flutter devices",
        "flutter pub get",
        "flutter test",
        "flutter build web",
        "flutter run -d web-server --web-hostname 0.0.0.0 --web-port 19090",
    ]
    assert report["file_changes"] == []
    assert "pubspec.yaml" not in report["summary"]
    assert "He validado el proyecto con sus tests" in report["summary"]
    assert "No he podido completar `flutter run -d web-server --web-hostname 0.0.0.0 --web-port 19090`" in report["summary"]


def test_agent_runner_does_not_scaffold_flutter_project_when_model_is_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "json_repair_retries": 0},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class PlanningModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps({"action": "finish", "args": {"summary": "modelo planifico"}})

    report = run_agent(
        "Hazme un proyecto Flutter que se pueda ejecutar con un login basico.",
        settings,
        tmp_path,
        max_iters=1,
        model=PlanningModel(),
        permissions=permissions,
    )

    assert report["ok"] is False
    assert report["tools_ok"] is False
    assert "No he aplicado cambios" in report["summary"]
    assert report["file_changes"] == []
    assert not (tmp_path / "pubspec.yaml").exists()


def test_agent_runner_does_not_complete_command_request_without_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "json_repair_retries": 0},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class PlanningModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps({"action": "finish", "args": {"summary": "finished"}})

    report = run_agent(
        "Ejecuta el proyecto en el emulador de Android.",
        settings,
        tmp_path,
        max_iters=1,
        model=PlanningModel(),
        permissions=permissions,
    )

    assert report["ok"] is False
    assert report["tools_ok"] is False
    assert "No he ejecutado acciones" in report["summary"]
    assert report["tool_calls"] == []


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


def test_agent_runner_completes_flutter_project_when_model_returns_only_main_dart(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
                "void main() => runApp(const MaterialApp(home: Text('Login')));\n"
                "```"
            )

    report = run_agent(
        "Hazme un proyecto Flutter que se pueda ejecutar con un login basico.",
        settings,
        tmp_path,
        max_iters=3,
        model=MarkdownCodeModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert (tmp_path / "lib" / "main.dart").exists()
    assert (tmp_path / "pubspec.yaml").exists()
    assert (tmp_path / "test" / "widget_test.dart").exists()
    assert {change["path"] for change in report["file_changes"]} >= {
        "lib/main.dart",
        "pubspec.yaml",
        "test/widget_test.dart",
        "README.md",
    }


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
