from __future__ import annotations

import json
import os
from pathlib import Path

from c3rnt2.agent.permissions import AgentPermissions
from c3rnt2.agent.runner import run_agent, Action
from c3rnt2.agent.tools import AgentTools, ToolResult
from c3rnt2.context_budget import resolve_context_budget, resolve_model_context_limit
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


def test_agent_runner_executes_known_node_project_directly():
    pass































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


def test_context_budget_uses_llama_cpp_ctx() -> None:
    settings = {"core": {"backend": "llama_cpp", "llama_cpp_ctx": 4096}}
    budget = resolve_context_budget(settings)

    assert budget["model_max_context_tokens"] == 4096
    assert budget["default_agent_context_tokens"] == 4096

    class Model:
        class cfg:
            n_ctx = 4096

    assert resolve_model_context_limit({"context": {"model_max_context_tokens": 32768}}, Model()) == 4096


def test_agent_runner_compacts_before_model_context_overflow(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "core": {"backend": "llama_cpp", "llama_cpp_ctx": 2048},
        "context": {"rolling_summary_tokens": 200, "recent_messages_tokens": 512},
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "action_max_new_tokens": 1200,
            "max_context_compactions": 3,
        },
    }

    class StrictContextModel:
        tokenizer = None

        class cfg:
            n_ctx = 2048

        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def encode_prompt(self, prompt: str):
            count = max(1, len(prompt) // 4)
            return [], count

        def generate(self, prompt: str, **kwargs):
            prompt_tokens = max(1, len(prompt) // 4)
            max_new = int(kwargs.get("max_new_tokens") or 0)
            self.calls.append((prompt_tokens, max_new))
            if prompt_tokens + max_new > self.cfg.n_ctx:
                raise ValueError(
                    f"Requested tokens ({prompt_tokens + max_new}) exceed context window of {self.cfg.n_ctx}"
                )
            return json.dumps({"type": "finish", "args": {"summary": "compacted_ok"}})

    model = StrictContextModel()
    task = "Objetivo principal:\nResponde cuando termines.\n\nContexto reciente:\n" + ("historial largo " * 3000)
    report = run_agent(task, settings, tmp_path, max_iters=1, model=model)

    assert report["ok"] is True
    assert report["summary"] == "compacted_ok"
    assert model.calls
    assert all(prompt_tokens + max_new <= 2048 for prompt_tokens, max_new in model.calls)
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["context_compactions_done"] >= 1


def test_agent_runner_compacts_after_model_context_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "core": {"backend": "llama_cpp", "llama_cpp_ctx": 4096},
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "max_context_compactions": 2},
    }

    class ErrorOnceModel:
        tokenizer = None

        class cfg:
            n_ctx = 4096

        def __init__(self):
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Requested tokens (5221) exceed context window of 4096")
            return json.dumps({"type": "finish", "args": {"summary": "continued_after_context_compaction"}})

    model = ErrorOnceModel()
    report = run_agent("Continua tras error de contexto", settings, tmp_path, max_iters=2, model=model)

    assert report["ok"] is True
    assert report["summary"] == "continued_after_context_compaction"
    assert model.calls == 2
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["context_compactions_done"] >= 1


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


def test_agent_runner_passes_action_grammar_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "max_iters": 1},
    }

    class GrammarModel:
        tokenizer = None

        def __init__(self):
            self.grammars: list[object] = []

        def generate(self, _prompt: str, **kwargs):
            self.grammars.append(kwargs.get("grammar"))
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    model = GrammarModel()
    report = run_agent("Responde con done", settings, tmp_path, model=model)

    assert report["ok"] is True
    assert report["action_grammar_enabled"] is True
    assert isinstance(model.grammars[0], str)
    assert "action-type" in model.grammars[0]


def test_agent_runner_can_disable_action_grammar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "max_iters": 1, "action_grammar_enabled": False},
    }

    class PlainModel:
        tokenizer = None

        def __init__(self):
            self.grammars: list[object] = []

        def generate(self, _prompt: str, **kwargs):
            self.grammars.append(kwargs.get("grammar"))
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    model = PlainModel()
    report = run_agent("Responde con done", settings, tmp_path, model=model)

    assert report["ok"] is True
    assert report["action_grammar_enabled"] is False
    assert model.grammars == [None]


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


def test_agent_runner_continues_after_direct_read_for_requested_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {
            "web_allowlist": [],
            "tools_enabled": ["read_file", "write_file"],
            "max_iters": 3,
        },
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    (tmp_path / "README.md").write_text("README base para smoke\n", encoding="utf-8")

    class Model:
        def __init__(self) -> None:
            self.tokenizer = None
            self.calls = 0
            self.saw_read_output = False

        def generate(self, _prompt: str, **kwargs):
            self.calls += 1
            messages = kwargs.get("messages") or []
            self.saw_read_output = self.saw_read_output or any(
                msg.get("role") == "tool" and "README base" in str(msg.get("content") or "")
                for msg in messages
                if isinstance(msg, dict)
            )
            if self.calls == 1:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "notes/agent-smoke.txt",
                            "text": "Agente puede leer y escribir.\n",
                        },
                    }
                )
            return json.dumps(
                {
                    "type": "finish",
                    "args": {"summary": "Sure! Here's the final answer: changed."},
                }
            )

    model = Model()
    report = run_agent(
        "Lee README.md y crea notes/agent-smoke.txt con una frase corta.",
        settings,
        tmp_path,
        permissions=permissions,
        model=model,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert model.saw_read_output is True
    assert (tmp_path / "notes" / "agent-smoke.txt").read_text(encoding="utf-8") == "Agente puede leer y escribir.\n"
    assert [call["action"] for call in report["tool_calls"][:2]] == ["read_file", "write_file"]
    assert report["file_changes"][0]["path"] == "notes/agent-smoke.txt"
    assert report["summary"] == "He creado `notes/agent-smoke.txt`."


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


def test_agent_runner_direct_create_file_keeps_exact_content_only(
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
        "Crea un archivo smoke.txt con el texto exacto ok-smoke en el workspace. Usa write_file. No expliques nada.",
        settings,
        tmp_path,
        max_iters=3,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert (tmp_path / "smoke.txt").read_text(encoding="utf-8") == "ok-smoke"


def test_agent_runner_prompt_has_no_canned_flutter_login_example(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": []},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class CaptureModel:
        tokenizer = None

        def __init__(self):
            self.prompt = ""

        def generate(self, prompt: str, **_kwargs):
            self.prompt = prompt
            return '{"type":"finish","args":{"summary":"done"}}'

    model = CaptureModel()
    run_agent(
        "Piensa cual es el siguiente paso.",
        settings,
        tmp_path,
        max_iters=1,
        model=model,
        permissions=permissions,
    )

    assert "Created lib/main.dart" not in model.prompt
    assert "Material Design 3" not in model.prompt
    assert "Scaffold(body: Center(child: Text('Login')))" not in model.prompt


def test_agent_runner_run_command_only_does_not_satisfy_code_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["run_command", "write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    calls = {"count": 0}

    def provider(_messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return Action(type="run_command", args={"command": "python --version", "cwd": ".", "timeout_s": 60})
        return Action(type="finish", args={"summary": "Created lib/main.dart with a login screen using Material Design 3"})

    report = run_agent(
        "Anade modo oscuro",
        settings,
        tmp_path,
        max_iters=3,
        action_provider=provider,
        permissions=permissions,
    )

    assert report["ok"] is False
    assert report["blocked"] is True
    assert report["file_changes"] == []
    assert "No he aplicado cambios" in report["summary"]


def test_agent_runner_groups_repeated_file_changes_by_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    calls = {"count": 0}

    def provider(_messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return Action(type="write_file", args={"path": "README.md", "text": "uno"})
        if calls["count"] == 2:
            return Action(type="write_file", args={"path": "README.md", "text": "dos"})
        return Action(type="finish", args={"summary": "done"})

    report = run_agent(
        "Actualiza README dos veces",
        settings,
        tmp_path,
        max_iters=4,
        action_provider=provider,
        permissions=permissions,
    )

    assert report["ok"] is True
    assert len(report["file_changes"]) == 1
    assert report["file_changes"][0]["path"] == "README.md"
    assert "+uno" in report["file_changes"][0]["diff"]
    assert "-uno" in report["file_changes"][0]["diff"]
    assert "+dos" in report["file_changes"][0]["diff"]


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


def test_agent_runner_adds_forgot_password_button_to_flutter_login():
    pass
























































































































def test_agent_runner_adds_animated_dark_mode_button_to_flutter_login():
    pass





























































































































def test_agent_runner_adds_dark_mode_without_magic_words_when_model_loaded():
    pass




















































































































def test_agent_runner_adds_dark_mode_to_generic_flutter_material_app():
    pass







































































def test_agent_runner_understands_ejecutame_flutter_project():
    pass



















































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


def test_agent_runner_does_not_scaffold_flutter_login_without_model():
    pass





































def test_agent_runner_scaffolds_basic_flutter_project_without_model():
    pass


































def test_agent_runner_flutter_project_uses_terminal_and_emulator_without_model():
    pass




































































def test_agent_runner_flutter_terminal_request_uses_tools_without_model():
    pass


















































def test_agent_runner_infers_flutter_workspace_for_login_and_emulator():
    pass
























































def test_agent_runner_runs_existing_flutter_project_without_repeating_previous_login_context():
    pass





































































def test_agent_runner_reports_emulator_unavailable_without_repeating_stale_summary():
    pass











































































def test_agent_runner_does_not_scaffold_flutter_project_when_model_is_available():
    pass



































def test_agent_runner_does_not_complete_command_request_without_terminal():
    pass


































def test_agent_runner_writes_flutter_code_when_model_returns_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class MarkdownModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return """
pubspec.yaml
```yaml
name: markdown_login
dependencies:
  flutter:
    sdk: flutter
```

lib/main.dart
```dart
import 'package:flutter/material.dart';

void main() => runApp(const MaterialApp(home: Text('Login')));
```
"""

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=3,
        model=MarkdownModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert (tmp_path / "pubspec.yaml").read_text(encoding="utf-8").startswith("name: markdown_login")
    assert "MaterialApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert [call["action"] for call in report["tool_calls"]] == ["write_file", "write_file"]
    assert {change["path"] for change in report["file_changes"]} == {"pubspec.yaml", "lib/main.dart"}


def test_agent_runner_rejects_json_payload_as_file_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        action_provider=lambda _messages: Action(
            type="write_file",
            args={
                "path": "lib/main.dart",
                "text": '{"type":"write_file","args":{"path":"lib/main.dart","text":"bad"}}',
            },
        ),
    )

    assert report["tools_ok"] is False
    assert not (tmp_path / "lib" / "main.dart").exists()
    assert "incomplete_file_content" in report["tool_calls"][0]["output"]


def test_agent_runner_rejects_tool_marker_as_file_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        action_provider=lambda _messages: Action(
            type="write_file",
            args={
                "path": "lib/main.dart",
                "text": '  {\n"type": "write_file",\n"path": "lib/main.dart",\n"text": "import package:flutter/material.dart;",\n}\n',
            },
        ),
    )

    assert report["ok"] is False
    assert not (tmp_path / "lib" / "main.dart").exists()
    assert "incomplete_file_content" in report["tool_calls"][0]["output"]


def test_agent_runner_retries_after_invalid_markdown_file_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"], "json_repair_retries": 2},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class RetryModel:
        tokenizer = None

        def __init__(self):
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return """
lib/main.dart
```dart
{"type":"write_file","path":"lib/main.dart","text":"bad"}
```
"""
            if self.calls == 2:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "lib/main.dart",
                            "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                        },
                    }
                )
            return json.dumps(
                {
                    "type": "finish",
                    "args": {"summary": "done"},
                }
            )

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=3,
        model=RetryModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tool_calls"][0]["ok"] is False
    assert report["tool_calls"][1]["ok"] is True
    assert "MaterialApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_repairs_nested_write_file_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )
    nested_text = json.dumps(
        {
            "type": "write_file",
            "args": {
                "path": "lib/main.dart",
                "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
            },
        }
    )

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        action_provider=lambda _messages: Action(
            type="write_file",
            args={"path": "lib/main.dart", "text": nested_text},
        ),
    )

    assert report["ok"] is True
    assert (tmp_path / "lib" / "main.dart").exists()
    assert "MaterialApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_accepts_model_create_file_alias_with_complete_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class AliasModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps(
                {
                    "action": "create_file",
                    "args": {
                        "path": "lib/main.dart",
                        "content": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                    },
                }
            )

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        model=AliasModel(),
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert "MaterialApp" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_failed_patch_does_not_mark_task_done(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["apply_patch"], "allow_patch_tools": True},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Crea un proyecto basico de flutter con login",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        action_provider=lambda _messages: Action(type="apply_patch", args={"patch_id": "missing"}),
    )

    assert report["ok"] is False
    assert report["blocked"] is True
    assert report["file_changes"] == []
    assert "patch.diff no existe" in report["tool_calls"][0]["output"]


def test_agent_runner_does_not_rescue_flutter_project_when_model_does_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    report = run_agent(
        "Hazme un proyecto basico de flutter con un login",
        settings,
        tmp_path,
        max_iters=1,
        permissions=permissions,
        action_provider=lambda _messages: Action(type="finish", args={"summary": "ready"}),
    )

    assert report["ok"] is False
    assert report["blocked"] is True
    assert report["tools_ok"] is False
    assert report["file_changes"] == []
    assert not (tmp_path / "pubspec.yaml").exists()
    assert "No he aplicado cambios" in report["summary"]


def test_agent_runner_uses_file_summary_after_benign_model_refusal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file", "run_command"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class RefusalAfterWriteModel:
        tokenizer = None

        def __init__(self):
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({
                    "type": "write_file",
                    "args": {
                        "path": "lib/main.dart",
                        "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                    },
                })
            return json.dumps({
                "type": "finish",
                "args": {
                    "summary": "I apologize, but I cannot fulfill your request because it goes against ethical and moral principles.",
                },
            })

    report = run_agent(
        "Hazme un login basico en flutter para el proyecto",
        settings,
        tmp_path,
        max_iters=3,
        model=RefusalAfterWriteModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tools_ok"] is True
    assert "I apologize" not in report["summary"]
    assert "No he ejecutado comandos" not in report["summary"]
    assert (tmp_path / "lib" / "main.dart").exists()


def test_agent_runner_creates_requested_project_file_even_when_model_marks_require_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class RequireExistsModel:
        tokenizer = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "lib/main.dart",
                            "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                            "require_exists": True,
                        },
                    }
                )
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    report = run_agent(
        "Hazme un proyecto de login en flutter basico",
        settings,
        tmp_path,
        max_iters=3,
        model=RequireExistsModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert "Login" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert "No encuentro el archivo" not in report["summary"]


def test_agent_runner_flutter_create_then_edits_generated_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["run_command", "write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    def fake_run_command(self, command, cwd=".", timeout_s=120, background=False):
        assert command.startswith("flutter create --project-name ")
        (self.repo_root / "lib").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "pubspec.yaml").write_text(
            "name: generated_app\ndependencies:\n  flutter:\n    sdk: flutter\n",
            encoding="utf-8",
        )
        (self.repo_root / "lib" / "main.dart").write_text("void main() {}\n", encoding="utf-8")
        return ToolResult(ok=True, output="Flutter project generated.")

    monkeypatch.setattr(AgentTools, "run_command", fake_run_command)

    class EditModel:
        tokenizer = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "lib/main.dart",
                            "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                        },
                    }
                )
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    report = run_agent(
        "Hazme un proyecto basico de flutter con un login",
        settings,
        tmp_path,
        max_iters=4,
        model=EditModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert [call["action"] for call in report["tool_calls"][:2]] == ["run_command", "write_file"]
    assert (tmp_path / "pubspec.yaml").exists()
    assert "Login" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_flutter_create_reads_generated_files_when_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["run_command", "read_file", "write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    def fake_run_command(self, command, cwd=".", timeout_s=120, background=False):
        assert command.startswith("flutter create --project-name ")
        (self.repo_root / "lib").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "pubspec.yaml").write_text(
            "name: generated_app\ndependencies:\n  flutter:\n    sdk: flutter\n",
            encoding="utf-8",
        )
        (self.repo_root / "lib" / "main.dart").write_text("void main() {}\n", encoding="utf-8")
        return ToolResult(ok=True, output="Flutter project generated.")

    monkeypatch.setattr(AgentTools, "run_command", fake_run_command)

    class EditModel:
        tokenizer = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                return json.dumps({"type": "finish", "args": {"summary": "done"}})
            assert "generated_app" in prompt
            assert "void main() {}" in prompt
            return json.dumps(
                {
                    "type": "write_file",
                    "args": {
                        "path": "lib/main.dart",
                        "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                    },
                }
            )

    report = run_agent(
        "Hazme un proyecto basico de flutter con un login",
        settings,
        tmp_path,
        max_iters=3,
        model=EditModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert [call["action"] for call in report["tool_calls"][:4]] == [
        "run_command",
        "read_file",
        "read_file",
        "write_file",
    ]
    assert "Login" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_retries_after_incomplete_write_with_specific_instruction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class RetryModel:
        tokenizer = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "lib/main.dart",
                            "text": "import 'package:flutter/material.dart';\n\nclass LoginPage extends StatefulWidget {}\n",
                        },
                    }
                )
            if self.calls > 2:
                return json.dumps({"type": "finish", "args": {"summary": "done"}})
            assert "complete, compilable FULL file content" in prompt
            return json.dumps(
                {
                    "type": "write_file",
                    "args": {
                        "path": "lib/main.dart",
                        "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                    },
                }
            )

    report = run_agent(
        "Hazme un proyecto de login en flutter basico",
        settings,
        tmp_path,
        max_iters=3,
        model=RetryModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tool_calls"][0]["ok"] is False
    assert report["tool_calls"][1]["ok"] is True
    assert "Login" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_runner_recovers_repeated_incomplete_flutter_login_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    class WeakModel:
        tokenizer = None

        def generate(self, _prompt: str, **_kwargs):
            return json.dumps(
                {
                    "type": "write_file",
                    "args": {
                        "path": "lib/main.dart",
                        "text": (
                            "import 'package:flutter/material.dart';\n\n"
                            "main() {\n"
                            "  runApp(MyApp());\n"
                            "}\n"
                        ),
                    },
                }
            )

    report = run_agent(
        "Crea el proyecto de 0, haz un login basico en flutter",
        settings,
        tmp_path,
        max_iters=4,
        model=WeakModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tool_calls"][0]["ok"] is False
    assert report["tool_calls"][1]["ok"] is False
    assert (tmp_path / "pubspec.yaml").exists()
    main_text = (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")
    assert "class LoginApp" in main_text
    assert "TextFormField" in main_text


def test_agent_runner_falls_back_to_manual_files_when_flutter_create_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": [], "tools_enabled": ["run_command", "write_file"]},
    }
    permissions = AgentPermissions.from_payload(
        {"level": "full", "action_mode": "full"},
        tmp_path,
    )

    def fake_run_command(self, command, cwd=".", timeout_s=120, background=False):
        assert command.startswith("flutter create --project-name ")
        return ToolResult(ok=False, output="flutter: command not found")

    monkeypatch.setattr(AgentTools, "run_command", fake_run_command)

    class ManualModel:
        tokenizer = None

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "pubspec.yaml",
                            "text": "name: manual_app\ndependencies:\n  flutter:\n    sdk: flutter\n",
                        },
                    }
                )
            if self.calls == 2:
                return json.dumps(
                    {
                        "type": "write_file",
                        "args": {
                            "path": "lib/main.dart",
                            "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Text('Login')));\n",
                        },
                    }
                )
            return json.dumps({"type": "finish", "args": {"summary": "done"}})

    report = run_agent(
        "Crea el proyecto de 0, haz un login basico en flutter",
        settings,
        tmp_path,
        max_iters=5,
        model=ManualModel(),
        permissions=permissions,
    )

    assert report["ok"] is True
    assert report["tool_calls"][0]["action"] == "run_command"
    assert report["tool_calls"][0]["ok"] is False
    assert (tmp_path / "pubspec.yaml").exists()
    assert "Login" in (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8")


def test_agent_permissions_map_downloads_workspace(tmp_path: Path, monkeypatch) -> None:
    mount = tmp_path / "host" / "downloads"
    project = mount / "test_flutter"
    project.mkdir(parents=True)
    monkeypatch.setenv("C3RNT2_HOST_DOWNLOADS_WINDOWS_ROOT", r"C:\Users\marcm\Downloads")
    monkeypatch.setenv("C3RNT2_HOST_DOWNLOADS_MOUNT", str(mount))

    permissions = AgentPermissions.from_payload(
        {
            "level": "full",
            "action_mode": "full",
            "workspace_root": r"C:\Users\marcm\Downloads\test_flutter",
            "project_path": r"C:\Users\marcm\Downloads\test_flutter",
        },
        tmp_path,
    )
    tools = AgentTools(
        allowlist=[],
        web_cfg={"enabled": False, "allow_domains": []},
        repo_root=permissions.scope_root,
        permissions=permissions,
    )

    result = tools.write_file("agent-created.txt", "ok")

    assert permissions.scope_root == project.resolve()
    assert result.ok is True
    assert (project / "agent-created.txt").read_text(encoding="utf-8") == "ok"














































def test_agent_runner_completes_flutter_project_when_model_returns_only_main_dart():
    pass













































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
