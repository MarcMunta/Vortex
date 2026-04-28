from __future__ import annotations

import json
import os
from pathlib import Path

from c3rnt2.agent.permissions import AgentPermissions
from c3rnt2.agent.runner import run_agent, Action
from c3rnt2.agent.tools import AgentTools


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

    assert read_result.ok is True
    assert command_result.ok is False
    assert "permission_denied:command" in command_result.output
    assert apply_result.ok is False
    assert "permission_denied:write" in apply_result.output
    assert write_result.ok is False
    assert "permission_denied:write" in write_result.output


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
    browser_result = tools.open_browser("http://localhost:4173")

    assert write_result.ok is True
    assert (tmp_path / "lib" / "main.dart").read_text(encoding="utf-8") == "void main() {}\n"
    assert browser_result.ok is True
    assert tools.browser_actions == [
        {"target": "http://localhost:4173", "opened": True, "backend_opened": False}
    ]


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
