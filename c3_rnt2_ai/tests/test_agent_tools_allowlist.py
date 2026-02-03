from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.agent.runner import Action, run_agent


def test_agent_tool_disabled_is_blocked(tmp_path: Path, monkeypatch) -> None:
    settings = {
        "agent": {"tools_enabled": ["read_file"]},
        "tools": {"web": {"allow_domains": []}},
        "selfimprove": {"sandbox_root": str(tmp_path / "sandbox")},
    }

    from c3rnt2.agent import tools as tools_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("run_tests should not be called")

    monkeypatch.setattr(tools_mod.AgentTools, "run_tests", _boom)

    def _provider(_messages):
        if not hasattr(_provider, "called"):
            _provider.called = True
            return Action(type="run_tests", args={})
        return Action(type="finish", args={"summary": "done"})

    run_agent("task", settings, tmp_path, max_iters=2, action_provider=_provider)

    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    payload = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    call = payload["tool_calls"][0]
    assert call["ok"] is False
    assert call["output"].startswith("tool_disabled:run_tests")


def test_agent_tool_unsupported_is_blocked(tmp_path: Path) -> None:
    settings = {"agent": {"tools_enabled": ["read_file"]}}

    def _provider(_messages):
        if not hasattr(_provider, "called"):
            _provider.called = True
            return Action(type="edit_repo", args={})
        return Action(type="finish", args={"summary": "done"})

    run_agent("task", settings, tmp_path, max_iters=2, action_provider=_provider)
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    payload = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    call = payload["tool_calls"][0]
    assert call["ok"] is False
    assert call["output"].startswith("tool_unsupported:edit_repo")
