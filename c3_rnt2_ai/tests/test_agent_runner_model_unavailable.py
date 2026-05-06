from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.agent.runner import run_agent


def test_agent_runner_returns_report_when_model_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3RNT2_NO_NET", "1")
    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": []},
    }

    report = run_agent(
        "Arregla el modo agente.",
        settings,
        tmp_path,
        max_iters=1,
        model=None,
        allow_model_load=False,
    )

    assert report["ok"] is False
    assert report["model_unavailable"] is True
    assert "agent_model_unavailable" in report["summary"]
    episode_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[-1])
    assert episode["model_unavailable"] is True
