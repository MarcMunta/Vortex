from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.agent.runner import run_agent, Action


def test_episode_schema_v2_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path
    target = repo / "src" / "c3rnt2" / "demo.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("value = 1\n", encoding="utf-8")

    settings = {
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "agent": {"web_allowlist": []},
        "self_patch": {"allowed_paths": ["src/"], "forbidden_globs": ["data/**"], "max_patch_kb": 64},
    }

    calls = {"count": 0}

    def provider(_messages):
        if calls["count"] == 0:
            calls["count"] += 1
            return Action(
                type="propose_patch",
                args={"goal": "update demo", "changes": {"src/c3rnt2/demo.py": "value = 2\n"}},
            )
        return Action(type="finish", args={"summary": "done"})

    report = run_agent("Update demo value", settings, repo, max_iters=2, action_provider=provider)
    assert report["ok"]

    episodes = repo / "data" / "episodes" / "agent.jsonl"
    payload = json.loads(episodes.read_text(encoding="utf-8").splitlines()[-1])
    assert payload.get("version") == 2
    assert "Task: Update demo value" in payload.get("prompt", "")
    assert payload.get("patch_id")
    patch = payload.get("patch", "")
    assert "demo.py" in patch
    assert patch.strip()
