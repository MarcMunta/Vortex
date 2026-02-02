from __future__ import annotations

import json
import time
from pathlib import Path

from c3rnt2.learning_loop.data_collector import collect_from_episodes


def test_learning_loop_collect_from_episodes_reads_patch_or_patch_id(tmp_path: Path) -> None:
    patch_id = "abc123"
    queue_dir = tmp_path / "data" / "self_patch" / "queue" / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_text = "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@\n-old\n+new\n"
    (queue_dir / "patch.diff").write_text(patch_text, encoding="utf-8")

    episodes_path = tmp_path / "data" / "episodes" / "agent.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episode = {
        "version": 2,
        "ts": time.time(),
        "task": "demo task",
        "prompt": "Task: demo task",
        "patch_id": patch_id,
        "patch": "",
        "tests_ok": True,
        "tools_ok": False,
        "summary": "ok",
        "tool_calls": [],
    }
    episodes_path.write_text(json.dumps(episode, ensure_ascii=True) + "\n", encoding="utf-8")

    result = collect_from_episodes(tmp_path, settings={}, max_events=10)
    assert result.ok
    payload = json.loads(result.output_path.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["response"] == patch_text
