from __future__ import annotations

import difflib
import json
import time
from pathlib import Path
from typing import Dict

from .memory import MemoryStore
from .tools import AgentTools


def _setup_demo_repo(base: Path) -> Path:
    repo = base / "agent_demo_repo"
    repo.mkdir(parents=True, exist_ok=True)
    calc = repo / "calc.py"
    test = repo / "test_calc.py"
    if not calc.exists():
        calc.write_text(
            "def add(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
    if not test.exists():
        test.write_text(
            "from calc import add\n\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
    return repo


def _log_episode(base_dir: Path, payload: dict) -> None:
    path = base_dir / "data" / "episodes" / "agent.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_demo_agent(settings: dict) -> Dict[str, str]:
    agent_cfg = settings.get("agent", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    allowlist = web_cfg.get("allow_domains") or agent_cfg.get("web_allowlist", ["docs.python.org"])
    sandbox_root = Path(settings.get("selfimprove", {}).get("sandbox_root", "data/workspaces"))
    rate_limit = int(web_cfg.get("rate_limit_per_min", agent_cfg.get("rate_limit_per_min", 30)))
    tools = AgentTools(
        allowlist=allowlist,
        sandbox_root=sandbox_root,
        rate_limit_per_min=rate_limit,
        web_cfg=tools_cfg,
    )
    repo = _setup_demo_repo(sandbox_root)

    memory = MemoryStore(Path("data") / "memory" / "agent_memory.sqlite")
    memory.add("Bugfix add function to use addition, not subtraction")

    # Step 1: open docs (best-effort)
    docs = tools.open_docs("https://docs.python.org/3/faq/programming.html")

    # Step 2: fix bug
    before = (repo / "calc.py").read_text(encoding="utf-8")
    fixed = "def add(a, b):\n    return a + b\n"
    edit_result = tools.edit_repo(repo / "calc.py", fixed)
    patch_id_result = tools.propose_patch(repo, {Path("calc.py"): fixed}, goal="fix add bug")
    after = fixed
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="calc.py",
            tofile="calc.py",
            lineterm="",
        )
    )

    # Step 3: run tests
    test_result = tools.run_tests(repo)

    summary = {
        "docs_ok": str(docs.ok),
        "docs_excerpt": docs.output[:200],
        "edit_ok": str(edit_result.ok),
        "patch_id": patch_id_result.output,
        "tests_ok": str(test_result.ok),
        "tests_output": test_result.output[:400],
    }
    episode = {
        "task": "Fix calc.add bug",
        "prompt": docs.output[:400],
        "patch": diff,
        "tests_ok": test_result.ok,
        "tests_output_excerpt": test_result.output[:400],
        "timestamp": time.time(),
    }
    _log_episode(Path("."), episode)
    return summary
