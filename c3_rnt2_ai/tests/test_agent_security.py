from __future__ import annotations

from pathlib import Path

from c3rnt2.agent.policies import WebPolicy
from c3rnt2.agent.tools import AgentTools


def test_webpolicy_domain_boundary():
    policy = WebPolicy(allowlist=["github.com"])
    assert policy.allow_url("https://github.com/openai")
    assert policy.allow_url("https://sub.github.com/repo")
    assert not policy.allow_url("https://evilgithub.com")
    assert not policy.allow_url("https://github.com.evil.com")


def test_edit_repo_sandbox_only(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    tools = AgentTools(allowlist=["github.com"], sandbox_root=sandbox)
    outside = tmp_path / "outside" / "file.txt"
    result = tools.edit_repo(outside, "data")
    assert result.ok
    assert not outside.exists()
    created = Path(result.output)
    assert sandbox in created.parents
    assert created.read_text(encoding="utf-8") == "data"
