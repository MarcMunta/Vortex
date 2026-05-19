from __future__ import annotations

import json
from pathlib import Path

import pytest

from c3rnt2.agent.grammar import build_agent_action_json_grammar
from c3rnt2.training.agent_tool_dataset import build_agent_tool_samples, write_agent_tool_dataset


def test_agent_action_grammar_loads_in_llama_cpp_when_available() -> None:
    llama_cpp = pytest.importorskip("llama_cpp")

    grammar = llama_cpp.LlamaGrammar.from_string(build_agent_action_json_grammar())

    assert grammar is not None


def test_agent_tool_dataset_contains_valid_action_json() -> None:
    rows = build_agent_tool_samples(target_count=24)

    assert len(rows) >= 24
    for row in rows:
        payload = json.loads(row["response"])
        assert set(payload) == {"type", "args"}
        assert isinstance(payload["type"], str)
        assert isinstance(payload["args"], dict)
        assert row["source_kind"] == "agent_tool_use_sft"


def test_write_agent_tool_dataset(tmp_path: Path) -> None:
    out = tmp_path / "agent_tool_use_sft.jsonl"

    result = write_agent_tool_dataset(out, target_count=24)

    assert result["ok"] is True
    assert result["samples"] >= 24
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result["samples"]
    assert json.loads(lines[0])["source_kind"] == "agent_tool_use_sft"
