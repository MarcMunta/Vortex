from __future__ import annotations

from pathlib import Path

from c3rnt2.continuous.knowledge_store import KnowledgeStore
from c3rnt2.server import _inject_rag_context


def test_rag_disabled_no_injection(tmp_path: Path) -> None:
    settings = {"rag": {"enabled": False}}
    messages = [{"role": "user", "content": "hello"}]
    new_messages, prompt, rag = _inject_rag_context(tmp_path, settings, messages, None)
    assert new_messages == messages
    assert prompt is None
    assert rag["enabled"] is False


def test_rag_enabled_injects_context(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "data" / "continuous" / "knowledge.sqlite"
    store = KnowledgeStore(knowledge_path)
    store.ingest_text("web", "local", "alpha beta gamma", quality=0.8)
    settings = {
        "continuous": {"knowledge_path": knowledge_path},
        "rag": {"enabled": True, "top_k": 1, "max_chars": 10},
    }
    messages = [{"role": "user", "content": "alpha"}]
    new_messages, prompt, rag = _inject_rag_context(tmp_path, settings, messages, None)
    assert prompt is None
    assert new_messages[0]["role"] == "system"
    assert "CONTEXT" in new_messages[0]["content"]
    assert "UNTRUSTED CONTEXT" in new_messages[0]["content"]
    content = new_messages[0]["content"]
    ctx = content.split("CONTEXT:\n", 1)[-1].split("\nEND_CONTEXT", 1)[0]
    assert len(ctx) <= 10
    assert rag["enabled"] is True


def test_rag_respects_max_chars_and_system_prompt(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "data" / "continuous" / "knowledge.sqlite"
    store = KnowledgeStore(knowledge_path)
    store.ingest_text("web", "local", "alpha beta gamma delta epsilon", quality=0.8)
    settings = {
        "continuous": {"knowledge_path": knowledge_path},
        "rag": {"enabled": True, "top_k": 1, "max_chars": 5},
    }
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "alpha"}]
    new_messages, prompt, rag = _inject_rag_context(tmp_path, settings, messages, None)
    assert prompt is None
    assert new_messages[0]["content"] == "SYS"
    assert new_messages[1]["role"] == "system"
    content = new_messages[1]["content"]
    ctx = content.split("CONTEXT:\n", 1)[-1].split("\nEND_CONTEXT", 1)[0]
    assert len(ctx) <= 5
    assert rag["enabled"] is True
    again, _, _ = _inject_rag_context(tmp_path, settings, messages, None)
    assert again[1]["content"] == new_messages[1]["content"]


def test_rag_does_not_double_inject(tmp_path: Path) -> None:
    settings = {"rag": {"enabled": True, "top_k": 1}}
    messages = [{"role": "system", "content": "CONTEXT:\nfoo\nEND_CONTEXT"}, {"role": "user", "content": "foo"}]
    new_messages, prompt, rag = _inject_rag_context(tmp_path, settings, messages, None)
    assert new_messages == messages
    assert prompt is None
    assert rag["enabled"] is True
