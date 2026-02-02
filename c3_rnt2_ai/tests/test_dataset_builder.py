from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.training.dataset_builder import build_sft_dataset
from c3rnt2.continuous.knowledge_store import KnowledgeChunk


def test_dataset_builder_dedup(tmp_path: Path) -> None:
    chunks = [
        KnowledgeChunk(text="alpha beta gamma", score=0.9, source_kind="web", source_ref="local"),
        KnowledgeChunk(text="alpha beta gamma", score=0.9, source_kind="web", source_ref="local"),
    ]
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(
        chunks=chunks,
        episodes_path=tmp_path / "episodes.jsonl",
        output_path=out,
        system_prompt="You are a helpful assistant.",
        min_chars=5,
        max_repeat_ratio=0.9,
        semantic_dedup_threshold=0.95,
        embedding_backend=None,
    )
    assert stats.written == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload.get("messages")
    roles = [m.get("role") for m in payload["messages"]]
    assert "system" in roles and "user" in roles
    assert payload.get("response")


def test_dataset_builder_doc_prompt_coherent(tmp_path: Path) -> None:
    chunks = [KnowledgeChunk(text="doc sample", score=0.9, source_kind="web", source_ref="local")]
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(
        chunks=chunks,
        episodes_path=tmp_path / "episodes.jsonl",
        output_path=out,
        system_prompt="You are a helpful assistant.",
        min_chars=1,
        max_repeat_ratio=0.99,
        semantic_dedup_threshold=0.99,
        embedding_backend=None,
    )
    assert stats.written == 1
    payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    user_msg = [m.get("content") for m in payload["messages"] if m.get("role") == "user"][0]
    assert "EXACTLY" in user_msg
    assert payload.get("response") == "doc sample"


def test_dataset_builder_chat_feedback(tmp_path: Path) -> None:
    chat_path = tmp_path / "data" / "episodes" / "chat.jsonl"
    feedback_path = tmp_path / "data" / "episodes" / "feedback.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    chat_path.write_text(
        json.dumps(
            {
                "version": 1,
                "ts": 1.0,
                "request_id": "req1",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_text": "hi",
                "response_text": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_path.write_text(
        json.dumps({"version": 1, "ts": 2.0, "request_id": "req1", "rating": "up", "ideal_response": "ideal"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sft_chat.jsonl"
    stats = build_sft_dataset(
        chunks=[],
        episodes_path=episodes_path,
        output_path=out,
        system_prompt="You are a helpful assistant.",
        chat_path=chat_path,
        feedback_path=feedback_path,
        training_path=None,
        min_chars=1,
        max_repeat_ratio=0.99,
        semantic_dedup_threshold=0.99,
        embedding_backend=None,
    )
    assert stats.written == 1
    payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert payload.get("response") == "ideal"
