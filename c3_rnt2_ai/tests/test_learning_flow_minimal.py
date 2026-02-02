from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.training.dataset_builder import build_sft_dataset


def test_learning_flow_builds_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "episodes"
    data_dir.mkdir(parents=True, exist_ok=True)
    chat_path = data_dir / "chat.jsonl"
    feedback_path = data_dir / "feedback.jsonl"
    training_path = data_dir / "training.jsonl"

    chat_payload = {
        "request_id": "req-1",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt_text": "hello",
        "response_text": "response text " * 5,
    }
    feedback_payload = {"request_id": "req-1", "rating": "up", "ideal_response": "ideal response " * 5}
    training_payload = {"messages": [{"role": "user", "content": "hi"}], "response": "training response " * 5}

    chat_path.write_text(json.dumps(chat_payload, ensure_ascii=True) + "\n", encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback_payload, ensure_ascii=True) + "\n", encoding="utf-8")
    training_path.write_text(json.dumps(training_payload, ensure_ascii=True) + "\n", encoding="utf-8")

    output_path = tmp_path / "data" / "registry" / "hf_train" / "sft_samples.jsonl"
    stats = build_sft_dataset(
        chunks=[],
        episodes_path=tmp_path / "data" / "episodes" / "agent.jsonl",
        output_path=output_path,
        system_prompt="SYS",
        chat_path=chat_path,
        feedback_path=feedback_path,
        training_path=training_path,
        min_chars=10,
    )
    assert output_path.exists()
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    assert stats.written > 0
