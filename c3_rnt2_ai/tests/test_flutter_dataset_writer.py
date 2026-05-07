from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.flutter_docs.dataset_writer import write_datasets


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_domain_config(path: Path) -> None:
    config = {
        "system_prompt": "You are Vortex, a test assistant.",
        "domain_name": "flutter_official_docs",
        "dataset_files": {
            "general": "flutter_official_docs_sft.jsonl",
            "code": "flutter_official_docs_code_sft.jsonl",
            "debugging": "flutter_official_docs_debugging_sft.jsonl",
            "architecture": "flutter_official_docs_architecture_sft.jsonl",
        },
        "topics": {
            "debugging": ["constraints", "layout", "debugging"],
            "code": ["widgets", "forms_validation", "constraints"],
            "architecture": ["architecture", "clean_architecture"],
        },
        "hard_eval_templates": [
            {
                "prompt": "Mi `ListView` dentro de `Column` lanza error. Dame causa, fix y test.",
                "topics": ["constraints", "layout"],
                "rubric": ["mentions bounded constraints", "uses Expanded"],
            },
            {
                "prompt": "Tengo `RenderFlex overflowed`. Corrige.",
                "topics": ["layout"],
                "rubric": ["explains overflow", "uses Expanded"],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=True), encoding="utf-8")


def test_dataset_writer_generates_valid_jsonl(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    out = tmp_path / "datasets"
    config_path = tmp_path / "domain_config.json"
    _write_domain_config(config_path)
    _write_jsonl(
        chunks,
        [
            {
                "id": "c1",
                "url": "https://docs.flutter.dev/ui/layout/constraints",
                "title": "Constraints",
                "heading_path": ["UI", "Layout"],
                "text": "Flutter constraints are passed down and sizes go up. RenderBox needs bounded constraints.",
                "code_blocks": ["Expanded(child: ListView())"],
                "topic": "constraints",
                "difficulty": "intermediate",
                "source_kind": "flutter_official_docs",
            },
            {
                "id": "c2",
                "url": "https://docs.flutter.dev/app-architecture/guide",
                "title": "Architecture",
                "heading_path": ["Architecture"],
                "text": "Separate data, domain, and presentation layers for maintainable Flutter apps.",
                "code_blocks": [],
                "topic": "architecture",
                "difficulty": "intermediate",
                "source_kind": "flutter_official_docs",
            },
        ],
    )

    result = write_datasets(chunks, out, config_path)

    assert result["chunks"] == 2
    general = _read_jsonl(out / "flutter_official_docs_sft.jsonl")
    debugging = _read_jsonl(out / "flutter_official_docs_debugging_sft.jsonl")
    architecture = _read_jsonl(out / "flutter_official_docs_architecture_sft.jsonl")
    hard_eval = _read_jsonl(out / "flutter_official_docs_official_hard_eval.jsonl")
    assert general
    assert debugging
    assert architecture
    assert len(hard_eval) == 80
    row = general[0]
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["response"]
    assert row["source_ref"].startswith("https://docs.flutter.dev/")
    assert "flutter_official_docs" in row["source_kind"]
