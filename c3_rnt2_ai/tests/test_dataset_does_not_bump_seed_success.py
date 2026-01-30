from __future__ import annotations

import sqlite3
from pathlib import Path

from c3rnt2.continuous.dataset import collect_samples
from c3rnt2.continuous.knowledge_store import KnowledgeStore


def test_dataset_does_not_bump_seed_success(tmp_path: Path) -> None:
    base_dir = tmp_path
    data_dir = base_dir / "data"
    cont_dir = data_dir / "continuous"
    cont_dir.mkdir(parents=True, exist_ok=True)

    knowledge_path = cont_dir / "knowledge.sqlite"
    replay_path = cont_dir / "replay.sqlite"

    store = KnowledgeStore(knowledge_path)
    store.ingest_text("memory", "unit", "Seed chunk for testing.", quality=0.9)

    settings = {
        "continuous": {
            "knowledge_path": str(knowledge_path),
            "ingest_web": False,
            "filter": {"min_quality": 0.0, "max_repeat_ratio": 1.0},
            "replay": {"path": str(replay_path), "seed_chunks": 1, "sample_size": 1},
        },
        "rag": {"enabled": False},
    }

    collect_samples(base_dir, allowlist=[], settings=settings)

    with sqlite3.connect(replay_path) as conn:
        cur = conn.execute("SELECT COALESCE(SUM(success_count), 0) FROM replay")
        total = int(cur.fetchone()[0])
    assert total == 0
