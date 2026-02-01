from __future__ import annotations

from pathlib import Path

from c3rnt2.continuous.knowledge_store import KnowledgeStore


def test_knowledge_store_fts_rebuild_on_create(tmp_path: Path) -> None:
    db_path = tmp_path / "knowledge.sqlite"
    store = KnowledgeStore(db_path, embedding_backend="hash", index_backend="none")
    store.ingest_text("logs", "unit", "hello world", quality=0.5)
    reopened = KnowledgeStore(db_path, embedding_backend="hash", index_backend="none")
    assert reopened.fts_rebuilds == 0
