from __future__ import annotations

from pathlib import Path

from c3rnt2.continuous.knowledge_store import IngestPolicy, KnowledgeStore


def test_knowledge_store_policy_domains(tmp_path: Path) -> None:
    db_path = tmp_path / "ks.sqlite"
    policy = IngestPolicy(allow_domains=["example.com"], deny_domains=["blocked.com"])
    store = KnowledgeStore(db_path, embedding_backend="hash", index_backend="none", policy=policy)
    added_ok = store.ingest_text("web", "https://example.com/doc", "alpha beta", quality=0.8)
    added_blocked = store.ingest_text("web", "https://blocked.com/doc", "gamma delta", quality=0.8)
    assert added_ok == 1
    assert added_blocked == 0


def test_retrieve_deterministic(tmp_path: Path) -> None:
    db_path = tmp_path / "ks.sqlite"
    store = KnowledgeStore(db_path, embedding_backend="hash", index_backend="none")
    store.ingest_text("web", "local", "alpha beta gamma", quality=0.8)
    store.ingest_text("web", "local", "alpha beta", quality=0.8)
    first = store.retrieve("alpha", top_k=2)
    second = store.retrieve("alpha", top_k=2)
    assert [c.text for c in first] == [c.text for c in second]
