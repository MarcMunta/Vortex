from __future__ import annotations

from pathlib import Path

from c3rnt2.agent.tools import ToolResult
from c3rnt2.continuous import dataset as dataset_mod
from c3rnt2.continuous.dataset import ingest_sources
from c3rnt2.continuous.knowledge_store import KnowledgeStore


def test_web_canonicalize_url() -> None:
    url = "https://Example.com/docs/page/#section"
    assert dataset_mod._canonicalize_url(url) == "https://example.com/docs/page"


def test_web_ingest_chunks_and_source_ref(tmp_path: Path, monkeypatch) -> None:
    class DummyTools:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def open_docs(self, url: str) -> ToolResult:
            text = " ".join([f"word{i}" for i in range(120)])
            return ToolResult(ok=True, output=text)

    monkeypatch.setattr("c3rnt2.continuous.dataset.AgentTools", DummyTools)
    monkeypatch.setattr("c3rnt2.continuous.dataset._cosine_sim", lambda _a, _b: 0.0)

    knowledge_path = tmp_path / "data" / "continuous" / "knowledge.sqlite"
    settings = {
        "tools": {"web": {"enabled": True, "allow_domains": ["example.com"]}},
        "continuous": {
            "knowledge_path": str(knowledge_path),
            "ingest_web": True,
            "ingest_urls": ["https://Example.com/docs/page/#section"],
            "ingest": {"web": {"cooldown_minutes": 0, "sanitize": {"max_chars": 80}}},
        },
        "knowledge": {"embedding_backend": "hash"},
    }
    new_docs = ingest_sources(tmp_path, ["example.com"], settings)
    assert new_docs >= 1
    store = KnowledgeStore(knowledge_path)
    chunks = store.sample_chunks(limit=10, source_kinds=["web"])
    assert len(chunks) > 1
    assert all(chunk.source_ref.startswith("https://example.com/docs/page") for chunk in chunks)
    assert any("#chunk=" in chunk.source_ref for chunk in chunks)
