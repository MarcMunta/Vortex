from c3rnt2.continuous.dataset import retrieve_context
from c3rnt2.continuous.knowledge_store import KnowledgeStore


def test_rag_retrieve(tmp_path) -> None:
    knowledge_path = tmp_path / "knowledge.sqlite"
    store = KnowledgeStore(knowledge_path)
    store.ingest_text("web", "local", "alpha beta gamma", quality=0.8)
    store.ingest_text("web", "local", "delta epsilon zeta", quality=0.8)

    settings = {"continuous": {"knowledge_path": knowledge_path}, "rag": {"max_chars": 200}}
    context = retrieve_context(tmp_path, "alpha", settings, top_k=1)
    assert "alpha" in context

