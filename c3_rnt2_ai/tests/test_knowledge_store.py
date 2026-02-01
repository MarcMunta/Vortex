import sqlite3

from c3rnt2.continuous.knowledge_store import KnowledgeStore, _vec_to_blob


def test_retrieve_basic(tmp_path) -> None:
    db_path = tmp_path / "ks.sqlite"
    store = KnowledgeStore(db_path, embedding_backend="hash", index_backend="none")
    store.ingest_text("web", "local", "alpha beta gamma", quality=0.8)
    store.ingest_text("web", "local", "delta epsilon zeta", quality=0.8)
    results = store.retrieve("alpha", top_k=1)
    assert results
    assert "alpha" in results[0].text


def test_vec_blob_migration(tmp_path) -> None:
    db_path = tmp_path / "ks.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                source_kind TEXT,
                source_ref TEXT,
                text TEXT,
                hash TEXT UNIQUE,
                ts REAL,
                vec_json TEXT,
                quality REAL,
                tokens_est INTEGER,
                use_count INTEGER DEFAULT 0
            )
            """
        )
        conn.execute("INSERT INTO docs (source_kind, source_ref, text, hash, ts, vec_json, quality, tokens_est) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     ("web", "local", "hello", "hash", 0.0, "[1.0, 0.0]", 0.5, 2))
        conn.commit()
    store = KnowledgeStore(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT vec_blob FROM docs WHERE hash = ?", ("hash",))
        row = cur.fetchone()
        assert row is not None
        assert row[0] is not None
