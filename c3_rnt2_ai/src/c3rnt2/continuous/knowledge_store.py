from __future__ import annotations

import json
import math
import sqlite3
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def embed_text(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        digest = hashlib.sha256(tok.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    return vec


def _normalize_vec(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-9:
        return vec
    return [v / norm for v in vec]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class KnowledgeChunk:
    text: str
    score: float
    source_kind: str
    source_ref: str


class KnowledgeStore:
    def __init__(self, db_path: Path, dim: int = 128, cache_limit: int = 2000):
        self.db_path = db_path
        self.dim = dim
        self.cache_limit = cache_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON docs(hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_tokens ON docs(tokens_est)")
            conn.commit()

    def _chunk_text(self, text: str, min_chars: int = 800, max_chars: int = 1500, overlap: int = 200) -> List[str]:
        cleaned = _normalize_text(text)
        if len(cleaned) <= max_chars:
            return [cleaned] if cleaned else []
        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + max_chars)
            chunk = cleaned[start:end]
            if len(chunk) < min_chars and chunks:
                chunks[-1] = (chunks[-1] + " " + chunk).strip()
                break
            chunks.append(chunk.strip())
            if end >= len(cleaned):
                break
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    def ingest_text(
        self,
        source_kind: str,
        source_ref: str,
        text: str,
        quality: float = 0.5,
        ts: Optional[float] = None,
    ) -> int:
        ts_val = ts if ts is not None else time.time()
        added = 0
        for chunk in self._chunk_text(text):
            normalized = _normalize_text(chunk)
            if not normalized:
                continue
            digest = _hash_text(normalized)
            vec = _normalize_vec(embed_text(normalized, dim=self.dim))
            tokens_est = max(1, len(_tokenize(normalized)))
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO docs (source_kind, source_ref, text, hash, ts, vec_json, quality, tokens_est)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (source_kind, source_ref, normalized, digest, ts_val, json.dumps(vec), quality, tokens_est),
                )
                if cur.rowcount:
                    added += 1
                conn.commit()
        return added

    def count_new_since(self, since_ts: float) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM docs WHERE ts > ?", (since_ts,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def retrieve(self, query: str, top_k: int = 5, min_quality: float = 0.0) -> List[KnowledgeChunk]:
        qvec = _normalize_vec(embed_text(query, dim=self.dim))
        qtokens = max(1, len(_tokenize(query)))
        min_tokens = max(1, int(qtokens * 0.3))
        max_tokens = max(50, int(qtokens * 8.0))
        candidates: List[Tuple[str, str, str, str]] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT text, vec_json, source_kind, source_ref
                FROM docs
                WHERE tokens_est BETWEEN ? AND ? AND quality >= ?
                """,
                (min_tokens, max_tokens, min_quality),
            )
            candidates = cur.fetchall()
        scored: List[KnowledgeChunk] = []
        for text, vec_json, source_kind, source_ref in candidates:
            try:
                vec = json.loads(vec_json)
            except Exception:
                continue
            score = _dot(qvec, vec)
            scored.append(KnowledgeChunk(text=text, score=score, source_kind=source_kind, source_ref=source_ref))
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[:top_k]
        self._bump_use_count(top)
        return top

    def _bump_use_count(self, chunks: List[KnowledgeChunk]) -> None:
        if not chunks:
            return
        hashes = [_hash_text(chunk.text) for chunk in chunks]
        with sqlite3.connect(self.db_path) as conn:
            for digest in hashes:
                conn.execute("UPDATE docs SET use_count = use_count + 1 WHERE hash = ?", (digest,))
            conn.commit()

    def sample_chunks(self, limit: int = 50, min_quality: float = 0.0) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT text, source_kind, source_ref, quality
                FROM docs
                WHERE quality >= ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (min_quality, limit),
            )
            for text, source_kind, source_ref, quality in cur.fetchall():
                chunks.append(KnowledgeChunk(text=text, score=float(quality), source_kind=source_kind, source_ref=source_ref))
        return chunks
