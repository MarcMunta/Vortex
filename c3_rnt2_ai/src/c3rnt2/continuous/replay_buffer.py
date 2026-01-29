from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .types import Sample


def _hash_sample(prompt: str, response: str) -> str:
    text = f"{prompt}\n{response}".strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ReplayItem:
    sample: Sample
    source_kind: str
    quality_score: float
    novelty_score: float
    success_count: int


class ReplayBuffer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay (
                    hash TEXT PRIMARY KEY,
                    prompt TEXT,
                    response TEXT,
                    source_kind TEXT,
                    quality_score REAL,
                    novelty_score REAL,
                    last_used_ts REAL,
                    use_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_ts REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_quality ON replay(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_novelty ON replay(novelty_score)")
            conn.commit()

    def add(self, item: ReplayItem, max_items: Optional[int] = None) -> bool:
        digest = _hash_sample(item.sample.prompt, item.sample.response)
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO replay
                (hash, prompt, response, source_kind, quality_score, novelty_score, last_used_ts, use_count, success_count, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    item.sample.prompt,
                    item.sample.response,
                    item.source_kind,
                    float(item.quality_score),
                    float(item.novelty_score),
                    0.0,
                    0,
                    int(item.success_count),
                    now,
                ),
            )
            inserted = cur.rowcount > 0
            conn.commit()
        if inserted and max_items:
            self._enforce_max_items(max_items)
        return inserted

    def _enforce_max_items(self, max_items: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM replay")
            count = int(cur.fetchone()[0])
            if count <= max_items:
                return
            overflow = count - max_items
            conn.execute(
                """
                DELETE FROM replay
                WHERE hash IN (
                    SELECT hash FROM replay
                    ORDER BY (quality_score + novelty_score + success_count) ASC, created_ts ASC
                    LIMIT ?
                )
                """,
                (overflow,),
            )
            conn.commit()

    def recent_texts(self, limit: int = 50) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT prompt, response FROM replay
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [f"{row[0]}\n{row[1]}".strip() for row in cur.fetchall()]

    def sample(self, batch_size: int, top_frac: float = 0.7, random_frac: float = 0.3) -> List[Sample]:
        if batch_size <= 0:
            return []
        top_n = max(0, int(batch_size * top_frac))
        rand_n = max(0, int(batch_size * random_frac))
        if top_n + rand_n < batch_size:
            rand_n = batch_size - top_n
        if top_n == 0 and rand_n == 0:
            top_n = min(1, batch_size)
        samples: List[Sample] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT hash, prompt, response FROM replay
                ORDER BY (quality_score + novelty_score + success_count) DESC
                LIMIT ?
                """,
                (top_n,),
            )
            top_rows = cur.fetchall()
            remaining = batch_size - len(top_rows)
            rand_n = max(rand_n, remaining)
            for _hash, prompt, response in top_rows:
                samples.append(Sample(prompt=prompt, response=response))
            if rand_n > 0:
                cur = conn.execute(
                    """
                    SELECT hash, prompt, response FROM replay
                    ORDER BY RANDOM()
                    LIMIT ?
                    """,
                    (rand_n,),
                )
                for _hash, prompt, response in cur.fetchall():
                    samples.append(Sample(prompt=prompt, response=response))
        unique: dict[str, Sample] = {}
        for sample in samples:
            digest = _hash_sample(sample.prompt, sample.response)
            unique[digest] = sample
        final = list(unique.values())[:batch_size]
        self._mark_used(final)
        return final

    def _mark_used(self, samples: List[Sample]) -> None:
        if not samples:
            return
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            for sample in samples:
                digest = _hash_sample(sample.prompt, sample.response)
                conn.execute(
                    """
                    UPDATE replay
                    SET use_count = use_count + 1, last_used_ts = ?
                    WHERE hash = ?
                    """,
                    (now, digest),
                )
            conn.commit()

    def count_new_since(self, since_ts: float) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM replay WHERE created_ts > ?", (since_ts,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
