from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

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
    bucket: str | None = None
    difficulty_score: float | None = None
    metadata: dict[str, Any] | None = None


DEFAULT_BUCKET_BY_SOURCE: dict[str, str] = {
    "feedback": "chat_feedback",
    "chat_feedback": "chat_feedback",
    "chat_feedback_soft": "chat_feedback_soft",
    "episode": "episode",
    "repo": "repo",
    "docs": "docs",
    "lesson": "docs",
    "memory": "docs",
    "self_edit": "self_edit",
    "patch": "self_edit",
    "reflection": "autonomy_reflection",
    "autonomy_reflection": "autonomy_reflection",
}


class ReplayBuffer:
    @staticmethod
    def hash_sample(prompt: str, response: str) -> str:
        return _hash_sample(prompt, response)


    def __init__(self, db_path: Path, age_weight: float = 0.01):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.age_weight = float(age_weight)
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
                    created_ts REAL,
                    bucket TEXT DEFAULT 'chat_feedback',
                    difficulty_score REAL DEFAULT 0.5,
                    metadata_json TEXT DEFAULT '{}',
                    outcome_score REAL DEFAULT 0.0,
                    regression_count INTEGER DEFAULT 0,
                    rollback_count INTEGER DEFAULT 0,
                    last_outcome TEXT DEFAULT 'unknown',
                    last_train_ts REAL DEFAULT 0.0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_quality ON replay(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_novelty ON replay(novelty_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_created ON replay(created_ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_bucket ON replay(bucket)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_outcome ON replay(outcome_score)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id TEXT PRIMARY KEY,
                    ts REAL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_events (
                    event_id TEXT PRIMARY KEY,
                    sample_hash TEXT,
                    delta INTEGER,
                    ts REAL
                )
                """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_sample ON pending_events(sample_hash)")
            self._migrate_schema(conn)
            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(replay)").fetchall()
            if len(row) > 1
        }
        migrations = {
            "bucket": "ALTER TABLE replay ADD COLUMN bucket TEXT DEFAULT 'chat_feedback'",
            "difficulty_score": "ALTER TABLE replay ADD COLUMN difficulty_score REAL DEFAULT 0.5",
            "metadata_json": "ALTER TABLE replay ADD COLUMN metadata_json TEXT DEFAULT '{}'",
            "outcome_score": "ALTER TABLE replay ADD COLUMN outcome_score REAL DEFAULT 0.0",
            "regression_count": "ALTER TABLE replay ADD COLUMN regression_count INTEGER DEFAULT 0",
            "rollback_count": "ALTER TABLE replay ADD COLUMN rollback_count INTEGER DEFAULT 0",
            "last_outcome": "ALTER TABLE replay ADD COLUMN last_outcome TEXT DEFAULT 'unknown'",
            "last_train_ts": "ALTER TABLE replay ADD COLUMN last_train_ts REAL DEFAULT 0.0",
        }
        for column, ddl in migrations.items():
            if column not in columns:
                conn.execute(ddl)

    def _infer_bucket(self, item: ReplayItem) -> str:
        bucket = str(item.bucket or item.sample.bucket or "").strip().lower()
        if bucket:
            return bucket
        source_kind = str(item.source_kind or item.sample.source_kind or "unknown").strip().lower()
        return DEFAULT_BUCKET_BY_SOURCE.get(source_kind, source_kind or "chat_feedback")

    def _infer_difficulty(self, item: ReplayItem) -> float:
        if item.difficulty_score is not None:
            return max(0.0, min(1.0, float(item.difficulty_score)))
        if item.sample.difficulty is not None:
            return max(0.0, min(1.0, float(item.sample.difficulty)))
        prompt_len = len((item.sample.prompt or "").split())
        response_len = len((item.sample.response or "").split())
        source_kind = str(item.source_kind or item.sample.source_kind or "unknown").lower()
        base = 0.35
        if source_kind in {"episode", "repo", "self_edit", "autonomy_reflection"}:
            base += 0.15
        complexity = min(0.4, ((prompt_len * 0.015) + (response_len * 0.005)))
        return max(0.05, min(0.95, base + complexity))

    def _metadata_json(self, item: ReplayItem) -> str:
        payload = item.metadata or item.sample.metadata or {}
        try:
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except Exception:
            return "{}"

    def add(self, item: ReplayItem, max_items: Optional[int] = None) -> bool:
        digest = _hash_sample(item.sample.prompt, item.sample.response)
        now = time.time()
        bucket = self._infer_bucket(item)
        difficulty = self._infer_difficulty(item)
        metadata_json = self._metadata_json(item)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO replay
                (hash, prompt, response, source_kind, quality_score, novelty_score, last_used_ts, use_count, success_count, created_ts, bucket, difficulty_score, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    bucket,
                    float(difficulty),
                    metadata_json,
                ),
            )
            inserted = cur.rowcount > 0
            self._apply_pending(digest, conn)
            conn.commit()
        if inserted and max_items:
            self._enforce_max_items(max_items)
        return inserted

    def mark_event_processed(self, event_id: str) -> bool:
        if not event_id:
            return False
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO processed_events (event_id, ts) VALUES (?, ?)",
                (event_id, now),
            )
            conn.commit()
            return cur.rowcount > 0

    def bump_success_once(self, sample_hash: str, event_id: str, delta: int = 1) -> bool:
        if not event_id:
            return False
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cur_proc = conn.execute(
                "INSERT OR IGNORE INTO processed_events (event_id, ts) VALUES (?, ?)",
                (event_id, now),
            )
            if cur_proc.rowcount == 0:
                return False
            cur_upd = conn.execute(
                "UPDATE replay SET success_count = success_count + ? WHERE hash = ?",
                (int(delta), sample_hash),
            )
            if cur_upd.rowcount == 0:
                conn.execute("DELETE FROM processed_events WHERE event_id = ?", (event_id,))
                conn.execute(
                    "INSERT OR IGNORE INTO pending_events (event_id, sample_hash, delta, ts) VALUES (?, ?, ?, ?)",
                    (event_id, sample_hash, int(delta), now),
                )
                conn.commit()
                return False
            conn.commit()
            return True

    def _apply_pending(self, digest: str, conn: sqlite3.Connection | None = None) -> None:
        own_conn = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT event_id, delta FROM pending_events WHERE sample_hash = ?",
            (digest,),
        ).fetchall()
        if not rows:
            if own_conn:
                conn.commit()
                conn.close()
            return
        now = time.time()
        for event_id, delta in rows:
            cur_proc = conn.execute(
                "INSERT OR IGNORE INTO processed_events (event_id, ts) VALUES (?, ?)",
                (event_id, now),
            )
            if cur_proc.rowcount == 0:
                conn.execute("DELETE FROM pending_events WHERE event_id = ?", (event_id,))
                continue
            cur_upd = conn.execute(
                "UPDATE replay SET success_count = success_count + ? WHERE hash = ?",
                (int(delta), digest),
            )
            if cur_upd.rowcount == 0:
                conn.execute("DELETE FROM processed_events WHERE event_id = ?", (event_id,))
                continue
            conn.execute("DELETE FROM pending_events WHERE event_id = ?", (event_id,))
        if own_conn:
            conn.commit()
            conn.close()

    def update_success(self, digest: str, delta: int = 1) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE replay SET success_count = success_count + ? WHERE hash = ?",
                (int(delta), digest),
            )
            conn.commit()

    def update_outcome(
        self,
        digest: str,
        *,
        improved_eval: bool = False,
        improved_bench: bool = False,
        regression: bool = False,
        rollback: bool = False,
        outcome_label: str | None = None,
    ) -> None:
        delta = 0.0
        if improved_eval:
            delta += 0.6
        if improved_bench:
            delta += 0.8
        if regression:
            delta -= 0.8
        if rollback:
            delta -= 1.0
        label = outcome_label or (
            "rollback"
            if rollback
            else "regression"
            if regression
            else "improved"
            if (improved_eval or improved_bench)
            else "observed"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE replay
                SET
                    outcome_score = outcome_score + ?,
                    regression_count = regression_count + ?,
                    rollback_count = rollback_count + ?,
                    last_outcome = ?,
                    last_train_ts = ?
                WHERE hash = ?
                """,
                (
                    float(delta),
                    int(1 if regression else 0),
                    int(1 if rollback else 0),
                    str(label),
                    time.time(),
                    digest,
                ),
            )
            conn.commit()

    def _composite_score_sql(self) -> str:
        age_days = "((strftime('%s','now') - created_ts) / 86400.0)"
        recency_bonus = f"MAX(0.0, 1.8 - ({age_days} * {self.age_weight}))"
        use_penalty = "((use_count + 1.0) * 0.42)"
        return (
            "(quality_score * 4.8)"
            " + (novelty_score * 3.1)"
            " + (success_count * 1.75)"
            " + (outcome_score * 2.4)"
            " + (difficulty_score * 0.9)"
            f" + ({recency_bonus})"
            " - (regression_count * 2.6)"
            " - (rollback_count * 3.0)"
            f" - ({use_penalty} * 1.2)"
        )

    def _enforce_max_items(self, max_items: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM replay")
            count = int(cur.fetchone()[0])
            if count <= max_items:
                return
            overflow = count - max_items
            conn.execute(
                f"""
                DELETE FROM replay
                WHERE hash IN (
                    SELECT hash FROM replay
                    ORDER BY ({self._composite_score_sql()}) ASC,
                             created_ts ASC
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

    def _sample_random_rows(
        self,
        conn: sqlite3.Connection,
        count: int,
        source_weights: dict[str, float] | None = None,
    ) -> List[Sample]:
        if count <= 0:
            return []
        source_weights = source_weights or {}

        # If source weights are configured, sample from a randomized pool with weighted picks.
        if source_weights:
            pool_limit = max(200, count * 25)
            cur = conn.execute(
                """
                SELECT prompt, response, source_kind
                FROM replay
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (pool_limit,),
            )
            pool_rows = cur.fetchall()
            if not pool_rows:
                return []

            candidates: list[tuple[str, str, str]] = [
                (str(prompt), str(response), str(source_kind or "unknown"))
                for prompt, response, source_kind in pool_rows
            ]
            samples: List[Sample] = []
            while candidates and len(samples) < count:
                weights = [
                    max(0.0, float(source_weights.get(kind, 1.0)))
                    for _prompt, _response, kind in candidates
                ]
                if sum(weights) <= 0.0:
                    picked_idx = random.randrange(len(candidates))
                else:
                    picked_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                prompt, response, source_kind = candidates.pop(picked_idx)
                samples.append(Sample(prompt=prompt, response=response, source_kind=source_kind))
            return samples

        cur = conn.execute("SELECT MIN(rowid), MAX(rowid) FROM replay")
        row = cur.fetchone()
        if not row or row[0] is None or row[1] is None:
            return []
        min_id, max_id = int(row[0]), int(row[1])
        samples: List[Sample] = []
        seen: set[int] = set()
        tries = 0
        while len(samples) < count and tries < count * 5:
            tries += 1
            rid = random.randint(min_id, max_id)
            if rid in seen:
                continue
            cur = conn.execute(
                "SELECT rowid, prompt, response, source_kind FROM replay WHERE rowid >= ? LIMIT 1",
                (rid,),
            )
            found = cur.fetchone()
            if not found:
                continue
            rowid, prompt, response, source_kind = found
            if rowid in seen:
                continue
            seen.add(int(rowid))
            samples.append(Sample(prompt=prompt, response=response, source_kind=str(source_kind or "unknown")))
        return samples

    def sample(
        self,
        batch_size: int,
        top_frac: float = 0.7,
        random_frac: float = 0.3,
        source_weights: dict[str, float] | None = None,
    ) -> List[Sample]:
        if batch_size <= 0:
            return []
        top_n = max(0, int(batch_size * top_frac))
        rand_n = max(0, int(batch_size * random_frac))
        if top_n + rand_n < batch_size:
            rand_n = batch_size - top_n
        if top_n == 0 and rand_n == 0:
            top_n = min(1, batch_size)
        samples: List[Sample] = []
        score_sql = self._composite_score_sql()
        with sqlite3.connect(self.db_path) as conn:
            if top_n > 0:
                elite_n = max(1, int(math.ceil(top_n * 0.35)))
                recent_n = max(1, int(math.ceil(top_n * 0.25)))
                underused_n = max(1, int(math.ceil(top_n * 0.20)))
                negatives_n = max(0, top_n - elite_n - recent_n - underused_n)

                queries: list[tuple[str, tuple[Any, ...]]] = [
                    (
                        f"""
                        SELECT prompt, response, source_kind
                        FROM replay
                        ORDER BY ({score_sql}) DESC, created_ts DESC
                        LIMIT ?
                        """,
                        (elite_n,),
                    ),
                    (
                        f"""
                        SELECT prompt, response, source_kind
                        FROM replay
                        WHERE created_ts >= ?
                        ORDER BY ({score_sql}) DESC, created_ts DESC
                        LIMIT ?
                        """,
                        (time.time() - 86400.0 * 7.0, recent_n),
                    ),
                    (
                        f"""
                        SELECT prompt, response, source_kind
                        FROM replay
                        WHERE use_count <= 1 AND (quality_score + novelty_score + difficulty_score) >= 1.1
                        ORDER BY ({score_sql}) DESC, created_ts DESC
                        LIMIT ?
                        """,
                        (underused_n,),
                    ),
                ]
                if negatives_n > 0:
                    queries.append(
                        (
                            """
                            SELECT prompt, response, source_kind
                            FROM replay
                            WHERE regression_count > 0 OR rollback_count > 0 OR outcome_score < 0
                            ORDER BY created_ts DESC, outcome_score ASC
                            LIMIT ?
                            """,
                            (negatives_n,),
                        )
                    )
                for sql, params in queries:
                    for prompt, response, source_kind in conn.execute(sql, params).fetchall():
                        samples.append(
                            Sample(
                                prompt=prompt,
                                response=response,
                                source_kind=str(source_kind or "unknown"),
                            )
                        )
            if rand_n > 0:
                samples.extend(
                    self._sample_random_rows(
                        conn,
                        rand_n,
                        source_weights=source_weights,
                    )
                )
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
