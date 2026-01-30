from __future__ import annotations

import sqlite3
from pathlib import Path

from c3rnt2.continuous.replay_buffer import ReplayBuffer, ReplayItem
from c3rnt2.continuous.types import Sample


def _get_success_count(db_path: Path, digest: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT success_count FROM replay WHERE hash = ?", (digest,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def test_replay_success_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.sqlite"
    replay = ReplayBuffer(db_path)
    sample = Sample(prompt="p", response="r")
    item = ReplayItem(sample=sample, source_kind="episode", quality_score=0.9, novelty_score=0.9, success_count=0)
    replay.add(item)
    digest = replay.hash_sample(sample.prompt, sample.response)

    assert replay.bump_success_once(digest, "event-1", delta=1) is True
    assert replay.bump_success_once(digest, "event-1", delta=1) is False
    assert replay.bump_success_once(digest, "event-1", delta=1) is False
    assert _get_success_count(db_path, digest) == 1

    assert replay.bump_success_once(digest, "event-2", delta=1) is True
    assert _get_success_count(db_path, digest) == 2
