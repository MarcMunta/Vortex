import sqlite3

from c3rnt2.continuous.types import Sample
from c3rnt2.continuous.replay_buffer import ReplayBuffer, ReplayItem


def test_replay_dedup_priority(tmp_path) -> None:
    path = tmp_path / "replay.sqlite"
    buffer = ReplayBuffer(path)

    high = ReplayItem(sample=Sample(prompt="p1", response="r1"), source_kind="episode", quality_score=0.9, novelty_score=0.9, success_count=1)
    low = ReplayItem(sample=Sample(prompt="p2", response="r2"), source_kind="logs", quality_score=0.1, novelty_score=0.1, success_count=0)

    assert buffer.add(high) is True
    assert buffer.add(high) is False  # dedup
    assert buffer.add(low) is True

    with sqlite3.connect(path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM replay")
        assert int(cur.fetchone()[0]) == 2

    samples = buffer.sample(batch_size=1, top_frac=1.0, random_frac=0.0)
    assert len(samples) == 1
    assert samples[0].response == "r1"
