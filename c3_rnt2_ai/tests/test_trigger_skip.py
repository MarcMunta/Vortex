from pathlib import Path

from c3rnt2.continuous.trainer import ContinualTrainer


def test_trigger_skip(tmp_path) -> None:
    settings = {
        "agent": {"web_allowlist": []},
        "continuous": {
            "knowledge_path": tmp_path / "knowledge.sqlite",
            "ingest_web": False,
            "trigger": {"enabled": True, "min_new_docs": 2, "min_novelty": 0.5, "min_successes": 1},
            "filter": {"min_quality": 0.9, "max_repeat_ratio": 0.1},
            "replay": {"path": tmp_path / "replay.sqlite", "sample_size": 2},
            "adapters": {"target_modules": ["lm_head"], "strict_target_modules": False},
            "eval": {"anchors_path": tmp_path / "anchors.jsonl", "max_regression": 0.2, "min_improvement": 0.0},
        },
    }
    trainer = ContinualTrainer(settings=settings, base_dir=tmp_path)
    result = trainer.run_tick()
    assert result.promoted is False
    assert result.samples == 0

