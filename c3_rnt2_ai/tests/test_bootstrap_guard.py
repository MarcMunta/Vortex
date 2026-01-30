from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.continuous.trainer import ContinualTrainer


def test_train_requires_bootstrap(tmp_path: Path) -> None:
    settings = {"continuous": {}, "core": {}}
    trainer = ContinualTrainer(settings=settings, base_dir=tmp_path)
    result = trainer.run_tick(ingest=False)
    meta_path = tmp_path / "data" / "registry" / "runs" / result.run_id / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("reason") == "not_bootstrapped"
