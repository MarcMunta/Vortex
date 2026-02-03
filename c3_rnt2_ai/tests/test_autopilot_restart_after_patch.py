from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from c3rnt2 import autopilot as ap


def test_autopilot_restart_after_patch_writes_marker_and_exits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ap, "ingest_sources", lambda *_args, **_kwargs: 0)

    def _fake_autopatch(*_args, **_kwargs):
        return {"ok": True, "promoted": True, "branch": "autopilot/ts"}

    monkeypatch.setattr(ap, "_maybe_autopatch", _fake_autopatch)

    state_path = tmp_path / "data" / "state" / "autopilot.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "last_tick_ts": 0.0,
                "last_ingest_ts": 0.0,
                "last_train_ts": time.time(),
                "last_eval_ts": 0.0,
                "last_patch_ts": 0.0,
            }
        ),
        encoding="utf-8",
    )

    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "knowledge": {"embedding_backend": "hash"},
        "autopilot": {
            "enabled": True,
            "restart_after_patch": True,
            "patch_cooldown_minutes": 0,
            "train_cooldown_minutes": 9999,
            "training_jsonl_max_items": 0,
            "autopatch_enabled": True,
        },
        "hf_train": {"enabled": False},
    }

    with pytest.raises(SystemExit) as exc:
        ap.run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False)
    assert exc.value.code == 23

    restart_path = tmp_path / "data" / "state" / "restart.json"
    assert restart_path.exists()
    payload = json.loads(restart_path.read_text(encoding="utf-8"))
    assert payload.get("reason") == "autopatch_promoted"

