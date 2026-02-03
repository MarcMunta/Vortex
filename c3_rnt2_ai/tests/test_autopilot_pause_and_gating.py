from __future__ import annotations

import json
from pathlib import Path

import pytest

from c3rnt2.autopilot import run_autopilot_tick


def test_autopilot_pause_file_skips_tick(tmp_path: Path) -> None:
    pause = tmp_path / "data" / "state" / "PAUSE_AUTOPILOT"
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text("1", encoding="utf-8")

    settings = {"_profile": "autonomous_4080_hf", "autopilot": {"enabled": True}}
    res = run_autopilot_tick(settings, tmp_path, no_web=True, mock=True, force=False)
    assert res.ok is True
    assert res.steps.get("skipped") == "paused"


def test_autopilot_gates_training_when_no_new_samples(tmp_path: Path, monkeypatch) -> None:
    # Force a stable dataset size across ticks -> delta=0 -> training must be skipped.
    monkeypatch.setattr("c3rnt2.autopilot.ingest_sources", lambda *_a, **_k: 0)
    monkeypatch.setattr("c3rnt2.autopilot._build_sft_dataset", lambda *_a, **_k: {"ok": True, "stats": {"written": 10}})

    called = {"train": 0}

    def _train_runner(_profile: str, reuse_dataset: bool, max_steps: int | None):
        _ = _profile, reuse_dataset, max_steps
        called["train"] += 1
        return {"ok": True, "ok_train": True, "ok_eval": True, "eval_ok": True, "improvement": 0.0}

    settings = {
        "_profile": "autonomous_4080_hf",
        "autopilot": {
            "enabled": True,
            "train_cooldown_minutes": 0,
            "min_new_samples_per_tick": 1,
            "training_jsonl_max_items": 0,
        },
        "continuous": {"ingest_web": False},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "hf_train": {"enabled": False},
    }
    # First tick sets baseline written=10.
    _ = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_runner)
    # Second tick has delta=0 -> skip training.
    res2 = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_runner)
    assert res2.ok is True
    assert isinstance(res2.steps.get("train"), dict)
    assert res2.steps["train"].get("skipped") == "insufficient_new_samples"
    assert called["train"] == 1


def test_autopilot_enters_safe_mode_after_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("c3rnt2.autopilot.ingest_sources", lambda *_a, **_k: 0)
    monkeypatch.setattr("c3rnt2.autopilot._build_sft_dataset", lambda *_a, **_k: {"ok": True, "stats": {"written": 10}})

    def _train_fail(_profile: str, reuse_dataset: bool, max_steps: int | None):
        _ = _profile, reuse_dataset, max_steps
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": "fail"}

    settings = {
        "_profile": "autonomous_4080_hf",
        "autopilot": {
            "enabled": True,
            "train_cooldown_minutes": 0,
            "min_new_samples_per_tick": 0,
            "max_consecutive_failures": 2,
            "safe_mode_cooldown_minutes": 0,
            "training_jsonl_max_items": 0,
        },
        "continuous": {"ingest_web": False},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "hf_train": {"enabled": False},
    }
    _ = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_fail)
    res2 = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_fail)
    assert res2.ok is True
    assert isinstance(res2.steps.get("safe_mode"), dict)
    assert res2.steps["safe_mode"].get("active") is True

    # Next tick should skip training due to safe_mode.
    res3 = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_fail)
    assert res3.steps.get("train", {}).get("skipped") == "safe_mode"

