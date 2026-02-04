from __future__ import annotations

from pathlib import Path

from c3rnt2.autopilot import run_autopilot_tick


def test_autopilot_skips_training_when_gpu_lock_unavailable(tmp_path: Path, monkeypatch) -> None:
    import c3rnt2.autopilot as ap
    from c3rnt2.utils.locks import FileLock as RealFileLock, LockUnavailable

    # Create the lock file path up-front so the test doesn't depend on lock acquisition creating it.
    gpu_lock_path = tmp_path / "data" / "locks" / "gpu.lock"
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock_path.write_text("", encoding="utf-8")

    class _FailGpuLock(RealFileLock):
        def acquire(self, blocking: bool = False, timeout_s: float | None = None) -> None:  # type: ignore[override]
            if self.path.name == "gpu.lock":
                raise LockUnavailable("busy")
            return super().acquire(blocking=blocking)

    monkeypatch.setattr(ap, "FileLock", _FailGpuLock)
    monkeypatch.setattr(ap, "ingest_sources", lambda *_a, **_k: 0)
    monkeypatch.setattr(ap, "_build_sft_dataset", lambda *_a, **_k: {"ok": True, "stats": {"written": 1}})

    calls: list[int] = []

    def _train_runner(_profile: str, reuse_dataset: bool, max_steps: int | None):
        _ = _profile, reuse_dataset, max_steps
        calls.append(1)
        return {"ok": True, "ok_train": True, "ok_eval": True, "eval_ok": True, "improvement": 0.0}

    settings = {
        "_profile": "autonomous_4080_hf",
        "autopilot": {"enabled": True, "train_cooldown_minutes": 0, "training_jsonl_max_items": 0},
        "continuous": {"ingest_web": False},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "hf_train": {"enabled": False},
    }
    res = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_runner)
    assert res.ok is True
    assert res.steps.get("train", {}).get("skipped") == "gpu_lock_unavailable"
    assert calls == []


def test_autopilot_skips_training_when_vram_insufficient(tmp_path: Path, monkeypatch) -> None:
    import c3rnt2.autopilot as ap

    monkeypatch.setattr(ap, "ingest_sources", lambda *_a, **_k: 0)
    monkeypatch.setattr(ap, "_build_sft_dataset", lambda *_a, **_k: {"ok": True, "stats": {"written": 1}})
    monkeypatch.setattr(ap, "get_vram_free_mb", lambda: 100.0)

    calls: list[int] = []

    def _train_runner(_profile: str, reuse_dataset: bool, max_steps: int | None):
        _ = _profile, reuse_dataset, max_steps
        calls.append(1)
        return {"ok": True, "ok_train": True, "ok_eval": True, "eval_ok": True, "improvement": 0.0}

    settings = {
        "_profile": "autonomous_4080_hf",
        "core": {"vram_safety_margin_mb": 512, "vram_threshold_mb": 1200},
        "autopilot": {"enabled": True, "train_cooldown_minutes": 0, "training_jsonl_max_items": 0},
        "continuous": {"ingest_web": False},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "hf_train": {"enabled": False},
    }
    res = run_autopilot_tick(settings, tmp_path, no_web=True, mock=False, force=False, train_runner=_train_runner)
    assert res.ok is True
    assert res.steps.get("train", {}).get("skipped") == "vram_insufficient"
    assert res.steps.get("train", {}).get("vram_free_mb") == 100.0
    assert calls == []
