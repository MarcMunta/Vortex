from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

from c3rnt2 import __main__ as main_mod


def test_ensure_runtime_app_state_populates_defaults() -> None:
    app = SimpleNamespace()

    state = main_mod._ensure_runtime_app_state(app)

    assert state.model is None
    assert state.models == {}
    assert state.model_lock is None
    assert state.training_active is False
    assert state.maintenance_until == 0.0


def test_serve_self_train_mock_loop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_load_and_validate(_profile, override=None):
        settings = {
            "continuous": {"interval_minutes": 0.01, "ingest_web": False, "trigger": {"enabled": False}},
            "core": {"backend": "hf"},
        }
        return override(settings) if override else settings

    monkeypatch.setattr(main_mod, "_load_and_validate", _fake_load_and_validate)
    monkeypatch.setattr(main_mod, "ingest_sources", lambda base_dir, allowlist, settings: 1)

    args = argparse.Namespace(
        profile=None,
        backend=None,
        model=None,
        device=None,
        host="127.0.0.1",
        port=8000,
        once=True,
        interval_minutes=0.01,
        reuse_dataset=False,
        maintenance_window_s=0.01,
        mock=True,
    )
    main_mod.cmd_serve_self_train(args)

    lock_path = tmp_path / "data" / "locks" / "train.db"
    assert lock_path.exists()
    gpu_lock_path = tmp_path / "data" / "locks" / "gpu.db"
    assert gpu_lock_path.exists()


def test_self_train_tick_sets_and_clears_training_active(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import __main__ as main_mod

    monkeypatch.setattr(main_mod, "ingest_sources", lambda base_dir, allowlist, settings: 0)

    app = SimpleNamespace(state=SimpleNamespace())

    def _fake_train(settings, base_dir, reuse_dataset=False):
        assert app.state.training_active is True
        return SimpleNamespace(
            ok=True,
            run_id="r1",
            adapter_dir=None,
            loss=0.0,
            steps=1,
            samples=1,
            tokens_per_sec=1.0,
        )

    settings = {"server": {"block_during_training": True}, "continuous": {"ingest_web": False, "trigger": {"enabled": False}}}
    result = main_mod._run_self_train_tick(
        app,
        settings,
        tmp_path,
        reuse_dataset=False,
        maintenance_window_s=0.0,
        reload_fn=None,
        train_fn=_fake_train,
    )
    assert result.get("ok") is True
    assert app.state.training_active is False


def test_self_train_tick_skips_when_vram_insufficient(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import __main__ as main_mod

    monkeypatch.setattr(main_mod, "ingest_sources", lambda base_dir, allowlist, settings: 0)
    monkeypatch.setattr(main_mod, "get_vram_free_mb", lambda: 100.0)

    app = SimpleNamespace(state=SimpleNamespace())
    train_calls: list[int] = []

    def _fake_train(settings, base_dir, reuse_dataset=False):
        train_calls.append(1)
        return SimpleNamespace(ok=True, ok_eval=True, ok_train=True, eval_ok=True)

    settings = {
        "core": {"vram_safety_margin_mb": 512, "vram_threshold_mb": 1200},
        "server": {"block_during_training": True, "train_strategy": "inprocess"},
        "continuous": {"ingest_web": False, "trigger": {"enabled": False}},
    }
    result = main_mod._run_self_train_tick(
        app,
        settings,
        tmp_path,
        reuse_dataset=False,
        maintenance_window_s=0.0,
        reload_fn=None,
        train_fn=_fake_train,
    )
    assert result.get("ok") is True
    assert result.get("skipped") == "vram_insufficient"
    assert train_calls == []
    assert (tmp_path / "data" / "locks" / "gpu.db").exists()


def test_self_train_tick_skips_when_host_ram_insufficient(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import __main__ as main_mod

    monkeypatch.setattr(main_mod, "ingest_sources", lambda base_dir, allowlist, settings: 0)
    monkeypatch.setattr(main_mod, "get_vram_free_mb", lambda: 10_000.0)
    monkeypatch.setattr(main_mod, "_host_ram_free_mb", lambda: 1024.0)

    app = SimpleNamespace(state=SimpleNamespace())
    train_calls: list[int] = []

    def _fake_train(settings, base_dir, reuse_dataset=False):
        train_calls.append(1)
        return SimpleNamespace(ok=True, ok_eval=True, ok_train=True, eval_ok=True)

    settings = {
        "core": {"vram_safety_margin_mb": 0, "vram_threshold_mb": 0},
        "server": {"block_during_training": True, "train_strategy": "inprocess", "train_host_ram_threshold_mb": 8192},
        "continuous": {"ingest_web": False, "trigger": {"enabled": False}},
    }
    result = main_mod._run_self_train_tick(
        app,
        settings,
        tmp_path,
        reuse_dataset=False,
        maintenance_window_s=0.0,
        reload_fn=None,
        train_fn=_fake_train,
    )
    assert result.get("ok") is True
    assert result.get("skipped") == "host_ram_insufficient"
    assert train_calls == []
