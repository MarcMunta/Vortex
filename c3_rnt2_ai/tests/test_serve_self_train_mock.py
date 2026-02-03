from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from c3rnt2 import __main__ as main_mod


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
    monkeypatch.setattr(
        main_mod,
        "train_hf_once",
        lambda settings, base_dir, reuse_dataset=False: SimpleNamespace(
            ok=True,
            run_id="r1",
            adapter_dir=None,
            loss=0.0,
            steps=1,
            samples=1,
            tokens_per_sec=1.0,
        ),
    )

    args = SimpleNamespace(
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

    lock_path = tmp_path / "data" / "locks" / "train.lock"
    assert lock_path.exists()


def test_self_train_tick_sets_and_clears_training_active(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import __main__ as main_mod

    monkeypatch.setattr(main_mod, "ingest_sources", lambda base_dir, allowlist, settings: 0)

    app = SimpleNamespace(state=SimpleNamespace())

    def _fake_train(_settings, _base_dir, reuse_dataset=False):
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

    monkeypatch.setattr(main_mod, "train_hf_once", _fake_train)

    settings = {"server": {"block_during_training": True}, "continuous": {"ingest_web": False, "trigger": {"enabled": False}}}
    result = main_mod._run_self_train_tick(
        app,
        settings,
        tmp_path,
        reuse_dataset=False,
        maintenance_window_s=0.0,
        reload_fn=None,
    )
    assert result.get("ok") is True
    assert app.state.training_active is False
