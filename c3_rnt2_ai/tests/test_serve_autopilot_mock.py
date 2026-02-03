from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from c3rnt2 import __main__ as main_mod
from c3rnt2 import autopilot as ap


def test_serve_autopilot_mock_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def _fake_load_and_validate(_profile, override=None):
        settings = {
            "_profile": "safe_selftrain_4080_hf",
            "autopilot": {"enabled": False, "interval_minutes": 0.01, "training_jsonl_max_items": 0},
            "continuous": {"interval_minutes": 0.01, "ingest_web": False, "trigger": {"enabled": False}},
            "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
            "knowledge": {"embedding_backend": "hash"},
            "core": {"backend": "hf"},
            "hf_train": {"enabled": False},
        }
        return override(settings) if override else settings

    monkeypatch.setattr(main_mod, "_load_and_validate", _fake_load_and_validate)
    monkeypatch.setattr(ap, "ingest_sources", lambda *_args, **_kwargs: 0)

    args = SimpleNamespace(
        profile=None,
        backend=None,
        model=None,
        device=None,
        host="127.0.0.1",
        port=8001,
        once=True,
        interval_minutes=0.01,
        no_web=True,
        mock=True,
        force=False,
    )
    main_mod.cmd_serve_autopilot(args)
    # No exception is success.
