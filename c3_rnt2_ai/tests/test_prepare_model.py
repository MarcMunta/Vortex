from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from c3rnt2 import __main__ as main_mod
from c3rnt2.prepare import prepare_model_state


def test_prepare_model_cmd_writes_state_json(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(profile="rtx4080_16gb_120b_like")
    main_mod.cmd_prepare_model(args)

    out_path = tmp_path / "data" / "models" / "prepared_rtx4080_16gb_120b_like.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload.get("profile") == "rtx4080_16gb_120b_like"
    assert "backend_resolved" in payload

    # CLI prints JSON (single line) too.
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("{") and printed.endswith("}")


def test_prepare_model_state_fails_closed_for_unsafe_windows_hf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "core": {"backend": "hf"},
        "experts": {"enabled": True},
        "adapters": {"enabled": True},
        "bench_thresholds": {"required_ctx": 4096},
    }
    out = prepare_model_state(settings, base_dir=tmp_path)
    assert out["ok"] is False
    assert "unsafe_hf_config_windows_120b_like" in (out.get("errors") or [])
    assert out.get("next_steps")


def test_prepare_model_cmd_exits_nonzero_when_invalid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32", raising=False)

    def _fake_load_and_validate(_profile, override=None):
        _ = override
        return {
            "_profile": "rtx4080_16gb_120b_like",
            "core": {"backend": "hf"},
            "experts": {"enabled": True},
            "adapters": {"enabled": True},
            "bench_thresholds": {"required_ctx": 4096},
        }

    monkeypatch.setattr(main_mod, "_load_and_validate", _fake_load_and_validate)
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_prepare_model(SimpleNamespace(profile="rtx4080_16gb_120b_like"))
    assert int(exc.value.code) == 1

