from __future__ import annotations

from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.doctor import _deep_check_120b_like_profile


def test_doctor_deep_120b_like_requires_peft_when_hf_experts_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings("rtx4080_16gb_120b_like")

    # Simulate PEFT missing even if the dev environment has it installed.
    import c3rnt2.doctor as doctor_mod

    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda _name: None)

    out = _deep_check_120b_like_profile(settings, tmp_path, mock=True)
    assert out["ok"] is False
    assert "peft_missing_for_hf_experts" in (out.get("errors") or [])
    info = out.get("info") or {}
    peft = info.get("peft") or {}
    assert peft.get("ok") is False
    assert "pip install" in str(peft.get("install") or "")

