from __future__ import annotations

from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.doctor import _deep_check_120b_like_profile


class _DummyHFModel:
    is_hf = True

    def __init__(self):
        self.quant_fallback = False


def test_doctor_deep_120b_like_requires_baseline_in_real_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    settings = load_settings("rtx4080_16gb_120b_like")

    import c3rnt2.doctor as doctor_mod
    import c3rnt2.model_loader as loader_mod

    # Make the check deterministic/offline: simulate PEFT present and avoid loading real weights.
    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda name: object() if name == "peft" else None)
    monkeypatch.setattr(loader_mod, "load_inference_model", lambda _settings: _DummyHFModel())

    out = _deep_check_120b_like_profile(settings, tmp_path, mock=False)
    assert out["ok"] is False
    assert "bench_baseline_missing" in (out.get("errors") or [])
    info = out.get("info") or {}
    baseline = info.get("bench_baseline") or {}
    assert baseline.get("ok") is False
    assert "bench --profile" in str(baseline.get("hint") or "")

