from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from c3rnt2 import doctor as doctor_mod
from c3rnt2 import model_loader as loader_mod


class _DummyHFModel:
    is_hf = True

    def __init__(self, *, quant_fallback: bool):
        self.quant_fallback = bool(quant_fallback)


def _patch_deep_checks_lightweight(monkeypatch) -> None:
    monkeypatch.setattr(doctor_mod.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(doctor_mod, "detect_device", lambda: SimpleNamespace(device="cpu", cuda_available=True, name="gpu", vram_gb=16.0, dtype="bf16"))
    monkeypatch.setattr(doctor_mod, "_profile_checks", lambda _base_dir: {})
    monkeypatch.setattr(doctor_mod, "_tokenizer_roundtrip_strict", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_inference_smoke", lambda *_a, **_k: ({"ok": True}, {"ok": True}))
    monkeypatch.setattr(doctor_mod, "_self_train_mock", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_promotion_gating_wiring_check", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_security_deep_check", lambda *_a, **_k: {"ok": True})


def test_doctor_deep_120b_like_fails_when_quant_fallback(tmp_path: Path, monkeypatch) -> None:
    _patch_deep_checks_lightweight(monkeypatch)
    monkeypatch.setattr(loader_mod, "load_inference_model", lambda _settings: _DummyHFModel(quant_fallback=True))

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "core": {"backend": "hf", "hf_load_in_4bit": True},
        "bench_thresholds": {"min_tokens_per_sec": 10.0, "max_regression": 0.15, "max_vram_peak_mb": 15500, "required_ctx": 4096},
        "adapters": {"enabled": True, "allow_empty": True, "max_loaded": 6, "router": {"mode": "hybrid", "top_k": 2, "mix_mode": "weighted"}},
        "experts": {"enabled": True, "max_loaded": 6, "paths": {}, "router": {"mode": "hybrid", "top_k": 2, "mix_mode": "weighted"}},
        "learning": {"require_bench_ok": False},
        "autopilot": {"bench_enabled": False},
    }
    report = doctor_mod.run_deep_checks(settings, base_dir=tmp_path, mock=False)
    assert report["deep_ok"] is False
    prof = report["checks"]["profile_120b_like"]
    assert prof["ok"] is False
    assert "120b_like_requires_quant_backend" in (prof.get("errors") or [])


def test_doctor_deep_120b_like_passes_when_quant_active(tmp_path: Path, monkeypatch) -> None:
    _patch_deep_checks_lightweight(monkeypatch)
    monkeypatch.setattr(loader_mod, "load_inference_model", lambda _settings: _DummyHFModel(quant_fallback=False))

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "core": {"backend": "hf", "hf_load_in_4bit": True},
        "bench_thresholds": {"min_tokens_per_sec": 10.0, "max_regression": 0.15, "max_vram_peak_mb": 15500, "required_ctx": 4096},
        "adapters": {"enabled": True, "allow_empty": True, "max_loaded": 6, "router": {"mode": "hybrid", "top_k": 2, "mix_mode": "weighted"}},
        "experts": {"enabled": True, "max_loaded": 6, "paths": {}, "router": {"mode": "hybrid", "top_k": 2, "mix_mode": "weighted"}},
        "learning": {"require_bench_ok": False},
        "autopilot": {"bench_enabled": False},
    }
    report = doctor_mod.run_deep_checks(settings, base_dir=tmp_path, mock=False)
    assert report["deep_ok"] is True

