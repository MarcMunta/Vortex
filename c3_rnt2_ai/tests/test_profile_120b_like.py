from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from c3rnt2.config import load_settings, validate_profile
from c3rnt2.promotion.gating import bench_gate, log_promotion_decision


def test_profile_120b_like_loads_and_is_coherent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings("rtx4080_16gb_120b_like")
    validate_profile(settings, base_dir=tmp_path)

    thresholds = settings.get("bench_thresholds", {}) or {}
    for key in ("min_tokens_per_sec", "max_regression", "max_vram_peak_mb", "required_ctx"):
        assert thresholds.get(key) is not None

    adapters = settings.get("adapters", {}) or {}
    assert adapters.get("enabled") is True
    assert int(adapters.get("max_loaded") or 0) >= 2
    router = adapters.get("router", {}) or {}
    assert int(router.get("top_k") or 0) >= 1
    assert int(adapters.get("max_loaded") or 0) >= int(router.get("top_k") or 1)
    assert str(router.get("mode") or "").lower() == "hybrid"
    assert str(router.get("mix_mode") or "").lower() == "weighted"

    experts = settings.get("experts", {}) or {}
    assert experts.get("enabled") is True
    assert int(experts.get("max_loaded") or 0) >= 2
    ex_router = experts.get("router", {}) or {}
    assert int(ex_router.get("top_k") or 0) >= 1
    assert int(experts.get("max_loaded") or 0) >= int(ex_router.get("top_k") or 1)
    assert str(ex_router.get("mode") or "").lower() == "hybrid"
    assert str(ex_router.get("mix_mode") or "").lower() == "weighted"


def test_doctor_deep_mock_profile_120b_like_passes(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2 import __main__ as main_mod
    from c3rnt2 import doctor as doctor_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda name: object() if name == "peft" else None)

    # Skip base (dependency) checks to keep CI offline/fast.
    monkeypatch.setattr(main_mod, "_run_doctor_checks", lambda *_a, **_k: {"ok": True, "errors": [], "warnings": [], "info": {}})

    # Keep deep checks lightweight while still exercising the 120B-like profile-specific validator + mock bench.
    monkeypatch.setattr(
        doctor_mod,
        "detect_device",
        lambda: SimpleNamespace(device="cuda", cuda_available=True, name="gpu", vram_gb=16.0, dtype="bf16"),
    )
    monkeypatch.setattr(doctor_mod, "_profile_checks", lambda _base_dir: {})
    monkeypatch.setattr(doctor_mod, "_tokenizer_roundtrip_strict", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_inference_smoke", lambda *_a, **_k: ({"ok": True}, {"ok": True}))
    monkeypatch.setattr(doctor_mod, "_self_train_mock", lambda *_a, **_k: {"ok": True, "run_id": "r1"})
    monkeypatch.setattr(doctor_mod, "_promotion_gating_wiring_check", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_security_deep_check", lambda *_a, **_k: {"ok": True})

    args = SimpleNamespace(profile="rtx4080_16gb_120b_like", deep=True, mock=True)
    main_mod.cmd_doctor(args)

    bench_path = tmp_path / "data" / "bench" / "doctor_120b_like_mock.json"
    assert bench_path.exists()
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
    for key in (
        "tokens_per_sec",
        "vram_peak_mb",
        "backend",
        "backend_resolved",
        "quant_mode",
        "offload_enabled",
        "active_adapters",
        "adapter_load_ms",
    ):
        assert key in payload


def test_gating_blocks_and_logs_reason(tmp_path: Path) -> None:
    verdict = bench_gate(
        {"tokens_per_sec": 1.0, "vram_peak_mb": 99999.0},
        baseline={"tokens_per_sec": 10.0},
        thresholds={"min_tokens_per_sec": 10.0, "max_regression": 0.05, "max_vram_peak_mb": 15500},
    )
    assert verdict["ok"] is False
    assert "below_min_tokens_per_sec" in (verdict["reason"] or "")

    log_promotion_decision(
        tmp_path,
        {
            "kind": "promotion_gate",
            "backend": "test",
            "promote_ok": False,
            "reason": verdict.get("reason"),
            "verdict": verdict,
        },
    )
    log_path = tmp_path / "data" / "logs" / "promotions.jsonl"
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    rec = json.loads(line)
    assert rec.get("promote_ok") is False
    assert "below_min_tokens_per_sec" in (rec.get("reason") or "")
