from __future__ import annotations

from pathlib import Path

from c3rnt2 import __main__ as main_mod


def test_doctor_checks_ok(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "load_inference_model", lambda _settings: object())
    settings = {
        "_profile": "rtx4080_16gb",
        "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
        "tools": {"web": {"enabled": False, "cache_dir": str(tmp_path / "data" / "web_cache")}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "core": {"backend": "hf", "hf_device": "cpu"},
        "decode": {"max_new_tokens": 64},
    }
    report = main_mod._run_doctor_checks(settings, tmp_path)
    assert report["ok"] is True


def test_doctor_detects_strict_web_ingest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "load_inference_model", lambda _settings: object())
    settings = {
        "_profile": "safe_selftrain_4080",
        "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
        "tools": {"web": {"enabled": False, "allow_domains": ["example.com"]}},
        "continuous": {"ingest_web": True, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "core": {"backend": "hf", "hf_device": "cpu"},
        "decode": {"max_new_tokens": 64},
    }
    report = main_mod._run_doctor_checks(settings, tmp_path)
    assert report["ok"] is False
    assert any("ingest_web enabled but tools.web.enabled=false" in err for err in report["errors"])


def test_doctor_detects_autopilot_nested_under_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "load_inference_model", lambda _settings: object())
    settings = {
        "_profile": "safe_selftrain_4080_hf",
        "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
        "tools": {"web": {"enabled": False}, "autopilot": {"enabled": True}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "core": {"backend": "hf", "hf_device": "cpu"},
        "decode": {"max_new_tokens": 64},
    }
    report = main_mod._run_doctor_checks(settings, tmp_path)
    assert report["ok"] is False
    assert any("tools.autopilot" in err for err in report["errors"])


def test_doctor_windows_cuda_gate_fails_when_cuda_missing(tmp_path: Path, monkeypatch) -> None:
    import sys as _sys
    from types import ModuleType, SimpleNamespace

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    monkeypatch.setattr(main_mod.sys, "platform", "win32", raising=False)

    settings = {
        "_profile": "rtx4080_16gb_safe_windows_core",
        "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
        "tools": {"web": {"enabled": False, "cache_dir": str(tmp_path / "data" / "web_cache")}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "core": {"backend": "vortex", "device": "cuda"},
        "decode": {"max_new_tokens": 64},
    }
    report = main_mod._run_doctor_checks(settings, tmp_path)
    assert report["ok"] is False
    assert any("cuda_missing" in err for err in report["errors"])


def test_doctor_hf_train_missing_deps_fails_with_actionable_message(tmp_path: Path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in {"peft", "accelerate", "bitsandbytes"}:
            raise ImportError("missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    settings = {
        "_profile": "rtx4080_16gb_safe_windows_hf",
        "agent": {"tools_enabled": list(main_mod._supported_agent_tools())},
        "tools": {"web": {"enabled": False, "cache_dir": str(tmp_path / "data" / "web_cache")}},
        "continuous": {"ingest_web": False, "knowledge_path": str(tmp_path / "data" / "continuous" / "knowledge.sqlite")},
        "core": {"backend": "vortex", "device": "cuda"},
        "decode": {"max_new_tokens": 64},
        "hf_train": {"enabled": True, "load_in_4bit": True, "load_in_8bit": False},
    }
    report = main_mod._run_doctor_checks(settings, tmp_path)
    assert report["ok"] is False
    assert any(str(err).startswith("hf_train_deps_missing:") for err in report["errors"])


def test_doctor_deep_mock_skips_bench(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from c3rnt2 import doctor as doctor_mod

    monkeypatch.setattr(
        doctor_mod,
        "detect_device",
        lambda: SimpleNamespace(device="cuda", cuda_available=True, name="gpu", vram_gb=16.0, dtype="bf16"),
    )
    monkeypatch.setattr(doctor_mod, "_profile_checks", lambda _base_dir: {})
    monkeypatch.setattr(doctor_mod, "_tokenizer_roundtrip_strict", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(doctor_mod, "_inference_smoke", lambda *_a, **_k: ({"ok": True}, {"ok": True}))
    monkeypatch.setattr(doctor_mod, "_self_train_mock", lambda *_a, **_k: {"ok": True, "run_id": "r1"})

    settings = {"_profile": "p", "core": {"backend": "vortex"}}
    report = doctor_mod.run_deep_checks(settings, base_dir=tmp_path, mock=True)
    assert report["deep_ok"] is True
    assert (tmp_path / "data" / "doctor" / "last.json").exists()
