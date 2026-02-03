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
