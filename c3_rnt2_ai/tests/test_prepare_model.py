from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from c3rnt2 import __main__ as main_mod
from c3rnt2 import prepare as prepare_mod
from c3rnt2.prepare import prepare_model_state


def test_prepare_model_cmd_writes_state_json(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        main_mod,
        "_load_and_validate",
        lambda _profile, override=None: {
            "_profile": "rtx4080_16gb_120b_like",
            "core": {"backend": "hf", "hf_model": "Qwen/Qwen2.5-7B-Instruct", "hf_device": "cpu"},
            "experts": {"enabled": False},
            "adapters": {"enabled": False},
            "bench_thresholds": {"required_ctx": 4096},
        },
    )
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


def test_prepare_model_state_reports_programming_local_readiness(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_mod,
        "_http_json",
        lambda url, timeout_s=2.0: {
            "data": [{"id": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"}]
        },
    )
    monkeypatch.setattr(
        prepare_mod,
        "_docker_ready_status",
        lambda settings, base_dir=None: (True, "docker_ready", {}),
    )
    settings = {
        "_profile": "rtx4080_16gb_programming_local",
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://127.0.0.1:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        },
        "docker": {"enabled": True},
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "security": {"web": {"strict": True, "allowlist_domains": []}},
        "continuous": {"ingest_web": False, "local_sources": {"enabled": True}},
        "autolearn": {"web_ingest": False, "url_discovery": False},
        "hf_train": {"enabled": False},
        "profile_contract": {
            "offline_required": True,
            "require_web_disabled": True,
            "require_external_engine": "sglang",
            "require_docker": True,
            "require_local_base_url": True,
            "disable_fallbacks": True,
        },
    }
    out = prepare_model_state(settings, base_dir=tmp_path)
    assert out["ok"] is True
    assert out["offline_ready"] is True
    assert out["engine_ready"] is True
    assert out["engine_kind"] == "sglang"
    assert out["docker_ready"] is True
    assert out["active_model"] == "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
    assert out["web_disabled"] is True
    assert "training_ready" in out


def test_prepare_model_state_fails_when_sglang_model_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_mod,
        "_http_json",
        lambda url, timeout_s=2.0: {"data": [{"id": "other-model"}]},
    )
    monkeypatch.setattr(
        prepare_mod,
        "_docker_ready_status",
        lambda settings, base_dir=None: (True, "docker_ready", {}),
    )
    settings = {
        "_profile": "rtx4080_16gb_programming_local",
        "core": {
            "backend": "external",
            "external_engine": "sglang",
            "external_base_url": "http://127.0.0.1:30000",
            "external_model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        },
        "docker": {"enabled": True},
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "security": {"web": {"strict": True, "allowlist_domains": []}},
        "continuous": {"ingest_web": False},
        "autolearn": {"web_ingest": False, "url_discovery": False},
        "profile_contract": {
            "offline_required": True,
            "require_web_disabled": True,
            "require_external_engine": "sglang",
            "require_docker": True,
            "require_local_base_url": True,
            "disable_fallbacks": True,
        },
    }
    out = prepare_model_state(settings, base_dir=tmp_path)
    assert out["ok"] is False
    assert out["engine_ready"] is False
    assert out["model_ready"] is False
    assert out["model_reason"] == "sglang_model_missing"


def test_prepare_model_state_fails_when_gemma_cache_missing(tmp_path: Path) -> None:
    settings = {
        "_profile": "rtx4080_16gb_programming_gemma4_local",
        "core": {
            "backend": "hf",
            "hf_model": "google/gemma-4-E4B-it",
            "hf_cache_dir": str(tmp_path / "hf-cache"),
            "hf_local_files_only": True,
            "backend_fallback": None,
            "hf_fallback": None,
            "allow_implicit_hf_fallback": False,
        },
        "docker": {"enabled": False},
        "tools": {"web": {"enabled": False, "allow_domains": []}},
        "security": {"web": {"strict": True, "allowlist_domains": []}},
        "continuous": {"ingest_web": False},
        "autolearn": {"web_ingest": False, "url_discovery": False},
        "profile_contract": {
            "offline_required": True,
            "require_web_disabled": True,
            "disable_fallbacks": True,
        },
    }
    out = prepare_model_state(settings, base_dir=tmp_path)
    assert out["ok"] is False
    assert out["offline_ready"] is False
    assert out["model_ready"] is False
    assert out["model_reason"] == "hf_model_cache_missing"
    assert "hf_model_cache_missing" in str(out["errors"])
    assert "--download" in "\n".join(out["next_steps"] or [])


def test_docker_ready_status_can_be_assumed_inside_container(tmp_path: Path, monkeypatch) -> None:
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("C3RNT2_ASSUME_DOCKER_READY", "1")

    ready, reason, meta = prepare_mod._docker_ready_status(  # type: ignore[attr-defined]
        {"docker": {"enabled": True, "compose_path": str(compose_path)}},
        base_dir=tmp_path,
    )

    assert ready is True
    assert reason == "docker_managed_by_host"
    assert meta["assumed_from_env"] is True
