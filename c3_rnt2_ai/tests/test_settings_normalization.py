from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pylint: disable=import-error,no-name-in-module

from pathlib import Path

from c3rnt2.config import (
    load_settings,
    load_settings_document,
    resolve_settings_sources,
    validate_profile,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def _read_all_settings_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in resolve_settings_sources()
    )


def _assert_profile(profile: str) -> None:
    settings = load_settings(profile)
    validate_profile(settings, base_dir=BASE_DIR)
    tok = settings.get("tokenizer", {})
    assert tok.get("vortex_tok_path")
    runtime = settings.get("runtime", {})
    assert "cache_vram_budget_mb" in runtime
    c3 = settings.get("c3", {})
    if c3.get("paged_lm_head_stream_topk") is not None:
        assert runtime.get("paged_lm_head_stream_topk") == c3.get(
            "paged_lm_head_stream_topk"
        )
    cont = settings.get("continuous", {})
    if cont.get("run_interval_minutes") is not None:
        assert cont.get("interval_minutes") == cont.get("run_interval_minutes")


def test_settings_normalization_profiles() -> None:
    _assert_profile("dev_small")
    _assert_profile("rtx4080_16gb_gemma4_26b_a4b_hf")
    _assert_profile("rtx4080_16gb_gemma4_31b_hf")
    _assert_profile("rtx4080_16gb_gemma3_12b_hf")
    _assert_profile("rtx4080_16gb_vortexx_next")
    _assert_profile("safe_selftrain_4080")
    _assert_profile("rtx4080_16gb_programming_local")
    _assert_profile("rtx4080_16gb_programming_train_docker")
    _assert_profile("rtx4080_16gb_programming_train_wsl")


def test_settings_safety_defaults() -> None:
    settings = load_settings("dev_small")
    autopilot = settings.get("autopilot", {}) or {}
    autolearn = settings.get("autolearn", {}) or {}
    assert bool(autopilot.get("autopatch_require_approval", False)) is True
    assert bool(autolearn.get("enabled", False)) is False


def test_settings_manifest_loads_fragmented_profiles() -> None:
    document = load_settings_document()
    sources = resolve_settings_sources()

    assert "dev_small" in document["profiles"]
    assert "rtx4080_16gb_programming_train_docker" in document["profiles"]
    assert "ethical_security_lab_4080_offensive_lab" in document["profiles"]
    assert sources[-1].name == "settings.yaml"
    assert len(sources) == 6


def _assert_security_lab_profile_is_local_and_safe(profile: str) -> None:
    settings = load_settings(profile)
    core = settings.get("core", {}) or {}
    tools = settings.get("tools", {}) or {}
    web = tools.get("web", {}) or {}
    continuous = settings.get("continuous", {}) or {}
    autolearn = settings.get("autolearn", {}) or {}
    local_lab = settings.get("local_lab", {}) or {}

    assert core.get("backend") == "external"
    assert core.get("external_engine") == "ollama"
    assert core.get("external_base_url") == "http://127.0.0.1:11434"
    assert core.get("external_model") == "qwen3:14b"
    assert core.get("backend_fallback") is None
    assert "Security-Lab" in str(core.get("hf_system_prompt") or "")
    assert bool(web.get("enabled", True)) is False
    assert bool(continuous.get("ingest_web", True)) is False
    assert bool(autolearn.get("enabled", True)) is False
    assert bool(local_lab.get("enabled", False)) is True
    assert bool(local_lab.get("guardrails_enabled", False)) is True


def test_security_lab_profiles_are_local_and_safe() -> None:
    _assert_security_lab_profile_is_local_and_safe(
        "ethical_security_lab_4080_offensive_lab"
    )
    _assert_security_lab_profile_is_local_and_safe("ethical_security_lab_4080_defensive")


def test_legacy_offensive_profile_name_removed() -> None:
    raw = _read_all_settings_text()
    assert "ethical_hacking_autolearn_4080" not in raw
    assert "ethical_security_lab_4080:" not in raw
    assert "ethical_security_lab_4080_offensive_lab:" in raw
    assert "ethical_security_lab_4080_defensive:" in raw


def test_programming_profiles_are_local_and_offline() -> None:
    daily = load_settings("rtx4080_16gb_programming_local")
    train = load_settings("rtx4080_16gb_programming_train_docker")
    train_wsl = load_settings("rtx4080_16gb_programming_train_wsl")

    assert daily["core"]["backend"] == "external"
    assert daily["core"]["external_engine"] == "sglang"
    assert daily["core"]["external_base_url"] == "http://127.0.0.1:30000"
    assert daily["core"]["external_model"] == "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
    assert daily["core"]["backend_fallback"] is None
    assert daily["core"]["hf_fallback"] is None
    assert daily["core"]["allow_implicit_hf_fallback"] is False
    assert daily["docker"]["enabled"] is True
    assert daily["tools"]["web"]["enabled"] is False
    assert daily["continuous"]["ingest_web"] is False
    assert daily["autolearn"]["web_ingest"] is False
    assert daily["autolearn"]["url_discovery"] is False
    assert daily["profile_contract"]["require_external_engine"] == "sglang"
    assert daily["profile_contract"]["require_docker"] is True
    assert daily["profile_contract"]["disable_fallbacks"] is True
    assert daily["continuous"]["local_sources"]["include_repo"] is True
    assert daily["hf_train"]["enabled"] is False

    assert train["core"]["backend"] == "hf"
    assert train["core"]["hf_model"] == "google/gemma-4-E4B-it"
    assert train["core"]["backend_fallback"] is None
    assert train["core"].get("hf_fallback") is None
    assert train["core"]["allow_implicit_hf_fallback"] is False
    assert train["server"]["train_strategy"] == "inprocess"
    assert train["hf_train"]["enabled"] is True
    assert train["hf_train"]["model_name"] == "google/gemma-4-E4B-it"
    assert train["hf_train"]["manual_promotion_only"] is False
    assert train["hf_train"]["extra_training_paths"] == [
        "data/registry/hf_train/sft_samples.jsonl",
        "data/registry/hf_train/programming_eval.jsonl",
        "config/datasets/flutter_dart_seed.jsonl",
        "config/datasets/flutter_dart_eval.jsonl",
    ]
    assert train["profile_contract"]["require_docker"] is True
    assert train["profile_contract"]["disable_fallbacks"] is True
    assert train["profile_contract"]["require_wsl_training"] is False
    assert train["profile_contract"]["approved_training_sources_only"] is True

    assert train_wsl["server"]["train_strategy"] == "wsl_subprocess_unload"
    assert train_wsl["server"]["wsl_workdir"] == "/mnt/d/Vortex/c3_rnt2_ai"
    assert train_wsl["profile_contract"]["require_wsl_training"] is True
