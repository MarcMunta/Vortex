from __future__ import annotations

from c3rnt2.config import DEFAULT_PROFILE, load_settings, resolve_profile


def test_default_profile_points_to_qwen_coder_main(monkeypatch) -> None:
    monkeypatch.delenv("C3RNT2_PROFILE", raising=False)
    assert DEFAULT_PROFILE == "rtx4080_16gb_programming_qwen_coder_local"
    assert resolve_profile(None) == "rtx4080_16gb_programming_qwen_coder_local"


def test_default_qwen_profile_fails_closed_without_hf_access() -> None:
    settings = load_settings("rtx4080_16gb_programming_qwen_coder_local")
    core = settings.get("core", {}) or {}
    assert core.get("backend") == "hf"
    assert core.get("hf_model") == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert core.get("hf_model_loader") == "causal_lm"
    assert core.get("backend_fallback") is None
    assert core.get("allow_implicit_hf_fallback") is False
