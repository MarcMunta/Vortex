from __future__ import annotations

from c3rnt2.config import DEFAULT_PROFILE, load_settings, resolve_profile


def test_default_profile_points_to_llama2_local_main(monkeypatch) -> None:
    monkeypatch.delenv("C3RNT2_PROFILE", raising=False)
    assert DEFAULT_PROFILE == "rtx4080_16gb_llama2_7b_q4_local"
    assert resolve_profile(None) == "rtx4080_16gb_llama2_7b_q4_local"


def test_default_llama2_profile_uses_local_llama_cpp_quant() -> None:
    settings = load_settings("rtx4080_16gb_llama2_7b_q4_local")
    core = settings.get("core", {}) or {}
    assert core.get("backend") == "llama_cpp"
    assert core.get("llama_cpp_model_path") == "data/models/gguf/llama-2-7b-chat.Q4_K_M.gguf"
    assert core.get("llama_cpp_quant") == "Q4_K_M"
    assert core.get("llama_cpp_ctx") == 4096
    assert core.get("backend_fallback") is None
    assert core.get("allow_implicit_hf_fallback") is False
