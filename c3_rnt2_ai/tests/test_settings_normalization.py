from __future__ import annotations

from pathlib import Path

from c3rnt2.config import load_settings, load_settings_document, resolve_settings_sources, validate_profile

BASE_DIR = Path(__file__).resolve().parents[1]


def test_only_llama2_profile_is_loaded() -> None:
    document = load_settings_document()
    assert list(document["profiles"]) == ["rtx4080_16gb_llama2_7b_q4_local"]
    assert [path.name for path in resolve_settings_sources()] == ["settings.yaml"]


def test_llama2_profile_is_local_chat_agent_rag() -> None:
    settings = load_settings("rtx4080_16gb_llama2_7b_q4_local")
    validate_profile(settings, base_dir=BASE_DIR)

    core = settings["core"]
    assert core["backend"] == "llama_cpp"
    assert core["llama_cpp_model_path"] == "data/models/gguf/llama-2-7b-chat.Q4_K_M.gguf"
    assert core["llama_cpp_chat_format"] == "llama-2"
    assert core["backend_fallback"] is None
    assert core["allow_implicit_hf_fallback"] is False

    assert settings["rag"]["enabled"] is True
    assert settings["continuous"]["enabled"] is False
    assert settings["continuous"]["ingest_web"] is False
    assert settings["skills"]["enabled"] is True
    assert "run_command" in settings["agent"]["tools_enabled"]
    assert "write_file" in settings["agent"]["tools_enabled"]
