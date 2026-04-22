from __future__ import annotations

from c3rnt2.config import DEFAULT_PROFILE, load_settings, resolve_profile


def test_default_profile_points_to_gemma4_main(monkeypatch) -> None:
    monkeypatch.delenv("C3RNT2_PROFILE", raising=False)
    assert DEFAULT_PROFILE == "rtx4080_16gb_gemma4_26b_a4b_hf"
    assert resolve_profile(None) == "rtx4080_16gb_gemma4_26b_a4b_hf"


def test_default_gemma_profile_fails_closed_without_hf_access() -> None:
    settings = load_settings("rtx4080_16gb_gemma4_26b_a4b_hf")
    core = settings.get("core", {}) or {}
    assert core.get("backend") == "hf"
    assert core.get("hf_model") == "google/gemma-4-26B-A4B-it"
    assert core.get("hf_model_loader") == "image_text_to_text"
    assert core.get("backend_fallback") is None
    assert core.get("allow_implicit_hf_fallback") is False
