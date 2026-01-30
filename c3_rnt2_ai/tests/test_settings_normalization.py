from __future__ import annotations
from pathlib import Path

from c3rnt2.config import load_settings, validate_profile


def _assert_profile(profile: str) -> None:
    settings = load_settings(profile)
    validate_profile(settings, base_dir=Path("."))
    tok = settings.get("tokenizer", {})
    assert tok.get("vortex_tok_path")
    runtime = settings.get("runtime", {})
    assert "cache_vram_budget_mb" in runtime
    c3 = settings.get("c3", {})
    if c3.get("paged_lm_head_stream_topk") is not None:
        assert runtime.get("paged_lm_head_stream_topk") == c3.get("paged_lm_head_stream_topk")
    cont = settings.get("continuous", {})
    if cont.get("run_interval_minutes") is not None:
        assert cont.get("interval_minutes") == cont.get("run_interval_minutes")


def test_settings_normalization_profiles() -> None:
    _assert_profile("dev_small")
    _assert_profile("rtx4080_16gb_vortexx_next")
    _assert_profile("safe_selftrain_4080")
