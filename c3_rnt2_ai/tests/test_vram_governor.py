from __future__ import annotations

import types

from c3rnt2.runtime import vram_governor as vg


def test_vram_governor_reduces_tokens_when_low_free(monkeypatch) -> None:
    class DummyCuda:
        @staticmethod
        def mem_get_info():
            return (100 * 1024**2, 1000 * 1024**2)

        @staticmethod
        def max_memory_allocated():
            return 900 * 1024**2

        @staticmethod
        def is_available():
            return True

    dummy_torch = types.SimpleNamespace(cuda=DummyCuda())
    monkeypatch.setattr(vg, "torch", dummy_torch)

    settings = {"core": {"vram_threshold_mb": 200, "vram_floor_tokens": 32, "vram_ceil_tokens": 256, "vram_safety_margin_mb": 200}}
    decided = vg.decide_max_new_tokens(200, "cuda", "bf16", settings)
    assert decided == 32
