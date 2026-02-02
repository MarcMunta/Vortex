from __future__ import annotations

from c3rnt2.utils import vram


def test_vram_recommendation_scales(monkeypatch) -> None:
    monkeypatch.setattr(vram, "get_vram_free_mb", lambda: 500.0)
    free = vram.get_vram_free_mb()
    assert vram.should_reduce_decode(free, 1000.0) is True
    reduced = vram.recommended_max_new_tokens(200, free, floor=32, ceil=256)
    assert reduced <= 200
