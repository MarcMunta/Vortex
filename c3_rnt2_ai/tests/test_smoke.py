from __future__ import annotations

from c3rnt2.config import load_settings
from c3rnt2.device import detect_device


def test_smoke_settings():
    settings = load_settings("rtx4080_16gb_llama2_7b_q4_local")
    assert "tokenizer" in settings


def test_smoke_device():
    info = detect_device()
    assert info.device
