from __future__ import annotations

import random
from pathlib import Path

from c3rnt2.tokenizer import vortex_tok as vt


def _model(tmp_path: Path) -> vt.VortexTokModel:
    return vt.load_or_create(tmp_path / "vortex_tok.pt", block_size=32)


def test_vortex_tok_roundtrip_cases(tmp_path: Path) -> None:
    model = _model(tmp_path)
    cases = [
        "hello world",
        "unicode: ñ é ö 😀",
        "\x00\x01\t\ncontrol-chars",
        "a" * 5000,
        ("abc" * 2000) + " END",
    ]
    for text in cases:
        stream = vt.encode(text, model)
        assert vt.decode(stream, model) == text

        ids, total_len = vt.encode_to_ids(text, model)
        assert vt.decode_from_ids(ids, model, total_len=total_len) == text


def test_vortex_tok_fuzz_roundtrip(tmp_path: Path) -> None:
    model = _model(tmp_path)
    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t-_=+()[]{}<>/\\'\".,:;!?ñéö😀"
    for _ in range(100):
        n = rng.randint(0, 200)
        text = "".join(rng.choice(alphabet) for _ in range(n))
        stream = vt.encode(text, model)
        assert vt.decode(stream, model) == text

