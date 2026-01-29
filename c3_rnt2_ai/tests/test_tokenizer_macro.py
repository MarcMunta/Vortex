from __future__ import annotations

from c3rnt2.tokenizer.rnt2_model import RNT2Codebook
from c3rnt2.tokenizer.vortex_tok import VortexMacroCodebook, VortexTokModel, encode, decode, metrics


def test_macro_trie_roundtrip_and_hit_rate():
    codebook = RNT2Codebook(block_size=2, entries=[b"aa", b"bb", b"cc"])
    macro = VortexMacroCodebook(sequences=[[0, 1]])
    model = VortexTokModel(patch_codebook=codebook, macro_codebook=macro)
    text = "aabb"
    stream = encode(text, model)
    assert decode(stream, model) == text
    stats = metrics(stream)
    assert stats["macro_hit_rate"] > 0.0
