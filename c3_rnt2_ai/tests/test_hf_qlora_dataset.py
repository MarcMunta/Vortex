from __future__ import annotations

import torch

from c3rnt2.training.hf_qlora import SFTDataset, _collate_fn, build_sft_texts
from c3rnt2.continuous.types import Sample


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        tokens = text.split()
        ids = torch.arange(1, min(len(tokens), max_length) + 1, dtype=torch.long)
        attn = torch.ones_like(ids)
        return {"input_ids": ids.unsqueeze(0), "attention_mask": attn.unsqueeze(0)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m.get("content", "") for m in messages)


def test_hf_qlora_dataset_padding():
    tok = FakeTokenizer()
    samples = [Sample(prompt="hello", response="world"), Sample(prompt="one", response="two three")]
    texts = build_sft_texts(samples, tokenizer=tok, default_system=None)
    ds = SFTDataset(texts, tokenizer=tok, max_length=16)
    batch = _collate_fn([ds[0], ds[1]], pad_token_id=tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] == -100).any()
