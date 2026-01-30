from __future__ import annotations

import torch

from c3rnt2.model.bad_decode import bad_decode


class _FakeConfig:
    def __init__(self):
        self.dtype = "fp32"
        self.draft_layers = 0
        self.cuda_graphs = False


class _FakeModel:
    def __init__(self, vocab: int = 8) -> None:
        self.device = torch.device("cpu")
        self.config = _FakeConfig()
        self.escape_mode = "exact"
        self.runtime_cfg = {"paged_lm_head_stream_topk": 4}
        self.byte_token_start = 0
        self._vocab = vocab

    def encode_prompt(self, prompt: str):
        return [1], len(prompt)

    def decode_ids(self, ids, total_len=None):
        return "ok"

    def reset_state(self) -> None:
        return None

    def init_state(self, prompt_ids=None, return_logits=False, write_memory=True, num_layers=None):
        logits = torch.zeros(1, self._vocab)
        logits[0, 0] = 1.0
        state = {}
        if return_logits:
            return logits, state
        return state

    def step(self, token, state, num_layers=None, write_memory=True, return_mtp=False):
        logits = torch.zeros(1, self._vocab)
        logits[0, (int(token) + 1) % self._vocab] = 1.0
        if return_mtp:
            mtp = torch.zeros(1, 1, 1, self._vocab)
            return logits, state, mtp
        return logits, state

    def step_topk(self, token, state, top_k=4, write_memory=True):
        logits = torch.zeros(1, self._vocab)
        logits[0, (int(token) + 1) % self._vocab] = 1.0
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        return values, indices, state


def test_bad_decode_stream_topk_no_unbound() -> None:
    model = _FakeModel()
    text, stats = bad_decode(
        model,
        prompt="hi",
        max_new_tokens=2,
        block_size=2,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram=0,
        adaptive_granularity=True,
        entropy_threshold=3.5,
        entropy_top_k=4,
        penalty_window=8,
        top_p_min_k=4,
        top_p_max_k=4,
        exact_copy_mode=False,
        escape_restrict=False,
        use_mtp=False,
    )
    assert isinstance(text, str)
    assert stats.proposed >= 0
