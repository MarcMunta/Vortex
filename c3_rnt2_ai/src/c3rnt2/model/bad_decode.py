from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import torch

from ..device import autocast_context


@dataclass
class BADStats:
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    entropy_high: int = 0


class _RepetitionTracker:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue: Deque[int] = deque()
        self.counts: Dict[int, int] = {}

    def add(self, token: int) -> None:
        if self.window_size <= 0:
            return
        self.queue.append(token)
        self.counts[token] = self.counts.get(token, 0) + 1
        if len(self.queue) > self.window_size:
            old = self.queue.popleft()
            remaining = self.counts.get(old, 0) - 1
            if remaining <= 0:
                self.counts.pop(old, None)
            else:
                self.counts[old] = remaining

    def ids(self) -> List[int]:
        return list(self.counts.keys())

    def clone(self) -> "_RepetitionTracker":
        dup = _RepetitionTracker(self.window_size)
        dup.queue = deque(self.queue)
        dup.counts = dict(self.counts)
        return dup


class _NgramTracker:
    def __init__(self, n: int):
        self.n = n
        self.map: Dict[Tuple[int, ...], set[int]] = {}
        self.recent: Deque[int] = deque(maxlen=max(0, n - 1))

    def add(self, token: int) -> None:
        if self.n <= 0:
            return
        if len(self.recent) == self.n - 1:
            prefix = tuple(self.recent)
            self.map.setdefault(prefix, set()).add(token)
        self.recent.append(token)

    def banned(self) -> set[int]:
        if self.n <= 0 or len(self.recent) < self.n - 1:
            return set()
        return self.map.get(tuple(self.recent), set())


class _DraftNgramTracker:
    def __init__(self, base: _NgramTracker):
        self.base = base
        self.n = base.n
        self.recent: Deque[int] = deque(base.recent, maxlen=base.recent.maxlen)
        self.added: Dict[Tuple[int, ...], set[int]] = {}

    def add(self, token: int) -> None:
        if self.n <= 0:
            return
        if len(self.recent) == self.n - 1:
            prefix = tuple(self.recent)
            self.added.setdefault(prefix, set()).add(token)
        self.recent.append(token)

    def banned(self) -> set[int]:
        if self.n <= 0 or len(self.recent) < self.n - 1:
            return set()
        prefix = tuple(self.recent)
        banned = set()
        base_set = self.base.map.get(prefix)
        if base_set:
            banned.update(base_set)
        added_set = self.added.get(prefix)
        if added_set:
            banned.update(added_set)
        return banned


def _approx_entropy(logits: torch.Tensor, top_k: int = 64, eps: float = 1e-9) -> float:
    vocab = logits.size(-1)
    k = min(max(4, top_k), vocab)
    values, _ = torch.topk(logits.float(), k=k, dim=-1)
    lse_top = torch.logsumexp(values, dim=-1, keepdim=True)
    if vocab > k:
        kth = values[..., -1:]
        tail_mass = (vocab - k) * torch.exp(kth - lse_top)
        lse_total = lse_top + torch.log1p(tail_mass)
        probs = torch.exp(values - lse_total)
        p_top = probs.sum(dim=-1, keepdim=True)
        p_tail = torch.clamp(1.0 - p_top, min=eps)
        ent_top = -(probs * torch.log(torch.clamp(probs, min=eps))).sum(dim=-1)
        ent_tail = -p_tail.squeeze(-1) * torch.log(p_tail.squeeze(-1) / max(1, vocab - k))
        ent = ent_top + ent_tail
    else:
        probs = torch.softmax(values, dim=-1)
        ent = -(probs * torch.log(torch.clamp(probs, min=eps))).sum(dim=-1)
    return float(ent.mean().item())


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float) -> None:
    if penalty <= 1.0 or not token_ids:
        return
    idx = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    vals = logits[0, idx]
    adjusted = torch.where(vals > 0, vals / penalty, vals * penalty)
    logits[0, idx] = adjusted


def _filter_ids(token_ids: List[int], offset: int, vocab_size: int) -> List[int]:
    end = offset + vocab_size
    if offset == 0:
        return [tok for tok in token_ids if 0 <= tok < end]
    return [tok - offset for tok in token_ids if offset <= tok < end]


def _sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    rep_tracker: _RepetitionTracker,
    ngram_tracker,
    vocab_offset: int = 0,
    top_p_min_k: int = 128,
    top_p_max_k: int = 512,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    work = logits.float().clone()
    work = work / max(1e-6, temperature)
    vocab = work.size(-1)
    token_ids = _filter_ids(rep_tracker.ids(), vocab_offset, vocab)
    _apply_repetition_penalty(work, token_ids, repetition_penalty)
    banned = ngram_tracker.banned() if ngram_tracker is not None else set()
    if banned:
        banned_ids = _filter_ids(list(banned), vocab_offset, vocab)
        if banned_ids:
            work[0, torch.tensor(banned_ids, device=work.device)] = -float("inf")
    if not torch.isfinite(work).any():
        return int(torch.argmax(logits, dim=-1).item())
    if top_p < 1.0:
        k = min(max(8, top_p_min_k), vocab)
        max_k = min(max(8, top_p_max_k), vocab)
        while True:
            values, indices = torch.topk(work, k=k, dim=-1)
            lse_top = torch.logsumexp(values, dim=-1, keepdim=True)
            if k >= vocab:
                top_mass = torch.ones_like(lse_top)
            else:
                kth = values[..., -1:]
                tail_mass = (vocab - k) * torch.exp(kth - lse_top)
                top_mass = torch.exp(lse_top - (lse_top + torch.log1p(tail_mass)))
            if float(top_mass.item()) >= top_p or k >= max_k or k >= vocab:
                probs = torch.softmax(values, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = int(torch.argmax((cum > top_p).float(), dim=-1).item())
                values = values.clone()
                values[:, cutoff + 1 :] = -float("inf")
                probs = torch.softmax(values, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1)
                return int(indices.gather(1, sample_idx).item())
            k = min(k * 2, max_k, vocab)
    probs = torch.softmax(work, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _should_restrict_escape(escape_restrict: bool, exact_copy_mode: bool, escape_mode: str) -> bool:
    if exact_copy_mode:
        return True
    if not escape_restrict:
        return False
    return escape_mode in {"exact", "exact-copy"}


def bad_decode(
    model,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    entropy_threshold: float,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    adaptive_granularity: bool = True,
    entropy_top_k: int = 64,
    penalty_window: int = 512,
    top_p_min_k: int = 128,
    top_p_max_k: int = 512,
    exact_copy_mode: bool = False,
    escape_restrict: bool = False,
    use_mtp: bool = True,
) -> Tuple[str, BADStats]:
    ids, total_len = model.encode_prompt(prompt)
    stats = BADStats()
    generated = list(ids)
    rep_tracker = _RepetitionTracker(penalty_window)
    ngram_tracker = _NgramTracker(no_repeat_ngram)
    for tok in ids:
        rep_tracker.add(tok)
        ngram_tracker.add(tok)

    use_mtp = bool(use_mtp) and getattr(model, "mtp_k", 0) > 0
    escape_mode = getattr(model, "escape_mode", "")
    restrict_escape = _should_restrict_escape(escape_restrict, exact_copy_mode, escape_mode)

    with torch.inference_mode():
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            model.reset_state()
            last_logits_full, state_full = model.init_state(prompt_ids=ids, return_logits=True, write_memory=True)
            last_logits_draft, state_draft = model.init_state(
                prompt_ids=ids,
                num_layers=model.config.draft_layers,
                return_logits=True,
                write_memory=False,
            )
            last_mtp_logits = None

            remaining = max_new_tokens
            while remaining > 0:
                draft_block = min(block_size, remaining)
                draft_tokens: List[int] = []
                draft_rep = rep_tracker.clone()
                draft_ngram = _DraftNgramTracker(ngram_tracker)
                if use_mtp and last_mtp_logits is not None:
                    mtp_count = min(draft_block, last_mtp_logits.size(1))
                    for i in range(mtp_count):
                        next_tok = _sample_logits(
                            last_mtp_logits[:, i, :],
                            temperature,
                            top_p,
                            repetition_penalty,
                            draft_rep,
                            draft_ngram,
                            top_p_min_k=top_p_min_k,
                            top_p_max_k=top_p_max_k,
                        )
                        draft_tokens.append(next_tok)
                        draft_rep.add(next_tok)
                        draft_ngram.add(next_tok)
                for _ in range(len(draft_tokens), draft_block):
                    if last_logits_draft is None:
                        last_logits_draft, state_draft = model.step(
                            generated[-1], state_draft, num_layers=model.config.draft_layers, write_memory=False
                        )
                    next_tok = _sample_logits(
                        last_logits_draft,
                        temperature,
                        top_p,
                        repetition_penalty,
                        draft_rep,
                        draft_ngram,
                        top_p_min_k=top_p_min_k,
                        top_p_max_k=top_p_max_k,
                    )
                    draft_tokens.append(next_tok)
                    draft_rep.add(next_tok)
                    draft_ngram.add(next_tok)
                    last_logits_draft, state_draft = model.step(next_tok, state_draft, num_layers=model.config.draft_layers, write_memory=False)
                stats.proposed += len(draft_tokens)

                accepted = 0
                for tok in draft_tokens:
                    if last_logits_full is None:
                        if use_mtp:
                            last_logits_full, state_full, last_mtp_logits = model.step(
                                generated[-1], state_full, write_memory=True, return_mtp=True
                            )
                        else:
                            last_logits_full, state_full = model.step(generated[-1], state_full, write_memory=True)
                    ent = None
                    if adaptive_granularity and restrict_escape:
                        ent = _approx_entropy(last_logits_full, top_k=entropy_top_k)
                    if adaptive_granularity and restrict_escape and ent is not None and ent > entropy_threshold:
                        stats.entropy_high += 1
                        start = getattr(model, "byte_token_start", 0)
                        byte_slice = last_logits_full[:, start : start + 256]
                        next_tok = (
                            _sample_logits(
                                byte_slice,
                                temperature,
                                top_p,
                                repetition_penalty,
                                rep_tracker,
                                ngram_tracker,
                                vocab_offset=start,
                                top_p_min_k=top_p_min_k,
                                top_p_max_k=top_p_max_k,
                            )
                            + start
                        )
                    else:
                        next_tok = _sample_logits(
                            last_logits_full,
                            temperature,
                            top_p,
                            repetition_penalty,
                            rep_tracker,
                            ngram_tracker,
                            top_p_min_k=top_p_min_k,
                            top_p_max_k=top_p_max_k,
                        )
                    if next_tok == tok:
                        generated.append(tok)
                        accepted += 1
                        stats.accepted += 1
                        rep_tracker.add(tok)
                        ngram_tracker.add(tok)
                        if use_mtp:
                            last_logits_full, state_full, last_mtp_logits = model.step(tok, state_full, write_memory=True, return_mtp=True)
                        else:
                            last_logits_full, state_full = model.step(tok, state_full, write_memory=True)
                    else:
                        generated.append(next_tok)
                        stats.rejected += 1
                        rep_tracker.add(next_tok)
                        ngram_tracker.add(next_tok)
                        if use_mtp:
                            last_logits_full, state_full, last_mtp_logits = model.step(
                                next_tok, state_full, write_memory=True, return_mtp=True
                            )
                        else:
                            last_logits_full, state_full = model.step(next_tok, state_full, write_memory=True)
                        break
                remaining -= max(1, accepted)

            text = model.decode_ids(generated, total_len=None)
            return text, stats
