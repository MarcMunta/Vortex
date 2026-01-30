from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import torch

from ..device import autocast_context
try:  # optional
    from ..runtime.graph_runner import build_graph_step_block
except Exception:  # pragma: no cover
    build_graph_step_block = None


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


def _approx_entropy(logits: torch.Tensor, top_k: int = 64, eps: float = 1e-9) -> torch.Tensor:
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
    return ent.mean()


def _approx_entropy_seq(logits: torch.Tensor, top_k: int = 64, eps: float = 1e-9) -> torch.Tensor:
    # logits: [B, K, V]
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
    return ent


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
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    work = logits.float()
    work = work / max(1e-6, temperature)
    vocab = work.size(-1)
    token_ids = _filter_ids(rep_tracker.ids(), vocab_offset, vocab)
    _apply_repetition_penalty(work, token_ids, repetition_penalty)
    banned = ngram_tracker.banned() if ngram_tracker is not None else set()
    if banned:
        banned_ids = _filter_ids(list(banned), vocab_offset, vocab)
        if banned_ids:
            work[0, torch.tensor(banned_ids, device=work.device)] = -float("inf")
    work = torch.nan_to_num(work, neginf=-1e9, posinf=1e9)
    if top_p < 1.0:
        k = min(max(8, top_p_min_k), vocab)
        k = min(k, max(8, top_p_max_k))
        values, indices = torch.topk(work, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = torch.searchsorted(cum[0], torch.tensor(top_p, device=cum.device, dtype=cum.dtype))
        cutoff = torch.clamp(cutoff, max=k - 1)
        mask = torch.arange(k, device=values.device).unsqueeze(0) > cutoff
        values = values.masked_fill(mask, -float("inf"))
        probs = torch.softmax(values, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1)
        return indices.gather(1, sample_idx).squeeze(1)
    probs = torch.softmax(work, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)
    return sample_idx.squeeze(1)


def _sample_logits_topk(
    values: torch.Tensor,
    indices: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    rep_tracker: _RepetitionTracker,
    ngram_tracker,
    top_p_min_k: int = 128,
    top_p_max_k: int = 512,
) -> torch.Tensor:
    if temperature <= 0:
        return indices.gather(1, torch.argmax(values, dim=-1, keepdim=True)).squeeze(1)
    work = values.float()
    if temperature > 0:
        work = work / max(1e-6, temperature)
    # apply repetition penalty on available indices (vectorized)
    if repetition_penalty > 1.0:
        rep_ids = rep_tracker.ids()
        if rep_ids:
            rep_tensor = torch.tensor(list(rep_ids), device=indices.device)
            if rep_tensor.numel() > 0:
                mask = (indices.unsqueeze(-1) == rep_tensor).any(dim=-1)
                if mask.any():
                    adjusted = torch.where(work > 0, work / repetition_penalty, work * repetition_penalty)
                    work = torch.where(mask, adjusted, work)
    banned = ngram_tracker.banned() if ngram_tracker is not None else set()
    if banned:
        ban_tensor = torch.tensor(list(banned), device=indices.device)
        if ban_tensor.numel() > 0:
            mask = (indices.unsqueeze(-1) == ban_tensor).any(dim=-1)
            if mask.any():
                work = work.masked_fill(mask, -float("inf"))
    work = torch.nan_to_num(work, neginf=-1e9, posinf=1e9)
    if top_p < 1.0:
        k = min(max(8, top_p_min_k), work.size(-1))
        k = min(k, max(8, top_p_max_k))
        values_k, idx_k = torch.topk(work, k=k, dim=-1)
        probs = torch.softmax(values_k, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = torch.searchsorted(cum[0], torch.tensor(top_p, device=cum.device, dtype=cum.dtype))
        cutoff = torch.clamp(cutoff, max=k - 1)
        mask = torch.arange(k, device=values_k.device).unsqueeze(0) > cutoff
        values_k = values_k.masked_fill(mask, -float("inf"))
        probs = torch.softmax(values_k, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1)
        picked = idx_k.gather(1, sample_idx)
        return indices.gather(1, picked).squeeze(1)
    probs = torch.softmax(work, dim=-1)
    sample_idx = torch.multinomial(probs, num_samples=1)
    return indices.gather(1, sample_idx).squeeze(1)


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
    stream_topk = bool(getattr(model, "runtime_cfg", {}).get("paged_lm_head_stream_topk", False))
    if stream_topk:
        adaptive_granularity = False

    with torch.inference_mode():
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            model.reset_state()
            last_logits_full, state_full = model.init_state(prompt_ids=ids, return_logits=True, write_memory=True)
            draft_model = getattr(model, "draft_model", None)
            if draft_model is not None:
                last_logits_draft, state_draft = draft_model.init_state(prompt_ids=ids, return_logits=True, write_memory=False)
            else:
                last_logits_draft, state_draft = model.init_state(
                    prompt_ids=ids,
                    num_layers=model.config.draft_layers,
                    return_logits=True,
                    write_memory=False,
                )
            last_mtp_logits = None
            graph_runner = None
            if (
                build_graph_step_block is not None
                and getattr(model.config, "cuda_graphs", False)
                and model.device.type == "cuda"
                and block_size > 0
            ):
                graph_runner = build_graph_step_block(model, state_full, token_shape=(1, block_size), return_mtp=use_mtp)

            remaining = max_new_tokens
            while remaining > 0:
                draft_block = min(block_size, remaining)
                draft_tokens: List[int] = []
                draft_rep = rep_tracker.clone()
                draft_ngram = _DraftNgramTracker(ngram_tracker)
                if use_mtp and last_mtp_logits is not None:
                    mtp_count = min(draft_block, last_mtp_logits.size(1))
                    for i in range(mtp_count):
                        next_tok_t = _sample_logits(
                            last_mtp_logits[:, i, :],
                            temperature,
                            top_p,
                            repetition_penalty,
                            draft_rep,
                            draft_ngram,
                            top_p_min_k=top_p_min_k,
                            top_p_max_k=top_p_max_k,
                        )
                        next_tok = int(next_tok_t.item())
                        draft_tokens.append(next_tok)
                        draft_rep.add(next_tok)
                        draft_ngram.add(next_tok)
                while len(draft_tokens) < draft_block:
                    if last_logits_draft is None:
                        if draft_model is not None:
                            last_logits_draft, state_draft = draft_model.step(generated[-1], state_draft, write_memory=False)
                        else:
                            last_logits_draft, state_draft = model.step(
                                generated[-1], state_draft, num_layers=model.config.draft_layers, write_memory=False
                            )
                    next_tok_t = _sample_logits(
                        last_logits_draft,
                        temperature,
                        top_p,
                        repetition_penalty,
                        draft_rep,
                        draft_ngram,
                        top_p_min_k=top_p_min_k,
                        top_p_max_k=top_p_max_k,
                    )
                    next_tok = int(next_tok_t.item())
                    draft_tokens.append(next_tok)
                    draft_rep.add(next_tok)
                    draft_ngram.add(next_tok)
                    if draft_model is not None:
                        last_logits_draft, state_draft = draft_model.step(next_tok, state_draft, write_memory=False)
                    else:
                        last_logits_draft, state_draft = model.step(
                            next_tok, state_draft, num_layers=model.config.draft_layers, write_memory=False
                        )
                stats.proposed += len(draft_tokens)

                accepted = 0
                rejected = False
                if hasattr(model, "step_block") and draft_tokens:
                    token_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=model.device)
                    if stream_topk and hasattr(model, "step_block_topk"):
                        cfg = getattr(model, "runtime_cfg", {}).get("paged_lm_head_stream_topk", 64)
                        top_k = int(cfg) if isinstance(cfg, int) else 64
                        values_seq, indices_seq, state_full = model.step_block_topk(token_tensor, state_full, top_k=top_k, write_memory=True)
                        logits_seq = None
                        mtp_seq = None
                    elif graph_runner is not None and draft_block == block_size:
                        if use_mtp:
                            logits_seq, state_full, mtp_seq = graph_runner(token_tensor, state_full)
                        else:
                            logits_seq, state_full = graph_runner(token_tensor, state_full)
                            mtp_seq = None
                    else:
                        if use_mtp:
                            logits_seq, state_full, mtp_seq = model.step_block(
                                token_tensor, state_full, write_memory=True, return_mtp=True
                            )
                        else:
                            logits_seq, state_full = model.step_block(token_tensor, state_full, write_memory=True)
                            mtp_seq = None
                    ent_mask = None
                    if adaptive_granularity and restrict_escape and logits_seq is not None:
                        ent_mask = _approx_entropy_seq(logits_seq, top_k=entropy_top_k) > entropy_threshold
                        stats.entropy_high += int(ent_mask.sum().item())
                    for i, tok in enumerate(draft_tokens):
                        if stream_topk and logits_seq is None:
                            values_i = values_seq[:, i, :]
                            indices_i = indices_seq[:, i, :]
                            tok_full = _sample_logits_topk(
                                values_i,
                                indices_i,
                                temperature,
                                top_p,
                                repetition_penalty,
                                rep_tracker,
                                ngram_tracker,
                                top_p_min_k=top_p_min_k,
                                top_p_max_k=top_p_max_k,
                            )
                            start = getattr(model, "byte_token_start", 0)
                        else:
                            logits_i = logits_seq[:, i, :]
                            start = getattr(model, "byte_token_start", 0)
                            tok_full = _sample_logits(
                                logits_i,
                                temperature,
                                top_p,
                                repetition_penalty,
                                rep_tracker,
                                ngram_tracker,
                                top_p_min_k=top_p_min_k,
                                top_p_max_k=top_p_max_k,
                            )
                        if adaptive_granularity and restrict_escape and ent_mask is not None and logits_seq is not None:
                            byte_slice = logits_i[:, start : start + 256]
                            tok_byte = _sample_logits(
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
                            tok_choice = torch.where(ent_mask[:, i], tok_byte + start, tok_full)
                        else:
                            tok_choice = tok_full
                        next_tok = int(tok_choice.item())
                        if next_tok == tok:
                            generated.append(tok)
                            accepted += 1
                            stats.accepted += 1
                            rep_tracker.add(tok)
                            ngram_tracker.add(tok)
                            if logits_seq is not None:
                                last_logits_full = logits_i
                            if use_mtp and mtp_seq is not None:
                                last_mtp_logits = mtp_seq[:, i, :, :]
                        else:
                            generated.append(next_tok)
                            stats.rejected += 1
                            rejected = True
                            rep_tracker.add(next_tok)
                            ngram_tracker.add(next_tok)
                            if logits_seq is not None:
                                last_logits_full = logits_i
                            if use_mtp and mtp_seq is not None:
                                last_mtp_logits = mtp_seq[:, i, :, :]
                            break
                else:
                    for tok in draft_tokens:
                        ent_high = None
                        values_full = None
                        indices_full = None
                        if last_logits_full is None:
                            if stream_topk and hasattr(model, "step_topk"):
                                cfg = getattr(model, "runtime_cfg", {}).get("paged_lm_head_stream_topk", 64)
                                top_k = int(cfg) if isinstance(cfg, int) else 64
                                values_full, indices_full, state_full = model.step_topk(generated[-1], state_full, top_k=top_k, write_memory=True)
                                last_logits_full = None
                            elif use_mtp:
                                last_logits_full, state_full, last_mtp_logits = model.step(
                                    generated[-1], state_full, write_memory=True, return_mtp=True
                                )
                            else:
                                last_logits_full, state_full = model.step(generated[-1], state_full, write_memory=True)
                        if adaptive_granularity and restrict_escape and last_logits_full is not None:
                            ent_high = _approx_entropy(last_logits_full, top_k=entropy_top_k) > entropy_threshold
                            stats.entropy_high += int(ent_high.item())
                        if adaptive_granularity and restrict_escape and ent_high is not None:
                            start = getattr(model, "byte_token_start", 0)
                            byte_slice = last_logits_full[:, start : start + 256]
                            tok_byte = _sample_logits(
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
                            tok_choice = torch.where(ent_high, tok_byte + start, _sample_logits(
                                last_logits_full,
                                temperature,
                                top_p,
                                repetition_penalty,
                                rep_tracker,
                                ngram_tracker,
                                top_p_min_k=top_p_min_k,
                                top_p_max_k=top_p_max_k,
                            ))
                        else:
                            if stream_topk and last_logits_full is None:
                                tok_choice = _sample_logits_topk(
                                    values_full,
                                    indices_full,
                                    temperature,
                                    top_p,
                                    repetition_penalty,
                                    rep_tracker,
                                    ngram_tracker,
                                    top_p_min_k=top_p_min_k,
                                    top_p_max_k=top_p_max_k,
                                )
                            else:
                                tok_choice = _sample_logits(
                                    last_logits_full,
                                    temperature,
                                    top_p,
                                    repetition_penalty,
                                    rep_tracker,
                                    ngram_tracker,
                                    top_p_min_k=top_p_min_k,
                                    top_p_max_k=top_p_max_k,
                                )
                        next_tok = int(tok_choice.item())
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
                            rejected = True
                            rep_tracker.add(next_tok)
                            ngram_tracker.add(next_tok)
                            if use_mtp:
                                last_logits_full, state_full, last_mtp_logits = model.step(
                                    next_tok, state_full, write_memory=True, return_mtp=True
                                )
                            else:
                                last_logits_full, state_full = model.step(next_tok, state_full, write_memory=True)
                            break
                if rejected:
                    remaining -= accepted + 1
                else:
                    remaining -= max(1, accepted)

            text = model.decode_ids(generated, total_len=None)
            return text, stats
