from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, cast

import torch
from torch import nn

from ..device import autocast_context


@dataclass
class LavaStats:
    reads: int = 0
    writes: int = 0


class LAVAMemory(nn.Module):
    """Latent-Addressed Vector Attention with fixed slots."""

    addresses: torch.Tensor
    contents: torch.Tensor
    addresses_norm: torch.Tensor
    addresses_norm_t: torch.Tensor
    age: torch.Tensor
    importance: torch.Tensor
    cluster_centroids: torch.Tensor
    cluster_centroids_norm: torch.Tensor
    cluster_centroids_norm_t: torch.Tensor
    cluster_assignments: torch.Tensor
    addr_proj: nn.Linear
    read_proj: nn.Linear
    stats: LavaStats
    _step: int

    def __init__(
        self,
        hidden_size: int,
        latent_slots: int = 128,
        top_k: int = 4,
        dtype: str | None = None,
        lava_clusters: int = 0,
        lava_cluster_top: int = 1,
        ann_mode: str | None = None,
        cluster_ema: float = 0.1,
        cluster_reassign_threshold: float = 0.0,
        read_every: int = 1,
        write_every: int = 1,
        write_on_surprise: bool = False,
        surprise_threshold: float = 0.0,
        kv_quant_bits: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_slots = latent_slots
        self.top_k = top_k
        self.dtype = dtype
        self.lava_clusters = int(lava_clusters)
        self.lava_cluster_top = max(1, int(lava_cluster_top))
        self.ann_mode = (ann_mode or ("ivf" if self.lava_clusters > 0 else "flat")).lower()
        if self.lava_clusters <= 0:
            self.ann_mode = "flat"
        self.cluster_ema = float(cluster_ema)
        self.cluster_reassign_threshold = float(cluster_reassign_threshold)
        self.read_every = max(1, int(read_every))
        self.write_every = max(1, int(write_every))
        self.write_on_surprise = bool(write_on_surprise)
        self.surprise_threshold = float(surprise_threshold)
        self.kv_quant_bits = int(kv_quant_bits)
        self._step = 0
        self._cluster_dirty = False
        self._centroid_dirty = False

        self.register_buffer("addresses", torch.randn(latent_slots, hidden_size) * 0.02)
        self.register_buffer("contents", torch.randn(latent_slots, hidden_size) * 0.02)
        self.register_buffer("contents_q", torch.zeros(latent_slots, hidden_size, dtype=torch.int8))
        self.register_buffer("contents_scale", torch.ones(latent_slots, dtype=torch.float32))
        self.register_buffer("addresses_q", torch.zeros(latent_slots, hidden_size, dtype=torch.int8))
        self.register_buffer("addresses_scale", torch.ones(latent_slots, dtype=torch.float32))
        self.register_buffer("addresses_norm", torch.empty(latent_slots, hidden_size))
        self.register_buffer("addresses_norm_t", torch.empty(hidden_size, latent_slots))
        if self.lava_clusters > 0:
            self.register_buffer("cluster_centroids", torch.empty(self.lava_clusters, hidden_size))
            self.register_buffer("cluster_centroids_norm", torch.empty(self.lava_clusters, hidden_size))
            self.register_buffer("cluster_centroids_norm_t", torch.empty(hidden_size, self.lava_clusters))
            self.register_buffer("cluster_assignments", torch.zeros(latent_slots, dtype=torch.long))
            self.register_buffer("cluster_indices", torch.full((self.lava_clusters, latent_slots), -1, dtype=torch.long))
            self.register_buffer("cluster_counts", torch.zeros(self.lava_clusters, dtype=torch.long))
        else:
            self.register_buffer("cluster_centroids", torch.empty(0, hidden_size))
            self.register_buffer("cluster_centroids_norm", torch.empty(0, hidden_size))
            self.register_buffer("cluster_centroids_norm_t", torch.empty(hidden_size, 0))
            self.register_buffer("cluster_assignments", torch.empty(0, dtype=torch.long))
            self.register_buffer("cluster_indices", torch.empty(0, latent_slots, dtype=torch.long))
            self.register_buffer("cluster_counts", torch.empty(0, dtype=torch.long))
        self.addr_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.register_buffer("age", torch.zeros(latent_slots))
        self.register_buffer("importance", torch.zeros(latent_slots))
        self.stats = LavaStats()
        self.enable_write = True
        self._refresh_address_cache()
        self._init_clusters()
        if self.kv_quant_bits > 0:
            for slot in range(self.latent_slots):
                self._quantize_slot(slot)

    def _refresh_address_cache(self) -> None:
        with torch.no_grad():
            eps = 1e-8
            norm = torch.linalg.vector_norm(self.addresses, dim=-1, keepdim=True).clamp_min(eps)
            addr_norm = self.addresses / norm
            self.addresses_norm.copy_(addr_norm)
            self.addresses_norm_t.copy_(addr_norm.transpose(0, 1))

    def _update_address_cache_row(self, slot: int | torch.Tensor) -> None:
        with torch.no_grad():
            eps = 1e-8
            row = self.addresses[slot]
            denom = torch.linalg.vector_norm(row, dim=-1).clamp_min(eps)
            row_norm = row / denom
            self.addresses_norm[slot].copy_(row_norm)
            if isinstance(slot, torch.Tensor):
                slot_idx = slot.reshape(-1).to(torch.long)
            else:
                slot_idx = torch.tensor([slot], device=self.addresses.device, dtype=torch.long)
            if slot_idx.numel() == 1:
                self.addresses_norm_t.index_copy_(1, slot_idx, row_norm.view(-1, 1))

    def _init_clusters(self) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids.numel() == 0:
            return
        with torch.no_grad():
            assignments = torch.arange(self.latent_slots, device=self.addresses.device) % self.lava_clusters
            self.cluster_assignments.copy_(assignments)
            for cluster_id in range(self.lava_clusters):
                mask = self.cluster_assignments == cluster_id
                if mask.any():
                    centroid = self.addresses[mask].mean(dim=0)
                else:
                    centroid = self.addresses[cluster_id % self.latent_slots]
                self.cluster_centroids[cluster_id].copy_(centroid)
            self._refresh_centroid_cache()
            self._refresh_cluster_indices()

    def _refresh_centroid_cache(self) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids.numel() == 0:
            return
        with torch.no_grad():
            eps = 1e-8
            norm = torch.linalg.vector_norm(self.cluster_centroids, dim=-1, keepdim=True).clamp_min(eps)
            cent_norm = self.cluster_centroids / norm
            self.cluster_centroids_norm.copy_(cent_norm)
            self.cluster_centroids_norm_t.copy_(cent_norm.transpose(0, 1))
            self._centroid_dirty = False

    def _refresh_cluster_indices(self) -> None:
        if self.lava_clusters <= 0 or self.cluster_indices.numel() == 0:
            return
        with torch.no_grad():
            self.cluster_indices.fill_(-1)
            self.cluster_counts.zero_()
            for cluster_id in range(self.lava_clusters):
                idx = torch.nonzero(self.cluster_assignments == cluster_id, as_tuple=False).squeeze(-1)
                count = int(idx.numel())
                if count > 0:
                    self.cluster_indices[cluster_id, :count].copy_(idx)
                self.cluster_counts[cluster_id] = count
            self._cluster_dirty = False

    def _ensure_cluster_indices(self) -> None:
        if self._cluster_dirty:
            self._refresh_cluster_indices()

    def _ensure_centroid_cache(self) -> None:
        if self._centroid_dirty:
            self._refresh_centroid_cache()

    def _maybe_reassign_cluster(self, slot: int, row_norm: torch.Tensor) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids_norm_t.numel() == 0:
            return
        self._ensure_centroid_cache()
        scores = torch.matmul(row_norm, self.cluster_centroids_norm_t)
        best = torch.argmax(scores, dim=-1)
        current = self.cluster_assignments[slot] if self.cluster_assignments.numel() > 0 else best
        if self.cluster_reassign_threshold > 0:
            cur_cent = self.cluster_centroids_norm[current]
            sim = (row_norm * cur_cent).sum()
            mask = (1.0 - sim) >= self.cluster_reassign_threshold
        else:
            mask = torch.tensor(True, device=row_norm.device)
        new_cluster = torch.where(mask, best, current)
        self.cluster_assignments[slot] = new_cluster
        ema = self.cluster_ema
        scale = mask.to(self.cluster_centroids.dtype)
        self.cluster_centroids[best].mul_(1.0 - ema * scale).add_(ema * scale * self.addresses[slot])
        self._centroid_dirty = True
        self._cluster_dirty = True


    def _pack_2bit(self, q: torch.Tensor) -> torch.Tensor:
        total = q.numel()
        pad = (4 - total % 4) % 4
        if pad:
            q = torch.cat([q, torch.zeros(pad, device=q.device, dtype=q.dtype)])
        q = q.view(-1, 4).to(torch.uint8)
        packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
        return packed

    def _unpack_2bit(self, packed: torch.Tensor, total: int) -> torch.Tensor:
        packed = packed.to(torch.uint8)
        q0 = packed & 0x3
        q1 = (packed >> 2) & 0x3
        q2 = (packed >> 4) & 0x3
        q3 = (packed >> 6) & 0x3
        q = torch.stack([q0, q1, q2, q3], dim=1).reshape(-1)
        return q[:total]

    def _quantize_vec(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_quant_bits == 2:
            # Experimental 2-bit quantization. TODO: per-channel scaling and better packing.
            scale = vec.abs().max() / 1.5 if vec.numel() else torch.tensor(1.0, device=vec.device)
            if scale == 0:
                scale = torch.tensor(1.0, device=vec.device)
            q = torch.round(vec / scale).clamp(-2, 1).to(torch.int8)
            q = (q + 2).to(torch.uint8)
            return q, scale
        # int8 default
        scale = vec.abs().max() / 127.0 if vec.numel() else torch.tensor(1.0, device=vec.device)
        if scale == 0:
            scale = torch.tensor(1.0, device=vec.device)
        q = torch.round(vec / scale).clamp(-127, 127).to(torch.int8)
        return q, scale

    def _dequantize_vec(self, q: torch.Tensor, scale: torch.Tensor, total: int) -> torch.Tensor:
        if self.kv_quant_bits == 2:
            return (q.to(torch.int8) - 2).to(torch.float32) * scale
        return q.to(torch.float32) * scale

    def _quantize_slot(self, slot: int) -> None:
        if self.kv_quant_bits <= 0:
            return
        with torch.no_grad():
            vec = self.contents[slot].float()
            q, scale = self._quantize_vec(vec)
            if self.kv_quant_bits == 2:
                self.contents_q[slot] = q
            else:
                self.contents_q[slot].copy_(q)
            self.contents_scale[slot] = scale.float()
            addr_vec = self.addresses[slot].float()
            q_addr, scale_addr = self._quantize_vec(addr_vec)
            if self.kv_quant_bits == 2:
                self.addresses_q[slot] = q_addr
            else:
                self.addresses_q[slot].copy_(q_addr)
            self.addresses_scale[slot] = scale_addr.float()

    def _dequantize_contents(self, slots: torch.Tensor) -> torch.Tensor:
        # slots: [B, K]
        if self.kv_quant_bits <= 0:
            return self.contents[slots]
        if self.kv_quant_bits == 2:
            q = self.contents_q[slots].to(torch.int8) - 2
            scales = self.contents_scale[slots].unsqueeze(-1)
            return q.to(torch.float32) * scales
        q = self.contents_q[slots].to(torch.float32)
        scales = self.contents_scale[slots].unsqueeze(-1)
        return q * scales

    def reset_state(self) -> None:
        with torch.no_grad():
            self.addresses.copy_(torch.randn_like(self.addresses) * 0.02)
            self.contents.copy_(torch.randn_like(self.contents) * 0.02)
            self.age.zero_()
            self.importance.zero_()
            self.stats = LavaStats()
        self._refresh_address_cache()
        self._init_clusters()
        if self.kv_quant_bits > 0:
            for slot in range(self.latent_slots):
                self._quantize_slot(slot)
        self._step = 0
        self._cluster_dirty = False
        self._centroid_dirty = False

    def should_read(self) -> bool:
        return self.read_every <= 1 or (self._step % self.read_every) == 0

    def should_write(self) -> bool:
        return self.write_every <= 1 or (self._step % self.write_every) == 0

    def step_advance(self) -> None:
        self._step += 1

    def step_advance_block(self, steps: int) -> None:
        self._step += int(steps)

    def reset_stats(self) -> None:
        self.stats = LavaStats()

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "addresses": self.addresses.detach().cpu(),
            "contents": self.contents.detach().cpu(),
            "age": self.age.detach().cpu(),
            "importance": self.importance.detach().cpu(),
        }
        torch.save(payload, path)

    def load_state(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        payload_t = cast(Dict[str, torch.Tensor], payload)
        with torch.no_grad():
            self.addresses.copy_(payload_t["addresses"].to(self.addresses.device))
            self.contents.copy_(payload_t["contents"].to(self.contents.device))
            self.age.copy_(payload_t["age"].to(self.age.device))
            self.importance.copy_(payload_t["importance"].to(self.importance.device))
        self._refresh_address_cache()
        self._init_clusters()
        return True

    def read(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.read_step(x)
        return self.read_block(x)

    def write(self, x: torch.Tensor) -> None:
        if not getattr(self, "enable_write", True):
            return
        if x.dim() == 2:
            self.write_step(x)
            return
        self.write_block(x)

    def read_step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        mem = self.read_block(x.unsqueeze(1)).squeeze(1)
        return mem

    def write_step(self, x: torch.Tensor, surprise: torch.Tensor | None = None) -> None:
        if not getattr(self, "enable_write", True):
            return
        if self.write_every > 1 and (self._step % self.write_every) != 0:
            return
        with torch.no_grad():
            if self.write_on_surprise:
                if surprise is None:
                    return
                mask = (surprise >= self.surprise_threshold).to(x.dtype)
                mask_sum = mask.sum()
                write_scale = (mask_sum > 0).to(x.dtype)
                batch_mean = (x * mask.unsqueeze(-1)).sum(dim=0) / mask_sum.clamp_min(1.0)
            else:
                write_scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                batch_mean = x.mean(dim=0)
            score = self.age - self.importance
            slot = torch.argmax(score)
            self.contents[slot].mul_(1.0 - 0.1 * write_scale).add_(0.1 * write_scale * batch_mean)
            self.addresses[slot].mul_(1.0 - 0.05 * write_scale).add_(0.05 * write_scale * batch_mean)
            self.importance[slot] = torch.clamp(self.importance[slot] + 0.05 * write_scale, max=1.0)
            self.age.add_(write_scale)
            self.age[slot] = self.age[slot] * (1.0 - write_scale)
            if not self.write_on_surprise:
                self.stats.writes += int(x.shape[0])
            self._update_address_cache_row(slot)
            if self.kv_quant_bits > 0:
                self._quantize_slot(slot)
            self._maybe_reassign_cluster(slot, self.addresses_norm[slot])

    def read_block(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("LAVAMemory.read_block expects [B, K, H]")
        with autocast_context(enabled=x.is_cuda, dtype=self.dtype):
            q = self.addr_proj(x)
            q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            batch, seq, hidden = q_norm.shape
            positions: list[int] | None = None
            if self.read_every > 1:
                positions = [i for i in range(seq) if (self._step + i) % self.read_every == 0]
                if not positions:
                    return torch.zeros_like(x)
                q_sel = q_norm[:, positions, :].reshape(-1, hidden)
            else:
                q_sel = q_norm.reshape(batch * seq, hidden)
            if self.lava_clusters > 0 and self.ann_mode == "ivf":
                self._ensure_centroid_cache()
                self._ensure_cluster_indices()
                vals, idx = self._ann_topk_batch(q_sel)
            else:
                scores = torch.matmul(q_sel, self.addresses_norm_t)
                top_k = min(self.top_k, self.latent_slots)
                vals, idx = torch.topk(scores, k=top_k, dim=-1)
            attn = torch.softmax(vals, dim=-1)
            if self.kv_quant_bits > 0:
                selected = self._dequantize_contents(idx)
            else:
                selected = self.contents[idx]
            mem = (attn.unsqueeze(-1) * selected).sum(dim=-2)
            mem = self.read_proj(mem)
            if positions is None:
                self.stats.reads += int(batch * seq)
                return mem.view(batch, seq, hidden)
            mem_full = torch.zeros(batch, seq, hidden, device=mem.device, dtype=mem.dtype)
            mem_full[:, positions, :] = mem.view(batch, len(positions), hidden)
            self.stats.reads += int(batch * len(positions))
            return mem_full

    def write_block(self, x: torch.Tensor, surprise: torch.Tensor | None = None) -> None:
        if not getattr(self, "enable_write", True):
            return
        if x.dim() != 3:
            raise ValueError("LAVAMemory.write_block expects [B, K, H]")
        if self.write_every > 1:
            positions = [i for i in range(x.size(1)) if (self._step + i) % self.write_every == 0]
            if not positions:
                return
            x_sel = x[:, positions, :].reshape(-1, x.size(-1))
        else:
            x_sel = x.reshape(-1, x.size(-1))
        with torch.no_grad():
            if self.write_on_surprise and surprise is not None:
                if surprise.dim() == 2:
                    surprise_sel = surprise[:, : x.size(1)]
                    if self.write_every > 1:
                        surprise_sel = surprise_sel[:, positions]
                    mask = (surprise_sel.reshape(-1) >= self.surprise_threshold).to(x.dtype)
                else:
                    mask = (surprise >= self.surprise_threshold).to(x.dtype)
                mask_sum = mask.sum()
                write_scale = (mask_sum > 0).to(x.dtype)
                batch_mean = (x_sel * mask.unsqueeze(-1)).sum(dim=0) / mask_sum.clamp_min(1.0)
            else:
                write_scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                batch_mean = x_sel.mean(dim=0)
            score = self.age - self.importance
            slot = torch.argmax(score)
            self.contents[slot].mul_(1.0 - 0.1 * write_scale).add_(0.1 * write_scale * batch_mean)
            self.addresses[slot].mul_(1.0 - 0.05 * write_scale).add_(0.05 * write_scale * batch_mean)
            self.importance[slot] = torch.clamp(self.importance[slot] + 0.05 * write_scale, max=1.0)
            self.age.add_(write_scale)
            self.age[slot] = self.age[slot] * (1.0 - write_scale)
            if not self.write_on_surprise:
                self.stats.writes += 1
            self._update_address_cache_row(slot)
            if self.kv_quant_bits > 0:
                self._quantize_slot(slot)
            self._maybe_reassign_cluster(slot, self.addresses_norm[slot])

    def _ann_topk_batch(self, q_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q_norm: [N, H]
        top_clusters = min(self.lava_cluster_top, self.lava_clusters)
        scores_c = torch.matmul(q_norm, self.cluster_centroids_norm_t)
        _cluster_vals, cluster_idx = torch.topk(scores_c, k=top_clusters, dim=-1)
        cand = self.cluster_indices[cluster_idx]
        cand_flat = cand.reshape(q_norm.size(0), -1)
        valid = cand_flat >= 0
        safe = cand_flat.clamp_min(0)
        candidate_vecs = self.addresses_norm[safe]
        scores = (candidate_vecs * q_norm.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(~valid, -float("inf"))
        top_k = min(self.top_k, self.latent_slots)
        vals, rel_idx = torch.topk(scores, k=top_k, dim=-1)
        idx = safe.gather(1, rel_idx)
        valid_count = valid.sum(dim=-1)
        full_scores = torch.matmul(q_norm, self.addresses_norm_t)
        vals_full, idx_full = torch.topk(full_scores, k=top_k, dim=-1)
        use_full = (valid_count < top_k).unsqueeze(-1)
        vals = torch.where(use_full, vals_full, vals)
        idx = torch.where(use_full, idx_full, idx)
        return vals, idx
