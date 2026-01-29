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

    def __init__(
        self,
        hidden_size: int,
        latent_slots: int = 128,
        top_k: int = 4,
        dtype: str | None = None,
        lava_clusters: int = 0,
        lava_cluster_top: int = 1,
        read_every: int = 1,
        write_every: int = 1,
        write_on_surprise: bool = False,
        surprise_threshold: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_slots = latent_slots
        self.top_k = top_k
        self.dtype = dtype
        self.lava_clusters = int(lava_clusters)
        self.lava_cluster_top = max(1, int(lava_cluster_top))
        self.read_every = max(1, int(read_every))
        self.write_every = max(1, int(write_every))
        self.write_on_surprise = bool(write_on_surprise)
        self.surprise_threshold = float(surprise_threshold)
        self._step = 0

        self.addresses: torch.Tensor
        self.contents: torch.Tensor
        self.age: torch.Tensor
        self.importance: torch.Tensor
        self.cluster_centroids: torch.Tensor
        self.cluster_centroids_norm: torch.Tensor
        self.cluster_centroids_norm_t: torch.Tensor
        self.cluster_assignments: torch.Tensor

        self.register_buffer("addresses", torch.randn(latent_slots, hidden_size) * 0.02)
        self.register_buffer("contents", torch.randn(latent_slots, hidden_size) * 0.02)
        self.register_buffer("addresses_norm", torch.empty(latent_slots, hidden_size))
        self.register_buffer("addresses_norm_t", torch.empty(hidden_size, latent_slots))
        if self.lava_clusters > 0:
            self.register_buffer("cluster_centroids", torch.empty(self.lava_clusters, hidden_size))
            self.register_buffer("cluster_centroids_norm", torch.empty(self.lava_clusters, hidden_size))
            self.register_buffer("cluster_centroids_norm_t", torch.empty(hidden_size, self.lava_clusters))
            self.register_buffer("cluster_assignments", torch.zeros(latent_slots, dtype=torch.long))
        else:
            self.register_buffer("cluster_centroids", torch.empty(0, hidden_size))
            self.register_buffer("cluster_centroids_norm", torch.empty(0, hidden_size))
            self.register_buffer("cluster_centroids_norm_t", torch.empty(hidden_size, 0))
            self.register_buffer("cluster_assignments", torch.empty(0, dtype=torch.long))
        self.addr_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.register_buffer("age", torch.zeros(latent_slots))
        self.register_buffer("importance", torch.zeros(latent_slots))
        self.stats = LavaStats()
        self._refresh_address_cache()
        self._init_clusters()

    def _refresh_address_cache(self) -> None:
        with torch.no_grad():
            eps = 1e-8
            norm = torch.linalg.vector_norm(self.addresses, dim=-1, keepdim=True).clamp_min(eps)
            addr_norm = self.addresses / norm
            self.addresses_norm.copy_(addr_norm)
            self.addresses_norm_t.copy_(addr_norm.transpose(0, 1))

    def _update_address_cache_row(self, slot: int) -> None:
        with torch.no_grad():
            eps = 1e-8
            row = self.addresses[slot]
            denom = torch.linalg.vector_norm(row, dim=-1).clamp_min(eps)
            row_norm = row / denom
            self.addresses_norm[slot].copy_(row_norm)
            self.addresses_norm_t[:, slot].copy_(row_norm)

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

    def _refresh_centroid_cache(self) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids.numel() == 0:
            return
        with torch.no_grad():
            eps = 1e-8
            norm = torch.linalg.vector_norm(self.cluster_centroids, dim=-1, keepdim=True).clamp_min(eps)
            cent_norm = self.cluster_centroids / norm
            self.cluster_centroids_norm.copy_(cent_norm)
            self.cluster_centroids_norm_t.copy_(cent_norm.transpose(0, 1))

    def _update_centroid_cache_row(self, cluster_id: int) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids.numel() == 0:
            return
        with torch.no_grad():
            eps = 1e-8
            row = self.cluster_centroids[cluster_id]
            denom = torch.linalg.vector_norm(row, dim=-1).clamp_min(eps)
            row_norm = row / denom
            self.cluster_centroids_norm[cluster_id].copy_(row_norm)
            self.cluster_centroids_norm_t[:, cluster_id].copy_(row_norm)

    def _maybe_reassign_cluster(self, slot: int, row_norm: torch.Tensor) -> None:
        if self.lava_clusters <= 0 or self.cluster_centroids_norm_t.numel() == 0:
            return
        scores = torch.matmul(row_norm, self.cluster_centroids_norm_t)
        best = int(torch.argmax(scores, dim=-1).item())
        current = int(self.cluster_assignments[slot].item()) if self.cluster_assignments.numel() > 0 else best
        if best != current:
            self.cluster_assignments[slot] = best
        # EMA update centroid for assigned cluster
        ema = 0.1
        self.cluster_centroids[best].mul_(1.0 - ema).add_(ema * self.addresses[slot])
        self._update_centroid_cache_row(best)

    def reset_state(self) -> None:
        with torch.no_grad():
            self.addresses.copy_(torch.randn_like(self.addresses) * 0.02)
            self.contents.copy_(torch.randn_like(self.contents) * 0.02)
            self.age.zero_()
            self.importance.zero_()
            self.stats = LavaStats()
        self._refresh_address_cache()
        self._init_clusters()
        self._step = 0

    def should_read(self) -> bool:
        return self.read_every <= 1 or (self._step % self.read_every) == 0

    def should_write(self, surprise: float | None = None) -> bool:
        if self.write_every > 1 and (self._step % self.write_every) != 0:
            return False
        if self.write_on_surprise:
            if surprise is None:
                return False
            return surprise >= self.surprise_threshold
        return True

    def step_advance(self) -> None:
        self._step += 1

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
        with autocast_context(enabled=x.is_cuda, dtype=self.dtype):
            q = self.addr_proj(x)
            q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            batch, seq, hidden = q_norm.shape
            if self.lava_clusters > 0 and self.cluster_centroids_norm_t.numel() > 0:
                top_k = min(self.top_k, self.latent_slots)
                scores = torch.empty(batch, seq, top_k, device=q.device, dtype=q.dtype)
                idx = torch.empty(batch, seq, top_k, device=q.device, dtype=torch.long)
                for b in range(batch):
                    for t in range(seq):
                        vals, picked = self._ann_topk(q_norm[b, t])
                        scores[b, t].copy_(vals)
                        idx[b, t].copy_(picked)
                vals = scores
            else:
                q_flat = q_norm.reshape(batch * seq, hidden)
                scores = torch.matmul(q_flat, self.addresses_norm_t).view(batch, seq, self.latent_slots)
                top_k = min(self.top_k, self.latent_slots)
                vals, idx = torch.topk(scores, k=top_k, dim=-1)
            attn = torch.softmax(vals, dim=-1)
            selected = self.contents[idx]
            mem = (attn.unsqueeze(-1) * selected).sum(dim=-2)
            self.stats.reads += int(x.numel() // self.hidden_size)
            return self.read_proj(mem)

    def write(self, x: torch.Tensor) -> None:
        if x.dim() == 2:
            return self.write_step(x)
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 1))
            # choose slot with max age or min importance
            score = self.age - self.importance
            slot = int(torch.argmax(score).item())
            self.contents[slot].mul_(0.9).add_(0.1 * batch_mean)
            self.addresses[slot].mul_(0.95).add_(0.05 * batch_mean)
            self.importance[slot] = min(1.0, float(self.importance[slot] + 0.05))
            self.age.add_(1.0)
            self.age[slot] = 0.0
            self.stats.writes += 1
            self._update_address_cache_row(slot)
            self._maybe_reassign_cluster(slot, self.addresses_norm[slot])

    def read_step(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        with autocast_context(enabled=x.is_cuda, dtype=self.dtype):
            q = self.addr_proj(x)
            q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            if self.lava_clusters > 0 and self.cluster_centroids_norm_t.numel() > 0:
                vals_list = []
                idx_list = []
                for b in range(q_norm.shape[0]):
                    vals, picked = self._ann_topk(q_norm[b])
                    vals_list.append(vals)
                    idx_list.append(picked)
                vals = torch.stack(vals_list, dim=0)
                idx = torch.stack(idx_list, dim=0)
            else:
                scores = torch.matmul(q_norm, self.addresses_norm_t)
                top_k = min(self.top_k, self.latent_slots)
                vals, idx = torch.topk(scores, k=top_k, dim=-1)
            attn = torch.softmax(vals, dim=-1)
            selected = self.contents[idx]
            mem = (attn.unsqueeze(-1) * selected).sum(dim=-2)
            self.stats.reads += int(x.numel() // self.hidden_size)
            return self.read_proj(mem)

    def write_step(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            score = self.age - self.importance
            slot = int(torch.argmax(score).item())
            self.contents[slot].mul_(0.9).add_(0.1 * batch_mean)
            self.addresses[slot].mul_(0.95).add_(0.05 * batch_mean)
            self.importance[slot] = min(1.0, float(self.importance[slot] + 0.05))
            self.age.add_(1.0)
            self.age[slot] = 0.0
            self.stats.writes += 1
            self._update_address_cache_row(slot)
            self._maybe_reassign_cluster(slot, self.addresses_norm[slot])

    def _ann_topk(self, q_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q_norm: [H]
        top_clusters = min(self.lava_cluster_top, self.lava_clusters)
        scores_c = torch.matmul(q_norm, self.cluster_centroids_norm_t)
        cluster_vals, cluster_idx = torch.topk(scores_c, k=top_clusters, dim=-1)
        mask = torch.zeros(self.latent_slots, device=q_norm.device, dtype=torch.bool)
        for cid in cluster_idx.tolist():
            mask |= self.cluster_assignments == cid
        candidates = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        top_k = min(self.top_k, self.latent_slots)
        if candidates.numel() < top_k:
            scores = torch.matmul(q_norm, self.addresses_norm_t)
            vals, idx = torch.topk(scores, k=top_k, dim=-1)
            return vals, idx
        scores = torch.matmul(q_norm, self.addresses_norm_t[:, candidates])
        vals, rel_idx = torch.topk(scores, k=top_k, dim=-1)
        idx = candidates[rel_idx]
        return vals, idx
