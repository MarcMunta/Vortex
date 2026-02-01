from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .local_mixer import LocalMixer, LocalMixerState
from .ssm_track import SSMTrack, SSMState
from .lava_memory import LAVAMemory


@dataclass
class VBlockConfig:
    hidden_size: int
    window_size: int
    latent_slots: int
    lava_top_k: int
    lava_clusters: int
    lava_cluster_top: int
    lava_ann_mode: str
    lava_cluster_ema: float
    lava_cluster_reassign_threshold: float
    lava_read_every: int
    lava_write_every: int
    lava_write_on_surprise: bool
    lava_surprise_threshold: float
    local_mixer_kernel: int
    ssm_state_size: int
    gated_mlp_ratio: int
    kv_quant_bits: int = 0
    dtype: str | None = None


@dataclass
class VBlockState:
    local: LocalMixerState
    ssm: SSMState
    prev: torch.Tensor


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, ratio: int):
        super().__init__()
        inner = hidden_size * ratio
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        self.gate = nn.Linear(hidden_size, inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        ff = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(ff * gate)


class VBlock(nn.Module):
    """LocalMixer -> SSM -> LAVA -> GatedMLP with residuals."""

    def __init__(self, config: VBlockConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.norm4 = nn.LayerNorm(config.hidden_size)

        self.local = LocalMixer(config.hidden_size, kernel_size=config.local_mixer_kernel)
        self.ssm = SSMTrack(config.hidden_size, state_size=config.ssm_state_size)
        self.lava = LAVAMemory(
            hidden_size=config.hidden_size,
            latent_slots=config.latent_slots,
            top_k=config.lava_top_k,
            dtype=config.dtype,
            lava_clusters=config.lava_clusters,
            lava_cluster_top=config.lava_cluster_top,
            ann_mode=config.lava_ann_mode,
            cluster_ema=config.lava_cluster_ema,
            cluster_reassign_threshold=config.lava_cluster_reassign_threshold,
            read_every=config.lava_read_every,
            write_every=config.lava_write_every,
            write_on_surprise=config.lava_write_on_surprise,
            surprise_threshold=config.lava_surprise_threshold,
            kv_quant_bits=int(getattr(config, "kv_quant_bits", 0)),
        )
        self.mlp = GatedMLP(config.hidden_size, ratio=config.gated_mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(self.norm1(x))
        x = x + self.ssm(self.norm2(x))
        mem = self.lava.read(self.norm3(x))
        x = x + mem
        self.lava.write(x)
        x = x + self.mlp(self.norm4(x))
        return x

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> VBlockState:
        prev = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        return VBlockState(
            local=self.local.init_state(batch, device, dtype),
            ssm=self.ssm.init_state(batch, device, dtype),
            prev=prev,
        )

    def step(self, x: torch.Tensor, state: VBlockState, write_memory: bool = True) -> tuple[torch.Tensor, VBlockState]:
        # x: [B, H]
        local_out, local_state = self.local.step(self.norm1(x), state.local)
        x = x + local_out
        ssm_out, ssm_state = self.ssm.step(self.norm2(x), state.ssm)
        x = x + ssm_out
        surprise = None
        if write_memory and self.lava.write_on_surprise and state.prev is not None:
            delta = x - state.prev
            surprise = delta.norm(dim=-1)
        if self.lava.should_read():
            mem = self.lava.read_step(self.norm3(x))
            x = x + mem
        if write_memory and self.lava.should_write():
            self.lava.write_step(x, surprise=surprise)
        x = x + self.mlp(self.norm4(x))
        new_prev = x.detach() if not self.training else x
        self.lava.step_advance()
        return x, VBlockState(local=local_state, ssm=ssm_state, prev=new_prev)

    def step_block(self, x: torch.Tensor, state: VBlockState, write_memory: bool = True) -> tuple[torch.Tensor, VBlockState]:
        # x: [B, K, H]
        local_out, local_state = self.local.step_block(self.norm1(x), state.local)
        x = x + local_out
        ssm_out, ssm_state = self.ssm.step_block(self.norm2(x), state.ssm)
        x = x + ssm_out
        surprise = None
        if write_memory and self.lava.write_on_surprise and state.prev is not None:
            prevs = torch.cat([state.prev.unsqueeze(1), x[:, :-1, :]], dim=1)
            delta = x - prevs
            surprise = delta.norm(dim=-1)
        mem = self.lava.read_block(self.norm3(x))
        x = x + mem
        if write_memory and self.lava.should_write():
            self.lava.write_block(x, surprise=surprise)
        x = x + self.mlp(self.norm4(x))
        new_prev = x[:, -1, :].detach() if not self.training else x[:, -1, :]
        self.lava.step_advance_block(x.size(1))
        return x, VBlockState(local=local_state, ssm=ssm_state, prev=new_prev)
