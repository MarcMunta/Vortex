from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ..device import autocast_context
from .vblock import VBlock, VBlockConfig, VBlockState


@dataclass
class DraftConfig:
    hidden_size: int
    layers: int
    vocab_size: int
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
    dtype: str | None = None
    device: str | None = None


class DraftModel(nn.Module):
    """Lightweight draft model for speculative decoding."""

    def __init__(
        self,
        config: DraftConfig,
        shared_embed: nn.Embedding | None = None,
        shared_lm_head: nn.Module | None = None,
        base_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device) if config.device else torch.device("cpu")
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16 if config.dtype == "fp16" else torch.float32
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.in_proj: nn.Linear | None = None
        self.out_proj: nn.Linear | None = None
        self.embed: nn.Embedding
        if shared_embed is not None and base_hidden is not None:
            self.embed = shared_embed
            if base_hidden != config.hidden_size:
                self.in_proj = nn.Linear(base_hidden, config.hidden_size, bias=False)
        else:
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                VBlock(
                    VBlockConfig(
                        hidden_size=config.hidden_size,
                        window_size=config.window_size,
                        latent_slots=config.latent_slots,
                        lava_top_k=config.lava_top_k,
                        lava_clusters=config.lava_clusters,
                        lava_cluster_top=config.lava_cluster_top,
                        lava_ann_mode=config.lava_ann_mode,
                        lava_cluster_ema=config.lava_cluster_ema,
                        lava_cluster_reassign_threshold=config.lava_cluster_reassign_threshold,
                        lava_read_every=config.lava_read_every,
                        lava_write_every=config.lava_write_every,
                        lava_write_on_surprise=config.lava_write_on_surprise,
                        lava_surprise_threshold=config.lava_surprise_threshold,
                        local_mixer_kernel=config.local_mixer_kernel,
                        ssm_state_size=config.ssm_state_size,
                        gated_mlp_ratio=config.gated_mlp_ratio,
                        kv_quant_bits=int(getattr(config, "kv_quant_bits", 0)),
                        dtype=config.dtype,
                    )
                )
                for _ in range(config.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

        if shared_lm_head is not None and base_hidden is not None:
            if base_hidden != config.hidden_size:
                self.out_proj = nn.Linear(config.hidden_size, base_hidden, bias=False)
            self.lm_head = shared_lm_head
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def init_state(
        self,
        prompt_ids: List[int] | torch.Tensor | None = None,
        batch: int = 1,
        write_memory: bool = True,
        return_logits: bool = False,
    ) -> List[VBlockState] | tuple[torch.Tensor | None, List[VBlockState]]:
        dtype = self.embed.weight.dtype
        state = [block.init_state(batch, self.device, dtype) for block in self.blocks]
        if prompt_ids is None:
            return (None, state) if return_logits else state
        if isinstance(prompt_ids, torch.Tensor):
            prompt_list = prompt_ids.flatten().tolist()
        else:
            prompt_list = list(prompt_ids)
        last_logits: torch.Tensor | None = None
        with torch.inference_mode():
            for tok in prompt_list:
                last_logits, state = self.step(int(tok), state, write_memory=write_memory)
        if return_logits:
            return last_logits, state
        return state

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids)
            if self.in_proj is not None:
                x = self.in_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            if self.out_proj is not None:
                x = self.out_proj(x)
            return self.lm_head(x)

    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        write_memory: bool = True,
    ) -> tuple[torch.Tensor, List[VBlockState]]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids).squeeze(1)
            if self.in_proj is not None:
                x = self.in_proj(x)
            new_state: List[VBlockState] = []
            for idx, block in enumerate(self.blocks):
                x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            if self.out_proj is not None:
                x = self.out_proj(x)
            logits = self.lm_head(x)
        return logits, new_state

    def step_block(
        self,
        token_ids: torch.Tensor,
        state: List[VBlockState],
        write_memory: bool = True,
    ) -> tuple[torch.Tensor, List[VBlockState]]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(token_ids)
            if self.in_proj is not None:
                x = self.in_proj(x)
            new_state: List[VBlockState] = []
            for idx, block in enumerate(self.blocks):
                x, layer_state = block.step_block(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            if self.out_proj is not None:
                x = self.out_proj(x)
            logits = self.lm_head(x)
        return logits, new_state

    def reset_state(self) -> None:
        for block in self.blocks:
            block.lava.reset_state()
