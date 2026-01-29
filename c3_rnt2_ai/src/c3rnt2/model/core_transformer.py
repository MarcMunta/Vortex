from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn

from ..device import detect_device, autocast_context
from ..tokenizer.vortex_tok import VortexTokModel, load_or_create, encode_to_ids, decode_from_ids
from .vblock import VBlock, VBlockConfig, VBlockState
from .bad_decode import bad_decode
from .kv_hybrid import KVHybridCache


@dataclass
class VortexXConfig:
    hidden_size: int
    layers: int
    heads: int
    vocab_size: int
    window_size: int
    latent_slots: int
    lava_top_k: int
    local_mixer_kernel: int
    ssm_state_size: int
    gated_mlp_ratio: int
    draft_layers: int
    dtype: str | None = None
    device: str | None = None


class CoreTransformer(nn.Module):
    """VORTEX-X core using V-Blocks and LAVA memory."""

    def __init__(self, config: VortexXConfig, tokenizer: VortexTokModel):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device) if config.device else torch.device("cpu")
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16 if config.dtype == "fp16" else torch.float32
        self.escape_mode = "exact"
        sub_size = tokenizer.sub_codebook.size if tokenizer.sub_codebook else 0
        self.byte_token_start = tokenizer.patch_codebook.size + tokenizer.macro_codebook.size + sub_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                VBlock(
                    VBlockConfig(
                        hidden_size=config.hidden_size,
                        window_size=config.window_size,
                        latent_slots=config.latent_slots,
                        lava_top_k=config.lava_top_k,
                        local_mixer_kernel=config.local_mixer_kernel,
                        ssm_state_size=config.ssm_state_size,
                        gated_mlp_ratio=config.gated_mlp_ratio,
                        dtype=config.dtype,
                    )
                )
                for _ in range(config.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.kv_cache = KVHybridCache(window_size=config.window_size, kv_quant_bits=8, latent_slots=32)

    @staticmethod
    def from_settings(settings: dict) -> "CoreTransformer":
        core = settings.get("core", {})
        tok_cfg = settings.get("tokenizer", {})
        vx_cfg = settings.get("vortex_model", {})
        bad_cfg = settings.get("bad", {})
        decode_cfg = settings.get("decode", {})
        device_info = detect_device()

        model_path = Path(tok_cfg.get("vortex_tok_path", tok_cfg.get("vortex_model_path", "data/runs/vortex_tok.pt")))
        block_size = int(tok_cfg.get("block_size", 64))
        tokenizer = load_or_create(model_path, block_size)

        patch_size = tokenizer.patch_codebook.size
        macro_size = tokenizer.macro_codebook.size
        sub_size = tokenizer.sub_codebook.size if tokenizer.sub_codebook else 0
        vocab_size = max(int(core.get("vocab_size", 1024)), patch_size + macro_size + sub_size + 256)
        layers = int(core.get("layers", 4))
        draft_layers = max(1, layers // 2)

        cfg = VortexXConfig(
            hidden_size=int(core.get("hidden_size", 256)),
            layers=layers,
            heads=int(core.get("heads", 4)),
            vocab_size=vocab_size,
            window_size=int(vx_cfg.get("window_size", 128)),
            latent_slots=int(vx_cfg.get("latent_slots", 64)),
            lava_top_k=int(vx_cfg.get("lava_top_k", 4)),
            local_mixer_kernel=int(vx_cfg.get("local_mixer_kernel", 5)),
            ssm_state_size=int(vx_cfg.get("ssm_state_size", 128)),
            gated_mlp_ratio=int(vx_cfg.get("gated_mlp_ratio", 4)),
            draft_layers=int(vx_cfg.get("draft_layers", draft_layers)),
            dtype=device_info.dtype,
            device=device_info.device,
        )
        model = CoreTransformer(cfg, tokenizer=tokenizer)
        model.escape_mode = tok_cfg.get("escape_mode", "exact")
        model.bad_block_size = int(bad_cfg.get("block_size", decode_cfg.get("draft_block", 8)))
        model.bad_entropy = float(bad_cfg.get("entropy_threshold", decode_cfg.get("entropy_threshold", 3.5)))
        model.decode_cfg = decode_cfg
        model.to(device_info.device, dtype=model.dtype if device_info.device.startswith("cuda") else None)
        # Sync device/dtype with actual parameters for downstream helpers
        param = next(model.parameters(), None)
        if param is not None:
            model.device = param.device
            model.dtype = param.dtype
        else:
            model.device = torch.device(device_info.device)
        if core.get("compile") and hasattr(torch, "compile"):
            model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
        if core.get("compile_step") and hasattr(torch, "compile"):
            model.step = torch.compile(model.step)  # type: ignore[method-assign]
        return model

    def forward(self, input_ids: torch.Tensor, num_layers: int | None = None) -> torch.Tensor:
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids)
            depth = num_layers or self.config.layers
            for block in self.blocks[:depth]:
                x = block(x)
            x = self.norm(x)
            return self.lm_head(x)

    def full_next_logits(self, ids: List[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.forward(input_ids)
        return logits[:, -1, :]

    def draft_next_tokens(self, ids: List[int], count: int) -> List[int]:
        working = list(ids)
        tokens = []
        for _ in range(count):
            input_ids = torch.tensor([working], dtype=torch.long, device=self.device)
            logits = self.forward(input_ids, num_layers=self.config.draft_layers)[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            working.append(next_id)
            tokens.append(next_id)
        return tokens

    def encode_prompt(self, prompt: str) -> Tuple[List[int], int]:
        return encode_to_ids(prompt, self.tokenizer)

    def decode_ids(self, ids: List[int], total_len: int | None = None) -> str:
        return decode_from_ids(ids, self.tokenizer, total_len=total_len)

    def init_state(
        self,
        prompt_ids: Iterable[int] | torch.Tensor | None = None,
        batch: int = 1,
        num_layers: int | None = None,
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
                last_logits, state = self.step(int(tok), state, num_layers=num_layers, write_memory=write_memory)
        if return_logits:
            return last_logits, state
        return state

    def reset_state(self) -> None:
        for block in self.blocks:
            block.lava.reset_state()

    def save_memory_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for idx, block in enumerate(self.blocks):
            block.lava.save_state(path.with_suffix(f".layer{idx}.pt"))

    def load_memory_state(self, path: str | Path) -> None:
        path = Path(path)
        for idx, block in enumerate(self.blocks):
            block.lava.load_state(path.with_suffix(f".layer{idx}.pt"))

    def step(self, token_id: int, state: List[VBlockState], num_layers: int | None = None, write_memory: bool = True) -> tuple[torch.Tensor, List[VBlockState]]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids).squeeze(1)
            depth = num_layers or self.config.layers
            new_state: List[VBlockState] = []
            for idx, block in enumerate(self.blocks[:depth]):
                x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            logits = self.lm_head(x)
        return logits, new_state + state[depth:]

    def generate(self, prompt: str, max_new_tokens: int = 32, **kwargs) -> str:
        bad_cfg = {
            "block_size": getattr(self, "bad_block_size", 8),
            "entropy_threshold": getattr(self, "bad_entropy", 3.5),
        }
        bad_cfg.update(kwargs)
        with torch.inference_mode():
            with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
                text, _stats = bad_decode(
                    self,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    block_size=bad_cfg["block_size"],
                    entropy_threshold=bad_cfg["entropy_threshold"],
                    temperature=bad_cfg.get("temperature", 1.0),
                    top_p=bad_cfg.get("top_p", 1.0),
                    repetition_penalty=bad_cfg.get("repetition_penalty", 1.0),
                    no_repeat_ngram=bad_cfg.get("no_repeat_ngram", 0),
                    adaptive_granularity=bad_cfg.get("adaptive_granularity", True),
                    entropy_top_k=bad_cfg.get("entropy_top_k", 64),
                    penalty_window=bad_cfg.get("penalty_window", 512),
                    top_p_min_k=bad_cfg.get("top_p_min_k", 128),
                    top_p_max_k=bad_cfg.get("top_p_max_k", 512),
                )
        return text
