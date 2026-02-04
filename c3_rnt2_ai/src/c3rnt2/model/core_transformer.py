from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple, overload, Literal

import torch
from torch import nn

from ..device import detect_device, autocast_context
from ..utils.oom import is_oom_error, clear_cuda_cache
from ..tokenizer.vortex_tok import VortexTokModel, load_or_create, encode_to_ids, decode_from_ids
from .vblock import VBlock, VBlockConfig, VBlockState
from .lava_memory import LAVAMemory
from .draft_model import DraftModel, DraftConfig


def _maybe_enable_paged_lm_head(model: "CoreTransformer", settings: dict) -> None:
    runtime_cfg = settings.get("runtime", {}) or {}
    c3_cfg = settings.get("c3", {}) or {}
    if not bool(runtime_cfg.get("paged_lm_head", False)):
        return
    try:
        from ..nn.paged_linear import PagedLinear
    except Exception:
        return
    tile_out_val = runtime_cfg.get("paged_tile_out", c3_cfg.get("tile_size", 128))
    if tile_out_val is None:
        tile_out_val = c3_cfg.get("tile_size", 128)
    tile_out = int(tile_out_val)
    tile_in_val = runtime_cfg.get("paged_tile_in", c3_cfg.get("tile_in", model.config.hidden_size))
    if tile_in_val is None:
        tile_in_val = model.config.hidden_size
    tile_in = int(tile_in_val)
    cache_budget_mb = int(runtime_cfg.get("cache_vram_budget_mb", c3_cfg.get("cache_vram_budget_mb", 2048)))
    prefetch_depth = int(runtime_cfg.get("prefetch_depth", c3_cfg.get("prefetch_depth", 2)))
    compression = runtime_cfg.get("compression", c3_cfg.get("compression"))
    pin_memory = bool(runtime_cfg.get("pinned_memory", c3_cfg.get("pinned_memory", True)))
    gpu_decompress = runtime_cfg.get("gpu_decompress", "none")
    model.lm_head = PagedLinear.from_linear(
        model.lm_head,
        tile_out=tile_out,
        tile_in=tile_in,
        cache_budget_bytes=cache_budget_mb * 1024 * 1024,
        compression=compression,
        device=str(model.device),
        prefetch_depth=prefetch_depth,
        pin_memory=pin_memory,
        gpu_decompress=gpu_decompress,
    )
from .bad_decode import bad_decode
from .kv_hybrid import KVHybridCache


def _approx_entropy_logits(logits: torch.Tensor, top_k: int = 64, eps: float = 1e-9) -> float:
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


def _approx_entropy_tensor(logits: torch.Tensor, top_k: int = 64, eps: float = 1e-9) -> torch.Tensor:
    vocab = logits.size(-1)
    k = min(max(4, top_k), vocab)
    values, _ = torch.topk(logits, k=k, dim=-1)
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

@dataclass
class VortexXConfig:
    hidden_size: int
    layers: int
    heads: int
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
    lava_shared_groups: int
    local_mixer_kernel: int
    ssm_state_size: int
    gated_mlp_ratio: int
    draft_layers: int
    draft_hidden: int
    mtp_k: int
    cuda_graphs: bool
    kv_quant_bits: int = 0
    dtype: str | None = None
    device: str | None = None


def _settings_hash(settings: dict) -> str:
    try:
        payload = json.dumps(settings, sort_keys=True, default=str)
    except Exception:
        payload = str(settings)
    return str(abs(hash(payload)))


def save_checkpoint(model: "CoreTransformer", path: str | Path, settings: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok_cfg = settings.get("tokenizer", {})
    payload = {
        "model_state": model.state_dict(),
        "settings_hash": _settings_hash(settings),
        "tokenizer_paths": {
            "vortex_tok_path": tok_cfg.get("vortex_tok_path", tok_cfg.get("vortex_model_path")),
            "rnt2_model_path": tok_cfg.get("rnt2_model_path"),
        },
    }
    torch.save(payload, path)


def _load_checkpoint(model: "CoreTransformer", path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=False)
    model.checkpoint_meta = {
        "settings_hash": payload.get("settings_hash"),
        "tokenizer_paths": payload.get("tokenizer_paths", {}),
    }



class CoreTransformer(nn.Module):
    """Vortex core using V-Blocks and LAVA memory."""

    def __init__(self, config: VortexXConfig, tokenizer: VortexTokModel):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device) if config.device else torch.device("cpu")
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16 if config.dtype == "fp16" else torch.float32
        self.escape_mode = "exact"
        self.bad_block_size: int = 8
        self.bad_entropy: float = 3.5
        self.decode_cfg: dict[str, Any] = {}
        sub_size = tokenizer.sub_size_total
        self.byte_token_start = tokenizer.patch_codebook.size + tokenizer.macro_codebook.size + sub_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList()
        shared_lavas: list[LAVAMemory] = []
        group_size = max(1, int(getattr(config, "lava_shared_groups", 1)))
        for layer_idx in range(config.layers):
            block = VBlock(
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
            if group_size > 1:
                group_id = layer_idx // group_size
                if group_id < len(shared_lavas):
                    block.lava = shared_lavas[group_id]
                else:
                    shared_lavas.append(block.lava)
            self.blocks.append(block)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.mtp_k = int(getattr(config, "mtp_k", 0))
        self.mtp_head: nn.Linear | None = nn.Linear(config.hidden_size, config.vocab_size * self.mtp_k) if self.mtp_k > 0 else None
        self.kv_cache = KVHybridCache(window_size=config.window_size, kv_quant_bits=8, latent_slots=32)
        self.depth_gating: dict[str, Any] = {}
        self._depth_last: int | None = None
        self._depth_tokens = 0
        self._depth_total = 0
        self.draft_model: DraftModel | None = None
        self.runtime_cfg: dict[str, Any] = {}
        self.checkpoint_meta: dict[str, Any] = {}

    @staticmethod
    def from_settings(settings: dict) -> "CoreTransformer":
        core = settings.get("core", {})
        tok_cfg = settings.get("tokenizer", {})
        vx_cfg = settings.get("vortex_model", {})
        bad_cfg = settings.get("bad", {})
        decode_cfg = settings.get("decode", {})
        device_info = detect_device()
        if device_info.cuda_available:
            tf32 = core.get("tf32")
            if tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
                torch.backends.cudnn.allow_tf32 = bool(tf32)
            cudnn_benchmark = core.get("cudnn_benchmark")
            if cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        model_path = Path(tok_cfg.get("vortex_tok_path", tok_cfg.get("vortex_model_path", "data/runs/vortex_tok.pt")))
        block_size = int(tok_cfg.get("block_size", 64))
        tokenizer = load_or_create(model_path, block_size)

        patch_size = tokenizer.patch_codebook.size
        macro_size = tokenizer.macro_codebook.size
        sub_size = tokenizer.sub_size_total
        vocab_size = max(int(core.get("vocab_size", 1024)), patch_size + macro_size + sub_size + 256)
        layers = int(core.get("layers", 4))
        draft_layers = max(1, layers // 2)

        precision = core.get("precision")
        dtype_override = None
        if isinstance(precision, str):
            precision = precision.lower()
            if precision in {"bf16", "bfloat16"}:
                dtype_override = "bf16"
            elif precision in {"fp16", "float16"}:
                dtype_override = "fp16"
            elif precision in {"fp32", "float32"}:
                dtype_override = "fp32"

        def _vx(key: str, default):
            if key in vx_cfg:
                return vx_cfg.get(key, default)
            return core.get(key, default)

        lava_ann_mode = str(_vx("lava_ann_mode", vx_cfg.get("lava_ann_mode", "ivf" if int(_vx("lava_clusters", 0)) > 0 else "flat")))
        runtime_cfg = settings.get("runtime", {}) or {}
        kv_quant = str(runtime_cfg.get("kv_quant", "")).lower()
        if kv_quant == "2bit" and not bool(runtime_cfg.get("kv_quant_2bit_experimental", False)):
            kv_quant = "none"
        kv_bits = None
        if kv_quant == "int8":
            kv_bits = 8
        elif kv_quant == "2bit":
            kv_bits = 2
        elif kv_quant == "none":
            kv_bits = 0
        if kv_bits is None:
            kv_bits = int(settings.get("kv", {}).get("kv_quant_bits", 0) or 0)
        cfg = VortexXConfig(
            hidden_size=int(core.get("hidden_size", 256)),
            layers=layers,
            heads=int(core.get("heads", 4)),
            vocab_size=vocab_size,
            window_size=int(_vx("window_size", vx_cfg.get("local_window", core.get("local_window", 128)))),
            latent_slots=int(_vx("latent_slots", vx_cfg.get("lava_slots", core.get("lava_slots", 64)))),
            lava_top_k=int(_vx("lava_top_k", 4)),
            lava_clusters=int(_vx("lava_clusters", 0)),
            lava_cluster_top=int(_vx("lava_cluster_top", 1)),
            lava_ann_mode=lava_ann_mode,
            lava_cluster_ema=float(_vx("lava_cluster_ema", 0.1)),
            lava_cluster_reassign_threshold=float(_vx("lava_cluster_reassign_threshold", 0.0)),
            lava_read_every=int(_vx("lava_read_every", 1)),
            lava_write_every=int(_vx("lava_write_every", 1)),
            lava_write_on_surprise=bool(_vx("lava_write_on_surprise", False)),
            lava_surprise_threshold=float(_vx("lava_surprise_threshold", 0.0)),
            lava_shared_groups=int(_vx("lava_shared_groups", 1)),
            local_mixer_kernel=int(_vx("local_mixer_kernel", 5)),
            ssm_state_size=int(_vx("ssm_state_size", vx_cfg.get("ssm", {}).get("state_dim", core.get("ssm", {}).get("state_dim", 128)))),
            gated_mlp_ratio=int(vx_cfg.get("gated_mlp_ratio", core.get("mlp_ratio", 4))),
            draft_layers=int(vx_cfg.get("draft_layers", draft_layers)),
            draft_hidden=int(vx_cfg.get("draft_hidden", core.get("draft_hidden", core.get("hidden_size", 256)))),
            mtp_k=int(core.get("mtp_k", 0)),
            cuda_graphs=bool(core.get("cuda_graphs", vx_cfg.get("cuda_graphs", False))),
            kv_quant_bits=int(kv_bits),
            dtype=dtype_override or device_info.dtype,
            device=device_info.device,
        )
        model = CoreTransformer(cfg, tokenizer=tokenizer)
        kv_cfg = settings.get("kv", {}) or {}
        kv_window = int(kv_cfg.get("window_size", cfg.window_size))
        kv_latent = int(kv_cfg.get("latent_slots", 32))
        model.kv_cache = KVHybridCache(window_size=kv_window, kv_quant_bits=kv_bits, latent_slots=kv_latent)
        model.escape_mode = tok_cfg.get("escape_mode", "exact")
        model.bad_block_size = int(bad_cfg.get("block_size", decode_cfg.get("draft_block", 8)))
        model.bad_entropy = float(bad_cfg.get("entropy_threshold", decode_cfg.get("entropy_threshold", 3.5)))
        model.decode_cfg = decode_cfg
        model.depth_gating = settings.get("depth_gating", {}) or {}
        model.runtime_cfg = settings.get("runtime", {}) or {}
        kv_mode = str(model.runtime_cfg.get("kv_quant", "none")).lower()
        for block in model.blocks:
            if hasattr(block, "lava") and hasattr(block.lava, "set_kv_quant"):
                block.lava.set_kv_quant(kv_mode)
        ckpt = core.get("checkpoint_path")
        if ckpt:
            _load_checkpoint(model, ckpt)
        model.to(device_info.device, dtype=model.dtype if device_info.device.startswith("cuda") else None)
        lava_state_path = vx_cfg.get("lava_state_path")
        if lava_state_path and bool(vx_cfg.get("lava_state_autoload", False)):
            model.load_memory_state(lava_state_path)
        _maybe_enable_paged_lm_head(model, settings)
        draft_cfg = decode_cfg.get("draft_model", {}) or vx_cfg.get("draft_model", {})
        if draft_cfg.get("enabled"):
            draft_layers = int(draft_cfg.get("draft_layers", cfg.draft_layers))
            draft_hidden = int(draft_cfg.get("draft_hidden", cfg.draft_hidden))
            share_embed = bool(draft_cfg.get("share_embeddings", True))
            share_head = bool(draft_cfg.get("share_lm_head", True))
            draft_config = DraftConfig(
                hidden_size=draft_hidden,
                layers=draft_layers,
                vocab_size=cfg.vocab_size,
                window_size=cfg.window_size,
                latent_slots=int(draft_cfg.get("latent_slots", cfg.latent_slots)),
                lava_top_k=int(draft_cfg.get("lava_top_k", cfg.lava_top_k)),
                lava_clusters=int(draft_cfg.get("lava_clusters", cfg.lava_clusters)),
                lava_cluster_top=int(draft_cfg.get("lava_cluster_top", cfg.lava_cluster_top)),
                lava_ann_mode=str(draft_cfg.get("lava_ann_mode", cfg.lava_ann_mode)),
                lava_cluster_ema=float(draft_cfg.get("lava_cluster_ema", cfg.lava_cluster_ema)),
                lava_cluster_reassign_threshold=float(draft_cfg.get("lava_cluster_reassign_threshold", cfg.lava_cluster_reassign_threshold)),
                lava_read_every=int(draft_cfg.get("lava_read_every", cfg.lava_read_every)),
                lava_write_every=int(draft_cfg.get("lava_write_every", cfg.lava_write_every)),
                lava_write_on_surprise=bool(draft_cfg.get("lava_write_on_surprise", cfg.lava_write_on_surprise)),
                lava_surprise_threshold=float(draft_cfg.get("lava_surprise_threshold", cfg.lava_surprise_threshold)),
                local_mixer_kernel=int(draft_cfg.get("local_mixer_kernel", cfg.local_mixer_kernel)),
                ssm_state_size=int(draft_cfg.get("ssm_state_size", cfg.ssm_state_size)),
                gated_mlp_ratio=int(draft_cfg.get("gated_mlp_ratio", cfg.gated_mlp_ratio)),
                dtype=cfg.dtype,
                device=device_info.device,
            )
            draft_model = DraftModel(
                draft_config,
                shared_embed=model.embed if share_embed else None,
                shared_lm_head=model.lm_head if share_head else None,
                base_hidden=cfg.hidden_size,
            )
            draft_model.to(device_info.device, dtype=model.dtype if device_info.device.startswith("cuda") else None)
            draft_path = draft_cfg.get("path")
            if draft_path:
                try:
                    draft_model.load_state_dict(torch.load(draft_path, map_location="cpu"), strict=False)
                except Exception:
                    pass
            model.draft_model = draft_model
            if kv_mode:
                for block in draft_model.blocks:
                    if hasattr(block, "lava") and hasattr(block.lava, "set_kv_quant"):
                        block.lava.set_kv_quant(kv_mode)
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
            model.step = torch.compile(model.step)  # type: ignore[method-assign,assignment]
        if core.get("compile_local_mixer_step") and hasattr(torch, "compile"):
            for block in model.blocks:
                block.local.step = torch.compile(block.local.step)  # type: ignore[method-assign,assignment]
        return model

    def forward(self, input_ids: torch.Tensor, num_layers: int | None = None) -> torch.Tensor:
        return self.forward_with_aux(input_ids, num_layers=num_layers)[0]

    def forward_with_aux(
        self,
        input_ids: torch.Tensor,
        num_layers: int | None = None,
        labels: torch.Tensor | None = None,
        return_mtp: bool = False,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        mtp_logits = None
        aux_loss = None
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids)
            depth = num_layers or self.config.layers
            for block in self.blocks[:depth]:
                x = block(x)
            x = self.norm(x)
            logits = self.lm_head(x)
            if self.mtp_head is not None and (return_mtp or (self.training and labels is not None) or return_aux):
                mtp_logits = self.mtp_head(x).view(x.shape[0], x.shape[1], self.mtp_k, -1)
            if labels is not None and self.training and self.mtp_k > 0 and mtp_logits is not None:
                losses: List[torch.Tensor] = []
                for offset in range(1, self.mtp_k + 1):
                    if labels.size(1) <= offset:
                        break
                    targets = labels[:, offset:]
                    preds = mtp_logits[:, :-offset, offset - 1, :]
                    loss = torch.nn.functional.cross_entropy(
                        preds.reshape(-1, preds.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100,
                    )
                    losses.append(loss)
                if losses:
                    aux_loss = torch.stack(losses).mean()
            depth_cfg = self.depth_gating if isinstance(self.depth_gating, dict) else {}
            if labels is not None and self.training and depth_cfg.get("enabled"):
                weight = float(depth_cfg.get("compute_cost_weight", 0.0))
                if weight > 0:
                    min_depth = int(depth_cfg.get("min_depth", 1))
                    max_depth = int(depth_cfg.get("max_depth", self.config.layers))
                    max_depth = max(1, min(max_depth, self.config.layers))
                    min_depth = max(1, min(min_depth, max_depth))
                    ent = _approx_entropy_tensor(logits, top_k=int(depth_cfg.get("entropy_top_k", 64)))
                    threshold = float(depth_cfg.get("entropy_threshold", 3.5))
                    smooth = float(depth_cfg.get("smoothness", 1.0))
                    prob = torch.sigmoid((ent - threshold) / max(1e-3, smooth))
                    depth_est = min_depth + (max_depth - min_depth) * prob
                    cost = depth_est / max_depth
                    aux_loss = (aux_loss + weight * cost) if aux_loss is not None else (weight * cost)
            if return_mtp or return_aux:
                return logits, mtp_logits, aux_loss
            return logits, None, None

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
        if self.draft_model is not None:
            self.draft_model.reset_state()

    def save_memory_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for idx, block in enumerate(self.blocks):
            block.lava.save_state(path.with_suffix(f".layer{idx}.pt"))

    def load_memory_state(self, path: str | Path) -> None:
        path = Path(path)
        for idx, block in enumerate(self.blocks):
            block.lava.load_state(path.with_suffix(f".layer{idx}.pt"))

    @overload
    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: Literal[False] = False,
        return_depth: Literal[False] = False,
    ) -> tuple[torch.Tensor, List[VBlockState]]: ...

    @overload
    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: Literal[True] = True,
        return_depth: Literal[False] = False,
    ) -> tuple[torch.Tensor, List[VBlockState], torch.Tensor | None]: ...

    @overload
    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: Literal[False] = False,
        return_depth: Literal[True] = True,
    ) -> tuple[torch.Tensor, List[VBlockState], int]: ...

    @overload
    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: Literal[True] = True,
        return_depth: Literal[True] = True,
    ) -> tuple[torch.Tensor, List[VBlockState], torch.Tensor | None, int]: ...

    def step(
        self,
        token_id: int,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: bool = False,
        return_depth: bool = False,
    ):
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids).squeeze(1)
            new_state: List[VBlockState] = []
            depth_used = num_layers or self.config.layers
            depth_cfg = self.depth_gating if num_layers is None else {}
            if depth_cfg.get("enabled"):
                min_depth = int(depth_cfg.get("min_depth", 1))
                max_depth = int(depth_cfg.get("max_depth", self.config.layers))
                max_depth = min(max_depth, self.config.layers)
                min_depth = max(1, min(min_depth, max_depth))
                for idx, block in enumerate(self.blocks[:min_depth]):
                    x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                    new_state.append(layer_state)
                logits_temp = self.lm_head(self.norm(x))
                ent = _approx_entropy_logits(logits_temp, top_k=int(depth_cfg.get("entropy_top_k", 64)))
                threshold = float(depth_cfg.get("entropy_threshold", 3.5))
                hysteresis = float(depth_cfg.get("hysteresis", 0.0))
                depth_last = self._depth_last if self._depth_last is not None else min_depth
                if ent > threshold + hysteresis:
                    depth_used = max_depth
                elif ent < threshold - hysteresis:
                    depth_used = min_depth
                else:
                    depth_used = int(depth_last)
                depth_used = int(min(max(depth_used, min_depth), max_depth))
                self._depth_last = depth_used
                if depth_used > min_depth:
                    for idx, block in enumerate(self.blocks[min_depth:depth_used], start=min_depth):
                        x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                        new_state.append(layer_state)
                logits = self.lm_head(self.norm(x))
            else:
                depth_used = num_layers or self.config.layers
                for idx, block in enumerate(self.blocks[:depth_used]):
                    x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                    new_state.append(layer_state)
                x = self.norm(x)
                logits = self.lm_head(x)
            if depth_cfg.get("enabled"):
                self._depth_tokens += 1
                self._depth_total += depth_used
            mtp_logits = None
            if return_mtp and self.mtp_k > 0 and self.mtp_head is not None:
                mtp_logits = self.mtp_head(x).view(x.shape[0], self.mtp_k, -1)
        outputs: list[object] = [logits, new_state + state[depth_used:]]
        if return_mtp:
            outputs.append(mtp_logits)
        if return_depth:
            outputs.append(depth_used)
        if len(outputs) == 2:
            return outputs[0], outputs[1]  # type: ignore[return-value]
        return tuple(outputs)

    def step_topk(
        self,
        token_id: int,
        state: List[VBlockState],
        top_k: int,
        num_layers: int | None = None,
        write_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, List[VBlockState]]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(input_ids).squeeze(1)
            new_state: List[VBlockState] = []
            depth_used = num_layers or self.config.layers
            for idx, block in enumerate(self.blocks[:depth_used]):
                x, layer_state = block.step(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            values, indices = self.lm_head_topk(x, top_k)
        return values, indices, new_state + state[depth_used:]

    def step_block_topk(
        self,
        token_ids: torch.Tensor,
        state: List[VBlockState],
        top_k: int,
        num_layers: int | None = None,
        write_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, List[VBlockState]]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(token_ids)
            new_state: List[VBlockState] = []
            depth_used = num_layers or self.config.layers
            for idx, block in enumerate(self.blocks[:depth_used]):
                x, layer_state = block.step_block(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            values, indices = self.lm_head_topk(x, top_k)
        return values, indices, new_state + state[depth_used:]

    def step_block(
        self,
        token_ids: torch.Tensor,
        state: List[VBlockState],
        num_layers: int | None = None,
        write_memory: bool = True,
        return_mtp: bool = False,
    ) -> tuple[torch.Tensor, List[VBlockState]] | tuple[torch.Tensor, List[VBlockState], torch.Tensor | None]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        depth_cfg = self.depth_gating if num_layers is None else {}
        if depth_cfg.get("enabled"):
            logits_list: list[torch.Tensor] = []
            new_state = state
            for t in range(token_ids.size(1)):
                tok = int(token_ids[0, t].item())
                if return_mtp:
                    logits_t, new_state, mtp_logits = self.step(tok, new_state, num_layers=num_layers, write_memory=write_memory, return_mtp=True)
                else:
                    logits_t, new_state = self.step(tok, new_state, num_layers=num_layers, write_memory=write_memory)
                    mtp_logits = None
                logits_list.append(logits_t)
            logits_seq = torch.stack(logits_list, dim=1)
            if return_mtp:
                return logits_seq, new_state, mtp_logits
            return logits_seq, new_state
        with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
            x = self.embed(token_ids)
            new_state: List[VBlockState] = []
            depth_used = num_layers or self.config.layers
            for idx, block in enumerate(self.blocks[:depth_used]):
                x, layer_state = block.step_block(x, state[idx], write_memory=write_memory)
                new_state.append(layer_state)
            x = self.norm(x)
            logits = self.lm_head(x)
            mtp_logits = None
            if return_mtp and self.mtp_k > 0 and self.mtp_head is not None:
                mtp_logits = self.mtp_head(x).view(x.shape[0], x.shape[1], self.mtp_k, -1)
        outputs: list[object] = [logits, new_state + state[depth_used:]]
        if return_mtp:
            outputs.append(mtp_logits)
        if len(outputs) == 2:
            return outputs[0], outputs[1]  # type: ignore[return-value]
        return tuple(outputs)

    def reset_depth_stats(self) -> None:
        self._depth_tokens = 0
        self._depth_total = 0
        self._depth_last = None

    def depth_stats(self) -> dict:
        avg_depth = (self._depth_total / self._depth_tokens) if self._depth_tokens else 0.0
        return {"avg_depth_used": avg_depth, "tokens": self._depth_tokens}

    def lm_head_topk(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.lm_head, "forward_topk"):
            return self.lm_head.forward_topk(x, k)
        logits = self.lm_head(x)
        values, indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
        return values, indices

    def generate(self, prompt: str, max_new_tokens: int = 32, return_stats: bool = False, **kwargs):
        bad_cfg = {
            "block_size": getattr(self, "bad_block_size", 8),
            "entropy_threshold": getattr(self, "bad_entropy", 3.5),
        }
        bad_cfg.update(kwargs)
        decode_defaults = getattr(self, "decode_cfg", {}) or {}
        for key in [
            "temperature",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram",
            "adaptive_granularity",
            "entropy_top_k",
            "penalty_window",
            "top_p_min_k",
            "top_p_max_k",
            "exact_copy_mode",
            "escape_restrict",
            "use_mtp",
        ]:
            if key not in bad_cfg and key in decode_defaults:
                bad_cfg[key] = decode_defaults[key]
        if "entropy_top_k" not in bad_cfg and "entropy_topk" in decode_defaults:
            bad_cfg["entropy_top_k"] = decode_defaults["entropy_topk"]
        if "penalty_window" not in bad_cfg and "repetition_window" in decode_defaults:
            bad_cfg["penalty_window"] = decode_defaults["repetition_window"]
        if "top_p_min_k" not in bad_cfg and "topk_start" in decode_defaults:
            bad_cfg["top_p_min_k"] = decode_defaults["topk_start"]
        if "top_p_max_k" not in bad_cfg and "topk_max" in decode_defaults:
            bad_cfg["top_p_max_k"] = decode_defaults["topk_max"]
        if "no_repeat_ngram" not in bad_cfg and "no_repeat_ngram_n" in decode_defaults:
            bad_cfg["no_repeat_ngram"] = decode_defaults["no_repeat_ngram_n"]
        block_size = int(bad_cfg["block_size"])
        entropy_threshold = float(bad_cfg["entropy_threshold"])
        temperature = float(bad_cfg.get("temperature", 1.0))
        top_p = float(bad_cfg.get("top_p", 1.0))
        repetition_penalty = float(bad_cfg.get("repetition_penalty", 1.0))
        no_repeat_ngram = int(bad_cfg.get("no_repeat_ngram", 0))
        adaptive_granularity = bool(bad_cfg.get("adaptive_granularity", True))
        entropy_top_k = int(bad_cfg.get("entropy_top_k", 64))
        penalty_window = int(bad_cfg.get("penalty_window", 512))
        top_p_min_k = int(bad_cfg.get("top_p_min_k", 128))
        top_p_max_k = int(bad_cfg.get("top_p_max_k", 512))
        exact_copy_mode = bool(bad_cfg.get("exact_copy_mode", False))
        escape_restrict = bool(bad_cfg.get("escape_restrict", False))
        use_mtp = bool(bad_cfg.get("use_mtp", True))
        def _run_decode(tokens: int, block: int):
            with torch.inference_mode():
                with autocast_context(enabled=self.device.type == "cuda", dtype=self.config.dtype):
                    return bad_decode(
                        self,
                        prompt=prompt,
                        max_new_tokens=tokens,
                        block_size=block,
                        entropy_threshold=entropy_threshold,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram=no_repeat_ngram,
                        adaptive_granularity=adaptive_granularity,
                        entropy_top_k=entropy_top_k,
                        penalty_window=penalty_window,
                        top_p_min_k=top_p_min_k,
                        top_p_max_k=top_p_max_k,
                        exact_copy_mode=exact_copy_mode,
                        escape_restrict=escape_restrict,
                        use_mtp=use_mtp,
                    )

        try:
            text, stats = _run_decode(max_new_tokens, block_size)
        except RuntimeError as exc:
            if is_oom_error(exc) and max_new_tokens > 1:
                clear_cuda_cache()
                retry_tokens = max(1, max_new_tokens // 2)
                retry_block = max(1, block_size // 2)
                text, stats = _run_decode(retry_tokens, retry_block)
            else:
                raise
        if return_stats:
            return text, stats
        return text

