from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, List

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0


DEFAULT_TARGET_MODULES: List[str] = [
    "lm_head",
    "fc",
    "proj",
    "gate",
    "read_proj",
    "addr_proj",
    "out_proj",
    "in_proj",
]


def resolve_target_modules(adapter_cfg: dict, strict: bool = False) -> List[str]:
    targets = adapter_cfg.get("target_modules") or []
    if targets:
        return list(targets)
    if strict:
        raise ValueError("target_modules must be set when strict_target_modules is enabled")
    return list(DEFAULT_TARGET_MODULES)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.config = config
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.A = nn.Parameter(torch.zeros(config.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + lora_out * (self.config.alpha / self.config.rank)


def _matches(name: str, target_modules: Optional[Iterable[str]]) -> bool:
    if not target_modules:
        return False
    return any(t in name for t in target_modules)


def inject_lora(model: nn.Module, config: LoRAConfig, target_modules: Optional[Iterable[str]] = None) -> Dict[str, LoRALinear]:
    wrapped: Dict[str, LoRALinear] = {}
    if not target_modules:
        return wrapped
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name, target_modules):
            parent = model
            if "." in name:
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                leaf_name = parts[-1]
            else:
                leaf_name = name
            wrapped_module = LoRALinear(module, config)
            setattr(parent, leaf_name, wrapped_module)
            wrapped[name] = wrapped_module
    return wrapped


def save_lora_state(model: nn.Module, path: Path) -> None:
    payload = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            payload[name] = {
                "A": module.A.detach().cpu(),
                "B": module.B.detach().cpu(),
                "rank": module.config.rank,
                "alpha": module.config.alpha,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_lora_state(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in payload:
            module.A.data.copy_(payload[name]["A"])
            module.B.data.copy_(payload[name]["B"])
    return True
