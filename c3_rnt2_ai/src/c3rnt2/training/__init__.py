from __future__ import annotations

from typing import Any

__all__ = ["train_router", "train_experts", "finetune_adapter"]


def __getattr__(name: str) -> Any:
    if name == "train_router":
        from .train_router import train_router

        return train_router
    if name == "train_experts":
        from .train_experts import train_experts

        return train_experts
    if name == "finetune_adapter":
        from .finetune_adapters import finetune_adapter

        return finetune_adapter
    raise AttributeError(name)
