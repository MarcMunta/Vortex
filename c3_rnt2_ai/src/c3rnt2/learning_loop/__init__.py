from __future__ import annotations

from typing import Any

__all__ = [
    "collect_from_episodes",
    "curate_dataset",
    "train_qlora",
    "evaluate_adapter",
    "promote_latest",
]


def __getattr__(name: str) -> Any:
    if name == "collect_from_episodes":
        from .data_collector import collect_from_episodes

        return collect_from_episodes
    if name == "curate_dataset":
        from .data_curator import curate_dataset

        return curate_dataset
    if name == "train_qlora":
        from .trainer import train_qlora

        return train_qlora
    if name == "evaluate_adapter":
        from .evaluator import evaluate_adapter

        return evaluate_adapter
    if name == "promote_latest":
        from .promoter import promote_latest

        return promote_latest
    raise AttributeError(name)
