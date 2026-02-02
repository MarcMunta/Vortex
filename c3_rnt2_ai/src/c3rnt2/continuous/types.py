from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    prompt: str
    response: str
    source_kind: str = "unknown"
    messages: list[dict[str, Any]] | None = None
    quality: float | None = None
