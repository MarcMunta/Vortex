from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .types import Sample


DEFAULT_ANCHORS = [
    {
        "prompt": "Write a Python function add(a, b) that returns the sum.",
        "response": "def add(a, b):\n    return a + b\n",
    },
    {
        "prompt": "Fix the bug in this function:\n\ndef mul(a, b):\n    return a + b\n",
        "response": "def mul(a, b):\n    return a * b\n",
    },
    {
        "prompt": "Write a Python function factorial(n) using a loop.",
        "response": "def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n",
    },
]


def load_anchors(path: Path) -> List[Sample]:
    if not path.exists():
        return []
    anchors: List[Sample] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        prompt = str(payload.get("prompt", "")).strip()
        response = str(payload.get("response", "")).strip()
        if prompt and response:
            anchors.append(Sample(prompt=prompt, response=response))
    return anchors


def write_default_anchors(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    lines = [json.dumps(anchor, ensure_ascii=True) for anchor in DEFAULT_ANCHORS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
