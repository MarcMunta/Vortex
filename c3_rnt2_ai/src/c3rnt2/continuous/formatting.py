from __future__ import annotations

from .types import Sample


def format_chat_sample(sample: Sample) -> str:
    prompt = (sample.prompt or "").strip()
    response = (sample.response or "").strip()
    if not prompt:
        return response
    return f"### User:\n{prompt}\n### Assistant:\n{response}"
