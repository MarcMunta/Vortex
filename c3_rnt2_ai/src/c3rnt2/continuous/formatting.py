from __future__ import annotations

from ..prompting.chat_format import build_chat_prompt
from .types import Sample


def _normalize_messages(messages: list[dict], default_system: str | None) -> list[dict]:
    normalized: list[dict] = []
    has_system = any((m.get("role") or "").lower() == "system" for m in messages)
    if not has_system and default_system:
        normalized.append({"role": "system", "content": default_system})
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        normalized.append({"role": role, "content": content})
    return normalized


def format_chat_sample(
    sample: Sample,
    *,
    backend: str = "vortex",
    tokenizer: object | None = None,
    default_system: str | None = None,
) -> str:
    prompt = (sample.prompt or "").strip()
    response = (sample.response or "").strip()
    messages = sample.messages or ([{"role": "user", "content": prompt}] if prompt else [])
    if not messages:
        return response
    backend = (backend or "vortex").lower()
    if backend == "hf" and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        full_messages = _normalize_messages(messages, default_system)
        full_messages.append({"role": "assistant", "content": response})
        try:
            return tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        except TypeError:
            return tokenizer.apply_chat_template(full_messages, add_generation_prompt=False)
    if backend == "vortex":
        user_text = prompt or "\n".join(str(item.get("content") or "") for item in messages if str(item.get("role") or "").lower() == "user")
        return f"### User\n{user_text.strip()}\n\n### Assistant\n{response}".strip()
    prompt_text = build_chat_prompt(messages, backend=backend, tokenizer=tokenizer, default_system=default_system)
    return f"{prompt_text}{response}".strip()
