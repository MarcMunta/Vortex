from __future__ import annotations

from typing import Any, Iterable, List


def _normalize_messages(messages: Iterable[dict], default_system: str | None) -> List[dict]:
    normalized: List[dict] = []
    has_system = any((m.get("role") or "").lower() == "system" for m in messages)
    if not has_system and default_system:
        normalized.append({"role": "system", "content": default_system})
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        normalized.append({"role": role, "content": content})
    return normalized


def _fallback_prompt(messages: Iterable[dict]) -> str:
    parts = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
        else:
            parts.append(f"### User:\n{content}")
    parts.append("### Assistant:\n")
    return "\n".join(parts).strip()


def build_chat_prompt(
    messages: Iterable[dict],
    backend: str,
    tokenizer: Any | None = None,
    default_system: str | None = None,
) -> str:
    normalized = _normalize_messages(messages, default_system)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(normalized, add_generation_prompt=True)
    return _fallback_prompt(normalized)
