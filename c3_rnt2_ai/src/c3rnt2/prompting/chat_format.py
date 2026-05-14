from __future__ import annotations

from typing import Any, Iterable, List


def _normalize_messages(messages: Iterable[dict], default_system: str | None) -> List[dict]:
    normalized: List[dict] = []
    raw_messages = list(messages)
    has_system = any((m.get("role") or "").lower() == "system" for m in raw_messages)
    if not has_system and default_system:
        normalized.append({"role": "system", "content": default_system})
    system_merged = False
    for msg in raw_messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if role == "system" and default_system and not system_merged:
            system_merged = True
            default_text = str(default_system or "").strip()
            content_text = str(content or "").strip()
            content = (
                f"{default_text}\n\n{content_text}"
                if default_text and content_text and default_text not in content_text
                else (content_text or default_text)
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _fallback_prompt(messages: Iterable[dict]) -> str:
    """Build a minimal prompt for models without a chat template.

    System instructions go at the top without special markers.
    Only the user question is presented last, so the model continues
    with its answer directly.
    """
    system_parts: List[str] = []
    conversation: List[dict] = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            conversation.append({"role": role, "content": content})

    system_text = "\n\n".join(system_parts)
    spanish_response = "Idioma obligatorio" in system_text and "espanol" in system_text
    user_label = "Pregunta" if spanish_response else "Q"
    assistant_label = "Respuesta" if spanish_response else "A"

    parts: List[str] = []
    if system_parts:
        # Single line instruction, no special markers to echo
        parts.append(" ".join(system_parts))
        parts.append("")

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            parts.append(f"{assistant_label}: {content}")
        else:
            parts.append(f"{user_label}: {content}")

    parts.append("Respuesta en espanol:" if spanish_response else "A:")
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
