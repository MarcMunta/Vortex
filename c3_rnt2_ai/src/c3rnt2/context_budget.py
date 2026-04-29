from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONTEXT_BUDGET: dict[str, Any] = {
    "enabled": True,
    "model_max_context_tokens": 32768,
    "default_chat_context_tokens": 16384,
    "default_agent_context_tokens": 24576,
    "max_input_tokens": 24576,
    "max_output_tokens": 2048,
    "max_agent_action_tokens": 2048,
    "max_agent_final_tokens": 4096,
    "reserve_output_tokens": 2048,
    "reserve_system_tokens": 1500,
    "reserve_tool_tokens": 4000,
    "compression_enabled": True,
    "summarize_old_messages": True,
    "rolling_summary_tokens": 2500,
    "recent_messages_tokens": 6000,
    "rag_tokens": 6000,
    "obsidian_tokens": 5000,
    "repo_context_tokens": 5000,
}


def _as_int(value: Any, default: int, *, low: int = 1, high: int = 65536) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(int(low), min(int(parsed), int(high)))


def resolve_context_budget(settings: dict[str, Any] | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONTEXT_BUDGET)
    raw = (settings or {}).get("context", {}) or {}
    if isinstance(raw, dict):
        cfg.update(raw)
    cfg["enabled"] = bool(cfg.get("enabled", True))
    for key, default in DEFAULT_CONTEXT_BUDGET.items():
        if isinstance(default, bool):
            cfg[key] = bool(cfg.get(key, default))
        elif isinstance(default, int):
            cfg[key] = _as_int(cfg.get(key), default)
    cfg["model_max_context_tokens"] = _as_int(cfg["model_max_context_tokens"], 32768, low=2048, high=65536)
    cfg["default_chat_context_tokens"] = min(cfg["default_chat_context_tokens"], cfg["model_max_context_tokens"])
    cfg["default_agent_context_tokens"] = min(cfg["default_agent_context_tokens"], cfg["model_max_context_tokens"])
    cfg["max_input_tokens"] = min(cfg["max_input_tokens"], cfg["model_max_context_tokens"])
    return cfg


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    raw = str(text)
    by_chars = max(1, len(raw) // 4)
    by_words = max(1, int(len(raw.split()) * 1.3))
    return max(by_chars, by_words)


def tokens_to_chars(tokens: int) -> int:
    return max(0, int(tokens) * 4)


def trim_to_tokens(text: str, max_tokens: int, *, tail: bool = True) -> str:
    value = str(text or "")
    max_chars = tokens_to_chars(max_tokens)
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[-max_chars:] if tail else value[:max_chars]


def context_limit_for_mode(settings: dict[str, Any] | None, mode: str = "chat") -> int:
    cfg = resolve_context_budget(settings)
    if str(mode or "").lower() == "agent":
        return int(cfg["default_agent_context_tokens"])
    return int(cfg["default_chat_context_tokens"])


def output_limit_for_mode(settings: dict[str, Any] | None, mode: str = "chat", *, final: bool = False) -> int:
    cfg = resolve_context_budget(settings)
    if str(mode or "").lower() == "agent":
        return int(cfg["max_agent_final_tokens"] if final else cfg["max_agent_action_tokens"])
    return int(cfg["max_output_tokens"])


def resolve_model_context_limit(settings: dict[str, Any] | None, model: object | None = None) -> int:
    cfg = resolve_context_budget(settings)
    for attr in ("max_position_embeddings", "n_ctx", "context_length", "model_max_length"):
        raw = getattr(model, attr, None) if model is not None else None
        try:
            value = int(raw) if raw is not None else 0
        except Exception:
            value = 0
        if value > 0 and value < 10_000_000:
            return min(value, int(cfg["model_max_context_tokens"]))
    model_cfg = getattr(model, "config", None) if model is not None else None
    for attr in ("max_position_embeddings", "n_ctx", "context_length", "model_max_length"):
        raw = getattr(model_cfg, attr, None) if model_cfg is not None else None
        try:
            value = int(raw) if raw is not None else 0
        except Exception:
            value = 0
        if value > 0 and value < 10_000_000:
            return min(value, int(cfg["model_max_context_tokens"]))
    return int(cfg["model_max_context_tokens"])


def apply_message_budget(
    messages: list[dict[str, Any]],
    settings: dict[str, Any] | None,
    *,
    mode: str = "chat",
) -> list[dict[str, Any]]:
    cfg = resolve_context_budget(settings)
    if not cfg.get("enabled", True):
        return list(messages)
    recent_budget = int(cfg["recent_messages_tokens"])
    summary_budget = int(cfg["rolling_summary_tokens"])
    system_messages = [m for m in messages if str(m.get("role") or "") == "system"]
    non_system = [m for m in messages if str(m.get("role") or "") != "system"]
    kept: list[dict[str, Any]] = []
    used = 0
    older: list[dict[str, Any]] = []
    for message in reversed(non_system):
        content = str(message.get("content") or "")
        cost = estimate_tokens(content)
        if kept and used + cost > recent_budget:
            older.append(message)
            continue
        kept.append(message)
        used += cost
    kept.reverse()
    older.reverse()
    if older and bool(cfg.get("summarize_old_messages", True)):
        bullets = []
        for message in older[-16:]:
            role = str(message.get("role") or "message")
            content = trim_to_tokens(str(message.get("content") or ""), max(64, summary_budget // 16), tail=False)
            if content:
                bullets.append(f"- {role}: {content}")
        if bullets:
            summary = "Rolling conversation summary. Preserve decisions, files, errors, and constraints:\n" + "\n".join(bullets)
            system_messages.append({"role": "system", "content": trim_to_tokens(summary, summary_budget, tail=False)})
    budget = min(context_limit_for_mode(settings, mode), int(cfg["max_input_tokens"]))
    out: list[dict[str, Any]] = []
    used = 0
    for message in system_messages + kept:
        content = str(message.get("content") or "")
        cost = estimate_tokens(content)
        if used + cost > budget:
            remaining = budget - used
            if remaining <= 0:
                continue
            clipped = dict(message)
            clipped["content"] = trim_to_tokens(content, remaining, tail=str(message.get("role") or "") != "system")
            out.append(clipped)
            break
        out.append(message)
        used += cost
    return out

