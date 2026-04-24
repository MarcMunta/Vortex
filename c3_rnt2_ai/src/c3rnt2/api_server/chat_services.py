from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .services import ChatSessionStoreLike


Messages = list[dict[str, Any]]
JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ChatContextBundle:
    messages: Messages
    chat_memory: JsonDict
    multimodal_context: str
    multimodal: JsonDict
    temporal_context: str
    direct_reply: str | None
    live_web_context: str
    live_web_refs: list[dict[str, str]]
    rag: JsonDict
    default_system: str


@dataclass(frozen=True)
class ChatContextService:
    base_dir: Path
    settings: dict[str, Any]
    chat_sessions_store: ChatSessionStoreLike | None
    multimodal_fusion: Any
    extract_query: Callable[[Messages, str | None], str]
    inject_chat_memory_context: Callable[[ChatSessionStoreLike | None, dict[str, Any], JsonDict, Messages], tuple[Messages, JsonDict]]
    build_temporal_system_context: Callable[[JsonDict], str]
    direct_temporal_response: Callable[[JsonDict, str], str | None]
    live_web_search_context: Callable[..., tuple[str, list[dict[str, str]]]]
    inject_rag_context: Callable[[Path, dict[str, Any], Messages, str | None], tuple[Messages, str | None, JsonDict]]

    def prepare(
        self,
        *,
        payload: JsonDict,
        messages: Messages,
        default_system_base: str,
    ) -> ChatContextBundle:
        messages, chat_memory = self.inject_chat_memory_context(
            self.chat_sessions_store,
            self.settings,
            payload,
            messages,
        )
        multimodal_context = ""
        multimodal: JsonDict = {"enabled": False, "refs": []}
        try:
            if self.multimodal_fusion is not None:
                multimodal = self.multimodal_fusion.build_context(messages=messages, payload=payload)
                multimodal_context = str(multimodal.get("text") or "").strip()
        except Exception as exc:
            multimodal = {"enabled": False, "refs": [], "error": str(exc)}

        temporal_context = self.build_temporal_system_context(payload)
        direct_reply = self.direct_temporal_response(payload, self.extract_query(messages, None))
        live_web_context = ""
        live_web_refs: list[dict[str, str]] = []
        if payload.get("web_ingest"):
            user_query = self.extract_query(messages, None)
            request_allowlist = payload.get("web_allowlist")
            scoped_allowlist = request_allowlist if isinstance(request_allowlist, list) else None
            if user_query:
                try:
                    live_web_context, live_web_refs = self.live_web_search_context(
                        self.base_dir,
                        user_query,
                        self.settings,
                        max_results=4,
                        extra_allowlist=scoped_allowlist,
                    )
                except Exception:
                    live_web_context, live_web_refs = "", []

        messages, _prompt_override, rag = self.inject_rag_context(
            self.base_dir,
            self.settings,
            messages,
            None,
        )
        if live_web_refs:
            rag["refs"] = live_web_refs
        refs = multimodal.get("refs")
        if isinstance(refs, list) and refs:
            rag["refs"] = list(rag.get("refs") or []) + list(refs)

        default_system = compose_dynamic_system_prompt(
            default_system_base,
            temporal_context=temporal_context,
            web_context=live_web_context,
            multimodal_context=multimodal_context,
        )
        return ChatContextBundle(
            messages=messages,
            chat_memory=chat_memory,
            multimodal_context=multimodal_context,
            multimodal=multimodal,
            temporal_context=temporal_context,
            direct_reply=direct_reply,
            live_web_context=live_web_context,
            live_web_refs=live_web_refs,
            rag=rag,
            default_system=default_system,
        )


def compose_dynamic_system_prompt(
    base_system: str,
    *,
    temporal_context: str | None,
    web_context: str | None,
    multimodal_context: str | None = None,
) -> str:
    parts = [str(base_system or "").strip()]
    if temporal_context:
        parts.append(temporal_context.strip())
    if web_context:
        parts.append(
            "When live web results are present, prefer them over stale prior knowledge for time-sensitive questions."
        )
        parts.append(web_context.strip())
    if multimodal_context:
        parts.append(
            "Use the multimodal workspace context as situational grounding for the current request."
        )
        parts.append(multimodal_context.strip())
    return "\n\n".join(part for part in parts if part)
