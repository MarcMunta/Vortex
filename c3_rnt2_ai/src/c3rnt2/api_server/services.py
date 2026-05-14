from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from fastapi import FastAPI


class ChatSessionStoreLike(Protocol):
    def list_sessions(self, account_id: str) -> list[dict[str, Any]]: ...
    def sync_sessions(
        self,
        account_id: str,
        sessions: list[dict[str, Any]],
        *,
        replace: bool = True,
    ) -> list[dict[str, Any]]: ...
    def delete_session(self, account_id: str, session_id: str) -> None: ...
    def clear_account(self, account_id: str) -> None: ...
    def render_memory_block(self, *args: Any, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]: ...


@dataclass
class ApiRuntimeServices:
    chat_sessions_store: ChatSessionStoreLike | None
    episode_index: Any
    episode_lock: RLock
    metrics: Any


def get_api_services(app: FastAPI) -> ApiRuntimeServices:
    services = getattr(app.state, "api_services", None)
    if not isinstance(services, ApiRuntimeServices):
        raise RuntimeError("api_services_not_configured")
    return services
