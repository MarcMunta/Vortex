from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OkResponse(BaseModel):
    ok: bool = True


class HealthResponse(OkResponse):
    service: str | None = None
    ts: float | None = None


class OperationalStatusResponse(BaseModel):
    ok: bool
    chat_ready: bool | None = None
    offline_ready: bool | None = None
    engine_ready: bool | None = None
    model_ready: bool | None = None
    training_ready: bool | None = None
    web_disabled: bool | None = None
    runtime_mode: str | None = None
    fallback_active: bool | None = None
    fallback_backend: str | None = None
    active_model: str | None = None


class ChatSessionSyncRequest(BaseModel):
    account_id: str
    sessions: list[dict[str, Any]]
    replace: bool = True


class ChatSessionsResponse(OkResponse):
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    count: int | None = None


class ErrorResponse(BaseModel):
    error: dict[str, Any]
