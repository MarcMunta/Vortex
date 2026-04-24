from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


OpenAIErrorFactory = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ApiDependencies:
    build_operational_status: Callable[[Any, dict[str, Any], Path], dict[str, Any]]
    models_list_payload: Callable[[Any, dict[str, Any], Path], dict[str, Any]]
    openai_error: OpenAIErrorFactory
    metrics_factory: Callable[[], Any]
    apply_voice_intent: Callable[[Any], dict[str, Any] | None]
    collect_local_lab_status: Callable[[dict[str, Any], Path], dict[str, Any]]
    ensure_host_layout: Callable[[dict[str, Any], Path], dict[str, Any]]
    list_modules: Callable[[dict[str, Any], Path], list[dict[str, Any]]]
    load_progress: Callable[[dict[str, Any], Path], dict[str, Any]]
    next_module: Callable[[dict[str, Any], Path], dict[str, Any]]
    write_roadmap: Callable[[dict[str, Any], Path], dict[str, Any]]
    write_bootstrap_plan: Callable[[dict[str, Any], Path], dict[str, Any]]
    write_rag_sources_manifest: Callable[[dict[str, Any], Path], dict[str, Any]]
    create_lesson: Callable[..., dict[str, Any]]
    check_lesson: Callable[..., dict[str, Any]]
    torch: ModuleType | None
