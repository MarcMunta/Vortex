from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, TypeAlias

YamlMapping: TypeAlias = dict[str, Any]
ProfileDefinition: TypeAlias = dict[str, Any]
ProfileMap: TypeAlias = dict[str, ProfileDefinition]


class SettingsDocument(TypedDict, total=False):
    imports: list[str]
    profiles: ProfileMap


class CoreSettings(TypedDict, total=False):
    backend: str
    backend_fallback: str | None
    hf_model: str
    hf_fallback: str | None
    external_engine: str
    external_base_url: str
    external_url: str


class RuntimeSettings(TypedDict, total=False):
    cache_vram_budget_mb: int
    kv_quant: str
    kv_quant_2bit_experimental: bool
    prefetch_depth: int
    paged_lm_head: bool
    paged_lm_head_stream_topk: int | bool


class ToolsWebSettings(TypedDict, total=False):
    enabled: bool
    allow_domains: list[str]
    search_domains: list[str]
    max_bytes: int
    timeout_s: float
    rate_limit_per_min: int
    cache_dir: str
    cache_ttl_s: int
    allow_content_types: list[str]


class ToolsSettings(TypedDict, total=False):
    web: ToolsWebSettings


class DockerSettings(TypedDict, total=False):
    enabled: bool
    compose_path: str
    runtime_service: str
    api_service: str
    trainer_service: str
    eval_service: str


class ProfileContractSettings(TypedDict, total=False):
    offline_required: bool
    require_web_disabled: bool
    require_ollama: bool
    require_external_engine: str | None
    require_local_base_url: bool
    require_wsl_training: bool
    require_docker: bool
    disable_fallbacks: bool
    approved_training_sources_only: bool
    min_host_ram_free_mb: int


class ResolvedSettings(TypedDict, total=False):
    _profile: str
    tokenizer: dict[str, Any]
    core: CoreSettings
    runtime: RuntimeSettings
    tools: ToolsSettings
    docker: DockerSettings
    profile_contract: ProfileContractSettings
    server: dict[str, Any]
    continuous: dict[str, Any]
    hf_train: dict[str, Any]
    local_lab: dict[str, Any]
    voice: dict[str, Any]
    camera: dict[str, Any]
    gesture: dict[str, Any]
    spatial_ui: dict[str, Any]
    obsidian: dict[str, Any]
    multimodal_memory: dict[str, Any]
    multimodal_context: dict[str, Any]
    presentation: dict[str, Any]
    workspace_panels: dict[str, Any]


SettingsPath: TypeAlias = str | Path | None
