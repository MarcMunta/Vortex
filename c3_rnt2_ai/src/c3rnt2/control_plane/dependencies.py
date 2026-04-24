from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ControlDependencies:
    state: Any
    status: Callable[[], dict[str, Any]]
    start_bootstrap: Callable[..., dict[str, Any]]
    restart_runtime: Callable[[], dict[str, Any]]
    get_allowlist: Callable[[], list[str]]
    set_allowlist: Callable[[list[str]], list[str]]
    start_training: Callable[..., dict[str, Any]]
    reset_training_state: Callable[..., dict[str, Any]]
    list_runs: Callable[..., list[dict[str, Any]]]
    get_run: Callable[..., dict[str, Any] | None]
    get_run_events: Callable[..., list[dict[str, Any]]]
    get_run_logs: Callable[..., dict[str, list[str]]]
    build_training_stream_payload: Callable[[], dict[str, Any]]
    runtime_status: Callable[[], dict[str, Any]]
    autonomy_status: Callable[..., dict[str, Any]]
    start_autonomy: Callable[[], dict[str, Any]]
    stop_autonomy: Callable[[], dict[str, Any]]
    configure_autonomy: Callable[[Any], dict[str, Any]]
    latest_autonomy_events: Callable[..., list[dict[str, Any]]]
    voice_status: Callable[[], dict[str, Any]]
    restart_voice: Callable[[], dict[str, Any]]
    obsidian_status: Callable[[], dict[str, Any]]
    configure_obsidian: Callable[[dict[str, Any]], dict[str, Any]]
    multimodal_status: Callable[[], dict[str, Any]]
    api_url: str

    @classmethod
    def from_state(cls, state: Any) -> "ControlDependencies":
        return cls(
            state=state,
            status=state.status,
            start_bootstrap=state.start_bootstrap,
            restart_runtime=state.restart_runtime,
            get_allowlist=state.get_allowlist,
            set_allowlist=state.set_allowlist,
            start_training=state.start_training,
            reset_training_state=state.reset_training_state,
            list_runs=state.list_runs,
            get_run=state.get_run,
            get_run_events=state.get_run_events,
            get_run_logs=state.get_run_logs,
            build_training_stream_payload=state._build_training_stream_payload,
            runtime_status=state.runtime_status,
            autonomy_status=state.autonomy_status,
            start_autonomy=state.start_autonomy,
            stop_autonomy=state.stop_autonomy,
            configure_autonomy=state.configure_autonomy,
            latest_autonomy_events=state._latest_autonomy_events,
            voice_status=state.voice_status,
            restart_voice=state.restart_voice,
            obsidian_status=state.obsidian_status,
            configure_obsidian=state.configure_obsidian,
            multimodal_status=state.multimodal_status,
            api_url=str(state.api_url),
        )
