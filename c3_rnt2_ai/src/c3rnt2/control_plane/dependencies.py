from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class ControlStateLike(Protocol):
    api_url: str

    def status(self) -> dict[str, Any]: ...
    def start_bootstrap(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def restart_runtime(self) -> dict[str, Any]: ...
    def get_allowlist(self) -> list[str]: ...
    def set_allowlist(self, domains: list[str]) -> list[str]: ...
    def start_training(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def reset_training_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def list_runs(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def get_run(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None: ...
    def get_run_events(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def get_run_logs(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]: ...
    def _build_training_stream_payload(self) -> dict[str, Any]: ...
    def runtime_status(self) -> dict[str, Any]: ...
    def autonomy_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def start_autonomy(self) -> dict[str, Any]: ...
    def stop_autonomy(self) -> dict[str, Any]: ...
    def configure_autonomy(self, payload: Any) -> dict[str, Any]: ...
    def _latest_autonomy_events(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def voice_status(self) -> dict[str, Any]: ...
    def restart_voice(self) -> dict[str, Any]: ...
    def obsidian_status(self) -> dict[str, Any]: ...
    def configure_obsidian(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def multimodal_status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ControlDependencies:
    state: ControlStateLike
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
    def from_state(cls, state: ControlStateLike) -> "ControlDependencies":
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
