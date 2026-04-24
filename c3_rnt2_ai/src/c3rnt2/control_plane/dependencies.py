from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .models import AutonomyConfigRequest


JsonDict = dict[str, Any]


class ControlStateLike(Protocol):
    api_url: str

    def status(self) -> JsonDict: ...
    def start_bootstrap(self, *args: Any, **kwargs: Any) -> JsonDict: ...
    def restart_runtime(self) -> JsonDict: ...
    def get_allowlist(self) -> list[str]: ...
    def set_allowlist(self, domains: list[str]) -> list[str]: ...
    def start_training(self, *args: Any, **kwargs: Any) -> JsonDict: ...
    def reset_training_state(self, *args: Any, **kwargs: Any) -> JsonDict: ...
    def list_runs(self, *args: Any, **kwargs: Any) -> list[JsonDict]: ...
    def get_run(self, *args: Any, **kwargs: Any) -> JsonDict | None: ...
    def get_run_events(self, *args: Any, **kwargs: Any) -> list[JsonDict]: ...
    def get_run_logs(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]: ...
    def _build_training_stream_payload(self) -> JsonDict: ...
    def runtime_status(self) -> JsonDict: ...
    def autonomy_status(self, *args: Any, **kwargs: Any) -> JsonDict: ...
    def start_autonomy(self) -> JsonDict: ...
    def stop_autonomy(self) -> JsonDict: ...
    def configure_autonomy(self, payload: AutonomyConfigRequest) -> JsonDict: ...
    def _latest_autonomy_events(self, *args: Any, **kwargs: Any) -> list[JsonDict]: ...
    def voice_status(self) -> JsonDict: ...
    def restart_voice(self) -> JsonDict: ...
    def obsidian_status(self) -> JsonDict: ...
    def configure_obsidian(self, payload: JsonDict) -> JsonDict: ...
    def multimodal_status(self) -> JsonDict: ...


@dataclass(frozen=True)
class ControlDependencies:
    state: ControlStateLike
    status: Callable[[], JsonDict]
    start_bootstrap: Callable[..., JsonDict]
    restart_runtime: Callable[[], JsonDict]
    get_allowlist: Callable[[], list[str]]
    set_allowlist: Callable[[list[str]], list[str]]
    start_training: Callable[..., JsonDict]
    reset_training_state: Callable[..., JsonDict]
    list_runs: Callable[..., list[JsonDict]]
    get_run: Callable[..., JsonDict | None]
    get_run_events: Callable[..., list[JsonDict]]
    get_run_logs: Callable[..., dict[str, list[str]]]
    build_training_stream_payload: Callable[[], JsonDict]
    runtime_status: Callable[[], JsonDict]
    autonomy_status: Callable[..., JsonDict]
    start_autonomy: Callable[[], JsonDict]
    stop_autonomy: Callable[[], JsonDict]
    configure_autonomy: Callable[[AutonomyConfigRequest], JsonDict]
    latest_autonomy_events: Callable[..., list[JsonDict]]
    voice_status: Callable[[], JsonDict]
    restart_voice: Callable[[], JsonDict]
    obsidian_status: Callable[[], JsonDict]
    configure_obsidian: Callable[[JsonDict], JsonDict]
    multimodal_status: Callable[[], JsonDict]
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
