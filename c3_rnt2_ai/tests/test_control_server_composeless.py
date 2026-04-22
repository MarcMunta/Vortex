import sys
from types import SimpleNamespace
from pathlib import Path

from c3rnt2.control_server import ControlState, _http_json


def _make_state(tmp_path: Path) -> ControlState:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    return ControlState(
        base_dir=tmp_path,
        compose_file=compose_file,
        api_profile="test-api",
        training_profile="test-train",
        api_url="http://api.internal:8000",
        runtime_url="http://runtime.internal:30000",
        frontend_port=4173,
        frontend_url="http://frontend.internal:4173",
        compose_actions_enabled=False,
        assume_docker_ready=True,
    )


def test_docker_status_can_be_assumed_ready_without_local_docker(tmp_path: Path) -> None:
    state = _make_state(tmp_path)

    status = state.docker_status()

    assert status["ready"] is True
    assert status["reason"] == "docker_managed_externally"


def test_bootstrap_is_non_destructive_when_compose_actions_are_disabled(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path)
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": True, "runtime_ready": True})

    result = state.start_bootstrap(mode="ensure")

    assert result["ok"] is True
    assert result["started"] is False
    assert result["reason"] == "compose_actions_disabled"
    assert result["stage"] == "ready"


def test_frontend_status_uses_frontend_url_when_provided(tmp_path: Path) -> None:
    state = _make_state(tmp_path)

    status = state.frontend_status()

    assert status["port"] == 4173
    assert status["url"] == "http://frontend.internal:4173"


def test_http_json_returns_dict_payload(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    fake_requests = SimpleNamespace(get=lambda _url, timeout=0.0: _FakeResponse({"ok": True, "timeout": timeout}))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    payload = _http_json("http://example.local/readyz", timeout=1.25)

    assert payload is not None
    assert payload["ok"] is True
    assert payload["timeout"] == 1.25
