from __future__ import annotations

from pathlib import Path


class _DummyModel:
    model_id = "ci-dummy"
    is_external = True

    def runtime_stats(self) -> dict[str, object]:
        return {"ok": True, "stub": True}


class _ControlState:
    api_url = "http://127.0.0.1:8000"

    def status(self): return {"ok": True, "api": {"ready": True}, "runs": []}
    def start_bootstrap(self, **_kwargs): return {"ok": True, "started": False}
    def restart_runtime(self): return {"ok": True}
    def get_allowlist(self): return ["example.com"]
    def set_allowlist(self, domains): return domains
    def start_training(self, *_args, **_kwargs): return {"ok": True, "run_id": "ci-run"}
    def reset_training_state(self, **_kwargs): return {"ok": True, "runs": []}
    def list_runs(self, **_kwargs): return []
    def get_run(self, *_args, **_kwargs): return None
    def get_run_events(self, *_args, **_kwargs): return []
    def get_run_logs(self, *_args, **_kwargs): return {}
    def _build_training_stream_payload(self): return {"ts": 0.0, "runs": []}
    def runtime_status(self): return {"ok": True, "runtime_ready": True}
    def autonomy_status(self, **_kwargs): return {"enabled": False, "boot_mode": "manual", "state": "idle", "active_agents": [], "autoedit_scope": "safe"}
    def start_autonomy(self): return {"ok": True, "enabled": True}
    def stop_autonomy(self): return {"ok": True, "enabled": False}
    def configure_autonomy(self, _payload): return {"ok": True, "autonomy": self.autonomy_status()}
    def _latest_autonomy_events(self, **_kwargs): return []
    def voice_status(self): return {"ok": True, "enabled": False}
    def restart_voice(self): return {"ok": True, "enabled": False}
    def obsidian_status(self): return {"ok": True, "enabled": False}
    def configure_obsidian(self, _payload): return {"ok": True}
    def multimodal_status(self): return {"ok": True}


def test_actual_api_and_control_apps_share_ci_safe_contracts(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import c3rnt2.server as api_server
    from c3rnt2.control_plane.app import create_control_app
    from c3rnt2.control_plane.dependencies import ControlDependencies

    dummy = _DummyModel()
    settings = {
        "server": {"lazy_model_load": False},
        "core": {"backend": "external", "external_model": "ci-dummy"},
        "rag": {"enabled": False},
        "continuous": {},
        "multimodal": {"voice": {"enabled": False}, "obsidian": {"enabled": False}},
    }
    monkeypatch.setattr(
        api_server,
        "_load_initial_models_with_boot_fallback",
        lambda *_args, **_kwargs: (dummy, {"external": dummy}, settings, None),
    )
    monkeypatch.setattr(
        api_server,
        "prepare_model_state",
        lambda *_args, **_kwargs: {
            "offline_ready": True,
            "engine_ready": True,
            "model_ready": True,
            "training_ready": True,
            "web_disabled": False,
            "docker_ready": True,
            "wsl_ready": True,
            "engine_kind": "external",
            "active_model": "ci-dummy",
        },
    )

    api_client = TestClient(api_server.create_app(settings, base_dir=tmp_path))
    control_client = TestClient(create_control_app(ControlDependencies.from_state(_ControlState())))

    assert api_client.get("/healthz").status_code == 200
    assert api_client.get("/readyz").status_code == 200
    assert api_client.get("/v1/status").json()["ok"] is True
    assert api_client.get("/v1/chat/sessions", params={"account_id": "ci"}).json()["ok"] is True
    assert api_client.get("/v1/voice/status").json()["ok"] is True

    assert control_client.get("/healthz").json()["ok"] is True
    assert control_client.get("/control/status").json()["ok"] is True
    assert control_client.get("/control/training/runs").json()["runs"] == []
    assert control_client.get("/control/autonomy/status").json()["autonomy"]["state"] == "idle"


def test_control_status_returns_degraded_json_on_state_error() -> None:
    from fastapi.testclient import TestClient

    from c3rnt2.control_plane.app import create_control_app
    from c3rnt2.control_plane.dependencies import ControlDependencies

    state = _ControlState()

    def _broken_status():
        raise RuntimeError("status_broken")

    deps = ControlDependencies.from_state(state)
    deps = ControlDependencies(
        **{**deps.__dict__, "status": _broken_status}
    )
    client = TestClient(create_control_app(deps))

    resp = client.get("/control/status")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["error"] == "status_broken"
