from __future__ import annotations

from pathlib import Path

import pytest


def _make_state(tmp_path: Path, monkeypatch):
    from c3rnt2.control_server import ControlState

    monkeypatch.setattr(ControlState, "_ensure_autonomy_worker", lambda self: None)
    return ControlState(
        base_dir=tmp_path,
        compose_file=tmp_path / "docker-compose.yml",
        api_profile="rtx4080_16gb_programming_runtime_docker",
        training_profile="rtx4080_16gb_programming_train_docker",
        api_url="http://127.0.0.1:8000",
        runtime_url="http://127.0.0.1:30000",
        frontend_port=4173,
    )


def test_status_uses_single_runtime_probe_and_summary_runs(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-summary-status"
    run_dir = state._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text("step=3 loss=0.12\n", encoding="utf-8")
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "quick",
            "status": "running",
            "stage": "training",
            "created_at": 10.0,
            "updated_at": 9999999999.0,
            "log_path": str(run_dir / "run.log"),
            "review_sections": [{"key": "review", "title": "heavy"}],
            "agent_dialogue": [{"speaker": "agent", "text": "heavy"}],
        },
    )
    state._append_run_event(run_id, phase="training", message="trainer_started", progress_pct=0.3)

    runtime_calls = {"count": 0}

    def _runtime_status():
        runtime_calls["count"] += 1
        return {
            "api_ready": True,
            "runtime_ready": True,
            "status": {"chat_mode": "primary", "active_backend": "external"},
            "runtime_models": {"data": [{"id": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"}]},
        }

    monkeypatch.setattr(state, "docker_status", lambda: {"ready": True, "reason": "docker_ready"})
    monkeypatch.setattr(state, "runtime_status", _runtime_status)
    monkeypatch.setattr(state, "frontend_status", lambda: {"ok": True})

    payload = state.status()

    assert runtime_calls["count"] == 1
    run = payload["runs"][0]
    assert run["run_id"] == run_id
    assert run["events"] == []
    assert run["logs"] == {}
    assert run["agent_dialogue"] == []
    assert run["review_sections"] == []
    assert run["latest_event"]["message"] == "trainer_started"
    assert run["log_tail"] == []


def test_training_runs_endpoint_returns_summary_payload(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2.control_server import create_control_app

    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-summary-endpoint"
    run_dir = state._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text("step=5 loss=0.25\n", encoding="utf-8")
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "completed",
            "stage": "done",
            "created_at": 1.0,
            "updated_at": 2.0,
            "log_path": str(run_dir / "run.log"),
            "review_sections": [{"key": "review", "title": "heavy"}],
            "agent_dialogue": [{"speaker": "agent", "text": "heavy"}],
        },
    )
    state._append_run_event(run_id, phase="done", message="trainer_finished", progress_pct=1.0)

    client = TestClient(create_control_app(state))
    resp = client.get("/control/training/runs")

    assert resp.status_code == 200
    run = resp.json()["runs"][0]
    assert run["run_id"] == run_id
    assert run["events"] == []
    assert run["logs"] == {}
    assert run["agent_dialogue"] == []
    assert run["review_sections"] == []
    assert run["latest_event"]["message"] == "trainer_finished"
    assert run["log_tail"] == []
