from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_state(tmp_path: Path, monkeypatch):
    from c3rnt2.control_server import ControlState

    monkeypatch.setattr(ControlState, "_ensure_autonomy_worker", lambda self: None)
    return ControlState(
        base_dir=tmp_path,
        compose_file=tmp_path / "docker-compose.yml",
        api_profile="rtx4080_16gb_programming_local",
        training_profile="rtx4080_16gb_programming_train_docker",
        api_url="http://127.0.0.1:8000",
        runtime_url="http://127.0.0.1:30000",
        frontend_port=4173,
    )


def _force_run_updated_at(state, run_id: str, updated_at: float) -> None:
    payload = state.storage.get_run(run_id) or {}
    payload["updated_at"] = float(updated_at)
    state.storage.put_run(run_id, payload)


def test_quick_training_queues_when_primary_runtime_busy(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2.utils.locks import acquire_exclusive_lock

    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})

    lock = acquire_exclusive_lock(tmp_path, "serve")
    try:
        payload = state.start_training("quick", source="feedback_queue")
    finally:
        lock.release()

    assert payload["ok"] is True
    assert payload["status"] == "queued"
    assert payload["queue_reason"] == "primary_runtime_busy"

    run = state.get_run(payload["run_id"])
    assert run is not None
    assert run["stage"] == "queued_waiting_resources"
    assert run["queue_reason"] == "primary_runtime_busy"
    assert "serve" in (run["queue_diagnostics"].get("blocking_roles") or [])


def test_compose_cmd_prefix_falls_back_to_docker_compose_when_plugin_missing(tmp_path: Path, monkeypatch) -> None:
    import c3rnt2.control_server as control_server

    state = _make_state(tmp_path, monkeypatch)

    def _which(name: str) -> str | None:
        if name == "docker":
            return "/usr/bin/docker"
        if name == "docker-compose":
            return "/usr/bin/docker-compose"
        return None

    def _run(cmd, **_kwargs):
        assert cmd == ["/usr/bin/docker", "compose", "version"]
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(control_server.shutil, "which", _which)
    monkeypatch.setattr(control_server.subprocess, "run", _run)

    assert state._compose_cmd_prefix() == ["docker-compose"]


def test_compose_cmd_includes_compose_file(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "_compose_cmd_prefix", lambda: ["docker-compose"])

    assert state._compose_cmd("images", "-q", "trainer") == [
        "docker-compose",
        "-f",
        str(state.compose_file),
        "images",
        "-q",
        "trainer",
    ]


def test_dispatch_queued_training_run_launches_when_resources_clear(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-queued-dispatch"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "quick",
            "status": "queued",
            "stage": "queued_waiting_resources",
            "created_at": 1.0,
            "queue_reason": "training_resources_busy",
        },
    )
    launched: list[tuple[str, str]] = []
    monkeypatch.setattr(state, "_queue_reason_for_mode", lambda mode: (None, {"blocking_roles": []}))
    monkeypatch.setattr(state, "_launch_training_thread", lambda run_id, mode: launched.append((run_id, mode)))

    payload = state._dispatch_queued_training_runs()

    assert payload == {"run_id": run_id, "mode": "quick"}
    assert launched == [(run_id, "quick")]
    run = state.get_run(run_id)
    assert run is not None
    assert run["stage"] == "queued"
    assert run["queue_reason"] is None


def test_stage_transition_recomputes_stale_progress_fields(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-progress-reset"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "queued",
            "stage": "queued",
            "lifecycle_state": "planned",
            "progress_pct": 0.02,
            "execution_progress_pct": 0.02,
            "pipeline_progress_pct": 0.08,
        },
    )

    updated = state._update_run_meta(
        run_id,
        {
            "status": "running",
            "stage": "training",
            "lifecycle_state": "training",
            "progress_pct": 0.52,
        },
    )

    assert updated["execution_progress_pct"] == pytest.approx(0.52)
    assert updated["pipeline_progress_pct"] == pytest.approx(0.52)


def test_promotion_skips_apply_when_eval_fails(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(type(state), "_training_manual_promotion_only", lambda self: False)
    monkeypatch.setattr(type(state), "_current_runtime_adapter_path", lambda self: "D:/previous")

    result = state._resolve_run_promotion(
        run_id="run-eval-fail",
        adapter_dir="D:/adapter",
        train_result={"promoted": True},
        eval_ok=False,
        bench_ok=True,
    )

    assert result["applied"] is False
    assert result["decision"] == "candidate_failed_eval"


def test_full_training_records_lock_diagnostics_when_serve_lock_stuck(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    control_cls = type(state)
    run_id = "run-lock-failure"
    state._active_run_id = run_id
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "queued",
            "stage": "queued",
            "created_at": 1.0,
        },
    )
    monkeypatch.setattr(
        state,
        "runtime_status",
        lambda: {
            "api_ready": True,
            "runtime_ready": True,
            "status": {"chat_mode": "primary", "active_backend": "external"},
            "runtime_models": {"data": [{"id": "sglang", "device": "cuda"}]},
        },
    )
    monkeypatch.setattr(control_cls, "_stop_runtime_stack", lambda self, **_kwargs: None)
    monkeypatch.setattr(control_cls, "_resume_runtime_stack", lambda self, **_kwargs: None)
    monkeypatch.setattr(
        control_cls,
        "_wait_for_lock_release",
        lambda self, role, timeout_s=45.0: (
            False,
            {
                "role": role,
                "released": False,
                "initially_held": True,
                "waited_s": float(timeout_s),
            },
        ),
    )
    monkeypatch.setattr(
        control_cls,
        "_run_compose",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("train should not start")),
    )

    state._training_worker(run_id, "full")

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "degraded"
    assert run["lifecycle_state"] == "degraded"
    assert run["terminal_reason"] == "serve_lock_not_released"
    assert run["lock_diagnostics"]["released"] is False


def test_full_training_fails_before_trainer_when_training_resources_never_clear(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    control_cls = type(state)
    run_id = "run-training-resources"
    state._active_run_id = run_id
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "queued",
            "stage": "queued",
            "created_at": 1.0,
        },
    )
    monkeypatch.setattr(control_cls, "_stop_runtime_stack", lambda self, **_kwargs: None)
    monkeypatch.setattr(control_cls, "_resume_runtime_stack", lambda self, **_kwargs: None)
    monkeypatch.setattr(
        control_cls,
        "_wait_for_lock_release",
        lambda self, role, timeout_s=45.0: (True, {"role": role, "released": True}),
    )
    monkeypatch.setattr(
        control_cls,
        "_start_fallback_runtime",
        lambda self, **_kwargs: {"fallback_backend": "hf", "fallback_active": True},
    )
    monkeypatch.setattr(
        control_cls,
        "_wait_for_training_resources",
        lambda self, **_kwargs: (False, {"blocking_roles": ["train"], "released": False}),
    )
    monkeypatch.setattr(
        control_cls,
        "_run_compose",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("trainer should not start")),
    )

    state._training_worker(run_id, "full")

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "queued"
    assert run["stage"] == "queued_waiting_resources"
    assert run["lifecycle_state"] == "blocked"
    assert run["blocked_reason"] == "training_resources_busy"
    assert run["queue_diagnostics"]["blocking_roles"] == ["train"]


def test_full_training_parallel_mode_skips_runtime_stop_and_resume(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    control_cls = type(state)
    run_id = "run-parallel-full"
    state._active_run_id = run_id
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "queued",
            "stage": "queued",
            "created_at": 1.0,
        },
    )

    runtime_calls: list[str] = []
    monkeypatch.setattr(
        state,
        "runtime_status",
        lambda: {
            "api_ready": True,
            "runtime_ready": True,
            "status": {"chat_mode": "fallback_degraded", "active_backend": "hf"},
            "runtime_models": {"data": [{"id": "hf", "device": "cpu"}]},
        },
    )
    monkeypatch.setattr(control_cls, "_stop_runtime_stack", lambda self, **_kwargs: runtime_calls.append("stop"))
    monkeypatch.setattr(control_cls, "_resume_runtime_stack", lambda self, **_kwargs: runtime_calls.append("resume"))
    monkeypatch.setattr(
        control_cls,
        "_wait_for_training_resources",
        lambda self, **_kwargs: (True, {"blocking_roles": [], "released": True}),
    )
    monkeypatch.setattr(
        control_cls,
        "_consume_learning_queue",
        lambda self, run_id, mode: {"run_id": run_id, "queued_count": 0, "consumed_count": 0, "items": []},
    )
    monkeypatch.setattr(
        control_cls,
        "_resolve_run_promotion",
        lambda self, **_kwargs: {
            "manual_only": False,
            "promoted": True,
            "applied": True,
            "decision": "auto_applied_runtime",
            "eval_ok": True,
            "bench_ok": True,
        },
    )
    monkeypatch.setattr(control_cls, "_should_use_local_job_runner", lambda self: False)

    def _run_compose(self, args, **_kwargs):
        if args[:3] == ["run", "--rm", "trainer"] and "train-once" in args:
            return 0, json.dumps({"ok": True, "adapter_dir": "D:/adapter", "promoted": True})
        if args[:3] == ["run", "--rm", "trainer"] and "eval" in args:
            return 0, json.dumps({"ok": True})
        if args[:3] == ["run", "--rm", "eval"] and "bench" in args:
            return 0, json.dumps({"ok": True})
        raise AssertionError(f"unexpected compose args: {args}")

    monkeypatch.setattr(control_cls, "_run_compose", _run_compose)

    state._training_worker(run_id, "full")

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["stage"] == "done"
    assert run["promotion"]["decision"] == "auto_applied_runtime"
    assert run["runtime_mode"] == "fallback_degraded"
    assert runtime_calls == []


def test_local_training_job_bypasses_serve_lock_in_parallel_mode(tmp_path: Path, monkeypatch) -> None:
    from c3rnt2.utils.locks import acquire_exclusive_lock

    state = _make_state(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        state,
        "_run_local_command",
        lambda cmd, **kwargs: (
            calls.append(cmd)
            or (0, json.dumps({"ok": True, "adapter_dir": str(tmp_path / "adapter"), "promoted": True, "steps": 25}))
        ),
    )

    serve_lock = acquire_exclusive_lock(tmp_path, "serve")
    try:
        code, output = state._run_local_training_job(
            mode="full",
            env={"C3RNT2_TRAIN_MAX_STEPS": "25"},
            log_path=tmp_path / "local-train.log",
            parallel_runtime_training=True,
        )
    finally:
        serve_lock.release()

    payload = json.loads(output)
    assert code == 0
    assert payload["ok"] is True
    assert payload["promoted"] is True
    assert payload["steps"] == 25
    assert calls
    assert "--allow-parallel-runtime" in calls[0]


def test_full_training_uses_local_runner_when_compose_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    control_cls = type(state)
    run_id = "run-local-full"
    state._active_run_id = run_id
    state.compose_actions_enabled = False
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "queued",
            "stage": "queued",
            "created_at": 1.0,
        },
    )

    local_commands: list[list[str]] = []
    monkeypatch.setattr(
        state,
        "runtime_status",
        lambda: {
            "api_ready": True,
            "runtime_ready": True,
            "status": {"chat_mode": "fallback_degraded", "active_backend": "hf"},
            "runtime_models": {"data": [{"id": "hf", "device": "cpu"}]},
        },
    )
    monkeypatch.setattr(
        control_cls,
        "_wait_for_training_resources",
        lambda self, **_kwargs: (True, {"blocking_roles": [], "released": True}),
    )
    monkeypatch.setattr(
        control_cls,
        "_consume_learning_queue",
        lambda self, run_id, mode: {"run_id": run_id, "queued_count": 0, "consumed_count": 0, "items": []},
    )
    monkeypatch.setattr(
        control_cls,
        "_resolve_run_promotion",
        lambda self, **_kwargs: {
            "manual_only": False,
            "promoted": True,
            "applied": True,
            "decision": "auto_applied_runtime",
            "eval_ok": True,
            "bench_ok": True,
        },
    )
    monkeypatch.setattr(
        control_cls,
        "_run_compose",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("docker compose should not be used")),
    )
    monkeypatch.setattr(
        control_cls,
        "_run_local_training_job",
        lambda self, **_kwargs: (0, json.dumps({"ok": True, "adapter_dir": "D:/adapter", "promoted": True})),
    )

    def _run_local_command(self, cmd, **_kwargs):
        local_commands.append(list(cmd))
        if "eval" in cmd:
            return 0, json.dumps({"ok": True})
        if "bench" in cmd:
            return 0, json.dumps({"ok": True})
        raise AssertionError(f"unexpected local command: {cmd}")

    monkeypatch.setattr(control_cls, "_run_local_command", _run_local_command)

    state._training_worker(run_id, "full")

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["stage"] == "done"
    assert run["promotion"]["decision"] == "auto_applied_runtime"
    assert any("eval" in cmd for cmd in local_commands)
    assert any("bench" in cmd for cmd in local_commands)


def test_start_training_adds_descriptive_metadata_and_agent_dialogue(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    launched: list[tuple[str, str]] = []
    monkeypatch.setattr(state, "_queue_reason_for_mode", lambda mode: (None, {"blocking_roles": []}))
    monkeypatch.setattr(state, "_launch_training_thread", lambda run_id, mode: launched.append((run_id, mode)))

    payload = state.start_training("quick", source="manual")

    assert payload["ok"] is True
    assert launched == [(payload["run_id"], "quick")]

    run = state.get_run(payload["run_id"])
    assert run is not None
    assert run["display_name"].startswith("Entrenamiento ")
    assert "operador" in run["display_description"].lower()
    assert run["base_model"] == run["served_model"]
    assert str(run["profile"]).startswith("continuous_descriptive::")
    assert run["objective"]
    assert run["learning_focus"]
    assert len(run["agent_dialogue"]) >= 2
    assert any(section["key"] == "dialogue" for section in run["review_sections"])
    assert state.autonomy_status()["latest_dialogue"]


def test_start_full_training_launches_when_runtime_is_degraded_but_ready(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    launched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        state,
        "runtime_status",
        lambda: {
            "api_ready": True,
            "runtime_ready": True,
            "status": {"chat_mode": "fallback_degraded", "active_backend": "hf"},
            "runtime_models": {"data": [{"id": "hf", "device": "cpu"}]},
        },
    )
    monkeypatch.setattr(state, "_launch_training_thread", lambda run_id, mode: launched.append((run_id, mode)))

    payload = state.start_training("full", source="manual")

    assert payload["ok"] is True
    assert payload["queue_reason"] is None
    assert launched == [(payload["run_id"], "full")]

    run = state.get_run(payload["run_id"])
    assert run is not None
    assert run["status"] == "queued"
    assert run["stage"] == "queued"


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_start_training_queues_when_runtime_is_unavailable(tmp_path: Path, monkeypatch, mode: str) -> None:
    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})

    payload = state.start_training(mode, source="manual")

    assert payload["ok"] is True
    assert payload["status"] == "queued"
    assert payload["queue_reason"] == "runtime_unavailable"

    run = state.get_run(payload["run_id"])
    assert run is not None
    assert run["stage"] == "queued_waiting_resources"
    assert run["queue_reason"] == "runtime_unavailable"


def test_autonomy_keeps_reflecting_and_defers_training_when_runtime_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    import c3rnt2.control_server as control_server

    state = _make_state(tmp_path, monkeypatch)
    now = time.time()
    autonomy = state._default_autonomy_state()
    autonomy["enabled"] = True
    autonomy["last_reflection_at"] = now - 30.0
    autonomy["last_train_at"] = now - 2.0
    autonomy["last_patch_at"] = now
    autonomy["config"]["reflection_interval_s"] = 1
    autonomy["config"]["quick_train_interval_s"] = 1
    autonomy["config"]["full_train_interval_s"] = 999999999
    autonomy["config"]["autoedit_enabled"] = False
    state._write_autonomy_state(autonomy)

    launched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(state, "_dispatch_queued_training_runs", lambda: None)
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})
    monkeypatch.setattr(
        state,
        "start_training",
        lambda mode, source=None: (
            launched.append((mode, source))
            or {"ok": True, "run_id": "run-deferred", "status": "queued", "queue_reason": "runtime_unavailable"}
        ),
    )
    monkeypatch.setattr(control_server.time, "sleep", lambda _seconds: state._autonomy_stop.set())

    state._autonomy_stop.clear()
    state._autonomy_worker()

    assert launched == [("quick", "autonomy_chain_quick")]

    autonomy_status = state.autonomy_status()
    assert autonomy_status["state"] == "learning"
    assert autonomy_status["latest_dialogue"]
    kinds = {event["kind"] for event in autonomy_status["latest_events"]}
    assert "reflection" in kinds
    assert "degraded_learning" in kinds
    assert "training_deferred" in kinds


def test_training_endpoints_and_stream_include_events_logs_and_metrics(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2.control_server import create_control_app

    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-demo"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "running",
            "stage": "training",
            "created_at": 1.0,
            "updated_at": 2.0,
            "max_steps": 10,
            "log_path": str(state._run_dir(run_id) / "run.log"),
        },
    )
    state._append_run_event(run_id, phase="training", message="trainer_started", progress_pct=0.52)
    run_dir = state._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.log").write_text("step=5 loss=0.25 tokens_per_sec=33.5 vram_peak_mb=2048\n", encoding="utf-8")
    state._active_run_id = run_id

    from c3rnt2.control_plane.dependencies import ControlDependencies

    app = create_control_app(ControlDependencies.from_state(state))
    client = TestClient(app)

    events = client.get(f"/control/training/runs/{run_id}/events")
    assert events.status_code == 200
    assert events.json()["events"][-1]["message"] == "trainer_started"

    logs = client.get(f"/control/training/runs/{run_id}/logs")
    assert logs.status_code == 200
    assert "step=5" in logs.json()["logs"]["run"][-1]

    payload = state._build_training_stream_payload()
    assert payload["active_run"]["run_id"] == run_id
    assert payload["phase"] == "training"
    assert payload["latest_metrics"]["step"] == 5
    assert payload["runtime_mode"] in {"maintenance", "fallback_degraded", "primary"}
    assert isinstance(payload["last_event"], dict)
    assert isinstance(payload["log_tail"], list)


def test_status_includes_learning_queue_summary_and_autonomy_events_are_deduped(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.storage.append_event(
        "learning_queue",
        "global",
        {"id": "item-1", "request_id": "req-1", "source_kind": "chat_feedback", "score": 1.0, "ts": 1.0},
    )
    state._write_learning_queue_state({"items": {"item-1": {"status": "queued"}}})

    event = state._append_autonomy_event(
        agent="builder",
        kind="hypothesis",
        title="Nueva hipotesis",
        detail="Detalle",
        state_name="learning",
    )

    payload = state.status()

    assert payload["learning_queue"]["queued_count"] == 1
    assert payload["learning_queue"]["items"][0]["id"] == "item-1"
    latest_events = payload["autonomy"]["latest_events"]
    assert latest_events[0]["id"] == event["id"]
    assert len({item["id"] for item in latest_events}) == len(latest_events)


def test_status_recovers_stale_runtime_when_previous_training_left_stack_drained(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    resumed: list[tuple[str, bool]] = []
    state.compose_actions_enabled = True
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})
    monkeypatch.setattr(
        state,
        "_resume_runtime_stack",
        lambda log_path, force_recreate=True: resumed.append((str(log_path), bool(force_recreate))),
    )

    run_id = "run-stale-recovery"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "maintenance",
            "stage": "draining_primary",
            "created_at": 1.0,
            "updated_at": 1.0,
        },
    )
    _force_run_updated_at(state, run_id, 1.0)
    payload = state.status()

    assert resumed
    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "interrupted"
    assert run["stage"] == "recovered_runtime"
    assert run["stale_recovery"]["status"] == "runtime_recovered"
    assert payload["runs"][0]["run_id"] == run_id


def test_status_marks_stale_run_interrupted_when_compose_actions_are_disabled(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.compose_actions_enabled = False
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})

    run_id = "run-stale-manual"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "maintenance",
            "stage": "draining_primary",
            "created_at": 1.0,
            "updated_at": 1.0,
        },
    )
    _force_run_updated_at(state, run_id, 1.0)
    state.status()

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "interrupted"
    assert run["stage"] == "manual_recovery_required"
    assert run["stale_recovery"]["status"] == "manual_recovery_required"


def test_status_reprocesses_completed_manual_recovery_if_run_is_still_active(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.compose_actions_enabled = False
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": False, "runtime_ready": False})

    run_id = "run-stale-recovery-completed"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "full",
            "status": "maintenance",
            "stage": "draining_primary",
            "created_at": 1.0,
            "updated_at": 1.0,
            "stale_recovery": {
                "completed": True,
                "status": "manual_recovery_required",
                "reason": "compose_actions_disabled",
            },
        },
    )
    _force_run_updated_at(state, run_id, 1.0)
    state.status()

    run = state.get_run(run_id)
    assert run is not None
    assert run["status"] == "interrupted"
    assert run["stage"] == "manual_recovery_required"
    assert run["stale_recovery"]["completed"] is True


def test_reset_training_state_clears_legacy_runs_queue_and_restores_continuous_autonomy(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    run_id = "run-legacy"
    state._update_run_meta(
        run_id,
        {
            "ok": True,
            "run_id": run_id,
            "mode": "quick",
            "status": "failed",
            "stage": "failed",
            "created_at": 1.0,
        },
    )
    state.storage.append_event(
        "learning_queue",
        "global",
        {"id": "item-legacy", "request_id": "req-legacy", "source_kind": "chat_feedback", "score": 1.0, "ts": 1.0},
    )
    state._write_learning_queue_state({"items": {"item-legacy": {"status": "queued"}}})
    state._append_autonomy_event(
        agent="system",
        kind="legacy_marker",
        title="Legacy",
        detail="Legacy event",
    )
    state._write_autonomy_state({"enabled": False, "state": "paused"})

    payload = state.reset_training_state()

    assert payload["ok"] is True
    assert payload["removed_runs"] >= 1
    assert state.list_runs() == []
    assert state._learning_queue_summary()["queued_count"] == 0
    autonomy = state.autonomy_status()
    assert autonomy["enabled"] is True
    assert autonomy["state"] in {"waiting_resources", "learning"}
    assert autonomy["config"]["quick_train_interval_s"] == 45
    assert autonomy["config"]["full_train_interval_s"] == 300


def test_bootstrap_ensure_skips_build_when_runtime_is_already_ready(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "docker_status", lambda: {"ready": True})
    monkeypatch.setattr(state, "runtime_status", lambda: {"api_ready": True, "runtime_ready": True})
    monkeypatch.setattr(
        state,
        "_run_compose",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compose should not run")),
    )

    state._bootstrap_worker("ensure")

    bootstrap = state.status()["bootstrap"]
    assert bootstrap["stage"] == "ready"
    assert bootstrap["mode"] == "ensure"


def test_start_bootstrap_returns_without_deadlock(tmp_path: Path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "_bootstrap_worker", lambda mode: None)

    result: dict[str, object] = {}

    def invoke() -> None:
        result.update(state.start_bootstrap(force=False, mode="ensure"))

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result["ok"] is True
    assert result["started"] is True
    assert result["stage"] == "queued"


def test_serve_fallback_lock_allows_train_but_blocks_primary_serve(tmp_path: Path) -> None:
    from c3rnt2.utils.locks import LockUnavailable, acquire_exclusive_lock

    fallback_lock = acquire_exclusive_lock(tmp_path, "serve_fallback")
    try:
        train_lock = acquire_exclusive_lock(tmp_path, "train")
        train_lock.release()
        with pytest.raises(LockUnavailable):
            acquire_exclusive_lock(tmp_path, "serve")
    finally:
        fallback_lock.release()
