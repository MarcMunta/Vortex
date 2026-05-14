from __future__ import annotations

import argparse
import asyncio
import contextlib
import ast
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import uvicorn

from .config import load_settings
from .control_plane import create_control_app
from .control_plane.dependencies import ControlDependencies
from .control_plane.services import RuntimeCommandService
from .control_plane.storage import OperationalStore
from .instructions import load_instruction_bundle
from .model_init import DEFAULT_MODEL_ID, model_cache_status, resolve_cache_dir


DEFAULT_CONTROL_PORT = 8765
DEFAULT_FRONTEND_PORT = 4173
DEFAULT_API_PORT = 8000
DEFAULT_RUNTIME_PORT = 30000
DEFAULT_API_PROFILE = "rtx4080_16gb_llama2_7b_q4_local"
DEFAULT_TRAINING_PROFILE = "rtx4080_16gb_llama2_7b_q4_local"
DEFAULT_FALLBACK_PROFILE = "rtx4080_16gb_safe_windows_hf"
DEFAULT_QUICK_QUEUE_THRESHOLD = 3
DEFAULT_QUICK_QUEUE_COOLDOWN_S = 900
DEFAULT_BOOTSTRAP_MODE = "ensure"

_RUN_PROGRESS_BY_STAGE: dict[str, float] = {
    "queued": 0.02,
    "queued_waiting_resources": 0.0,
    "draining_primary": 0.08,
    "fallback_ready": 0.16,
    "training": 0.52,
    "eval": 0.72,
    "apply": 0.86,
    "resume_primary": 0.86,
    "bench": 0.94,
    "done": 1.0,
    "completed": 1.0,
    "completed_with_warnings": 1.0,
    "failed": 1.0,
    "exception": 1.0,
    "runtime_resume_failed": 1.0,
}

_RUN_ACTIVE_STAGES = {
    "queued",
    "queued_waiting_resources",
    "draining_primary",
    "fallback_ready",
    "training",
    "eval",
    "resume_primary",
    "bench",
}

_RUN_PIPELINE_PROGRESS_BY_LIFECYCLE: dict[str, float] = {
    "collecting": 0.04,
    "curating": 0.08,
    "planned": 0.12,
    "blocked": 0.12,
    "training": 0.52,
    "evaluating": 0.72,
    "applying": 0.84,
    "verifying": 0.92,
    "completed": 1.0,
    "rolled_back": 1.0,
    "degraded": 1.0,
}

_RUN_TERMINAL_LIFECYCLES = {"completed", "rolled_back", "degraded"}


def _utc_ts() -> float:
    return float(time.time())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported_type:{type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return _json_safe(raw)
    return value


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail(path: Path, lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:]
    except Exception:
        return []


def _dedupe_event_items(items: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for payload in items:
        if not isinstance(payload, dict):
            continue
        event_id = str(payload.get("id") or "").strip()
        key = event_id or json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_default)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(payload)
        if limit is not None and len(deduped) >= int(limit):
            break
    return deduped


def _normalize_bootstrap_mode(raw_mode: str | None, *, force: bool = False) -> str:
    if force:
        return "rebuild"
    text = str(raw_mode or DEFAULT_BOOTSTRAP_MODE).strip().lower()
    if text in {"", "default"}:
        return DEFAULT_BOOTSTRAP_MODE
    if text not in {"ensure", "rebuild"}:
        raise HTTPException(status_code=400, detail="bootstrap_mode_invalid")
    return text


def _run_autopilot_tick_lazy(*args: Any, **kwargs: Any) -> Any:
    from .autopilot import run_autopilot_tick

    return run_autopilot_tick(*args, **kwargs)


def _parse_live_metrics(lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    window = lines[-80:]
    for line in window:
        lowered = line.lower()
        weight_match = re.search(
            r"loading weights:\s*([0-9]+(?:\.[0-9]+)?)%\|.*?\|\s*([0-9]+)\s*/\s*([0-9]+)",
            lowered,
        )
        if weight_match:
            try:
                metrics["load_weights_pct"] = float(weight_match.group(1))
                metrics["load_weights_loaded"] = int(weight_match.group(2))
                metrics["load_weights_total"] = int(weight_match.group(3))
            except Exception:
                pass
        for key, pattern in (
            ("step", r"(?:^|\b)(?:step|steps)\s*[:=]\s*(\d+)"),
            ("loss", r"(?:^|\b)loss\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"),
            ("tokens_per_sec", r"(?:tokens(?:/s|_per_sec)|tok/s)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"),
            ("vram_peak_mb", r"(?:vram(?:_peak)?_mb|gpu(?:_mem)?_mb)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"),
        ):
            match = re.search(pattern, lowered)
            if not match:
                continue
            value = match.group(1)
            if key == "step":
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)
    return metrics


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _http_json(url: str, *, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=float(timeout))
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.post(url, json=payload, timeout=float(timeout))
        data = resp.json()
        if isinstance(data, dict):
            data.setdefault("http_status", int(resp.status_code))
            data.setdefault("http_ok", bool(resp.ok))
            return data
    except Exception:
        return None
    return None


def _parse_url_target(raw_url: str | None, *, default_host: str, default_port: int) -> tuple[str, int, str]:
    text = str(raw_url or "").strip()
    if not text:
        return default_host, int(default_port), f"http://{default_host}:{int(default_port)}"
    parsed = urlparse(text if "://" in text else f"http://{text}")
    host = parsed.hostname or default_host
    port = int(parsed.port or default_port)
    scheme = parsed.scheme or "http"
    return host, port, f"{scheme}://{host}:{port}"


def _parse_structured_output(raw: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        try:
            payload = ast.literal_eval(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return None


def _iter_hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    seen = 0
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in sorted(root.rglob("*")) if p.is_file()]
        for path in files:
            try:
                digest.update(
                    str(path.relative_to(root.parent if root.parent.exists() else root)).encode(
                        "utf-8",
                        errors="ignore",
                    )
                )
            except Exception:
                digest.update(str(path).encode("utf-8", errors="ignore"))
            try:
                digest.update(path.read_bytes())
                seen += 1
            except Exception:
                continue
    return f"sha256:{digest.hexdigest()}" if seen else "sha256:empty"


class ControlState:
    def __init__(
        self,
        *,
        base_dir: Path,
        compose_file: Path,
        api_profile: str,
        training_profile: str,
        api_url: str,
        runtime_url: str,
        frontend_port: int,
        frontend_url: str | None = None,
        compose_actions_enabled: bool = True,
        assume_docker_ready: bool = False,
    ) -> None:
        self.base_dir = base_dir
        self.compose_file = compose_file
        self.api_profile = api_profile
        self.training_profile = training_profile
        self.api_url = api_url.rstrip("/")
        self.runtime_url = runtime_url.rstrip("/")
        self.frontend_port = int(frontend_port)
        _, _, normalized_frontend_url = _parse_url_target(
            frontend_url,
            default_host="127.0.0.1",
            default_port=int(frontend_port),
        )
        self.frontend_url = normalized_frontend_url.rstrip("/")
        self.compose_actions_enabled = bool(compose_actions_enabled)
        self.assume_docker_ready = bool(assume_docker_ready)

        self.control_dir = self.base_dir / "data" / "control"
        self.bootstrap_state_path = self.control_dir / "bootstrap_state.json"
        self.runs_dir = self.control_dir / "training_runs"
        self.internet_settings_path = self.control_dir / "internet_settings.json"
        self.autonomy_state_path = self.control_dir / "autonomy_state.json"
        self.autonomy_events_path = self.control_dir / "autonomy_events.jsonl"
        self.learning_queue_path = self.control_dir / "learning_queue.jsonl"
        self.learning_queue_state_path = self.control_dir / "learning_queue_state.json"
        self.runtime_state_path = self.control_dir / "runtime_state.json"
        self.storage_path = self.control_dir / "operational_state.sqlite3"
        self.log_dir = self.base_dir.parent / "logs"
        self.fallback_profile = DEFAULT_FALLBACK_PROFILE

        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.storage = OperationalStore(self.storage_path)
        self.runtime_commands = RuntimeCommandService(
            base_dir=self.base_dir,
            compose_file=self.compose_file,
            api_profile=self.api_profile,
            training_profile=self.training_profile,
            compose_actions_enabled=self.compose_actions_enabled,
        )

        self._lock = threading.RLock()
        self._bootstrap_thread: threading.Thread | None = None
        self._training_thread: threading.Thread | None = None
        self._autonomy_thread: threading.Thread | None = None
        self._autonomy_stop = threading.Event()
        self._active_run_id: str | None = None
        self._migrate_legacy_storage()

        if not self.bootstrap_state_path.exists():
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "idle",
                    "message": "control_ready",
                    "updated_at": _utc_ts(),
                }
            )
        self.storage.put_state("internet", {"domains": self.get_allowlist(), "updated_at": _utc_ts()})
        if not self.autonomy_state_path.exists():
            self._write_state_record("autonomy", self._default_autonomy_state(), self.autonomy_state_path)
        if not self.learning_queue_state_path.exists():
            self._write_state_record("learning_queue", self._default_learning_queue_state(), self.learning_queue_state_path)
        if not self.runtime_state_path.exists():
            self._write_state_record("runtime", self._default_runtime_state(), self.runtime_state_path)

        auto_start_autonomy = str(os.getenv("C3RNT2_CONTROL_AUTONOMY_AUTOSTART", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if auto_start_autonomy:
            self._ensure_autonomy_worker()

    def _migrate_legacy_storage(self) -> None:
        self.storage.import_state_file("bootstrap", self.bootstrap_state_path, {})
        self.storage.import_state_file("runtime", self.runtime_state_path, self._default_runtime_state())
        self.storage.import_state_file("autonomy", self.autonomy_state_path, self._default_autonomy_state())
        self.storage.import_state_file("learning_queue", self.learning_queue_state_path, self._default_learning_queue_state())
        self.storage.import_jsonl_events("autonomy", "global", self.autonomy_events_path)
        self.storage.import_jsonl_events("learning_queue", "global", self.learning_queue_path)
        self.storage.import_training_runs(self.runs_dir)

    def _read_state_record(self, key: str, default: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
        legacy = _load_json(path, default) if path is not None and path.exists() else default
        current = self.storage.get_state(key, legacy)
        return current if isinstance(current, dict) else dict(default)

    def _write_state_record(self, key: str, payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
        self.storage.put_state(key, payload)
        return payload

    def _set_bootstrap_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            current = self._read_state_record("bootstrap", {}, self.bootstrap_state_path)
            current.update(payload)
            current["updated_at"] = _utc_ts()
            self._write_state_record("bootstrap", current, self.bootstrap_state_path)

    def _compose_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return self.runtime_commands.compose_env(extra)

    def _compose_cmd_prefix(self) -> list[str]:
        return self.runtime_commands.compose_cmd_prefix()

    def _compose_cmd(self, *args: str) -> list[str]:
        return [*self._compose_cmd_prefix(), "-f", str(self.compose_file), *args]

    def _run_compose(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        line_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        return self.runtime_commands.run_compose(
            args,
            env=env,
            log_path=log_path,
            line_callback=line_callback,
        )

    def _should_use_local_job_runner(self) -> bool:
        return (not self.compose_actions_enabled) or self.runtime_commands.should_use_local_job_runner()

    def _run_local_command(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        line_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        return self.runtime_commands.run_local_command(
            cmd,
            env=env,
            log_path=log_path,
            line_callback=line_callback,
        )

    def _run_local_training_job(
        self,
        *,
        mode: str,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        parallel_runtime_training: bool = False,
        line_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        cmd = [sys.executable, "-m", "c3rnt2", "train-once", "--profile", self.training_profile]
        if mode == "quick":
            cmd.append("--reuse-dataset")
        if parallel_runtime_training:
            cmd.append("--allow-parallel-runtime")
        return self._run_local_command(
            cmd,
            env=env,
            log_path=log_path,
            line_callback=line_callback,
        )

    def _wait_runtime_ready(self, timeout_s: float = 240.0) -> bool:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            ready = _http_json(f"{self.api_url}/readyz", timeout=2.0)
            if ready and bool(ready.get("ok")):
                return True
            time.sleep(2.0)
        return False

    def _api_host_port(self) -> tuple[str, int]:
        host, port, _ = _parse_url_target(
            self.api_url,
            default_host="127.0.0.1",
            default_port=DEFAULT_API_PORT,
        )
        return host, port

    def _served_model_id(self, runtime: dict[str, Any] | None = None) -> str:
        status_payload = None
        if isinstance(runtime, dict):
            raw_status = runtime.get("status")
            if isinstance(raw_status, dict):
                status_payload = raw_status
        if isinstance(status_payload, dict):
            active_model = str(status_payload.get("active_model") or "").strip()
            if active_model:
                return active_model
        try:
            settings = load_settings(self.api_profile)
        except Exception:
            settings = {}
        core = settings.get("core", {}) if isinstance(settings, dict) else {}
        for key in ("external_model", "hf_model", "model_name"):
            model_id = str((core or {}).get(key) or "").strip()
            if model_id:
                return model_id
        return DEFAULT_MODEL_ID

    def _training_base_model_id(self) -> str:
        try:
            settings = load_settings(self.training_profile)
        except Exception:
            settings = {}
        hf_train = settings.get("hf_train", {}) if isinstance(settings, dict) else {}
        core = settings.get("core", {}) if isinstance(settings, dict) else {}
        for key in ("model_name",):
            model_id = str((hf_train or {}).get(key) or "").strip()
            if model_id:
                return model_id
        for key in ("hf_model", "external_model", "model_name"):
            model_id = str((core or {}).get(key) or "").strip()
            if model_id:
                return model_id
        return self._served_model_id()

    def _load_profile_settings(self, profile_name: str) -> dict[str, Any]:
        try:
            settings = load_settings(profile_name)
        except Exception:
            return {}
        return settings if isinstance(settings, dict) else {}

    def _training_manual_promotion_only(self) -> bool:
        settings = self._load_profile_settings(self.training_profile)
        hf_train = settings.get("hf_train", {}) if isinstance(settings, dict) else {}
        return bool((hf_train or {}).get("manual_promotion_only", False))

    def _current_runtime_mode(self, runtime: dict[str, Any] | None = None) -> str:
        current = runtime or self.runtime_status()
        status = current.get("status") if isinstance(current.get("status"), dict) else {}
        if isinstance(status, dict):
            chat_mode = str(status.get("chat_mode") or "").strip().lower()
            if chat_mode:
                return chat_mode
        return "fallback_degraded" if self._runtime_allows_parallel_training(current) else "primary"

    def _request_runtime_adapter_reload(self, adapter_dir: str | Path) -> dict[str, Any]:
        adapter_path = Path(str(adapter_dir))
        profile_settings = self._load_profile_settings(self.api_profile)
        core = profile_settings.get("core", {}) if isinstance(profile_settings, dict) else {}
        if not bool((core or {}).get("hf_use_latest_adapter", False)):
            return {
                "applied": False,
                "decision": "auto_apply_disabled",
                "error": "hf_use_latest_adapter_disabled",
                "runtime_profile": self.api_profile,
                "adapter_path": str(adapter_path),
            }

        request_path = self.base_dir / "data" / "state" / "reload.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            request_path.write_text(
                json.dumps(
                    {"adapter_path": str(adapter_path), "ts": _utc_ts()},
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            return {
                "applied": False,
                "decision": "auto_apply_failed",
                "error": str(exc),
                "runtime_profile": self.api_profile,
                "adapter_path": str(adapter_path),
            }

        reload_payload = _http_post_json(
            f"{self.api_url}/v1/reload_adapter",
            payload={"adapter_path": str(adapter_path)},
            timeout=20.0,
        )
        if isinstance(reload_payload, dict) and bool(reload_payload.get("ok", False)):
            return {
                "applied": True,
                "decision": "auto_applied_runtime",
                "reload": reload_payload,
                "runtime_profile": self.api_profile,
                "adapter_path": str(reload_payload.get("adapter_path") or adapter_path),
            }

        reload_error = None
        if isinstance(reload_payload, dict):
            reload_error = str(reload_payload.get("error") or "").strip() or None

        return {
            "applied": False,
            "decision": "reload_requested",
            "queued_reload": True,
            "reload": reload_payload,
            "runtime_profile": self.api_profile,
            "adapter_path": str(adapter_path),
        }

    def _repo_clean_for_autoedit(self) -> bool:
        repo_root = self.base_dir.parent
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0 and not str(result.stdout or "").strip()

    def _autoedit_scope_ok(self) -> bool:
        settings = self._load_profile_settings(self.training_profile)
        self_patch = settings.get("self_patch", {}) if isinstance(settings, dict) else {}
        allowed_paths = self_patch.get("allowed_paths") if isinstance(self_patch, dict) else []
        return bool(allowed_paths)

    def _smoke_check_runtime_adapter(self, adapter_dir: str | Path | None, *, timeout_s: float = 45.0) -> dict[str, Any]:
        target = str(adapter_dir or "").strip() or None
        started = time.time()
        last_runtime: dict[str, Any] | None = None
        while (time.time() - started) <= float(timeout_s):
            runtime = self.runtime_status()
            last_runtime = runtime
            if bool(runtime.get("api_ready")) and bool(runtime.get("runtime_ready")):
                current_adapter = self._current_runtime_adapter_path(runtime)
                if target is None or current_adapter == target:
                    return {
                        "ok": True,
                        "adapter_path": current_adapter,
                        "waited_s": round(time.time() - started, 3),
                    }
            time.sleep(1.5)
        return {
            "ok": False,
            "adapter_path": self._current_runtime_adapter_path(last_runtime),
            "waited_s": round(time.time() - started, 3),
            "runtime": _json_safe(last_runtime or {}),
        }

    def _rollback_runtime_adapter(self, adapter_dir: str | Path | None) -> dict[str, Any]:
        target = str(adapter_dir or "").strip()
        if not target:
            return {"ok": False, "reason": "no_previous_adapter"}
        request = self._request_runtime_adapter_reload(target)
        smoke = self._smoke_check_runtime_adapter(target, timeout_s=45.0)
        ok = bool(smoke.get("ok"))
        if ok:
            self._write_runtime_state({"active_adapter_path": target, "last_good_adapter_path": target})
        return {
            "ok": ok,
            "reason": "rollback_applied" if ok else "rollback_failed",
            "requested_adapter_path": target,
            "request": request,
            "smoke": smoke,
        }

    def _resolve_run_promotion(
        self,
        *,
        run_id: str,
        adapter_dir: str | Path | None,
        train_result: dict[str, Any],
        eval_ok: bool,
        bench_ok: bool,
    ) -> dict[str, Any]:
        manual_only = self._training_manual_promotion_only()
        promoted = bool((train_result or {}).get("promoted", False))
        previous_adapter = self._current_runtime_adapter_path()
        gate_results = {
            "manual_only": manual_only,
            "promoted": promoted,
            "eval_ok": bool(eval_ok),
            "bench_ok": bool(bench_ok),
            "smoke_check_required": True,
            "repo_clean_for_autoedit": self._repo_clean_for_autoedit(),
            "autoedit_scope_ok": self._autoedit_scope_ok(),
        }
        summary = {"gate_results": gate_results, "rollback_result": None}
        if not adapter_dir:
            return {
                **summary,
                "apply_result": {"applied": False, "decision": "no_adapter_artifact"},
                "applied": False,
                "decision": "no_adapter_artifact",
            }
        if not promoted:
            return {
                **summary,
                "apply_result": {
                    "applied": False,
                    "decision": "manual_review_required" if manual_only else "candidate_not_promoted",
                    "adapter_path": str(adapter_dir),
                },
                "applied": False,
                "decision": "manual_review_required" if manual_only else "candidate_not_promoted",
                "adapter_path": str(adapter_dir),
            }
        if not bool(eval_ok):
            return {
                **summary,
                "apply_result": {
                    "applied": False,
                    "decision": "candidate_failed_eval",
                    "adapter_path": str(adapter_dir),
                },
                "applied": False,
                "decision": "candidate_failed_eval",
                "adapter_path": str(adapter_dir),
            }
        if not bool(bench_ok):
            return {
                **summary,
                "apply_result": {
                    "applied": False,
                    "decision": "candidate_failed_bench",
                    "adapter_path": str(adapter_dir),
                },
                "applied": False,
                "decision": "candidate_failed_bench",
                "adapter_path": str(adapter_dir),
            }
        self._update_run_meta(
            run_id,
            {
                "status": "running",
                "stage": "apply",
                "lifecycle_state": "applying",
                "execution_progress_pct": 0.86,
                "gate_results": gate_results,
                "apply_result": {"requested_adapter_path": str(adapter_dir)},
            },
        )
        self._append_run_dialogue_turn(
            run_id,
            speaker="builder",
            speaker_label="Constructor",
            kind="apply",
            message="Los gates de entrenamiento ya estan cerrados; voy a intentar aplicar el adapter y validar el runtime con smoke check.",
        )
        self._append_run_event(
            run_id,
            phase="apply",
            message="runtime_apply_started",
            kind="phase",
            progress_pct=0.86,
            metadata={"adapter_dir": str(adapter_dir)},
        )
        self._append_notebook_section(
            run_id,
            phase="applying",
            title="Aplicacion del adapter",
            content="Se solicita la recarga del adapter entrenado en el runtime antes de verificar salud y consistencia.",
            kind="apply",
            metadata={"adapter_dir": str(adapter_dir)},
        )
        apply_result = self._request_runtime_adapter_reload(adapter_dir)
        smoke = self._smoke_check_runtime_adapter(adapter_dir, timeout_s=45.0)
        gate_results["smoke_ok"] = bool(smoke.get("ok"))
        gate_results["smoke_waited_s"] = smoke.get("waited_s")
        apply_result = {**apply_result, "smoke": smoke}
        if bool(smoke.get("ok")):
            self._write_runtime_state(
                {
                    "active_adapter_path": str(adapter_dir),
                    "last_good_adapter_path": str(adapter_dir),
                }
            )
            self._append_notebook_section(
                run_id,
                phase="verifying",
                title="Verificacion post-apply",
                content="El runtime sigue operativo y expone el adapter recien aplicado.",
                kind="verify",
                metadata={"adapter_dir": str(adapter_dir)},
            )
            self._append_run_dialogue_turn(
                run_id,
                speaker="analyst",
                speaker_label="Analista",
                kind="verify",
                message="El runtime ha quedado estable tras la recarga. Marco el ciclo como aplicable y verificable.",
            )
            return {
                **summary,
                "gate_results": gate_results,
                "apply_result": apply_result,
                "rollback_result": None,
                "applied": True,
                "decision": "auto_applied_runtime",
            }

        rollback_result = self._rollback_runtime_adapter(previous_adapter)
        self._append_run_dialogue_turn(
            run_id,
            speaker="analyst",
            speaker_label="Analista",
            kind="rollback",
            message=(
                "El smoke check ha fallado; fuerzo rollback al adapter anterior para no degradar el runtime."
                if rollback_result.get("ok")
                else "El smoke check ha fallado y el rollback no ha conseguido restaurar un estado valido."
            ),
        )
        self._append_notebook_section(
            run_id,
            phase="rollback" if rollback_result.get("ok") else "degraded",
            title="Rollback de seguridad",
            content=(
                "El smoke check ha fallado y se ha restaurado el adapter anterior."
                if rollback_result.get("ok")
                else "El smoke check ha fallado y el rollback no ha logrado restaurar un estado valido."
            ),
            kind="rollback",
            metadata=_json_safe(rollback_result),
        )
        return {
            **summary,
            "gate_results": gate_results,
            "apply_result": apply_result,
            "rollback_result": rollback_result,
            "applied": False,
            "decision": "rolled_back_after_smoke_failure" if rollback_result.get("ok") else "runtime_degraded_after_apply",
        }

    def _default_runtime_state(self) -> dict[str, Any]:
        return {
            "mode": "primary",
            "fallback_active": False,
            "fallback_backend": None,
            "fallback_profile": self.fallback_profile,
            "fallback_pid": None,
            "active_adapter_path": None,
            "last_good_adapter_path": None,
            "updated_at": _utc_ts(),
        }

    def _load_runtime_state(self) -> dict[str, Any]:
        current = self._read_state_record("runtime", self._default_runtime_state(), self.runtime_state_path)
        merged = self._default_runtime_state()
        if isinstance(current, dict):
            merged.update(current)
        return merged

    def _write_runtime_state(self, patch: dict[str, Any]) -> dict[str, Any]:
        current = self._load_runtime_state()
        current.update(patch)
        current["updated_at"] = _utc_ts()
        return self._write_state_record("runtime", current, self.runtime_state_path)

    def _current_runtime_adapter_path(self, runtime: dict[str, Any] | None = None) -> str | None:
        current = runtime or self.runtime_status()
        status = current.get("status") if isinstance(current.get("status"), dict) else {}
        if isinstance(status, dict):
            adapters = status.get("adapters")
            if isinstance(adapters, dict):
                for value in adapters.values():
                    text = str(value or "").strip()
                    if text:
                        return text
            adapter = str(status.get("adapter") or "").strip()
            if adapter:
                return adapter
        runtime_state = self._load_runtime_state()
        text = str(runtime_state.get("active_adapter_path") or "").strip()
        return text or None

    def _clamp_progress(self, value: Any, default: float = 0.0) -> float:
        try:
            raw = float(value)
        except Exception:
            raw = float(default)
        return max(0.0, min(1.0, raw))

    def _ensure_active_campaign(self, *, reason: str = "continuous") -> dict[str, Any]:
        autonomy = self._load_autonomy_state()
        current_campaign = autonomy.get("current_campaign") if isinstance(autonomy.get("current_campaign"), dict) else {}
        current_id = str(autonomy.get("active_campaign_id") or current_campaign.get("id") or "").strip()
        if current_id:
            campaign = dict(current_campaign or {})
            campaign.setdefault("id", current_id)
            campaign.setdefault("objective", "Aprendizaje continuo 24/7 orientado a fiabilidad, mejora y verificacion.")
            campaign.setdefault("started_at", autonomy.get("updated_at") or _utc_ts())
            campaign.setdefault("reason", reason)
            if campaign != current_campaign:
                self._write_autonomy_state({"active_campaign_id": current_id, "current_campaign": campaign})
            return campaign

        campaign = {
            "id": f"camp-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "objective": "Aprendizaje continuo 24/7 orientado a fiabilidad, mejora y verificacion.",
            "started_at": _utc_ts(),
            "reason": reason,
        }
        self._write_autonomy_state({"active_campaign_id": campaign["id"], "current_campaign": campaign})
        self._append_autonomy_event(
            agent="system",
            kind="campaign_started",
            title="Nueva campana de aprendizaje",
            detail="Se ha abierto una campana persistente para encadenar runs, aplicar mejoras seguras y mantener trazabilidad completa.",
            state_name="learning",
            metadata={"campaign_id": campaign["id"], "reason": reason},
        )
        return campaign

    def _campaign_summary(
        self,
        *,
        runs: list[dict[str, Any]] | None = None,
        autonomy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        autonomy = autonomy or self._load_autonomy_state()
        campaign = autonomy.get("current_campaign") if isinstance(autonomy.get("current_campaign"), dict) else {}
        campaign_id = str(autonomy.get("active_campaign_id") or campaign.get("id") or "").strip()
        if not campaign_id:
            return None
        recent_runs = runs if runs is not None else self.list_runs(include_details=False, limit=80)
        matching = [run for run in recent_runs if str(run.get("campaign_id") or "").strip() == campaign_id]
        matching.sort(key=lambda item: float(item.get("created_at") or 0.0))
        started_at = float((campaign or {}).get("started_at") or (matching[0].get("created_at") if matching else _utc_ts()) or _utc_ts())
        runtime_hours = max(1e-6, (_utc_ts() - started_at) / 3600.0)
        completed = [run for run in matching if str(run.get("lifecycle_state") or "") == "completed"]
        rolled_back = [run for run in matching if str(run.get("lifecycle_state") or "") == "rolled_back"]
        degraded = [run for run in matching if str(run.get("lifecycle_state") or "") == "degraded"]
        active = next((run for run in reversed(matching) if str(run.get("status") or "").strip().lower() in {"queued", "running", "maintenance"}), None)
        success_streak = 0
        failure_streak = 0
        for run in reversed(matching):
            lifecycle = str(run.get("lifecycle_state") or "").strip().lower()
            if lifecycle == "completed" and failure_streak == 0:
                success_streak += 1
                continue
            if lifecycle in {"rolled_back", "degraded"} and success_streak == 0:
                failure_streak += 1
                continue
            break
        return {
            "campaign_id": campaign_id,
            "objective": (campaign or {}).get("objective") or "Aprendizaje continuo 24/7 orientado a fiabilidad, mejora y verificacion.",
            "started_at": started_at,
            "run_count": len(matching),
            "completed_count": len(completed),
            "rolled_back_count": len(rolled_back),
            "degraded_count": len(degraded),
            "active_run_id": (active or {}).get("run_id"),
            "success_streak": success_streak,
            "failure_streak": failure_streak,
            "throughput_per_hour": round(len(matching) / runtime_hours, 2),
            "last_apply": next((run.get("apply_result") for run in reversed(matching) if isinstance(run.get("apply_result"), dict)), None),
            "last_rollback": next((run.get("rollback_result") for run in reversed(matching) if isinstance(run.get("rollback_result"), dict)), None),
        }

    def _derive_run_context(self, *, mode: str, source: str | None) -> dict[str, Any]:
        source_label = str(source or "manual").strip().lower()
        autonomy = self._load_autonomy_state()
        campaign = self._ensure_active_campaign(reason=source_label) if (autonomy.get("enabled", True) or source_label.startswith("autonomy")) else {
            "id": f"manual-{time.strftime('%Y%m%d-%H%M%S')}",
            "objective": "Ejecucion manual del pipeline de entrenamiento.",
            "started_at": _utc_ts(),
            "reason": source_label,
        }
        parent_run_id = str(autonomy.get("scheduled_parent_run_id") or "").strip() or None
        if parent_run_id is None:
            latest = next(
                (
                    run for run in self.list_runs(include_details=False, limit=80)
                    if str(run.get("campaign_id") or "").strip() == str(campaign.get("id") or "").strip()
                ),
                None,
            )
            if latest:
                parent_run_id = str(latest.get("run_id") or "").strip() or None
        parent_run = self.get_run(parent_run_id) if parent_run_id else None
        parent_lifecycle = str((parent_run or {}).get("lifecycle_state") or "").strip().lower()
        attempt = 1
        if parent_run and parent_lifecycle in {"rolled_back", "degraded"}:
            attempt = max(1, int((parent_run or {}).get("attempt") or 1) + 1)
        run_lineage = list((parent_run or {}).get("run_lineage") or [])
        if parent_run_id and (not run_lineage or run_lineage[-1] != parent_run_id):
            run_lineage.append(parent_run_id)
        return {
            "campaign_id": str(campaign.get("id") or "").strip() or None,
            "parent_run_id": parent_run_id,
            "attempt": attempt,
            "run_lineage": run_lineage[-8:],
        }

    def _append_notebook_section(
        self,
        run_id: str,
        *,
        phase: str,
        title: str,
        content: str,
        kind: str = "note",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.get_run(run_id) or {}
        sections = list(run.get("notebook_sections") or [])
        entry = {
            "id": f"{run_id}-note-{uuid.uuid4().hex[:8]}",
            "phase": phase,
            "kind": kind,
            "title": title,
            "content": content,
            "ts": _utc_ts(),
            "metadata": metadata or {},
        }
        sections.append(entry)
        self._update_run_meta(run_id, {"notebook_sections": sections[-48:]})
        return entry

    def _append_run_dialogue_turn(
        self,
        run_id: str,
        *,
        speaker: str,
        speaker_label: str,
        kind: str,
        message: str,
        cycle_id: str | None = None,
    ) -> dict[str, Any]:
        run = self.get_run(run_id) or {}
        dialogue = list(run.get("agent_dialogue") or [])
        turn = {
            "id": f"{run_id}-{speaker}-{uuid.uuid4().hex[:8]}",
            "speaker": speaker,
            "speaker_label": speaker_label,
            "kind": kind,
            "ts": _utc_ts(),
            "message": message,
            "cycle_id": cycle_id,
        }
        dialogue.append(turn)
        self._update_run_meta(run_id, {"agent_dialogue": dialogue[-24:]})
        return turn

    def _append_live_metrics_point(self, run_id: str, *, phase: str, metrics: dict[str, Any]) -> None:
        if not metrics:
            return
        run = self.get_run(run_id) or {}
        series = list(run.get("live_metrics_series") or [])
        point = {"ts": _utc_ts(), "phase": phase, "metrics": dict(metrics)}
        last = series[-1] if series else None
        if isinstance(last, dict) and json.dumps(last.get("metrics") or {}, ensure_ascii=True, sort_keys=True) == json.dumps(metrics, ensure_ascii=True, sort_keys=True):
            return
        series.append(point)
        self._update_run_meta(run_id, {"live_metrics_series": series[-160:]})

    def _next_followup_mode(self, latest_run: dict[str, Any] | None) -> str:
        lifecycle = str((latest_run or {}).get("lifecycle_state") or "").strip().lower()
        queue = self._learning_queue_summary()
        queued_count = int(queue.get("queued_count") or 0)
        quick_threshold = int(queue.get("quick_threshold") or DEFAULT_QUICK_QUEUE_THRESHOLD)
        if lifecycle in {"rolled_back", "degraded"}:
            return "quick"
        if queued_count >= quick_threshold or str((latest_run or {}).get("mode") or "").strip().lower() == "quick":
            return "full"
        return "quick"

    def _followup_delay_s(self, latest_run: dict[str, Any] | None, autonomy: dict[str, Any]) -> float:
        config = autonomy.get("config") if isinstance(autonomy.get("config"), dict) else {}
        lifecycle = str((latest_run or {}).get("lifecycle_state") or "").strip().lower()
        if lifecycle == "completed":
            return float(config.get("chain_run_delay_s", 6) or 6)
        attempt = max(1, int((latest_run or {}).get("attempt") or 1))
        base = float(config.get("failure_backoff_base_s", 8) or 8)
        cap = float(config.get("failure_backoff_max_s", 90) or 90)
        return min(cap, max(base, base * (2 ** max(0, attempt - 1))))

    def _default_learning_queue_state(self) -> dict[str, Any]:
        return {
            "items": {},
            "quick_threshold": DEFAULT_QUICK_QUEUE_THRESHOLD,
            "quick_cooldown_s": DEFAULT_QUICK_QUEUE_COOLDOWN_S,
            "last_quick_dispatch_at": None,
            "last_quick_dispatch_run_id": None,
            "updated_at": _utc_ts(),
        }

    def _load_learning_queue_state(self) -> dict[str, Any]:
        current = self._read_state_record("learning_queue", self._default_learning_queue_state(), self.learning_queue_state_path)
        merged = self._default_learning_queue_state()
        if isinstance(current, dict):
            merged.update(current)
        items = merged.get("items")
        merged["items"] = dict(items) if isinstance(items, dict) else {}
        return merged

    def _write_learning_queue_state(self, patch: dict[str, Any]) -> dict[str, Any]:
        current = self._load_learning_queue_state()
        existing_items = dict(current.get("items") or {})
        current.update(patch)
        if "items" in patch and isinstance(patch["items"], dict):
            merged_items = dict(existing_items)
            merged_items.update(patch["items"])
            current["items"] = merged_items
        current["updated_at"] = _utc_ts()
        return self._write_state_record("learning_queue", current, self.learning_queue_state_path)

    def _list_learning_queue(self, *, include_consumed: bool = True) -> list[dict[str, Any]]:
        stored = self.storage.list_events("learning_queue", reverse=False)
        items = _dedupe_event_items(stored)
        state = self._load_learning_queue_state()
        statuses = state.get("items") if isinstance(state.get("items"), dict) else {}
        merged: list[dict[str, Any]] = []
        for payload in items:
            item_id = str(payload.get("id") or "").strip()
            state_payload = statuses.get(item_id) if item_id else None
            enriched = dict(payload)
            if isinstance(state_payload, dict):
                enriched.update(state_payload)
            enriched.setdefault("status", "queued")
            if include_consumed or enriched.get("status") != "consumed":
                merged.append(enriched)
        merged.sort(key=lambda item: float(item.get("ts") or 0.0), reverse=True)
        return merged

    def _consume_learning_queue(self, run_id: str, *, mode: str) -> dict[str, Any]:
        queue = list(reversed(self._list_learning_queue(include_consumed=False)))
        if not queue:
            return {
                "run_id": run_id,
                "queued_count": 0,
                "consumed_count": 0,
                "source_kinds": {},
                "request_ids": [],
                "items": [],
            }
        limit = 12 if str(mode).strip().lower() == "quick" else len(queue)
        selected = queue[:limit]
        updates: dict[str, Any] = {}
        source_kinds: dict[str, int] = {}
        request_ids: list[str] = []
        for item in selected:
            item_id = str(item.get("id") or "").strip()
            if item_id:
                updates[item_id] = {
                    "status": "consumed",
                    "consumed_by": run_id,
                    "consumed_at": _utc_ts(),
                }
            kind = str(item.get("source_kind") or "unknown").strip() or "unknown"
            source_kinds[kind] = int(source_kinds.get(kind, 0)) + 1
            request_id = str(item.get("request_id") or "").strip()
            if request_id:
                request_ids.append(request_id)
        if updates:
            self._write_learning_queue_state({"items": updates})
        return {
            "run_id": run_id,
            "queued_count": len(queue),
            "consumed_count": len(selected),
            "source_kinds": source_kinds,
            "request_ids": request_ids,
            "items": selected[-6:],
        }

    def _restore_learning_queue(self, run_id: str) -> dict[str, Any]:
        state = self._load_learning_queue_state()
        items = state.get("items") if isinstance(state.get("items"), dict) else {}
        updates: dict[str, Any] = {}
        restored = 0
        for item_id, payload in items.items():
            if not isinstance(payload, dict):
                continue
            if str(payload.get("consumed_by") or "").strip() != run_id:
                continue
            updates[str(item_id)] = {
                "status": "queued",
                "consumed_by": None,
                "consumed_at": None,
            }
            restored += 1
        if updates:
            self._write_learning_queue_state({"items": updates})
        return {"run_id": run_id, "restored_count": restored}

    def _runtime_overlay(self, *, active_run: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime_state = self._load_runtime_state()
        stage = str((active_run or {}).get("stage") or "")
        mode = str((active_run or {}).get("mode") or "")
        explicit_runtime_mode = str((active_run or {}).get("runtime_mode") or "").strip().lower()
        if bool(runtime_state.get("fallback_active")):
            runtime_mode = "fallback_degraded"
        elif explicit_runtime_mode in {"primary", "maintenance", "fallback_degraded"}:
            runtime_mode = explicit_runtime_mode
        elif mode == "quick":
            runtime_mode = "primary"
        elif stage in {"draining_primary", "training", "eval", "resume_primary", "bench"}:
            runtime_mode = "maintenance"
        else:
            runtime_mode = "primary"
        return {
            "runtime_mode": runtime_mode,
            "fallback_active": bool(runtime_state.get("fallback_active")),
            "fallback_backend": runtime_state.get("fallback_backend"),
            "fallback_profile": runtime_state.get("fallback_profile"),
        }

    def _infer_lifecycle_state(self, meta: dict[str, Any] | None) -> str:
        if not isinstance(meta, dict):
            return "planned"
        explicit = str(meta.get("lifecycle_state") or "").strip().lower()
        if explicit:
            return explicit
        stage = str(meta.get("stage") or meta.get("status") or "queued").strip().lower()
        status = str(meta.get("status") or "").strip().lower()
        if stage in {"queued_waiting_resources"}:
            return "blocked"
        if stage in {"queued"}:
            return "planned"
        if stage in {"training", "draining_primary", "fallback_ready"}:
            return "training"
        if stage in {"eval"}:
            return "evaluating"
        if stage in {"resume_primary", "bench"}:
            return "verifying"
        if stage in {"apply"}:
            return "applying"
        if status in {"completed"} or stage in {"done"}:
            return "completed"
        if status in {"rolled_back"}:
            return "rolled_back"
        if status in {"degraded", "failed", "interrupted", "completed_with_warnings"}:
            return "degraded"
        return "planned"

    def _compute_execution_progress(self, meta: dict[str, Any] | None) -> float:
        if not isinstance(meta, dict):
            return 0.0
        explicit = meta.get("execution_progress_pct")
        if explicit is not None:
            try:
                return max(0.0, min(1.0, float(explicit)))
            except Exception:
                pass
        explicit = meta.get("progress_pct")
        if explicit is not None:
            try:
                return max(0.0, min(1.0, float(explicit)))
            except Exception:
                pass
        stage = str(meta.get("stage") or meta.get("status") or "queued").strip().lower()
        progress = _RUN_PROGRESS_BY_STAGE.get(stage, 0.0)
        latest_metrics = meta.get("latest_metrics")
        max_steps = meta.get("max_steps")
        if isinstance(latest_metrics, dict) and stage == "training" and not latest_metrics.get("step"):
            try:
                load_pct = float(latest_metrics.get("load_weights_pct") or 0.0)
            except Exception:
                load_pct = 0.0
            if load_pct > 0:
                progress = max(0.18, min(0.52, 0.18 + (load_pct / 100.0) * 0.34))
        if isinstance(latest_metrics, dict) and max_steps:
            try:
                step = float(latest_metrics.get("step") or 0.0)
                total = float(max_steps)
                if stage == "training" and total > 0:
                    progress = min(0.68, max(progress, 0.18 + (step / total) * 0.5))
            except Exception:
                pass
        return max(0.0, min(1.0, float(progress)))

    def _compute_pipeline_progress(self, meta: dict[str, Any] | None) -> float:
        if not isinstance(meta, dict):
            return 0.0
        explicit = meta.get("pipeline_progress_pct")
        if explicit is not None:
            try:
                return self._clamp_progress(explicit)
            except Exception:
                pass
        lifecycle = self._infer_lifecycle_state(meta)
        return self._clamp_progress(_RUN_PIPELINE_PROGRESS_BY_LIFECYCLE.get(lifecycle, 0.0))

    def _compute_run_progress(self, meta: dict[str, Any] | None) -> float:
        return self._compute_execution_progress(meta)

    def _terminal_reason(self, meta: dict[str, Any]) -> str | None:
        failure = meta.get("failure") if isinstance(meta.get("failure"), dict) else {}
        apply_result = meta.get("apply_result") if isinstance(meta.get("apply_result"), dict) else {}
        rollback_result = meta.get("rollback_result") if isinstance(meta.get("rollback_result"), dict) else {}
        promotion = meta.get("promotion") if isinstance(meta.get("promotion"), dict) else {}
        reason = (
            failure.get("reason")
            or apply_result.get("error")
            or rollback_result.get("reason")
            or promotion.get("decision")
            or meta.get("queue_reason")
        )
        text = str(reason or "").strip()
        return text or None

    def _normalize_source_mix(self, meta: dict[str, Any]) -> dict[str, Any]:
        if isinstance(meta.get("source_mix"), dict):
            return dict(meta.get("source_mix") or {})
        dataset_manifest = meta.get("dataset_manifest") if isinstance(meta.get("dataset_manifest"), dict) else {}
        source_kinds = dataset_manifest.get("source_kinds") if isinstance(dataset_manifest.get("source_kinds"), dict) else {}
        if source_kinds:
            return dict(source_kinds)
        dataset_mix = meta.get("dataset_mix") if isinstance(meta.get("dataset_mix"), dict) else {}
        return dict(dataset_mix or {})

    def _compare_with_parent_run(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        parent_run_id = str(meta.get("parent_run_id") or "").strip()
        if not parent_run_id:
            return None
        parent = self.get_run(parent_run_id)
        if not parent:
            return None
        current_metrics = meta.get("latest_metrics") if isinstance(meta.get("latest_metrics"), dict) else {}
        parent_metrics = parent.get("latest_metrics") if isinstance(parent.get("latest_metrics"), dict) else {}
        comparison: dict[str, Any] = {
            "parent_run_id": parent_run_id,
            "parent_lifecycle_state": parent.get("lifecycle_state"),
            "parent_terminal_reason": self._terminal_reason(parent),
            "source_mix_delta": {},
        }
        for key in ("loss", "tokens_per_sec", "vram_peak_mb", "step"):
            if key in current_metrics or key in parent_metrics:
                comparison[key] = {
                    "current": current_metrics.get(key),
                    "previous": parent_metrics.get(key),
                }
        current_sources = self._normalize_source_mix(meta)
        parent_sources = self._normalize_source_mix(parent)
        keys = sorted(set(current_sources.keys()) | set(parent_sources.keys()))
        comparison["source_mix_delta"] = {
            key: {
                "current": current_sources.get(key),
                "previous": parent_sources.get(key),
            }
            for key in keys
        }
        comparison["apply_changed"] = bool(meta.get("apply_result")) != bool(parent.get("apply_result"))
        return comparison

    def _build_run_artifacts(self, meta: dict[str, Any]) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        for key, label in (
            ("adapter_dir", "adapter"),
            ("log_path", "logs"),
            ("eval_log_path", "eval"),
            ("bench_log_path", "bench"),
            ("runtime_log_path", "runtime"),
            ("events_path", "events"),
            ("fallback_log_path", "fallback"),
        ):
            value = meta.get(key)
            if value:
                artifacts[label] = str(value)
        return artifacts

    def _run_failure_payload(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(meta, dict):
            return None
        if str(meta.get("status") or "").lower() not in {"failed", "completed_with_warnings"}:
            return None
        failure = meta.get("failure")
        if isinstance(failure, dict):
            return failure
        reason = (
            meta.get("runtime_resume_error")
            or meta.get("error")
            or meta.get("queue_reason")
            or ((meta.get("train_result") or {}).get("error") if isinstance(meta.get("train_result"), dict) else None)
        )
        if not reason:
            return None
        return {"reason": str(reason), "stage": meta.get("stage")}

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _append_run_event(
        self,
        run_id: str,
        *,
        phase: str,
        message: str,
        kind: str = "phase",
        latest_metrics: dict[str, Any] | None = None,
        progress_pct: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": uuid.uuid4().hex[:12],
            "ts": _utc_ts(),
            "run_id": run_id,
            "phase": phase,
            "kind": kind,
            "message": message,
            "latest_metrics": latest_metrics or {},
            "progress_pct": progress_pct,
            "metadata": metadata or {},
        }
        self.storage.append_event("training_run", run_id, event)
        meta_patch: dict[str, Any] = {"latest_event": event}
        if latest_metrics:
            meta_patch["latest_metrics"] = latest_metrics
        if progress_pct is not None:
            meta_patch["progress_pct"] = progress_pct
        self._update_run_meta(run_id, meta_patch)
        return event

    def _read_run_events(self, run_id: str, *, limit: int = 120) -> list[dict[str, Any]]:
        stored = self.storage.list_events("training_run", run_id, limit=limit, reverse=False)
        return _dedupe_event_items(stored)[-limit:]

    def _read_run_logs(self, run_id: str, *, lines: int = 80) -> dict[str, list[str]]:
        run_dir = self._run_dir(run_id)
        return {
            "run": _tail(run_dir / "run.log", lines=lines),
            "eval": _tail(run_dir / "eval.log", lines=lines),
            "bench": _tail(run_dir / "bench.log", lines=lines),
            "runtime": _tail(run_dir / "runtime.log", lines=lines),
            "fallback": _tail(run_dir / "fallback.log", lines=lines),
        }

    def _enrich_run(self, payload: dict[str, Any], *, include_details: bool = True) -> dict[str, Any]:
        meta = dict(payload)
        run_id = str(meta.get("run_id") or "").strip()
        active_run = run_id and run_id == self._active_run_id
        overlay = self._runtime_overlay(active_run=meta if active_run else None)
        lifecycle_state = self._infer_lifecycle_state(meta)
        execution_progress = self._compute_execution_progress(meta)
        pipeline_progress = self._compute_pipeline_progress(meta)
        meta.setdefault("lifecycle_state", lifecycle_state)
        meta.setdefault("execution_progress_pct", execution_progress)
        meta.setdefault("pipeline_progress_pct", pipeline_progress)
        meta.setdefault("progress_pct", execution_progress)
        meta["events_path"] = f"sqlite://training_run/{run_id}" if run_id else None
        events = self._read_run_events(run_id, limit=80) if (run_id and include_details) else []
        meta["events"] = events if include_details else []
        meta["latest_event"] = events[-1] if events else meta.get("latest_event")
        logs = self._read_run_logs(run_id, lines=60) if (run_id and include_details) else {}
        meta["log_tail"] = logs.get("run") if (include_details and logs) else []
        meta["logs"] = logs if include_details else {}
        meta["latest_metrics"] = dict(meta.get("latest_metrics") or {})
        if not meta["latest_metrics"] and include_details and logs:
            meta["latest_metrics"] = _parse_live_metrics(logs.get("run") or [])
        meta["live_metrics_series"] = list(meta.get("live_metrics_series") or []) if include_details else []
        meta["notebook_sections"] = list(meta.get("notebook_sections") or []) if include_details else []
        meta["gate_results"] = dict(meta.get("gate_results") or {})
        meta["apply_result"] = dict(meta.get("apply_result") or {}) if isinstance(meta.get("apply_result"), dict) else meta.get("apply_result")
        meta["rollback_result"] = dict(meta.get("rollback_result") or {}) if isinstance(meta.get("rollback_result"), dict) else meta.get("rollback_result")
        meta["blocked_reason"] = str(meta.get("blocked_reason") or meta.get("queue_reason") or "").strip() or None
        meta["blocked_since"] = meta.get("blocked_since")
        meta["retry_in_s"] = meta.get("retry_in_s")
        meta["next_run_scheduled_at"] = meta.get("next_run_scheduled_at")
        meta["terminal_reason"] = self._terminal_reason(meta)
        meta["artifacts"] = self._build_run_artifacts(meta)
        meta["failure"] = self._run_failure_payload(meta)
        meta["display_name"] = str(meta.get("display_name") or self._run_display_name(meta))
        meta["display_description"] = str(
            meta.get("display_description")
            or meta.get("objective")
            or (
                "Entrenamiento descriptivo listo para revisar en detalle."
                if str(meta.get("status") or "").strip().lower() in {"completed", "completed_with_warnings"}
                else "Entrenamiento descriptivo en preparacion o ejecucion."
            )
        )
        meta["source_mix"] = self._normalize_source_mix(meta)
        meta["run_lineage"] = list(meta.get("run_lineage") or [])
        meta["comparison"] = self._compare_with_parent_run(meta) if include_details else {}
        meta["learning_focus"] = [str(item) for item in (meta.get("learning_focus") or []) if str(item).strip()]
        meta["agent_dialogue"] = list(meta.get("agent_dialogue") or []) if include_details else []
        meta["review_sections"] = self._build_training_review_sections(meta) if include_details else []
        if active_run or str(meta.get("status") or "") in {"running", "maintenance"}:
            meta.update(overlay)
        else:
            meta.setdefault("runtime_mode", "primary")
            meta.setdefault("fallback_active", False)
            meta.setdefault("fallback_backend", None)
        return meta

    def get_run_events(self, run_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        if self.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="training_run_not_found")
        return self._read_run_events(run_id, limit=limit)

    def get_run_logs(self, run_id: str, *, lines: int = 160) -> dict[str, list[str]]:
        if self.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="training_run_not_found")
        return self._read_run_logs(run_id, lines=lines)

    def _wait_for_lock_release(self, role: str, *, timeout_s: float = 45.0) -> tuple[bool, dict[str, Any]]:
        started = time.time()
        initial = False
        try:
            from .utils.locks import is_lock_held

            initial = bool(is_lock_held(self.base_dir, role))
            while time.time() - started < float(timeout_s):
                if not bool(is_lock_held(self.base_dir, role)):
                    return True, {
                        "role": role,
                        "released": True,
                        "initially_held": initial,
                        "waited_s": round(time.time() - started, 3),
                        "checked_at": _utc_ts(),
                    }
                time.sleep(0.5)
        except Exception as exc:
            return False, {
                "role": role,
                "released": False,
                "initially_held": initial,
                "waited_s": round(time.time() - started, 3),
                "checked_at": _utc_ts(),
                "error": str(exc),
            }
        return False, {
            "role": role,
            "released": False,
            "initially_held": initial,
            "waited_s": round(time.time() - started, 3),
            "checked_at": _utc_ts(),
        }

    def _collect_blocking_roles(
        self,
        *,
        roles: tuple[str, ...] = ("serve", "train", "self_patch"),
        include_primary_runtime: bool = False,
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "checked_at": _utc_ts(),
            "locks": {},
            "lock_errors": {},
            "runtime": {},
        }
        blocking_roles: list[str] = []
        try:
            from .utils.locks import is_lock_held

            for role in roles:
                try:
                    held = bool(is_lock_held(self.base_dir, role))
                except Exception as exc:
                    held = False
                    diagnostics["lock_errors"][role] = str(exc)
                diagnostics["locks"][role] = held
                if held:
                    blocking_roles.append(role)
        except Exception as exc:
            diagnostics["lock_error"] = str(exc)
        current_runtime = runtime or self.runtime_status()
        runtime_status = current_runtime.get("status") if isinstance(current_runtime.get("status"), dict) else {}
        diagnostics["runtime"] = {
            "api_ready": bool(current_runtime.get("api_ready")),
            "runtime_ready": bool(current_runtime.get("runtime_ready")),
            "model_loading": bool(runtime_status.get("model_loading")),
            "model_loaded": bool(runtime_status.get("model_loaded")),
            "model_ready": bool(runtime_status.get("model_ready")),
        }
        if include_primary_runtime and self._primary_runtime_available(current_runtime):
            blocking_roles.insert(0, "primary_runtime")
        diagnostics["blocking_roles"] = blocking_roles
        return diagnostics

    def _queue_reason_for_mode(self, mode: str, *, runtime: dict[str, Any] | None = None) -> tuple[str | None, dict[str, Any]]:
        normalized = str(mode or "").strip().lower()
        current_runtime = runtime or self.runtime_status()
        allow_parallel = self._runtime_allows_parallel_training(current_runtime)
        include_primary_runtime = normalized == "quick" and not allow_parallel
        roles = ("serve", "train", "self_patch") if include_primary_runtime else ("train", "self_patch")
        diagnostics = self._collect_blocking_roles(
            roles=roles,
            include_primary_runtime=include_primary_runtime,
            runtime=current_runtime,
        )
        blocking_roles = diagnostics.get("blocking_roles") or []
        runtime_meta = diagnostics.get("runtime") or {}
        runtime_available = bool(runtime_meta.get("api_ready") and runtime_meta.get("runtime_ready"))
        runtime_loading = bool(runtime_meta.get("model_loading")) and not runtime_available
        if include_primary_runtime and ("primary_runtime" in blocking_roles or bool((diagnostics.get("locks") or {}).get("serve"))):
            return "primary_runtime_busy", diagnostics
        if normalized == "full" and not allow_parallel and not runtime_available:
            return ("runtime_loading" if runtime_loading else "runtime_unavailable"), diagnostics
        if include_primary_runtime and not bool(runtime_meta.get("api_ready") and runtime_meta.get("runtime_ready")):
            return ("runtime_loading" if runtime_loading else "runtime_unavailable"), diagnostics
        if blocking_roles:
            return "training_resources_busy", diagnostics
        return None, diagnostics

    def _wait_for_training_resources(
        self,
        *,
        roles: tuple[str, ...] = ("train", "self_patch"),
        include_primary_runtime: bool = False,
        timeout_s: float = 45.0,
    ) -> tuple[bool, dict[str, Any]]:
        started = time.time()
        initial = self._collect_blocking_roles(
            roles=roles,
            include_primary_runtime=include_primary_runtime,
        )
        last = initial
        while time.time() - started < float(timeout_s):
            current = self._collect_blocking_roles(
                roles=roles,
                include_primary_runtime=include_primary_runtime,
            )
            last = current
            if not current.get("blocking_roles"):
                return True, {
                    **current,
                    "released": True,
                    "initially_blocking_roles": initial.get("blocking_roles") or [],
                    "waited_s": round(time.time() - started, 3),
                }
            time.sleep(0.5)
        return False, {
            **last,
            "released": False,
            "initially_blocking_roles": initial.get("blocking_roles") or [],
            "waited_s": round(time.time() - started, 3),
        }

    def _launch_training_thread(self, run_id: str, mode: str) -> None:
        self._active_run_id = run_id
        thread = threading.Thread(target=self._training_worker, args=(run_id, mode), daemon=True)
        self._training_thread = thread
        thread.start()

    def _dispatch_queued_training_runs(self) -> dict[str, Any] | None:
        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                return None
            queued_runs = [
                run
                for run in self.list_runs(include_details=False, limit=120)
                if str(run.get("status") or "") == "queued"
                and str(run.get("stage") or "") == "queued_waiting_resources"
            ]
            queued_runs.sort(key=lambda item: float(item.get("created_at") or 0.0))
            for run in queued_runs:
                run_id = str(run.get("run_id") or "").strip()
                mode = str(run.get("mode") or "quick").strip().lower()
                queue_reason, diagnostics = self._queue_reason_for_mode(mode)
                if queue_reason:
                    self._update_run_meta(
                        run_id,
                        {
                            "queue_reason": queue_reason,
                            "queue_diagnostics": diagnostics,
                            "stage": "queued_waiting_resources",
                            "lifecycle_state": "blocked",
                            "blocked_reason": queue_reason,
                            "retry_in_s": float(
                                (
                                    (self._load_autonomy_state().get("config") or {}).get("failure_backoff_base_s", 8)
                                    if isinstance(self._load_autonomy_state().get("config"), dict)
                                    else 8
                                )
                                or 8
                            ),
                            "progress_pct": _RUN_PROGRESS_BY_STAGE["queued_waiting_resources"],
                        },
                    )
                    continue
                self._update_run_meta(
                    run_id,
                    {
                        "status": "queued",
                        "stage": "queued",
                        "lifecycle_state": "planned",
                        "queue_reason": None,
                        "queue_diagnostics": diagnostics,
                        "blocked_reason": None,
                        "blocked_since": None,
                        "retry_in_s": None,
                        "progress_pct": _RUN_PROGRESS_BY_STAGE["queued"],
                    },
                )
                self._append_run_event(
                    run_id,
                    phase="queued",
                    message="queued_run_dispatched",
                    kind="queue",
                    progress_pct=_RUN_PROGRESS_BY_STAGE["queued"],
                    metadata=diagnostics,
                )
                self._launch_training_thread(run_id, mode)
                return {"run_id": run_id, "mode": mode}
        return None

    def _queue_run_waiting_resources(
        self,
        run_id: str,
        *,
        mode: str,
        queue_reason: str,
        diagnostics: dict[str, Any] | None = None,
        event_message: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_run(run_id) or {}
        autonomy = self._load_autonomy_state()
        config = autonomy.get("config") if isinstance(autonomy.get("config"), dict) else {}
        retry_in_s = float(config.get("failure_backoff_base_s", 8) or 8)
        payload = {
            "status": "queued",
            "stage": "queued_waiting_resources",
            "lifecycle_state": "blocked",
            "queue_reason": queue_reason,
            "queue_diagnostics": diagnostics or {},
            "blocked_reason": queue_reason,
            "blocked_since": existing.get("blocked_since") or _utc_ts(),
            "retry_in_s": retry_in_s,
            "progress_pct": _RUN_PROGRESS_BY_STAGE["queued_waiting_resources"],
        }
        self._update_run_meta(run_id, payload)
        self._append_run_event(
            run_id,
            phase="queued_waiting_resources",
            message=event_message or queue_reason,
            kind="queue",
            progress_pct=_RUN_PROGRESS_BY_STAGE["queued_waiting_resources"],
            metadata=(diagnostics or {}),
        )
        with self._lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
        return payload

    def _start_fallback_runtime(self, *, log_path: Path) -> dict[str, Any]:
        state = self._load_runtime_state()
        pid = state.get("fallback_pid")
        if bool(state.get("fallback_active")) and pid:
            return state
        host, port = self._api_host_port()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.base_dir / "src")
        env["C3RNT2_SERVE_LOCK_ROLE"] = "serve_fallback"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sink = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "c3rnt2",
                "serve",
                "--profile",
                self.fallback_profile,
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(self.base_dir),
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        sink.close()
        next_state = self._write_runtime_state(
            {
                "mode": "fallback_degraded",
                "fallback_active": True,
                "fallback_backend": "hf",
                "fallback_profile": self.fallback_profile,
                "fallback_pid": int(proc.pid),
            }
        )
        if not self._wait_runtime_ready(timeout_s=240.0):
            self._stop_fallback_runtime()
            raise RuntimeError("fallback_runtime_not_ready")
        return next_state

    def _stop_fallback_runtime(self) -> dict[str, Any]:
        state = self._load_runtime_state()
        pid = state.get("fallback_pid")
        if pid:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    os.kill(int(pid), 15)
            except Exception:
                pass
        return self._write_runtime_state(
            {
                "mode": "primary",
                "fallback_active": False,
                "fallback_backend": None,
                "fallback_pid": None,
            }
        )

    def _build_training_stream_payload(self) -> dict[str, Any]:
        runs = self.list_runs(include_details=False, limit=60)
        active_run = self.get_run(self._active_run_id) if self._active_run_id else None
        runtime = self.runtime_status()
        autonomy = self.autonomy_status(runtime=runtime, runs=runs[:20])
        overlay = self._runtime_overlay(active_run=active_run)
        latest_logs = self._read_run_logs(self._active_run_id, lines=25) if self._active_run_id else {}
        latest_tail: list[str] = []
        for key in ("run", "eval", "bench", "runtime", "fallback"):
            tail = latest_logs.get(key) or []
            if tail:
                latest_tail = tail[-12:]
                break
        latest_event = (active_run or {}).get("latest_event") if isinstance(active_run, dict) else None
        pipeline_runs = [
            run for run in runs
            if str(run.get("lifecycle_state") or "").strip().lower() in {"planned", "blocked", "training", "evaluating", "applying", "verifying"}
        ][:12]
        blocked_runs = [
            run for run in runs
            if str(run.get("lifecycle_state") or "").strip().lower() == "blocked"
        ][:8]
        return {
            "ts": _utc_ts(),
            "active_run_id": self._active_run_id,
            "active_run": active_run,
            "phase": (active_run or {}).get("stage"),
            "progress_pct": (active_run or {}).get("progress_pct"),
            "execution_progress_pct": (active_run or {}).get("execution_progress_pct"),
            "pipeline_progress_pct": (active_run or {}).get("pipeline_progress_pct"),
            "latest_metrics": (active_run or {}).get("latest_metrics") or {},
            "runtime_mode": overlay.get("runtime_mode"),
            "fallback_active": overlay.get("fallback_active"),
            "fallback_backend": overlay.get("fallback_backend"),
            "last_event": latest_event,
            "log_tail": latest_tail,
            "campaign": autonomy.get("campaign"),
            "next_run_scheduled_at": autonomy.get("next_run_scheduled_at"),
            "scheduled_followup_reason": autonomy.get("scheduled_followup_reason"),
            "pipeline_runs": pipeline_runs,
            "blocked_runs": blocked_runs,
            "runs": runs,
        }

    def _resolve_dataset_hash(self) -> str:
        candidates = [
            self.base_dir / "data" / "registry" / "hf_train",
            self.base_dir / "data" / "episodes",
            self.base_dir / "data" / "corpora" / "programming",
            self.base_dir / "data" / "corpora" / "cybersecurity",
            self.base_dir / "data" / "local_lab" / "lessons",
        ]
        return _iter_hash_files(candidates)

    def _resolve_instruction_meta(self) -> dict[str, Any]:
        try:
            bundle = load_instruction_bundle({}, base_dir=self.base_dir)
        except Exception as exc:
            return {"digest": None, "sources": [], "error": str(exc)}

        if not bundle.get("sources") or bundle.get("sources", [{}])[0].get("kind") == "inline_fallback":
            fallback_sources = []
            for name in ("vortex_system.md", "domain_policy.md", "operator_notes.md"):
                candidate = self.base_dir / "config" / "instructions" / name
                if not candidate.exists():
                    continue
                text = candidate.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                fallback_sources.append(
                    {
                        "kind": name.replace(".md", ""),
                        "path": str(candidate),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
            if fallback_sources:
                joined = "\n\n".join(
                    Path(item["path"]).read_text(encoding="utf-8").strip()
                    for item in fallback_sources
                ).strip()
                bundle = {
                    "text": joined,
                    "digest": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
                    "sources": fallback_sources,
                }

        text = str(bundle.get("text") or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        return {
            "digest": digest,
            "sources": [str(item) for item in (bundle.get("sources") or [])],
        }

    def _default_dataset_mix(self) -> dict[str, float]:
        return {
            "chat_feedback": 0.24,
            "chat_feedback_soft": 0.12,
            "episode": 0.17,
            "repo": 0.14,
            "docs": 0.10,
            "self_edit": 0.09,
            "autonomy_reflection": 0.08,
            "failure_repair": 0.06,
        }

    def _summarize_run_result(self, run: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
        if not isinstance(run, dict):
            return None
        result = run.get(key)
        if not isinstance(result, dict) or not result:
            return None
        summary = result.get("summary") or result.get("detail") or result.get("message")
        return {
            "run_id": run.get("run_id"),
            "mode": run.get("mode"),
            "status": run.get("status"),
            "stage": run.get("stage"),
            "ok": result.get("ok"),
            "summary": summary,
            "exit_code": run.get(f"{key.replace('_result', '')}_exit_code"),
            "updated_at": run.get("updated_at") or run.get("created_at"),
        }

    def _next_autonomy_tasks(self, autonomy: dict[str, Any], now: float) -> tuple[list[str], float | None]:
        config = autonomy.get("config") if isinstance(autonomy.get("config"), dict) else {}
        scheduled: list[tuple[float, str]] = []
        if bool(config.get("reflection_enabled", True)):
            scheduled.append(
                (
                    float(autonomy.get("last_reflection_at") or 0.0) + float(config.get("reflection_interval_s", 300)),
                    "reflection",
                )
            )
        if bool(config.get("training_enabled", True)):
            scheduled.append(
                (
                    float(autonomy.get("last_train_at") or 0.0) + float(config.get("quick_train_interval_s", 1200)),
                    "quick_learning",
                )
            )
            scheduled.append(
                (
                    float(autonomy.get("last_train_at") or 0.0) + float(config.get("full_train_interval_s", 7200)),
                    "full_training",
                )
            )
        if bool(config.get("autoedit_enabled", True)):
            scheduled.append(
                (
                    float(autonomy.get("last_patch_at") or 0.0) + float(config.get("autoedit_interval_s", 1800)),
                    "autoedit",
                )
            )
        ordered = sorted(scheduled, key=lambda item: item[0])
        queue = [label for due_ts, label in ordered if due_ts <= now]
        if not queue:
            queue = [label for _due_ts, label in ordered[:3]]
        next_cycle_at = ordered[0][0] if ordered else None
        return queue[:4], next_cycle_at

    def _enrich_autonomy_state(
        self,
        autonomy: dict[str, Any],
        *,
        runtime: dict[str, Any] | None = None,
        runs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        enriched = dict(autonomy)
        runtime = runtime or self.runtime_status()
        runs = runs if runs is not None else self.list_runs(include_details=False, limit=12)
        active_run = self.get_run(self._active_run_id) if self._active_run_id else None
        latest_run = active_run or (runs[0] if runs else None)
        latest_eval_run = next((run for run in runs if isinstance(run.get("eval_result"), dict) and run.get("eval_result")), None)
        latest_bench_run = next((run for run in runs if isinstance(run.get("bench_result"), dict) and run.get("bench_result")), None)
        now = _utc_ts()
        training_queue, next_cycle_at = self._next_autonomy_tasks(enriched, now)
        campaign_summary = self._campaign_summary(runs=runs, autonomy=enriched)
        blocked_runs = [
            run for run in runs
            if str(run.get("lifecycle_state") or "").strip().lower() == "blocked"
        ]
        scheduled_at = enriched.get("next_run_scheduled_at")
        if scheduled_at:
            try:
                next_cycle_at = min(float(next_cycle_at), float(scheduled_at)) if next_cycle_at else float(scheduled_at)
            except Exception:
                pass
        if active_run:
            active_mode = str(active_run.get("mode") or "training").strip()
            active_stage = str(active_run.get("stage") or "queued").strip()
            training_queue = [f"{active_mode}:{active_stage}", *training_queue]
        elif enriched.get("scheduled_run_mode"):
            scheduled_mode = str(enriched.get("scheduled_run_mode") or "").strip()
            if scheduled_mode:
                training_queue = [f"{scheduled_mode}:scheduled", *training_queue]
        stage = str((active_run or {}).get("stage") or "")
        active_run_runtime_mode = str((active_run or {}).get("runtime_mode") or "").strip().lower()
        runtime_drained = bool(
            active_run
            and active_run_runtime_mode == "maintenance"
            and stage in {"draining_primary", "training", "eval", "resume_primary", "bench"}
            and not bool(runtime.get("api_ready"))
        )
        maintenance_mode = bool(
            runtime_drained
            or (active_run and active_run_runtime_mode == "maintenance")
            or str(enriched.get("state") or "") in {"training", "autoediting", "rollback", "restarting"}
        )
        enriched["training_queue"] = training_queue[:5]
        enriched["next_cycle_at"] = next_cycle_at
        enriched["maintenance_mode"] = maintenance_mode
        enriched["runtime_drained_for_training"] = runtime_drained
        enriched["current_dataset_mix"] = dict(
            (active_run or latest_run or {}).get("dataset_mix") or self._default_dataset_mix()
        )
        enriched["campaign"] = campaign_summary
        enriched["blocked_run_count"] = len(blocked_runs)
        enriched["blocked_runs"] = blocked_runs[:6]
        enriched["next_run_scheduled_at"] = enriched.get("next_run_scheduled_at")
        enriched["scheduled_run_mode"] = enriched.get("scheduled_run_mode")
        enriched["scheduled_parent_run_id"] = enriched.get("scheduled_parent_run_id")
        enriched["scheduled_followup_reason"] = enriched.get("scheduled_followup_reason")
        enriched["last_eval_summary"] = self._summarize_run_result(latest_eval_run, "eval_result")
        enriched["last_bench_summary"] = self._summarize_run_result(latest_bench_run, "bench_result")
        enriched["latest_dialogue"] = list(
            (active_run or latest_run or {}).get("agent_dialogue")
            or enriched.get("latest_dialogue")
            or []
        )[:12]
        enriched["last_training_outcome"] = {
            "run_id": (latest_run or {}).get("run_id"),
            "mode": (latest_run or {}).get("mode"),
            "status": (latest_run or {}).get("status"),
            "stage": (latest_run or {}).get("stage"),
            "updated_at": (latest_run or {}).get("updated_at") or (latest_run or {}).get("created_at"),
            "reason": (
                (latest_run or {}).get("terminal_reason")
                or (latest_run or {}).get("runtime_resume_error")
                or (latest_run or {}).get("error")
                or ((latest_run or {}).get("promotion") or {}).get("decision")
            ),
        } if latest_run else None
        return enriched

    def docker_status(self) -> dict[str, Any]:
        if self.assume_docker_ready:
            return {
                "ready": True,
                "reason": "docker_managed_externally",
                "server_version": None,
            }
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .ServerVersion}}"],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=4.0,
                check=False,
            )
        except FileNotFoundError:
            return {"ready": False, "reason": "docker_not_installed"}
        except Exception as exc:
            return {"ready": False, "reason": f"docker_unavailable:{exc}"}
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return {"ready": False, "reason": "docker_unavailable", "detail": detail}
        version = (result.stdout or "").strip().strip('"')
        return {"ready": True, "reason": "docker_ready", "server_version": version or None}

    def model_status(self, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        cache_dir = resolve_cache_dir(self.base_dir / "data" / "models" / "hf-cache")
        current_runtime = runtime or self.runtime_status()
        return model_cache_status(self._served_model_id(current_runtime), cache_dir)

    def runtime_status(self) -> dict[str, Any]:
        ready = _http_json(f"{self.api_url}/readyz", timeout=1.5)
        status = _http_json(f"{self.api_url}/v1/status", timeout=1.5)
        runtime_models = _http_json(f"{self.runtime_url}/v1/models", timeout=1.5)
        status_payload = status if isinstance(status, dict) else {}
        engine_kind = str((status_payload.get("engine_kind")) or "").strip().lower()
        model_ready = bool(status_payload.get("model_ready"))
        model_loaded = bool(status_payload.get("model_loaded"))
        model_loading = bool(status_payload.get("model_loading"))
        active_model = str(status_payload.get("active_model") or "").strip()
        api_ready = bool(ready and ready.get("ok"))
        if not api_ready and status_payload:
            api_ready = bool(
                status_payload.get("chat_ready")
                and status_payload.get("engine_ready")
                and model_ready
                and model_loaded
            )
        runtime_ready = api_ready
        if engine_kind == "hf":
            if active_model:
                runtime_models = {"data": [{"id": active_model}], "source": "api_status"}
            runtime_ready = bool(
                status_payload.get("engine_ready")
                and model_ready
                and model_loaded
                and not model_loading
            ) or api_ready
        if not runtime_ready and engine_kind in {"", "vllm", "external"}:
            runtime_ready = runtime_models is not None or bool(
                status_payload.get("engine_ready") and model_ready
            )
        if not runtime_ready and engine_kind not in {"", "vllm", "external"}:
            runtime_ready = bool(
                status_payload.get("engine_ready")
                and model_ready
                and model_loaded
                and not model_loading
            )
            if runtime_ready and active_model and runtime_models is None:
                runtime_models = {"data": [{"id": active_model}], "source": "api_status"}
        return {
            "api_ready": api_ready,
            "readyz": ready,
            "status": status,
            "runtime_ready": runtime_ready,
            "runtime_models": runtime_models,
        }

    def _runtime_allows_parallel_training(self, runtime: dict[str, Any] | None = None) -> bool:
        training_settings = self._load_profile_settings(self.training_profile)
        training_server = training_settings.get("server", {}) if isinstance(training_settings, dict) else {}
        if bool((training_server or {}).get("allow_parallel_runtime_training", False)):
            return True
        api_settings = self._load_profile_settings(self.api_profile)
        api_server = api_settings.get("server", {}) if isinstance(api_settings, dict) else {}
        if bool((api_server or {}).get("allow_parallel_runtime_training", False)):
            return True
        current = runtime or self.runtime_status()
        status = current.get("status") if isinstance(current.get("status"), dict) else {}
        if isinstance(status, dict):
            if str(status.get("chat_mode") or "").strip().lower() == "fallback_degraded":
                return True
            backend = str(status.get("active_backend") or status.get("backend") or "").strip().lower()
            if backend == "hf":
                runtime_models = current.get("runtime_models")
                if isinstance(runtime_models, dict):
                    data = runtime_models.get("data")
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and str(item.get("device") or "").strip().lower() == "cpu":
                                return True
        return False

    def _primary_runtime_available(self, runtime: dict[str, Any] | None = None) -> bool:
        current = runtime or self.runtime_status()
        return bool(current.get("api_ready") and current.get("runtime_ready"))

    def _compose_runtime_up(
        self,
        *,
        log_path: Path,
        no_build: bool = False,
        force_recreate: bool = False,
    ) -> tuple[int, str]:
        args = ["up", "-d"]
        if no_build:
            args.append("--no-build")
        if force_recreate:
            args.append("--force-recreate")
        args.extend(self._runtime_compose_services())
        return self._run_compose(args, log_path=log_path)

    def _runtime_compose_services(self) -> list[str]:
        settings = self._load_profile_settings(self.api_profile)
        docker_cfg = settings.get("docker", {}) if isinstance(settings, dict) else {}
        runtime_service = str(docker_cfg.get("runtime_service") or "").strip()
        api_service = str(docker_cfg.get("api_service") or "vortex-api").strip() or "vortex-api"
        services: list[str] = []
        if runtime_service:
            services.append(runtime_service)
        if api_service not in services:
            services.append(api_service)
        return services

    def _compose_local_images_available(self) -> bool:
        for service in ("model-init", "vortex-api", "trainer", "eval"):
            try:
                result = subprocess.run(
                    self._compose_cmd("images", "-q", service),
                    cwd=str(self.base_dir),
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                    check=False,
                    env=self._compose_env(),
                )
            except Exception:
                return False
            if result.returncode != 0:
                return False
            if not str(result.stdout or "").strip():
                return False
        return True

    def _learning_queue_summary(self, *, preview: int = 5) -> dict[str, Any]:
        queue_state = self._load_learning_queue_state()
        queued_items = self._list_learning_queue(include_consumed=False)
        preview_items = []
        for item in queued_items[:preview]:
            preview_items.append(
                {
                    "id": item.get("id"),
                    "request_id": item.get("request_id"),
                    "source_kind": item.get("source_kind"),
                    "score": item.get("score"),
                    "status": item.get("status"),
                    "queued_at": item.get("ts"),
                    "consumed_by": item.get("consumed_by"),
                }
            )
        return {
            "queued_count": len(queued_items),
            "quick_threshold": int(queue_state.get("quick_threshold") or DEFAULT_QUICK_QUEUE_THRESHOLD),
            "quick_cooldown_s": int(queue_state.get("quick_cooldown_s") or DEFAULT_QUICK_QUEUE_COOLDOWN_S),
            "last_quick_dispatch_at": queue_state.get("last_quick_dispatch_at"),
            "last_quick_dispatch_run_id": queue_state.get("last_quick_dispatch_run_id"),
            "items": preview_items,
        }

    def frontend_status(self) -> dict[str, Any]:
        host, port, url = _parse_url_target(
            self.frontend_url,
            default_host="127.0.0.1",
            default_port=self.frontend_port,
        )
        return {
            "ready": _port_open(host, port),
            "port": port,
            "url": url,
        }

    def get_allowlist(self) -> list[str]:
        payload = self._read_state_record("internet", {"domains": []}, self.internet_settings_path)
        raw = payload.get("domains", []) if isinstance(payload, dict) else []
        items = []
        for item in raw if isinstance(raw, list) else []:
            text = str(item or "").strip().lower()
            if text:
                items.append(text)
        return sorted(set(items))

    def set_allowlist(self, domains: list[str]) -> list[str]:
        cleaned = sorted(
            {
                str(item or "").strip().lower()
                for item in domains
                if str(item or "").strip()
            }
        )
        self._write_state_record("internet", {"domains": cleaned, "updated_at": _utc_ts()}, self.internet_settings_path)
        return cleaned

    def _default_autonomy_state(self) -> dict[str, Any]:
        now = _utc_ts()
        return {
            "enabled": True,
            "boot_mode": "always_on",
            "state": "waiting_resources",
            "active_agents": [
                {
                    "id": "analyst",
                    "name": "Analista",
                    "role": "reflection",
                    "status": "waiting",
                    "accent": "ask",
                    "last_event_at": now,
                },
                {
                    "id": "builder",
                    "name": "Constructor",
                    "role": "execution",
                    "status": "waiting",
                    "accent": "agent",
                    "last_event_at": now,
                },
            ],
            "current_cycle": None,
            "last_reflection_at": None,
            "last_train_at": None,
            "last_patch_at": None,
            "autoedit_scope": "repo_versioned",
            "last_rollback": None,
            "training_queue": [],
            "next_cycle_at": None,
            "next_run_scheduled_at": None,
            "scheduled_run_mode": None,
            "scheduled_parent_run_id": None,
            "scheduled_followup_reason": None,
            "active_campaign_id": None,
            "current_campaign": None,
            "maintenance_mode": False,
            "runtime_drained_for_training": False,
            "current_dataset_mix": self._default_dataset_mix(),
            "last_eval_summary": None,
            "last_bench_summary": None,
            "last_training_outcome": None,
            "latest_dialogue": [],
            "config": {
                "reflection_enabled": True,
                "training_enabled": True,
                "autoedit_enabled": True,
                "multi_agent_dialogue_enabled": True,
                "descriptive_reports_enabled": True,
                "live_autoedit_enabled": True,
                "reflection_interval_s": 20,
                "quick_train_interval_s": 45,
                "full_train_interval_s": 300,
                "autoedit_interval_s": 120,
                "chain_run_delay_s": 6,
                "failure_backoff_base_s": 8,
                "failure_backoff_max_s": 90,
            },
            "latest_events": [],
            "updated_at": now,
        }

    def _load_autonomy_state(self) -> dict[str, Any]:
        current = self._read_state_record("autonomy", self._default_autonomy_state(), self.autonomy_state_path)
        merged = self._default_autonomy_state()
        if isinstance(current, dict):
            merged.update(current)
            merged["config"] = {
                **self._default_autonomy_state()["config"],
                **(current.get("config") if isinstance(current.get("config"), dict) else {}),
            }
            merged["active_agents"] = current.get("active_agents") or merged["active_agents"]
            merged["latest_events"] = current.get("latest_events") or []
            merged["latest_dialogue"] = current.get("latest_dialogue") or []
        return merged

    def _write_autonomy_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self._load_autonomy_state()
        state.update(payload)
        if "config" in payload and isinstance(payload["config"], dict):
            state["config"] = {**state.get("config", {}), **payload["config"]}
        if "active_agents" in payload:
            state["active_agents"] = payload["active_agents"]
        if "latest_events" in payload:
            state["latest_events"] = payload["latest_events"][:80]
        if "latest_dialogue" in payload:
            state["latest_dialogue"] = payload["latest_dialogue"][:24]
        state["updated_at"] = _utc_ts()
        return self._write_state_record("autonomy", state, self.autonomy_state_path)

    def _latest_autonomy_events(self, limit: int = 24) -> list[dict[str, Any]]:
        stored = self.storage.list_events("autonomy", "global", limit=limit, reverse=True)
        ordered = sorted(
            [event for event in stored if isinstance(event, dict)],
            key=lambda item: float(item.get("ts") or 0.0),
            reverse=True,
        )
        return _dedupe_event_items(ordered, limit=limit)

    def _append_autonomy_event(
        self,
        *,
        agent: str,
        kind: str,
        title: str,
        detail: str,
        cycle_id: str | None = None,
        state_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "ts": _utc_ts(),
            "agent": agent,
            "kind": kind,
            "title": title,
            "detail": detail,
            "cycle_id": cycle_id,
            "state": state_name,
            "metadata": metadata or {},
        }
        self.storage.append_event("autonomy", "global", payload)
        events = _dedupe_event_items([payload, *self._latest_autonomy_events(limit=23)], limit=24)
        self._write_autonomy_state({"latest_events": events})
        return payload

    def _next_training_sequence(self) -> int:
        highest = 0
        for payload in self.storage.list_runs():
            if not isinstance(payload, dict):
                continue
            try:
                highest = max(highest, int(payload.get("sequence_id") or 0))
            except Exception:
                continue
        return highest + 1

    def _latest_incomplete_run(self) -> dict[str, Any] | None:
        for run in self.list_runs(include_details=False, limit=120):
            stage = str(run.get("stage") or "").strip().lower()
            status = str(run.get("status") or "").strip().lower()
            if stage in _RUN_ACTIVE_STAGES or status in {"queued", "running", "maintenance"}:
                return run
        return None

    def _build_dialogue_focus(self, mode: str, source: str | None) -> dict[str, Any]:
        runtime = self.runtime_status()
        learning_queue = self._learning_queue_summary(preview=4)
        latest_run = self.list_runs(include_details=False, limit=1)
        latest_failure = (latest_run[0].get("failure") if latest_run else None) or {}
        queued_count = int(learning_queue.get("queued_count") or 0)
        source_label = str(source or "manual").strip().lower()

        if not bool(runtime.get("api_ready")):
            problem = (
                "El runtime principal no responde; hay que recuperar servicio y aprender del ultimo corte."
            )
            objective = (
                "documentar el bloqueo del runtime, mantener aprendizaje local y preparar un entrenamiento recuperable"
            )
            focus = ["runtime", "recuperacion", "aprendizaje_local"]
        elif latest_failure:
            reason = str(latest_failure.get("reason") or "fallo_reciente").strip()
            problem = f"El ultimo entrenamiento termino con un fallo reciente: {reason}."
            objective = "extraer una leccion util del fallo y convertirla en un ciclo descriptivo verificable"
            focus = ["failure_repair", "descriptive_training", "validation"]
        elif queued_count > 0:
            problem = f"Hay {queued_count} muestras en cola esperando convertirse en mejora entrenable."
            objective = "agrupar muestras recientes y consolidarlas en un entrenamiento descriptivo"
            focus = ["chat_feedback", "queue_curation", "continuous_learning"]
        elif mode == "full":
            problem = "No hay bloqueo inmediato; toca consolidar lo aprendido en un entrenamiento profundo."
            objective = "consolidar conocimientos recientes y dejar una revision completa para inspeccion humana"
            focus = ["consolidation", "bench_eval", "descriptive_review"]
        else:
            problem = "No hay incidencias criticas; toca reforzar aprendizaje incremental de forma visible."
            objective = "reforzar conocimiento reciente y explicar con claridad que cambia y por que"
            focus = ["incremental_learning", "visibility", "self_reflection"]

        reason = {
            "manual": "El operador ha pedido un entrenamiento visible y descriptivo.",
            "feedback_queue": "La cola de aprendizaje ha pedido un refuerzo incremental.",
            "autonomy": "La autonomia ha detectado que ya toca ejecutar el siguiente ciclo.",
            "autonomy_quick": "La autonomia ha detectado un ciclo corto pendiente.",
            "autonomy_full": "La autonomia ha detectado un ciclo profundo pendiente.",
        }.get(source_label, f"Origen del ciclo: {source_label or 'manual'}.")

        return {
            "problem": problem,
            "objective": objective,
            "reason": reason,
            "focus": focus,
            "runtime_ready": bool(runtime.get("api_ready") and runtime.get("runtime_ready")),
            "queued_count": queued_count,
        }

    def _build_agent_dialogue(
        self,
        *,
        run_id: str,
        sequence_id: int,
        mode: str,
        source: str | None,
        cycle_id: str | None = None,
    ) -> list[dict[str, Any]]:
        focus = self._build_dialogue_focus(mode, source)
        turns = [
            {
                "id": f"{run_id}-analyst-1",
                "speaker": "analyst",
                "speaker_label": "Analista",
                "kind": "question",
                "ts": _utc_ts(),
                "message": (
                    f"Entrenamiento {sequence_id}: detecto este foco principal -> {focus['problem']} "
                    "Voy a abrir una hipotesis y un criterio de exito antes de consumir recursos."
                ),
                "cycle_id": cycle_id,
            },
            {
                "id": f"{run_id}-builder-1",
                "speaker": "builder",
                "speaker_label": "Constructor",
                "kind": "answer",
                "ts": _utc_ts(),
                "message": (
                    f"Plan inicial: {focus['objective']}. Motivo del arranque: {focus['reason']}. "
                    "Ire completando dataset, gates y verificacion por fases."
                ),
                "cycle_id": cycle_id,
            },
        ]
        return turns

    def _build_initial_notebook_sections(
        self,
        *,
        focus: dict[str, Any],
        run_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        now = _utc_ts()
        campaign_id = str(run_context.get("campaign_id") or "").strip() or "manual"
        sections = [
            {
                "id": f"note-{uuid.uuid4().hex[:8]}",
                "phase": "planned",
                "kind": "hypothesis",
                "title": "Hipotesis inicial",
                "content": str(focus.get("problem") or "Sin problema explicitado."),
                "ts": now,
                "metadata": {
                    "campaign_id": campaign_id,
                    "focus": list(focus.get("focus") or []),
                },
            },
            {
                "id": f"note-{uuid.uuid4().hex[:8]}",
                "phase": "planned",
                "kind": "goal",
                "title": "Objetivo y criterio de exito",
                "content": (
                    f"Objetivo: {focus.get('objective') or 'sin objetivo'}. "
                    "Se considerara valido si supera gates de eval, bench, smoke y seguridad de repo."
                ),
                "ts": now,
                "metadata": {
                    "campaign_id": campaign_id,
                    "parent_run_id": run_context.get("parent_run_id"),
                    "attempt": run_context.get("attempt"),
                },
            },
        ]
        return sections

    def _build_training_review_sections(self, meta: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = str(meta.get("run_id") or "").strip()
        dataset_manifest = meta.get("dataset_manifest") if isinstance(meta.get("dataset_manifest"), dict) else {}
        failure = meta.get("failure") if isinstance(meta.get("failure"), dict) else {}
        promotion = meta.get("promotion") if isinstance(meta.get("promotion"), dict) else {}
        events = meta.get("events") if isinstance(meta.get("events"), list) else []
        latest_metrics = meta.get("latest_metrics") if isinstance(meta.get("latest_metrics"), dict) else {}
        focus_items = [str(item) for item in (meta.get("learning_focus") or []) if str(item).strip()]
        dialogue = meta.get("agent_dialogue") if isinstance(meta.get("agent_dialogue"), list) else []
        artifact_labels = sorted(str(key) for key in (meta.get("artifacts") or {}).keys())
        notebook = meta.get("notebook_sections") if isinstance(meta.get("notebook_sections"), list) else []
        gate_results = meta.get("gate_results") if isinstance(meta.get("gate_results"), dict) else {}
        apply_result = meta.get("apply_result") if isinstance(meta.get("apply_result"), dict) else {}
        rollback_result = meta.get("rollback_result") if isinstance(meta.get("rollback_result"), dict) else {}
        comparison = meta.get("comparison") if isinstance(meta.get("comparison"), dict) else {}

        queue_summary = (
            f"muestras en cola: {int(dataset_manifest.get('queued_count') or 0)} | "
            f"consumidas: {int(dataset_manifest.get('consumed_count') or 0)}"
        )
        outcome = (
            f"estado={meta.get('status') or 'unknown'} | etapa={meta.get('stage') or 'unknown'} | "
            f"decision={promotion.get('decision') or 'pending'}"
        )
        if failure:
            outcome = f"{outcome} | fallo={failure.get('reason') or 'unknown'}"

        sections = [
            {
                "key": "objective",
                "title": "Objetivo del entrenamiento",
                "content": str(
                    meta.get("objective")
                    or meta.get("display_description")
                    or "Entrenamiento descriptivo sin objetivo registrado."
                ),
            },
            {
                "key": "focus",
                "title": "Foco de aprendizaje",
                "content": ", ".join(focus_items) if focus_items else "Sin focos explicitos; se usaran eventos, dataset y estado del runtime.",
            },
            {
                "key": "dataset",
                "title": "Dataset y fuentes",
                "content": (
                    f"{queue_summary}\n"
                    f"source_mix={json.dumps(self._normalize_source_mix(meta), ensure_ascii=True)}"
                ),
            },
            {
                "key": "notebook",
                "title": "Libreta de aprendizaje",
                "content": "\n".join(
                    f"[{entry.get('phase')}] {entry.get('title')}: {entry.get('content')}"
                    for entry in notebook[-8:]
                ) or "No hay notas de aprendizaje registradas todavia.",
            },
            {
                "key": "dialogue",
                "title": "Dialogo multiagente",
                "content": "\n".join(
                    f"{turn.get('speaker_label') or turn.get('speaker')}: {turn.get('message')}"
                    for turn in dialogue[-8:]
                ) or "No hay dialogo registrado para este run.",
            },
            {
                "key": "gates",
                "title": "Gates y aplicacion",
                "content": (
                    f"gate_results={json.dumps(gate_results, ensure_ascii=True)}\n"
                    f"apply_result={json.dumps(apply_result, ensure_ascii=True)}\n"
                    f"rollback_result={json.dumps(rollback_result, ensure_ascii=True)}"
                ),
            },
            {
                "key": "outcome",
                "title": "Resultado del ciclo",
                "content": outcome,
            },
            {
                "key": "comparison",
                "title": "Comparacion con el run anterior",
                "content": json.dumps(comparison, ensure_ascii=True) if comparison else "No hay un run padre comparable dentro de esta campana.",
            },
            {
                "key": "signals",
                "title": "Senales capturadas",
                "content": (
                    f"eventos={len(events)} | metricas={', '.join(sorted(latest_metrics.keys())) or 'ninguna'} | "
                    f"artefactos={', '.join(artifact_labels) if artifact_labels else 'ninguno'} | lifecycle={meta.get('lifecycle_state')} | run_id={run_id}"
                ),
            },
        ]
        return sections

    def _run_display_name(self, meta: dict[str, Any]) -> str:
        try:
            sequence_id = int(meta.get("sequence_id") or 0)
        except Exception:
            sequence_id = 0
        if sequence_id > 0:
            return f"Entrenamiento {sequence_id}"
        run_id = str(meta.get("run_id") or "run").strip() or "run"
        return f"Entrenamiento {run_id[-6:]}"

    def _agent_state(self, autonomy: dict[str, Any], *, analyst: str, builder: str) -> list[dict[str, Any]]:
        now = _utc_ts()
        return [
            {
                "id": "analyst",
                "name": "Analista",
                "role": "reflection",
                "status": analyst,
                "accent": "ask",
                "last_event_at": autonomy.get("last_reflection_at") or now,
            },
            {
                "id": "builder",
                "name": "Constructor",
                "role": "execution",
                "status": builder,
                "accent": "agent",
                "last_event_at": max(
                    float(autonomy.get("last_train_at") or 0.0),
                    float(autonomy.get("last_patch_at") or 0.0),
                    now,
                ),
            },
        ]

    def _ensure_autonomy_worker(self) -> None:
        with self._lock:
            if self._autonomy_thread and self._autonomy_thread.is_alive():
                return
            self._autonomy_stop.clear()
            self._autonomy_thread = threading.Thread(target=self._autonomy_worker, daemon=True)
            self._autonomy_thread.start()

    def autonomy_status(
        self,
        *,
        runtime: dict[str, Any] | None = None,
        runs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self._load_autonomy_state()
        state["latest_events"] = state.get("latest_events") or self._latest_autonomy_events(limit=20)
        return self._enrich_autonomy_state(state, runtime=runtime, runs=runs)

    def start_autonomy(self) -> dict[str, Any]:
        state = self._write_autonomy_state({"enabled": True, "state": "waiting_resources"})
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_start",
            title="Autonomia activada",
            detail="El bucle continuo vuelve a vigilar runtime, aprendizaje, entrenamiento y autoedicion.",
            state_name=state.get("state"),
        )
        self._ensure_autonomy_worker()
        return {"ok": True, "enabled": True, "autonomy": self.autonomy_status()}

    def stop_autonomy(self) -> dict[str, Any]:
        state = self._write_autonomy_state({"enabled": False, "state": "paused"})
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_stop",
            title="Autonomia en pausa",
            detail="Se ha detenido la reflexion continua y no se lanzaran nuevos ciclos autonomos hasta reactivarla.",
            state_name=state.get("state"),
        )
        return {"ok": True, "enabled": False, "autonomy": self.autonomy_status()}

    def configure_autonomy(self, payload: AutonomyConfigRequest) -> dict[str, Any]:
        patch: dict[str, Any] = {"config": {}}
        if payload.enabled is not None:
            patch["enabled"] = bool(payload.enabled)
            patch["state"] = "waiting_resources" if payload.enabled else "paused"
        if payload.reflection_enabled is not None:
            patch["config"]["reflection_enabled"] = bool(payload.reflection_enabled)
        if payload.training_enabled is not None:
            patch["config"]["training_enabled"] = bool(payload.training_enabled)
        if payload.autoedit_enabled is not None:
            patch["config"]["autoedit_enabled"] = bool(payload.autoedit_enabled)
        if payload.multi_agent_dialogue_enabled is not None:
            patch["config"]["multi_agent_dialogue_enabled"] = bool(payload.multi_agent_dialogue_enabled)
        if payload.descriptive_reports_enabled is not None:
            patch["config"]["descriptive_reports_enabled"] = bool(payload.descriptive_reports_enabled)
        if payload.live_autoedit_enabled is not None:
            patch["config"]["live_autoedit_enabled"] = bool(payload.live_autoedit_enabled)
        state = self._write_autonomy_state(patch)
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_config",
            title="Configuracion de autonomia actualizada",
            detail="Se han aplicado nuevos interruptores para reflexion, entrenamiento o autoedicion.",
            state_name=state.get("state"),
            metadata=patch["config"],
        )
        if state.get("enabled"):
            self._ensure_autonomy_worker()
        return {"ok": True, "autonomy": self.autonomy_status()}

    def reset_training_state(
        self,
        *,
        clear_runs: bool = True,
        clear_learning_queue: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                raise HTTPException(status_code=409, detail="training_reset_blocked_active_run")

            removed_runs = 0
            if clear_runs:
                removed_runs = len(self.storage.list_runs())
                if self.runs_dir.exists():
                    for path in list(self.runs_dir.iterdir()):
                        try:
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink(missing_ok=True)
                            removed_runs += 1
                        except Exception:
                            continue
                self.storage.delete_all_runs()

            if clear_learning_queue:
                for path in (self.learning_queue_path, self.learning_queue_state_path):
                    if path.exists():
                        path.unlink(missing_ok=True)
                self.storage.clear_events("learning_queue")
                self._write_state_record("learning_queue", self._default_learning_queue_state(), self.learning_queue_state_path)

            if self.autonomy_events_path.exists():
                self.autonomy_events_path.unlink(missing_ok=True)
            self.storage.clear_events("autonomy")

            fresh_state = self._default_autonomy_state()
            fresh_state["enabled"] = True
            fresh_state["state"] = "waiting_resources"
            fresh_state["current_cycle"] = None
            self._write_state_record("autonomy", fresh_state, self.autonomy_state_path)
            self._write_state_record(
                "runtime",
                {"mode": "primary", "fallback_active": False, "fallback_backend": None, "fallback_pid": None},
                self.runtime_state_path,
            )
            self._active_run_id = None

        self._append_autonomy_event(
            agent="system",
            kind="training_reset",
            title="Aprendizaje reiniciado desde cero",
            detail="Se ha limpiado todo el historial de entrenamiento anterior y la autonomia continua vuelve a arrancar en modo descriptivo y sin pausas.",
            state_name="waiting_resources",
            metadata={
                "removed_runs": removed_runs,
                "clear_runs": bool(clear_runs),
                "clear_learning_queue": bool(clear_learning_queue),
            },
        )
        self._ensure_autonomy_worker()
        return {
            "ok": True,
            "removed_runs": removed_runs,
            "autonomy": self.autonomy_status(runs=self.list_runs(include_details=False, limit=12)),
            "runs": self.list_runs(include_details=False, limit=120),
        }

    def _git_clean(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0 and not (result.stdout or "").strip()

    def _git_head(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        head = (result.stdout or "").strip()
        return head or None

    def _git_tag_snapshot(self, name: str) -> str | None:
        try:
            subprocess.run(
                ["git", "tag", "-f", name],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return None
        return name

    def _autonomy_settings(self) -> dict[str, Any]:
        settings = load_settings(
            profile=self.api_profile,
            settings_path=self.base_dir / "config" / "settings.yaml",
        )
        settings = json.loads(json.dumps(settings))
        settings.setdefault("continuous", {})["ingest_web"] = False
        settings.setdefault("autopilot", {})
        settings["autopilot"].update(
            {
                "enabled": True,
                "reuse_dataset": True,
                "autopatch_enabled": True,
                "autopatch_require_approval": False,
                "autopatch_goal": "continuous self-improvement",
                "autopatch_on_test_fail": True,
                "autopatch_on_doctor_fail": True,
                "autopatch_require_eval": True,
                "train_cooldown_minutes": 20,
                "patch_cooldown_minutes": 30,
                "eval_cooldown_minutes": 20,
                "safe_mode_cooldown_minutes": 30,
            }
        )
        settings.setdefault("self_patch", {})
        allowed_paths = [
            "c3_rnt2_ai/",
            "vortex-chat/",
            "scripts/",
            "docs/",
            "README.md",
            "docker-compose.yml",
        ]
        settings["self_patch"]["enabled"] = True
        settings["self_patch"]["allowed_paths"] = allowed_paths
        forbidden = list(settings["self_patch"].get("forbidden_globs", []))
        forbidden.extend(
            [
                ".git/**",
                "data/control/**",
                "data/models/**",
                "data/registry/**",
                "**/__pycache__/**",
                "logs/**",
            ]
        )
        settings["self_patch"]["forbidden_globs"] = sorted(set(forbidden))
        return settings

    def _run_autonomy_autoedit(self, cycle_id: str, *, live_mode: bool = True) -> dict[str, Any]:
        if not self._git_clean():
            detail = "workspace_dirty"
            self._append_autonomy_event(
                agent="system",
                kind="autoedit_skip",
                title="Autoedicion aplazada",
                detail="El repo tiene cambios sin confirmar. La autoedicion queda en espera para no pisar trabajo manual.",
                cycle_id=cycle_id,
                state_name="waiting_resources",
                metadata={"reason": detail},
            )
            return {"ok": True, "skipped": detail}

        snapshot = self._git_head()
        snapshot_tag = None
        if snapshot:
            snapshot_tag = self._git_tag_snapshot(f"autonomy/snapshot-{time.strftime('%Y%m%d_%H%M%S')}")

        settings = self._autonomy_settings()
        result = _run_autopilot_tick_lazy(
            settings,
            self.base_dir,
            no_web=True,
            mock=not bool(live_mode),
            force=True,
        )
        patch_info = result.steps.get("autopatch") if isinstance(result.steps, dict) else {}
        ok_patch = isinstance(patch_info, dict) and bool(patch_info.get("ok", False))
        promoted = isinstance(patch_info, dict) and bool(patch_info.get("promoted", False))
        rollback_status = None

        if ok_patch and promoted:
            self._append_autonomy_event(
                agent="builder",
                kind="autoedit_applied",
                title="Autoedicion promovida",
                detail="El constructor aplico cambios, ejecuto validaciones y dejo el repo en un estado promovido con snapshot previo.",
                cycle_id=cycle_id,
                state_name="autoediting",
                metadata={"branch": patch_info.get("branch"), "snapshot": snapshot_tag or snapshot},
            )
        elif isinstance(patch_info, dict) and patch_info.get("skipped"):
            self._append_autonomy_event(
                agent="builder",
                kind="autoedit_skipped",
                title="Autoedicion sin cambios",
                detail=(
                    f"No se aplicaron cambios: {patch_info.get('skipped')}."
                    + (" La autoedicion estaba en modo simulacion." if not live_mode else "")
                ),
                cycle_id=cycle_id,
                state_name="autoediting",
                metadata={"snapshot": snapshot_tag or snapshot, "live_mode": bool(live_mode)},
            )
        else:
            rollback_status = {
                "ts": _utc_ts(),
                "status": "rollback_ok" if self._git_clean() else "rollback_failed",
                "target": snapshot_tag or snapshot,
                "reason": (patch_info.get("error") if isinstance(patch_info, dict) else result.error) or "autopatch_failed",
            }
            self._append_autonomy_event(
                agent="system",
                kind="rollback",
                title="Rollback autonomo",
                detail="La autoedicion no paso las validaciones y el sistema ha vuelto al snapshot anterior o ha preservado el repo sin aplicar cambios inestables.",
                cycle_id=cycle_id,
                state_name="rollback",
                metadata=rollback_status,
            )
        return {
            "ok": result.ok,
            "steps": result.steps,
            "error": result.error,
            "rollback": rollback_status,
            "live_mode": bool(live_mode),
        }

    def _autonomy_worker(self) -> None:
        while not self._autonomy_stop.is_set():
            try:
                queued_dispatch = self._dispatch_queued_training_runs()
                if queued_dispatch:
                    time.sleep(1.5)
                    continue
                autonomy = self._load_autonomy_state()
                config = autonomy.get("config", {}) if isinstance(autonomy.get("config"), dict) else {}
                now = _utc_ts()
                runtime = self.runtime_status()
                training_busy = bool(self._training_thread and self._training_thread.is_alive())
                training_enabled = bool(config.get("training_enabled", True))
                reflection_enabled = bool(config.get("reflection_enabled", True))
                autoedit_enabled = bool(config.get("autoedit_enabled", True))

                if not autonomy.get("enabled", True):
                    self._write_autonomy_state(
                        {
                            "state": "paused",
                            "active_agents": self._agent_state(autonomy, analyst="paused", builder="paused"),
                        }
                    )
                    time.sleep(2.0)
                    continue

                campaign = self._ensure_active_campaign(reason="autonomy")
                runtime_ready = bool(runtime.get("api_ready") and runtime.get("runtime_ready"))
                cycle_id = str(autonomy.get("current_cycle") or "").strip() or f"cycle-{time.strftime('%Y%m%d-%H%M%S')}"

                if training_busy:
                    self._write_autonomy_state(
                        {
                            "state": "training",
                            "current_cycle": cycle_id,
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "active_agents": self._agent_state(autonomy, analyst="observing", builder="training"),
                        }
                    )
                    time.sleep(3.0)
                    continue

                pending_run = self._latest_incomplete_run()
                if pending_run:
                    pending_stage = str(pending_run.get("stage") or "").strip().lower()
                    pending_lifecycle = str(pending_run.get("lifecycle_state") or "").strip().lower()
                    waiting_on_queue = pending_stage in {"queued", "queued_waiting_resources"} or pending_lifecycle == "blocked"
                    self._write_autonomy_state(
                        {
                            "state": "learning" if waiting_on_queue else "training",
                            "current_cycle": str(pending_run.get("run_id") or cycle_id),
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "next_run_scheduled_at": None,
                            "scheduled_run_mode": None,
                            "scheduled_parent_run_id": None,
                            "scheduled_followup_reason": None,
                            "active_agents": self._agent_state(
                                autonomy,
                                analyst="monitoring" if waiting_on_queue else "observing",
                                builder="waiting" if waiting_on_queue else "training",
                            ),
                        }
                    )
                    time.sleep(2.0)
                    continue

                latest_run = self.list_runs(include_details=False, limit=1)
                latest_finished = latest_run[0] if latest_run else None
                latest_finished_status = str((latest_finished or {}).get("status") or "").strip().lower()
                latest_finished_lifecycle = str((latest_finished or {}).get("lifecycle_state") or "").strip().lower()
                scheduled_at = autonomy.get("next_run_scheduled_at")
                scheduled_mode = str(autonomy.get("scheduled_run_mode") or "").strip().lower() or None
                scheduled_parent_run_id = str(autonomy.get("scheduled_parent_run_id") or "").strip() or None
                scheduled_reason = str(autonomy.get("scheduled_followup_reason") or "").strip() or None

                reflection_due = (
                    reflection_enabled
                    and (now - float(autonomy.get("last_reflection_at") or 0.0)) >= float(config.get("reflection_interval_s", 300))
                )
                autoedit_due = (
                    autoedit_enabled
                    and (now - float(autonomy.get("last_patch_at") or 0.0)) >= float(config.get("autoedit_interval_s", 1800))
                )

                if training_enabled and not scheduled_at:
                    should_schedule = latest_finished is None or latest_finished_lifecycle in _RUN_TERMINAL_LIFECYCLES or latest_finished_status in {
                        "completed",
                        "completed_with_warnings",
                        "failed",
                        "rolled_back",
                        "degraded",
                        "interrupted",
                    }
                    if should_schedule:
                        scheduled_mode = self._next_followup_mode(latest_finished)
                        delay_s = 0.0 if latest_finished is None else self._followup_delay_s(latest_finished, autonomy)
                        scheduled_at = now + delay_s
                        scheduled_parent_run_id = str((latest_finished or {}).get("run_id") or "").strip() or None
                        scheduled_reason = "campaign_bootstrap" if latest_finished is None else f"followup_after_{latest_finished_lifecycle or latest_finished_status or 'terminal'}"
                        cycle_id = f"cycle-{time.strftime('%Y%m%d-%H%M%S')}"
                        self._write_autonomy_state(
                            {
                                "state": "learning",
                                "current_cycle": cycle_id,
                                "active_campaign_id": campaign.get("id"),
                                "current_campaign": campaign,
                                "next_run_scheduled_at": scheduled_at,
                                "scheduled_run_mode": scheduled_mode,
                                "scheduled_parent_run_id": scheduled_parent_run_id,
                                "scheduled_followup_reason": scheduled_reason,
                                "active_agents": self._agent_state(autonomy, analyst="planning", builder="planning"),
                            }
                        )
                        self._append_autonomy_event(
                            agent="system",
                            kind="followup_scheduled",
                            title="Siguiente run programado",
                            detail=(
                                "La campaña 24/7 ha dejado listo el siguiente entrenamiento."
                                f" Modo: {scheduled_mode}. Motivo: {scheduled_reason}."
                            ),
                            cycle_id=cycle_id,
                            state_name="learning",
                            metadata={
                                "scheduled_at": scheduled_at,
                                "mode": scheduled_mode,
                                "parent_run_id": scheduled_parent_run_id,
                            },
                        )
                        autonomy = self._load_autonomy_state()

                if reflection_due:
                    reflection_dialogue = []
                    if bool(config.get("multi_agent_dialogue_enabled", True)):
                        reflection_dialogue = self._build_agent_dialogue(
                            run_id=cycle_id,
                            sequence_id=int(self._next_training_sequence()),
                            mode=scheduled_mode or "quick",
                            source="autonomy",
                            cycle_id=cycle_id,
                        )
                    self._write_autonomy_state(
                        {
                            "state": "learning",
                            "current_cycle": cycle_id,
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "last_reflection_at": now,
                            "active_agents": self._agent_state(autonomy, analyst="reflecting", builder="planning"),
                            "latest_dialogue": reflection_dialogue,
                        }
                    )
                    self._append_autonomy_event(
                        agent="analyst",
                        kind="reflection",
                        title="Analista revisa sesiones y gaps",
                        detail="Cruza conversaciones, errores recientes, cobertura del repo y runs anteriores para detectar la siguiente mejora con mayor retorno.",
                        cycle_id=cycle_id,
                        state_name="learning",
                    )
                    if reflection_dialogue:
                        self._append_autonomy_event(
                            agent="system",
                            kind="multi_agent_sync",
                            title="Analista y Constructor sincronizan aprendizaje",
                            detail=(
                                "Los dos agentes han cruzado preguntas y respuestas para convertir el siguiente ciclo en un entrenamiento visible, descriptivo y trazable."
                            ),
                            cycle_id=cycle_id,
                            state_name="learning",
                            metadata={"turns": len(reflection_dialogue)},
                        )
                    self._append_autonomy_event(
                        agent="builder",
                        kind="hypothesis",
                        title="Constructor propone siguiente ciclo",
                        detail="Prepara hipotesis, dataset y verificacion del siguiente run sin dejar huecos entre ciclos.",
                        cycle_id=cycle_id,
                        state_name="learning",
                    )
                    if not runtime_ready:
                        self._append_autonomy_event(
                            agent="system",
                            kind="degraded_learning",
                            title="Aprendizaje local activo sin runtime",
                            detail=(
                                "La autonomia sigue reflexionando y preparando ciclos visibles aunque el runtime no este listo. "
                                "Los entrenamientos quedan en cola hasta recuperar servicio."
                            ),
                            cycle_id=cycle_id,
                            state_name="learning",
                            metadata={"runtime_ready": False},
                        )

                if training_enabled and scheduled_at:
                    try:
                        due_at = float(scheduled_at)
                    except Exception:
                        due_at = now
                    if now >= due_at:
                        launch_mode = scheduled_mode or self._next_followup_mode(latest_finished)
                        self._append_autonomy_event(
                            agent="builder",
                            kind=f"train_{launch_mode}",
                            title="Run continuo lanzado",
                            detail="La campaña 24/7 activa el siguiente run en cuanto queda libre el pipeline.",
                            cycle_id=cycle_id,
                            state_name="training",
                            metadata={
                                "mode": launch_mode,
                                "reason": scheduled_reason,
                                "parent_run_id": scheduled_parent_run_id,
                            },
                        )
                        result = self.start_training(launch_mode, source=f"autonomy_chain_{launch_mode}")
                        if result.get("ok"):
                            if result.get("queue_reason"):
                                self._write_autonomy_state(
                                    {
                                        "state": "learning",
                                        "current_cycle": cycle_id,
                                        "active_campaign_id": campaign.get("id"),
                                        "current_campaign": campaign,
                                        "last_train_at": now,
                                        "next_run_scheduled_at": None,
                                        "scheduled_run_mode": None,
                                        "scheduled_parent_run_id": None,
                                        "scheduled_followup_reason": None,
                                        "active_agents": self._agent_state(autonomy, analyst="planning", builder="waiting"),
                                    }
                                )
                                self._append_autonomy_event(
                                    agent="system",
                                    kind="training_deferred",
                                    title="Run continuo en espera",
                                    detail="El siguiente run de la campaña ya existe, pero ha quedado bloqueado por recursos o runtime.",
                                    cycle_id=cycle_id,
                                    state_name="learning",
                                    metadata={"queue_reason": result.get("queue_reason"), "run_id": result.get("run_id")},
                                )
                            else:
                                self._write_autonomy_state(
                                    {
                                        "state": "training",
                                        "current_cycle": cycle_id,
                                        "active_campaign_id": campaign.get("id"),
                                        "current_campaign": campaign,
                                        "last_train_at": now,
                                        "next_run_scheduled_at": None,
                                        "scheduled_run_mode": None,
                                        "scheduled_parent_run_id": None,
                                        "scheduled_followup_reason": None,
                                        "active_agents": self._agent_state(autonomy, analyst="observing", builder="training"),
                                    }
                                )
                            time.sleep(2.0)
                            continue

                if autoedit_due and not (self._training_thread and self._training_thread.is_alive()):
                    self._write_autonomy_state(
                        {
                            "state": "autoediting",
                            "current_cycle": cycle_id,
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "last_patch_at": now,
                            "active_agents": self._agent_state(autonomy, analyst="reviewing", builder="patching"),
                        }
                    )
                    self._append_autonomy_event(
                        agent="builder",
                        kind="autoedit_start",
                        title="Autoedicion del repo",
                        detail="El constructor abre un snapshot del repo versionado y ejecuta una ronda de autoedicion con tests, doctor y rollback si algo sale mal.",
                        cycle_id=cycle_id,
                        state_name="autoediting",
                    )
                    autoedit = self._run_autonomy_autoedit(
                        cycle_id,
                        live_mode=bool(config.get("live_autoedit_enabled", True)),
                    )
                    patch_state = "rollback" if autoedit.get("rollback") else "learning"
                    self._write_autonomy_state(
                        {
                            "state": patch_state,
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "last_rollback": autoedit.get("rollback"),
                            "active_agents": self._agent_state(
                                autonomy,
                                analyst="reviewing" if patch_state != "rollback" else "stabilizing",
                                builder="ready" if patch_state != "rollback" else "rollback",
                            ),
                        }
                    )
                    time.sleep(2.0)
                    continue

                if not reflection_due and not autoedit_due:
                    self._write_autonomy_state(
                        {
                            "state": "learning",
                            "current_cycle": cycle_id,
                            "active_campaign_id": campaign.get("id"),
                            "current_campaign": campaign,
                            "active_agents": self._agent_state(
                                autonomy,
                                analyst="monitoring",
                                builder="ready" if runtime_ready else "waiting",
                            ),
                        }
                    )
                time.sleep(2.0)
            except Exception as exc:
                self._write_autonomy_state({"state": "waiting_resources"})
                self._append_autonomy_event(
                    agent="system",
                    kind="autonomy_error",
                    title="Autonomia en espera",
                    detail=f"El bucle continuo ha detectado un error recuperable: {exc}",
                    state_name="waiting_resources",
                )
                time.sleep(5.0)

    def list_runs(self, *, include_details: bool = True, limit: int | None = None) -> list[dict[str, Any]]:
        wanted = max(1, int(limit)) if limit is not None else None
        runs: list[dict[str, Any]] = []
        for payload in self.storage.list_runs(limit=limit):
            if not isinstance(payload, dict):
                continue
            runs.append(self._enrich_run(payload, include_details=include_details))
        runs.sort(
            key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
            reverse=True,
        )
        if wanted is not None:
            runs = runs[:wanted]
        return runs

    def get_run(self, run_id: str, *, include_details: bool = True) -> dict[str, Any] | None:
        payload = self.storage.get_run(run_id)
        return self._enrich_run(payload, include_details=include_details) if isinstance(payload, dict) else None

    def _recover_stale_runtime_if_needed(self, runtime: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self._active_run_id or (self._training_thread and self._training_thread.is_alive()):
            return None
        candidate = self._latest_incomplete_run()
        if not candidate:
            return None

        stage = str(candidate.get("stage") or "").strip().lower()
        status = str(candidate.get("status") or "").strip().lower()
        recovery_state = candidate.get("stale_recovery") if isinstance(candidate.get("stale_recovery"), dict) else {}
        if recovery_state.get("completed") and stage not in _RUN_ACTIVE_STAGES and status not in {"queued", "running", "maintenance"}:
            return None

        updated_at = float(candidate.get("updated_at") or candidate.get("created_at") or 0.0)
        stale_for_s = max(0.0, _utc_ts() - updated_at)
        if stale_for_s < 20.0:
            return None

        run_id = str(candidate.get("run_id") or "").strip()
        runtime = runtime or self.runtime_status()

        recovery = {
            "attempted_at": _utc_ts(),
            "stale_for_s": round(stale_for_s, 3),
            "stage": stage,
        }

        if stage in {"draining_primary", "training", "eval", "resume_primary", "bench"} and not bool(runtime.get("api_ready")):
            if not self.compose_actions_enabled:
                recovery.update({"completed": True, "status": "manual_recovery_required", "reason": "compose_actions_disabled"})
                self._update_run_meta(
                    run_id,
                    {
                        "status": "interrupted",
                        "stage": "manual_recovery_required",
                        "failure": {
                            "reason": "compose_actions_disabled",
                            "stage": "manual_recovery_required",
                            "stale_for_s": round(stale_for_s, 3),
                        },
                        "stale_recovery": recovery,
                        "progress_pct": 1.0,
                    },
                )
                self._append_run_event(
                    run_id,
                    phase="resume_primary",
                    kind="recovery",
                    message="stale_runtime_manual_recovery_required",
                    progress_pct=1.0,
                    metadata=recovery,
                )
                self._append_autonomy_event(
                    agent="system",
                    kind="stale_recovery",
                    title="Run atascado cerrado para recuperacion manual",
                    detail="El control local ha cerrado el entrenamiento atascado para que deje de quedar pendiente. El runtime sigue necesitando recuperacion manual porque compose actions esta deshabilitado en este entorno.",
                    state_name="waiting_resources",
                    metadata={"run_id": run_id, **recovery},
                )
                return recovery
            try:
                runtime_log = self._run_dir(run_id) / "runtime.log"
                self._resume_runtime_stack(log_path=runtime_log, force_recreate=True)
                recovery.update({"completed": True, "status": "runtime_recovered"})
                self._update_run_meta(
                    run_id,
                    {
                        "status": "interrupted",
                        "stage": "recovered_runtime",
                        "failure": {
                            "reason": "stale_run_recovered",
                            "stage": "recovered_runtime",
                            "stale_for_s": round(stale_for_s, 3),
                        },
                        "stale_recovery": recovery,
                        "progress_pct": 1.0,
                    },
                )
                self._append_run_event(
                    run_id,
                    phase="resume_primary",
                    kind="recovery",
                    message="stale_runtime_recovered",
                    progress_pct=1.0,
                    metadata=recovery,
                )
                self._append_autonomy_event(
                    agent="system",
                    kind="stale_recovery",
                    title="Runtime recuperado tras run atascado",
                    detail="Se detecto un entrenamiento antiguo que habia dejado el runtime drenado. El control local ha restaurado el servicio y ha marcado el run como interrumpido.",
                    state_name="waiting_resources",
                    metadata={"run_id": run_id, **recovery},
                )
                return recovery
            except Exception as exc:
                recovery.update({"completed": True, "status": "runtime_recovery_failed", "reason": str(exc)})
                self._update_run_meta(
                    run_id,
                    {
                        "status": "failed",
                        "stage": "runtime_recovery_failed",
                        "failure": {
                            "reason": str(exc),
                            "stage": "runtime_recovery_failed",
                            "stale_for_s": round(stale_for_s, 3),
                        },
                        "stale_recovery": recovery,
                        "progress_pct": 1.0,
                    },
                )
                self._append_run_event(
                    run_id,
                    phase="failed",
                    kind="recovery",
                    message="stale_runtime_recovery_failed",
                    progress_pct=1.0,
                    metadata=recovery,
                )
                return recovery

        recovery.update({"completed": True, "status": "marked_interrupted"})
        self._update_run_meta(
            run_id,
            {
                "status": "interrupted",
                "stage": "stale_interrupted",
                "failure": {
                    "reason": "stale_run_interrupted",
                    "stage": "stale_interrupted",
                    "stale_for_s": round(stale_for_s, 3),
                },
                "stale_recovery": recovery,
                "progress_pct": 1.0,
            },
        )
        self._append_run_event(
            run_id,
            phase="failed",
            kind="recovery",
            message="stale_run_interrupted",
            progress_pct=1.0,
            metadata=recovery,
        )
        return recovery

    def status(self) -> dict[str, Any]:
        bootstrap = self._read_state_record("bootstrap", {}, self.bootstrap_state_path)
        docker = self.docker_status()
        runtime = self.runtime_status()
        recovery = self._recover_stale_runtime_if_needed(runtime)
        if recovery is not None:
            runtime = self.runtime_status()
        model = self.model_status(runtime)
        runs = self.list_runs(include_details=False, limit=12)
        active_run = self.get_run(self._active_run_id) if self._active_run_id else None
        overlay = self._runtime_overlay(active_run=active_run)

        runtime_models = runtime.get("runtime_models") if isinstance(runtime, dict) else None
        runtime_model_ids: list[str] = []
        if isinstance(runtime_models, dict):
            data = runtime_models.get("data")
            if isinstance(data, list):
                runtime_model_ids = [
                    str(item.get("id") or "").strip()
                    for item in data
                    if isinstance(item, dict) and str(item.get("id") or "").strip()
                ]

        if runtime.get("api_ready") and runtime.get("runtime_ready") and bootstrap.get("stage") != "ready":
            bootstrap = {
                **bootstrap,
                "running": False,
                "stage": "ready",
                "message": "stack_ready",
            }
            self._write_state_record("bootstrap", bootstrap, self.bootstrap_state_path)

        if runtime_model_ids and not bool(model.get("cached")):
            model = {
                **model,
                "cached": True,
                "snapshot_count": max(int(model.get("snapshot_count") or 0), len(runtime_model_ids)),
                "last_snapshot": model.get("last_snapshot") or runtime_model_ids[0],
            }

        if isinstance(runtime.get("status"), dict):
            runtime["status"] = {**runtime["status"], **overlay}
        else:
            runtime["status"] = overlay
        runtime.update(overlay)
        return {
            "ok": True,
            "bootstrap": bootstrap,
            "docker": docker,
            "model": model,
            "runtime": runtime,
            "frontend": self.frontend_status(),
            "internet": {"allowlist": self.get_allowlist()},
            "instructions": self._resolve_instruction_meta(),
            "learning_queue": self._learning_queue_summary(),
            "autonomy": self.autonomy_status(runtime=runtime, runs=runs),
            "active_run_id": self._active_run_id,
            "runs": runs,
        }

    def start_bootstrap(self, *, force: bool = False, mode: str | None = None) -> dict[str, Any]:
        normalized_mode = _normalize_bootstrap_mode(mode, force=force)
        log_path = self.log_dir / "control-bootstrap.log"
        if not self.compose_actions_enabled:
            runtime = self.runtime_status()
            ready = bool(runtime.get("api_ready") and runtime.get("runtime_ready"))
            stage = "ready" if ready else "external"
            message = "stack_ready" if ready else "compose_managed_externally"
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": stage,
                    "message": message,
                    "mode": normalized_mode,
                    "log_path": str(log_path),
                }
            )
            return {
                "ok": True,
                "started": False,
                "reason": "compose_actions_disabled",
                "mode": normalized_mode,
                "stage": stage,
                "log_path": str(log_path),
            }
        with self._lock:
            if self._bootstrap_thread and self._bootstrap_thread.is_alive() and not force:
                bootstrap = self._read_state_record("bootstrap", {}, self.bootstrap_state_path)
                return {
                    "ok": True,
                    "started": False,
                    "reason": "bootstrap_already_running",
                    "mode": bootstrap.get("mode") or normalized_mode,
                    "stage": bootstrap.get("stage"),
                    "log_path": bootstrap.get("log_path") or str(log_path),
                }
            self._set_bootstrap_state(
                {
                    "running": True,
                    "stage": "queued",
                    "message": "bootstrap_requested",
                    "mode": normalized_mode,
                    "log_path": str(log_path),
                }
            )
            thread = threading.Thread(target=self._bootstrap_worker, args=(normalized_mode,), daemon=True)
            self._bootstrap_thread = thread
            thread.start()
        return {"ok": True, "started": True, "mode": normalized_mode, "stage": "queued", "log_path": str(log_path)}

    def _bootstrap_worker(self, mode: str) -> None:
        log_path = self.log_dir / "control-bootstrap.log"
        self._set_bootstrap_state(
            {
                "running": True,
                "stage": "docker",
                "message": "checking_docker",
                "mode": mode,
                "log_path": str(log_path),
            }
        )

        docker = self.docker_status()
        if not docker.get("ready"):
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "failed",
                    "message": docker.get("reason"),
                    "error": docker,
                    "mode": mode,
                }
            )
            return

        runtime = self.runtime_status()
        if runtime.get("api_ready") and runtime.get("runtime_ready"):
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "ready",
                    "message": "stack_ready",
                    "mode": mode,
                    "tail": _tail(log_path),
                }
            )
            return

        runtime_services = self._runtime_compose_services()
        services_to_pull = [svc for svc in runtime_services if svc != "vortex-api"]
        if services_to_pull:
            code, _ = self._run_compose(["pull", *services_to_pull], log_path=log_path)
            if code != 0:
                self._set_bootstrap_state(
                    {
                        "running": False,
                        "stage": "failed",
                        "message": "image_pull_failed",
                        "mode": mode,
                        "tail": _tail(log_path),
                    }
                )
                return

        if mode == "ensure":
            self._set_bootstrap_state({"stage": "runtime", "message": "starting_runtime_no_build", "mode": mode})
            code, _ = self._compose_runtime_up(log_path=log_path, no_build=True)
            if code == 0 and self._wait_runtime_ready(timeout_s=120.0):
                self._set_bootstrap_state(
                    {
                        "running": False,
                        "stage": "ready",
                        "message": "stack_ready",
                        "mode": mode,
                        "tail": _tail(log_path),
                    }
                )
                return

            if self._compose_local_images_available():
                self._set_bootstrap_state({"stage": "model-init", "message": "ensuring_local_model_no_build", "mode": mode})
                code, _ = self._run_compose(["run", "--rm", "--no-build", "model-init"], log_path=log_path)
                if code == 0:
                    self._set_bootstrap_state({"stage": "runtime", "message": "starting_runtime_no_build_retry", "mode": mode})
                    code, _ = self._compose_runtime_up(log_path=log_path, no_build=True)
                    if code == 0 and self._wait_runtime_ready(timeout_s=120.0):
                        self._set_bootstrap_state(
                            {
                                "running": False,
                                "stage": "ready",
                                "message": "stack_ready",
                                "mode": mode,
                                "tail": _tail(log_path),
                            }
                        )
                        return
            self._set_bootstrap_state({"stage": "build", "message": "falling_back_to_rebuild", "mode": mode})
        else:
            self._set_bootstrap_state({"stage": "build", "message": "building_backend_images", "mode": mode})

        code, _ = self._run_compose(["build", "model-init", "vortex-api", "trainer", "eval"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "failed",
                    "message": "build_failed",
                    "mode": mode,
                    "tail": _tail(log_path),
                }
            )
            return

        self._set_bootstrap_state({"stage": "model-init", "message": "ensuring_local_model", "mode": mode})
        code, _ = self._run_compose(["run", "--rm", "model-init"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "failed",
                    "message": "model_init_failed",
                    "mode": mode,
                    "tail": _tail(log_path),
                }
            )
            return

        self._set_bootstrap_state({"stage": "runtime", "message": "starting_runtime", "mode": mode})
        code, _ = self._compose_runtime_up(log_path=log_path)
        if code != 0:
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "failed",
                    "message": "runtime_start_failed",
                    "mode": mode,
                    "tail": _tail(log_path),
                }
            )
            return

        self._set_bootstrap_state({"stage": "waiting", "message": "waiting_for_readyz", "mode": mode})
        if not self._wait_runtime_ready():
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "failed",
                    "message": "runtime_not_ready",
                    "mode": mode,
                    "tail": _tail(log_path),
                }
            )
            return

        self._set_bootstrap_state({"running": False, "stage": "ready", "message": "stack_ready", "mode": mode, "tail": _tail(log_path)})

    def restart_runtime(self) -> dict[str, Any]:
        if not self.compose_actions_enabled:
            return {"ok": False, "reason": "compose_actions_disabled"}
        log_path = self.log_dir / "control-runtime-restart.log"
        code, _ = self._compose_runtime_up(log_path=log_path, force_recreate=True)
        if code != 0:
            raise RuntimeError("runtime_restart_failed")
        ok = self._wait_runtime_ready(timeout_s=180.0)
        return {"ok": bool(ok), "log_path": str(log_path), "tail": _tail(log_path)}

    def _stop_runtime_stack(self, *, log_path: Path) -> None:
        if not self.compose_actions_enabled:
            raise RuntimeError("compose_actions_disabled")
        code, _ = self._run_compose(["stop", *self._runtime_compose_services()], log_path=log_path)
        if code != 0:
            raise RuntimeError("runtime_stop_failed")

    def _resume_runtime_stack(self, *, log_path: Path, force_recreate: bool = False) -> None:
        if not self.compose_actions_enabled:
            raise RuntimeError("compose_actions_disabled")
        if bool(self._load_runtime_state().get("fallback_active")):
            self._stop_fallback_runtime()
        args = ["up", "-d"]
        if force_recreate:
            args.append("--force-recreate")
        args.extend(self._runtime_compose_services())
        code, _ = self._run_compose(args, log_path=log_path)
        if code != 0:
            raise RuntimeError("runtime_resume_failed")
        if not self._wait_runtime_ready(timeout_s=240.0):
            raise RuntimeError("runtime_not_ready_after_training")

    def _update_run_meta(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.storage.get_run(run_id) or {}
        previous_stage = str(current.get("stage") or "").strip().lower()
        previous_lifecycle = str(current.get("lifecycle_state") or "").strip().lower()
        current.update(patch)
        next_stage = str(current.get("stage") or "").strip().lower()
        next_lifecycle = str(current.get("lifecycle_state") or "").strip().lower()
        if next_stage and next_stage != previous_stage and "execution_progress_pct" not in patch:
            current.pop("execution_progress_pct", None)
            if "progress_pct" not in patch:
                current.pop("progress_pct", None)
        if next_lifecycle and next_lifecycle != previous_lifecycle and "pipeline_progress_pct" not in patch:
            current.pop("pipeline_progress_pct", None)
        lifecycle = str(current.get("lifecycle_state") or "").strip().lower() or self._infer_lifecycle_state(current)
        current["lifecycle_state"] = lifecycle
        if lifecycle == "blocked":
            current["blocked_reason"] = str(current.get("blocked_reason") or current.get("queue_reason") or "waiting_resources").strip()
            current["blocked_since"] = current.get("blocked_since") or _utc_ts()
            current["retry_in_s"] = float(current.get("retry_in_s") or 2.0)
        else:
            current["blocked_reason"] = current.get("blocked_reason") if patch.get("blocked_reason") is not None else None
            current["blocked_since"] = current.get("blocked_since") if patch.get("blocked_since") is not None and lifecycle == "blocked" else None
            current["retry_in_s"] = current.get("retry_in_s") if patch.get("retry_in_s") is not None and lifecycle == "blocked" else None
        current["execution_progress_pct"] = self._compute_execution_progress(current)
        current["pipeline_progress_pct"] = self._compute_pipeline_progress(current)
        current["progress_pct"] = current["execution_progress_pct"]
        if isinstance(current.get("agent_dialogue"), list):
            current["agent_dialogue"] = list(current.get("agent_dialogue") or [])[-24:]
        if isinstance(current.get("notebook_sections"), list):
            current["notebook_sections"] = list(current.get("notebook_sections") or [])[-48:]
        if isinstance(current.get("live_metrics_series"), list):
            current["live_metrics_series"] = list(current.get("live_metrics_series") or [])[-160:]
        current["updated_at"] = _utc_ts()
        run_id_value = str(current.get("run_id") or run_id).strip()
        if run_id_value:
            current["run_id"] = run_id_value
            self.storage.put_run(run_id_value, current)
        return current

    def _make_run_log_callback(
        self,
        run_id: str,
        *,
        phase: str,
        log_path: Path,
    ) -> Callable[[str], None]:
        buffer: list[str] = []
        last_emit = 0.0

        def _callback(line: str) -> None:
            nonlocal last_emit
            buffer.append(line)
            if len(buffer) > 120:
                del buffer[:-120]
            now = time.time()
            metrics = _parse_live_metrics(buffer)
            if metrics or (now - last_emit) >= 2.0:
                execution_progress = self._compute_execution_progress(
                    {"stage": phase, "latest_metrics": metrics, "max_steps": (self.get_run(run_id) or {}).get("max_steps")}
                )
                self._update_run_meta(
                    run_id,
                    {
                        "latest_metrics": metrics,
                        "log_path": str(log_path),
                        "execution_progress_pct": execution_progress,
                    },
                )
                self._append_live_metrics_point(run_id, phase=phase, metrics=metrics)
                self._append_run_event(
                    run_id,
                    phase=phase,
                    kind="progress",
                    message=line[:320],
                    latest_metrics=metrics,
                    progress_pct=self._compute_execution_progress(self.get_run(run_id) or {}),
                )
                last_emit = now

        return _callback

    def _queue_reason_for_quick_run(self) -> tuple[str | None, dict[str, Any]]:
        return self._queue_reason_for_mode("quick")

    def start_training(self, mode: str, source: str | None = None) -> dict[str, Any]:
        normalized = str(mode or "quick").strip().lower()
        if normalized not in {"quick", "full"}:
            raise HTTPException(status_code=400, detail="training_mode_invalid")

        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                return {"ok": False, "error": "training_already_running", "run_id": self._active_run_id}

            for existing in self.list_runs(include_details=False, limit=120):
                if (
                    str(existing.get("mode") or "") == normalized
                    and str(existing.get("status") or "") == "queued"
                ):
                    return {
                        "ok": True,
                        "run_id": existing.get("run_id"),
                        "status": "queued",
                        "queue_reason": existing.get("queue_reason"),
                        "reused": True,
                    }

            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            queued_items = self._list_learning_queue(include_consumed=False)
            queue_state = self._load_learning_queue_state()
            quick_threshold = int(queue_state.get("quick_threshold") or DEFAULT_QUICK_QUEUE_THRESHOLD)
            sequence_id = self._next_training_sequence()
            focus = self._build_dialogue_focus(normalized, source)
            run_context = self._derive_run_context(mode=normalized, source=source)
            served_model = self._served_model_id()
            dialogue = self._build_agent_dialogue(
                run_id=run_id,
                sequence_id=sequence_id,
                mode=normalized,
                source=source,
                cycle_id=str(self._load_autonomy_state().get("current_cycle") or "") or None,
            )
            notebook_sections = self._build_initial_notebook_sections(focus=focus, run_context=run_context)
            source_mix = {
                "priority_1_failures_regressions_runtime": 0.5,
                "priority_2_chat_feedback_episodes": 0.3,
                "priority_3_repo_docs_lessons": 0.2,
            }
            meta = {
                "ok": True,
                "run_id": run_id,
                "sequence_id": sequence_id,
                "mode": normalized,
                "status": "queued",
                "stage": "queued",
                "lifecycle_state": "planned",
                "created_at": _utc_ts(),
                "profile": f"continuous_descriptive::{served_model}",
                "base_model": served_model,
                "served_model": served_model,
                "dataset_hash": self._resolve_dataset_hash(),
                "dataset_mix": self._default_dataset_mix(),
                "source_mix": source_mix,
                "instructions": self._resolve_instruction_meta(),
                "runtime_mode": "primary",
                "fallback_active": False,
                "fallback_backend": None,
                "campaign_id": run_context.get("campaign_id"),
                "parent_run_id": run_context.get("parent_run_id"),
                "attempt": run_context.get("attempt"),
                "run_lineage": run_context.get("run_lineage") or [],
                "progress_pct": _RUN_PROGRESS_BY_STAGE["queued"],
                "execution_progress_pct": _RUN_PROGRESS_BY_STAGE["queued"],
                "pipeline_progress_pct": _RUN_PIPELINE_PROGRESS_BY_LIFECYCLE["planned"],
                "queue_reason": None,
                "blocked_reason": None,
                "blocked_since": None,
                "retry_in_s": None,
                "latest_metrics": {},
                "live_metrics_series": [],
                "dataset_manifest": {
                    "queued_count": len(queued_items),
                    "quick_threshold": quick_threshold,
                    "consumed_count": 0,
                    "source_kinds": {},
                    "request_ids": [],
                    "items": [],
                },
                "promotion": {"manual_only": self._training_manual_promotion_only(), "decision": "pending"},
                "gate_results": {},
                "apply_result": None,
                "rollback_result": None,
                "trigger": source or "manual",
                "display_name": f"Entrenamiento {sequence_id}",
                "display_description": focus["reason"],
                "objective": focus["objective"],
                "learning_focus": focus["focus"],
                "agent_dialogue": dialogue,
                "notebook_sections": notebook_sections,
            }
            self._update_run_meta(run_id, meta)
            self._write_autonomy_state(
                {
                    "latest_dialogue": dialogue,
                    "current_cycle": run_id,
                    "next_run_scheduled_at": None,
                    "scheduled_run_mode": None,
                    "scheduled_parent_run_id": None,
                    "scheduled_followup_reason": None,
                }
            )
            for turn in dialogue[:2]:
                self._append_autonomy_event(
                    agent=str(turn.get("speaker") or "system"),
                    kind="dialogue_turn",
                    title=f"{turn.get('speaker_label') or turn.get('speaker')}: {turn.get('kind')}",
                    detail=str(turn.get("message") or ""),
                    cycle_id=str(turn.get("cycle_id") or "") or None,
                    state_name="learning",
                    metadata={"run_id": run_id, "mode": normalized},
                )
            self._append_run_event(
                run_id,
                phase="queued",
                message="training_run_created",
                kind="phase",
                progress_pct=_RUN_PROGRESS_BY_STAGE["queued"],
                metadata={"mode": normalized, "source": source or "manual"},
            )
            self._append_autonomy_event(
                agent="builder",
                kind="training_queued",
                title="Nuevo run en cola",
                detail=(
                    "Se ha programado un aprendizaje rapido para reforzar muestras recientes."
                    if normalized == "quick"
                    else "Se ha programado un entrenamiento completo con drenado de runtime."
                ),
                state_name="training",
                metadata={"run_id": run_id, "mode": normalized},
            )
            queue_reason, diagnostics = self._queue_reason_for_mode(normalized)
            if queue_reason:
                self._queue_run_waiting_resources(
                    run_id,
                    mode=normalized,
                    queue_reason=queue_reason,
                    diagnostics=diagnostics,
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "status": "queued",
                    "queue_reason": queue_reason,
                }
            self._launch_training_thread(run_id, normalized)
        return {"ok": True, "run_id": run_id, "status": "queued", "queue_reason": None}

    def _training_worker(self, run_id: str, mode: str) -> None:
        runtime_resumed = mode != "full"
        fallback_started = False
        parallel_runtime_training = False
        try:
            run_dir = self._run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "run.log"
            eval_log_path = run_dir / "eval.log"
            bench_log_path = run_dir / "bench.log"
            runtime_log_path = run_dir / "runtime.log"
            fallback_log_path = run_dir / "fallback.log"
            env = {"C3RNT2_TRAIN_MAX_STEPS": str(6 if mode == "quick" else 12)}
            args = [
                "run",
                "--rm",
                "trainer",
                "python",
                "-m",
                "c3rnt2",
                "train-once",
                "--profile",
                self.training_profile,
            ]
            if mode == "quick":
                args.append("--reuse-dataset")

            self._update_run_meta(
                run_id,
                {
                    "status": "queued",
                    "stage": "queued",
                    "max_steps": int(env["C3RNT2_TRAIN_MAX_STEPS"]),
                    "log_path": str(log_path),
                    "runtime_log_path": str(runtime_log_path),
                    "fallback_log_path": str(fallback_log_path),
                    "queue_reason": None,
                },
            )
            training_resource_diagnostics: dict[str, Any] = {}
            runtime_snapshot = self.runtime_status()
            allow_parallel_training = self._runtime_allows_parallel_training(runtime_snapshot)
            parallel_runtime_training = bool(allow_parallel_training)
            if parallel_runtime_training:
                args.append("--allow-parallel-runtime")

            if mode == "quick":
                queue_reason, diagnostics = self._queue_reason_for_mode("quick")
                if queue_reason:
                    self._queue_run_waiting_resources(
                        run_id,
                        mode=mode,
                        queue_reason=queue_reason,
                        diagnostics=diagnostics,
                    )
                    return
                training_resource_diagnostics = diagnostics
            else:
                if allow_parallel_training:
                    ready_for_training, training_resource_diagnostics = self._wait_for_training_resources(
                        roles=("train", "self_patch"),
                        include_primary_runtime=False,
                        timeout_s=45.0,
                    )
                    self._update_run_meta(
                        run_id,
                        {
                            "status": "queued",
                            "stage": "queued",
                            "runtime_mode": str((runtime_snapshot.get("status") or {}).get("chat_mode") or "primary"),
                            "queue_diagnostics": training_resource_diagnostics,
                        },
                    )
                    self._append_run_event(
                        run_id,
                        phase="queued",
                        message="parallel_runtime_training_enabled",
                        kind="phase",
                        progress_pct=_RUN_PROGRESS_BY_STAGE["queued"],
                        metadata={
                            "runtime_mode": str((runtime_snapshot.get("status") or {}).get("chat_mode") or "primary"),
                            "runtime_device": (
                                ((runtime_snapshot.get("runtime_models") or {}).get("data") or [{}])[0].get("device")
                                if isinstance((runtime_snapshot.get("runtime_models") or {}).get("data"), list)
                                else None
                            ),
                        },
                    )
                    if not ready_for_training:
                        self._queue_run_waiting_resources(
                            run_id,
                            mode=mode,
                            queue_reason="training_resources_busy",
                            diagnostics=training_resource_diagnostics,
                            event_message="parallel_runtime_training_deferred",
                        )
                        return
                else:
                    self._update_run_meta(
                        run_id,
                        {
                            "status": "maintenance",
                            "stage": "draining_primary",
                            "runtime_mode": "maintenance",
                            "progress_pct": _RUN_PROGRESS_BY_STAGE["draining_primary"],
                        },
                    )
                    self._append_run_event(
                        run_id,
                        phase="draining_primary",
                        message="stopping_primary_runtime",
                        progress_pct=_RUN_PROGRESS_BY_STAGE["draining_primary"],
                    )
                    self._stop_runtime_stack(log_path=runtime_log_path)
                    released, lock_diagnostics = self._wait_for_lock_release("serve", timeout_s=45.0)
                    self._update_run_meta(run_id, {"lock_diagnostics": lock_diagnostics})
                    if not released:
                        self._update_run_meta(
                            run_id,
                            {
                                "status": "degraded",
                                "stage": "failed",
                                "lifecycle_state": "degraded",
                                "progress_pct": 1.0,
                                "failure": {
                                    "reason": "serve_lock_not_released",
                                    "stage": "draining_primary",
                                    "lock_diagnostics": lock_diagnostics,
                                },
                                "terminal_reason": "serve_lock_not_released",
                            },
                        )
                        self._append_run_event(
                            run_id,
                            phase="failed",
                            message="serve_lock_not_released",
                            kind="failure",
                            progress_pct=1.0,
                            metadata=lock_diagnostics,
                        )
                        return
                    runtime_state = self._start_fallback_runtime(log_path=fallback_log_path)
                    fallback_started = True
                    self._update_run_meta(
                        run_id,
                        {
                            "stage": "fallback_ready",
                            "runtime_mode": "fallback_degraded",
                            "fallback_active": True,
                            "fallback_backend": runtime_state.get("fallback_backend"),
                            "progress_pct": _RUN_PROGRESS_BY_STAGE["fallback_ready"],
                        },
                    )
                    self._append_run_event(
                        run_id,
                        phase="fallback_ready",
                        message="fallback_runtime_ready",
                        progress_pct=_RUN_PROGRESS_BY_STAGE["fallback_ready"],
                        metadata={"fallback_backend": runtime_state.get("fallback_backend")},
                    )
                    ready_for_training, training_resource_diagnostics = self._wait_for_training_resources(
                        roles=("train", "self_patch"),
                        include_primary_runtime=False,
                        timeout_s=45.0,
                    )
                    self._update_run_meta(run_id, {"queue_diagnostics": training_resource_diagnostics})
                    if not ready_for_training:
                        self._queue_run_waiting_resources(
                            run_id,
                            mode=mode,
                            queue_reason="training_resources_busy",
                            diagnostics={**training_resource_diagnostics, "tail": _tail(fallback_log_path)},
                            event_message="fallback_training_deferred",
                        )
                        return

            dataset_manifest = self._consume_learning_queue(run_id, mode=mode)
            self._update_run_meta(
                run_id,
                {
                    "dataset_manifest": dataset_manifest,
                    "queue_reason": None,
                    "queue_diagnostics": training_resource_diagnostics,
                    "lifecycle_state": "curating",
                    "pipeline_progress_pct": _RUN_PIPELINE_PROGRESS_BY_LIFECYCLE["curating"],
                },
            )
            self._append_notebook_section(
                run_id,
                phase="curating",
                title="Curacion del dataset",
                content=(
                    f"Se han preparado {int(dataset_manifest.get('queued_count') or 0)} entradas visibles para el ciclo "
                    f"y {int(dataset_manifest.get('consumed_count') or 0)} ya quedan marcadas como consumidas."
                ),
                kind="dataset",
                metadata=_json_safe(dataset_manifest),
            )
            self._append_run_dialogue_turn(
                run_id,
                speaker="builder",
                speaker_label="Constructor",
                kind="dataset",
                message=(
                    "He curado el dataset priorizando fallos recientes, feedback de chat y lecciones locales antes de entrar a entrenamiento."
                ),
            )

            self._update_run_meta(
                run_id,
                {
                    "status": "running",
                    "stage": "training",
                    "lifecycle_state": "training",
                    "runtime_mode": "fallback_degraded" if fallback_started else "primary",
                    "fallback_active": fallback_started,
                    "fallback_backend": "hf" if fallback_started else None,
                    "progress_pct": _RUN_PROGRESS_BY_STAGE["training"],
                },
            )
            self._append_notebook_section(
                run_id,
                phase="training",
                title="Entrenamiento en curso",
                content="El adapter entra en fase de ajuste con seguimiento de metricas en vivo y un unico run activo en el pipeline.",
                kind="training",
            )
            self._append_run_event(
                run_id,
                phase="training",
                message="trainer_started",
                progress_pct=_RUN_PROGRESS_BY_STAGE["training"],
            )
            if self._should_use_local_job_runner():
                code, output = self._run_local_training_job(
                    mode=mode,
                    env=env,
                    log_path=log_path,
                    parallel_runtime_training=parallel_runtime_training,
                )
            else:
                code, output = self._run_compose(
                    args,
                    env=env,
                    log_path=log_path,
                    line_callback=self._make_run_log_callback(run_id, phase="training", log_path=log_path),
                )
            payload = _parse_structured_output(output) or {}
            success = code == 0 and bool(payload.get("ok", False))
            meta_patch: dict[str, Any] = {
                "train_result": payload,
                "exit_code": code,
                "latest_metrics": {**dict((self.get_run(run_id) or {}).get("latest_metrics") or {}), **dict(payload)},
            }
            adapter_dir = payload.get("adapter_dir")
            if adapter_dir:
                meta_patch["adapter_dir"] = str(adapter_dir)
            if not success:
                failure_reason = str(payload.get("error") or "train_failed")
                queue_reason, queue_diagnostics = self._queue_reason_for_mode(mode)
                retryable_quick_lock_failure = (
                    mode == "quick"
                    and "lock" in failure_reason.lower()
                    and bool(queue_reason)
                )
                if retryable_quick_lock_failure:
                    restore = self._restore_learning_queue(run_id)
                    combined_diagnostics = {
                        **(queue_diagnostics or {}),
                        "restored_queue_items": restore.get("restored_count"),
                        "trainer_error": failure_reason,
                        "exit_code": code,
                    }
                    self._queue_run_waiting_resources(
                        run_id,
                        mode=mode,
                        queue_reason=queue_reason or "training_resources_busy",
                        diagnostics=combined_diagnostics,
                        event_message="trainer_lock_contention_requeued",
                    )
                    return
                failure = {
                    "reason": failure_reason,
                    "stage": "training",
                    "exit_code": code,
                }
                self._update_run_meta(
                    run_id,
                    {
                        **meta_patch,
                        "status": "degraded",
                        "stage": "failed",
                        "lifecycle_state": "degraded",
                        "progress_pct": 1.0,
                        "failure": failure,
                        "terminal_reason": failure_reason,
                        "tail": _tail(log_path),
                    },
                )
                self._append_notebook_section(
                    run_id,
                    phase="degraded",
                    title="Fallo del entrenamiento",
                    content="El ciclo no ha superado la fase de entrenamiento y termina como degradado para permitir un follow-up correctivo.",
                    kind="failure",
                    metadata=failure,
                )
                self._append_run_event(
                    run_id,
                    phase="failed",
                    message=str(failure["reason"]),
                    kind="failure",
                    progress_pct=1.0,
                    metadata=failure,
                )
                self._append_autonomy_event(
                    agent="system",
                    kind="training_failed",
                    title="Run fallido",
                    detail="El entrenamiento no termino correctamente y se ha detenido antes de promocionar ningun cambio.",
                    state_name="training",
                    metadata={"run_id": run_id, "mode": mode, "exit_code": code},
                )
                return

            self._append_run_event(
                run_id,
                phase="training",
                message="trainer_finished",
                kind="phase",
                progress_pct=0.64,
                metadata={"adapter_dir": adapter_dir, "exit_code": code},
            )

            if adapter_dir:
                self._update_run_meta(
                    run_id,
                    {
                        **meta_patch,
                        "status": "running",
                        "stage": "eval",
                        "lifecycle_state": "evaluating",
                        "eval_log_path": str(eval_log_path),
                        "progress_pct": _RUN_PROGRESS_BY_STAGE["eval"],
                    },
                )
                self._append_notebook_section(
                    run_id,
                    phase="evaluating",
                    title="Evaluacion offline",
                    content="Se ejecutan evals del adapter antes de pensar en apply o rollback.",
                    kind="eval",
                    metadata={"adapter_dir": str(adapter_dir)},
                )
                self._append_run_dialogue_turn(
                    run_id,
                    speaker="analyst",
                    speaker_label="Analista",
                    kind="gate",
                    message="Empiezo los gates de evaluacion para decidir si este adapter merece apply, rollback o degradado.",
                )
                self._append_run_event(
                    run_id,
                    phase="eval",
                    message="eval_started",
                    progress_pct=_RUN_PROGRESS_BY_STAGE["eval"],
                )
                eval_args = [
                    sys.executable,
                    "-m",
                    "c3rnt2",
                    "learn",
                    "eval",
                    "--profile",
                    self.training_profile,
                    "--adapter",
                    str(adapter_dir),
                ]
                if self._should_use_local_job_runner():
                    eval_code, eval_output = self._run_local_command(
                        eval_args,
                        log_path=eval_log_path,
                        line_callback=self._make_run_log_callback(run_id, phase="eval", log_path=eval_log_path),
                    )
                else:
                    eval_code, eval_output = self._run_compose(
                        ["run", "--rm", "trainer", *eval_args],
                        log_path=eval_log_path,
                        line_callback=self._make_run_log_callback(run_id, phase="eval", log_path=eval_log_path),
                    )
                meta_patch["eval_result"] = _parse_structured_output(eval_output) or {}
                meta_patch["eval_exit_code"] = eval_code
                self._append_run_event(
                    run_id,
                    phase="eval",
                    message="eval_finished",
                    kind="phase",
                    progress_pct=0.8,
                    metadata={"exit_code": eval_code},
                )

            if mode == "full":
                if not parallel_runtime_training:
                    self._update_run_meta(
                        run_id,
                        {
                            **meta_patch,
                            "status": "maintenance",
                            "stage": "resume_primary",
                            "lifecycle_state": "verifying",
                            "runtime_mode": "maintenance",
                            "progress_pct": _RUN_PROGRESS_BY_STAGE["resume_primary"],
                        },
                    )
                    self._append_run_event(
                        run_id,
                        phase="resume_primary",
                        message="resuming_primary_runtime",
                        progress_pct=_RUN_PROGRESS_BY_STAGE["resume_primary"],
                    )
                    self._resume_runtime_stack(log_path=runtime_log_path, force_recreate=True)
                    runtime_resumed = True
                    fallback_started = False

                current_runtime_mode = self._current_runtime_mode()
                self._update_run_meta(
                    run_id,
                    {
                        **meta_patch,
                        "status": "running",
                        "stage": "bench",
                        "lifecycle_state": "evaluating",
                        "bench_log_path": str(bench_log_path),
                        "runtime_mode": current_runtime_mode,
                        "fallback_active": current_runtime_mode == "fallback_degraded",
                        "fallback_backend": "hf" if current_runtime_mode == "fallback_degraded" else None,
                        "progress_pct": _RUN_PROGRESS_BY_STAGE["bench"],
                    },
                )
                self._append_run_event(
                    run_id,
                    phase="bench",
                    message="bench_started",
                    progress_pct=_RUN_PROGRESS_BY_STAGE["bench"],
                )
                bench_args = [
                    sys.executable,
                    "-m",
                    "c3rnt2",
                    "bench",
                    "--profile",
                    self.api_profile,
                    "--scenario",
                    "default",
                ]
                if self._should_use_local_job_runner():
                    bench_code, bench_output = self._run_local_command(
                        bench_args,
                        log_path=bench_log_path,
                        line_callback=self._make_run_log_callback(run_id, phase="bench", log_path=bench_log_path),
                    )
                else:
                    bench_code, bench_output = self._run_compose(
                        ["run", "--rm", "eval", *bench_args],
                        log_path=bench_log_path,
                        line_callback=self._make_run_log_callback(run_id, phase="bench", log_path=bench_log_path),
                    )
                meta_patch["bench_result"] = _parse_structured_output(bench_output) or {}
                meta_patch["bench_exit_code"] = bench_code
                self._append_run_event(
                    run_id,
                    phase="bench",
                    message="bench_finished",
                    kind="phase",
                    progress_pct=0.98,
                    metadata={"exit_code": bench_code},
                )

            bench_ok = bool((meta_patch.get("bench_result") or {}).get("ok", mode != "full"))
            eval_ok = bool((meta_patch.get("eval_result") or {}).get("ok", True))
            promotion = self._resolve_run_promotion(
                run_id=run_id,
                adapter_dir=adapter_dir,
                train_result=payload,
                eval_ok=eval_ok,
                bench_ok=bench_ok,
            )
            promotion_decision = str(promotion.get("decision") or "").strip().lower()
            if promotion_decision == "rolled_back_after_smoke_failure":
                final_status = "rolled_back"
                final_lifecycle = "rolled_back"
                final_stage = "rolled_back"
            elif promotion_decision == "runtime_degraded_after_apply" or not eval_ok or not bench_ok:
                final_status = "degraded"
                final_lifecycle = "degraded"
                final_stage = "degraded"
            else:
                final_status = "completed"
                final_lifecycle = "completed"
                final_stage = "done"
            final_runtime_mode = self._current_runtime_mode()
            final_meta = {
                **meta_patch,
                "status": final_status,
                "stage": final_stage,
                "lifecycle_state": final_lifecycle,
                "runtime_mode": final_runtime_mode,
                "fallback_active": final_runtime_mode == "fallback_degraded",
                "fallback_backend": "hf" if final_runtime_mode == "fallback_degraded" else None,
                "progress_pct": 1.0,
                "promotion": promotion,
                "gate_results": promotion.get("gate_results") or {},
                "apply_result": promotion.get("apply_result"),
                "rollback_result": promotion.get("rollback_result"),
                "terminal_reason": promotion.get("decision") or ("eval_or_bench_warning" if (not eval_ok or not bench_ok) else "completed"),
                "tail": _tail(log_path),
            }
            if final_status == "degraded":
                final_meta["failure"] = {
                    "reason": promotion.get("decision") or "eval_or_bench_warning",
                    "stage": final_stage,
                    "eval_ok": eval_ok,
                    "bench_ok": bench_ok,
                }
            self._update_run_meta(run_id, final_meta)
            self._append_notebook_section(
                run_id,
                phase=final_lifecycle,
                title="Resultado terminal del ciclo",
                content=(
                    f"El run termina en estado {final_lifecycle}. "
                    f"Decision final: {promotion.get('decision') or 'sin decision'}."
                ),
                kind="outcome",
                metadata={
                    "status": final_status,
                    "eval_ok": eval_ok,
                    "bench_ok": bench_ok,
                    "apply_result": promotion.get("apply_result"),
                    "rollback_result": promotion.get("rollback_result"),
                },
            )
            self._append_run_dialogue_turn(
                run_id,
                speaker="builder",
                speaker_label="Constructor",
                kind="outcome",
                message=(
                    f"Cierro el ciclo como {final_lifecycle}. "
                    f"Decision final: {promotion.get('decision') or 'sin decision'}."
                ),
            )
            self._append_run_event(
                run_id,
                phase=final_stage,
                message="training_completed",
                kind="phase",
                progress_pct=1.0,
                metadata={
                    "eval_ok": eval_ok,
                    "bench_ok": bench_ok,
                    "decision": promotion.get("decision"),
                    "lifecycle_state": final_lifecycle,
                },
            )
            self._append_autonomy_event(
                agent="builder",
                kind="training_completed",
                title="Run completado",
                detail=(
                    "El entrenamiento ha terminado, ha dejado adapter y ha cerrado la decision final de promocion como "
                    f"{promotion.get('decision') or 'pending'}."
                ),
                state_name="learning",
                metadata={
                    "run_id": run_id,
                    "mode": mode,
                    "eval_ok": eval_ok,
                    "bench_ok": bench_ok,
                    "decision": promotion.get("decision"),
                },
            )
        except Exception as exc:
            failure = {"reason": str(exc), "stage": "exception"}
            self._update_run_meta(
                run_id,
                {
                    "status": "degraded",
                    "stage": "exception",
                    "lifecycle_state": "degraded",
                    "progress_pct": 1.0,
                    "failure": failure,
                    "error": str(exc),
                    "terminal_reason": str(exc),
                },
            )
            self._append_notebook_section(
                run_id,
                phase="degraded",
                title="Excepcion del ciclo",
                content="El run ha terminado con una excepcion recuperable y se marca como degradado para permitir un siguiente intento correctivo.",
                kind="exception",
                metadata=failure,
            )
            self._append_run_event(
                run_id,
                phase="failed",
                message=str(exc),
                kind="failure",
                progress_pct=1.0,
                metadata=failure,
            )
            self._append_autonomy_event(
                agent="system",
                kind="training_exception",
                title="Run interrumpido",
                detail=f"El entrenamiento ha terminado con una excepcion recuperable: {exc}",
                state_name="training",
                metadata={"run_id": run_id, "mode": mode},
            )
        finally:
            if fallback_started:
                try:
                    self._stop_fallback_runtime()
                except Exception:
                    pass
            if mode == "full" and not runtime_resumed and not parallel_runtime_training:
                try:
                    self._resume_runtime_stack(
                        log_path=(self._run_dir(run_id) / "runtime.log"),
                        force_recreate=True,
                    )
                except Exception as exc:
                    self._update_run_meta(
                        run_id,
                        {
                            "status": "degraded",
                            "stage": "runtime_resume_failed",
                            "lifecycle_state": "degraded",
                            "runtime_resume_error": str(exc),
                            "runtime_tail": _tail(self._run_dir(run_id) / "runtime.log"),
                            "failure": {"reason": str(exc), "stage": "runtime_resume_failed"},
                            "terminal_reason": str(exc),
                            "progress_pct": 1.0,
                        },
                    )
            self._write_runtime_state({"mode": "primary", "fallback_active": False, "fallback_backend": None, "fallback_pid": None})
            with self._lock:
                self._active_run_id = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vortex local control service")
    parser.add_argument("--base-dir", default=".", help="Backend repo root")
    parser.add_argument("--compose-file", default=None, help="Path to docker-compose.yml")
    parser.add_argument("--port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--runtime-port", type=int, default=DEFAULT_RUNTIME_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--runtime-url", default=None)
    parser.add_argument("--frontend-url", default=None)
    parser.add_argument("--api-profile", default=DEFAULT_API_PROFILE)
    parser.add_argument("--training-profile", default=DEFAULT_TRAINING_PROFILE)
    parser.add_argument("--disable-compose-actions", action="store_true")
    parser.add_argument("--assume-docker-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()
    compose_file = Path(args.compose_file).resolve() if args.compose_file else (base_dir / "docker-compose.yml").resolve()
    state = ControlState(
        base_dir=base_dir,
        compose_file=compose_file,
        api_profile=str(args.api_profile),
        training_profile=str(args.training_profile),
        api_url=str(args.api_url or f"http://127.0.0.1:{int(args.api_port)}"),
        runtime_url=str(args.runtime_url or f"http://127.0.0.1:{int(args.runtime_port)}"),
        frontend_port=int(args.frontend_port),
        frontend_url=str(args.frontend_url or f"http://127.0.0.1:{int(args.frontend_port)}"),
        compose_actions_enabled=not bool(args.disable_compose_actions),
        assume_docker_ready=bool(args.assume_docker_ready),
    )
    app = create_control_app(ControlDependencies.from_state(state))
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
