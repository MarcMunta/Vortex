from __future__ import annotations

# pylint: disable=broad-exception-caught
# ruff: noqa: BLE001

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests


@dataclass(frozen=True)
class ExternalEngineConfig:
    engine: str
    base_url: str
    model: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    autostart: bool = False
    start_command: str | None = None
    start_workdir: str | None = None
    startup_wait_s: float = 2.0
    retry_max_attempts: int = 2
    retry_backoff_base_s: float = 0.25
    retry_backoff_max_s: float = 2.0
    retry_on_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    circuit_enabled: bool = True
    circuit_fail_threshold: int = 4
    circuit_open_s: float = 20.0


def _extract_text(resp: dict) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message")
    if isinstance(msg, dict) and msg.get("content") is not None:
        return str(msg.get("content") or "")
    if first.get("text") is not None:
        return str(first.get("text") or "")
    return ""


def _extract_delta(evt: dict) -> str:
    choices = evt.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta.get("content") or "")
    if first.get("text") is not None:
        return str(first.get("text") or "")
    return ""


def _normalize_messages(messages: Iterable[dict[str, Any]] | None, prompt: str | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if messages:
        for raw in messages:
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "user").strip().lower() or "user"
            content = raw.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text is not None:
                            parts.append(str(text))
                    elif item is not None:
                        parts.append(str(item))
                content_text = "".join(parts).strip()
            else:
                content_text = str(content or "").strip()
            if not content_text:
                continue
            normalized.append({"role": role, "content": content_text})
    if normalized:
        return normalized
    return [{"role": "user", "content": str(prompt or "")}]


class ExternalEngineModel:
    is_external = True
    is_hf = False
    is_llama_cpp = False

    def __init__(self, cfg: ExternalEngineConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._requests_total = 0
        self._requests_failed = 0
        self._requests_succeeded = 0
        self._retries_total = 0
        self._consecutive_failures = 0
        self._circuit_open_until_ts = 0.0
        self._last_error: str | None = None

    def _is_retryable_status(self, status_code: int) -> bool:
        if status_code in set(self.cfg.retry_on_statuses):
            return True
        return int(status_code) >= 500

    def _is_retryable_exception(self, exc: Exception) -> bool:
        return isinstance(exc, requests.RequestException)

    def _backoff_sleep(self, attempt: int) -> None:
        base = max(0.0, float(self.cfg.retry_backoff_base_s))
        cap = max(base, float(self.cfg.retry_backoff_max_s))
        delay = min(cap, base * (2 ** max(0, int(attempt) - 1)))
        if delay > 0:
            time.sleep(delay)

    def _record_attempt(self) -> None:
        with self._lock:
            self._requests_total += 1

    def _record_retry(self) -> None:
        with self._lock:
            self._retries_total += 1

    def _record_success(self) -> None:
        with self._lock:
            self._requests_succeeded += 1
            self._consecutive_failures = 0
            self._last_error = None

    def _record_failure(self, exc: Exception) -> None:
        now = time.time()
        with self._lock:
            self._requests_failed += 1
            self._consecutive_failures += 1
            self._last_error = str(exc)
            if (
                bool(self.cfg.circuit_enabled)
                and self._consecutive_failures >= max(1, int(self.cfg.circuit_fail_threshold))
            ):
                self._circuit_open_until_ts = now + max(0.0, float(self.cfg.circuit_open_s))

    def _ensure_circuit_closed(self) -> None:
        if not bool(self.cfg.circuit_enabled):
            return
        now = time.time()
        with self._lock:
            if now < self._circuit_open_until_ts:
                raise RuntimeError("external_engine_circuit_open")

    def runtime_stats(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            open_for_s = max(0.0, float(self._circuit_open_until_ts - now))
            return {
                "requests_total": int(self._requests_total),
                "requests_failed": int(self._requests_failed),
                "requests_succeeded": int(self._requests_succeeded),
                "retries_total": int(self._retries_total),
                "consecutive_failures": int(self._consecutive_failures),
                "circuit_open": bool(open_for_s > 0.0),
                "circuit_open_for_s": float(open_for_s),
                "last_error": self._last_error,
            }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.engine in {"ollama", "lmstudio"}:
            return headers
        key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or os.environ.get("SGLANG_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _url(self, path: str) -> str:
        base = self.cfg.base_url.rstrip("/")
        suffix = str(path)
        if base.endswith("/v1") and suffix.startswith("/v1/"):
            suffix = suffix[3:]
        return base + suffix

    def _ensure_started(self) -> None:
        if not bool(self.cfg.autostart):
            return
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = str(self.cfg.start_command or "").strip()
        if not cmd:
            raise RuntimeError("external_engine_autostart_enabled_but_no_start_command")
        workdir = str(self.cfg.start_workdir) if self.cfg.start_workdir else None
        self._proc = subprocess.Popen(cmd, shell=True, cwd=workdir)
        time.sleep(max(0.0, float(self.cfg.startup_wait_s)))

    def generate(self, prompt: str, *, max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 1.0, **_kwargs) -> str:
        self._ensure_started()
        messages = _normalize_messages(_kwargs.get("messages"), prompt)
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if self.cfg.model:
            payload["model"] = str(self.cfg.model)
        url = self._url("/v1/chat/completions")
        attempts = max(1, int(self.cfg.retry_max_attempts))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            self._ensure_circuit_closed()
            self._record_attempt()
            try:
                resp = self.session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=float(self.cfg.timeout_s),
                )
                status = int(resp.status_code)
                if self._is_retryable_status(status) and attempt < attempts:
                    self._record_failure(RuntimeError(f"http_status_{status}"))
                    self._record_retry()
                    self._backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._record_success()
                if not isinstance(data, dict):
                    return ""
                return _extract_text(data)
            except Exception as exc:
                self._record_failure(exc)
                last_exc = exc
                if attempt < attempts and self._is_retryable_exception(exc):
                    self._record_retry()
                    self._backoff_sleep(attempt)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("external_engine_request_failed")

    def _open_stream_response(self, payload: dict[str, Any]) -> requests.Response:
        url = self._url("/v1/chat/completions")
        attempts = max(1, int(self.cfg.retry_max_attempts))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            self._ensure_circuit_closed()
            self._record_attempt()
            try:
                resp = self.session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=float(self.cfg.timeout_s),
                    stream=True,
                )
                status = int(resp.status_code)
                if self._is_retryable_status(status) and attempt < attempts:
                    self._record_failure(RuntimeError(f"http_status_{status}"))
                    self._record_retry()
                    try:
                        resp.close()
                    except Exception:
                        pass
                    self._backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                self._record_success()
                return resp
            except Exception as exc:
                self._record_failure(exc)
                last_exc = exc
                if attempt < attempts and self._is_retryable_exception(exc):
                    self._record_retry()
                    self._backoff_sleep(attempt)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("external_engine_request_failed")

    def stream_generate(self, prompt: str, *, max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 1.0, **_kwargs) -> Iterable[str]:
        self._ensure_started()
        messages = _normalize_messages(_kwargs.get("messages"), prompt)
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        if self.cfg.model:
            payload["model"] = str(self.cfg.model)
        with self._open_stream_response(payload) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                if not isinstance(evt, dict):
                    continue
                delta = _extract_delta(evt)
                if delta:
                    yield delta


def load_external_engine_model(settings: dict) -> ExternalEngineModel:
    core = settings.get("core", {}) or {}
    backend = str(core.get("backend", "external") or "external").strip().lower()
    engine = str(core.get("external_engine") or core.get("engine") or "").strip().lower()
    if backend in {"vllm"}:
        engine = backend
        backend = "external"
    if not engine:
        engine = "external"
    base_url = str(core.get("external_base_url") or core.get("external_url") or "http://127.0.0.1:30000").strip()
    model = core.get("external_model") or core.get("hf_model")
    api_key = core.get("external_api_key")
    timeout_s = float(core.get("external_timeout_s", 60.0) or 60.0)
    autostart = bool(core.get("external_autostart", False))
    start_command = core.get("external_start_command")
    workdir = core.get("external_start_workdir") or core.get("external_workdir")
    startup_wait_s = float(core.get("external_startup_wait_s", 2.0) or 2.0)
    retry_max_attempts = int(core.get("external_retry_max_attempts", 2) or 2)
    retry_backoff_base_s = float(core.get("external_retry_backoff_base_s", 0.25) or 0.25)
    retry_backoff_max_s = float(core.get("external_retry_backoff_max_s", 2.0) or 2.0)
    retry_on_statuses_raw = core.get("external_retry_on_statuses", [408, 429, 500, 502, 503, 504])
    retry_on_statuses: tuple[int, ...]
    if isinstance(retry_on_statuses_raw, list):
        parsed = []
        for item in retry_on_statuses_raw:
            try:
                parsed.append(int(item))
            except Exception:
                continue
        retry_on_statuses = tuple(parsed) if parsed else (408, 429, 500, 502, 503, 504)
    else:
        retry_on_statuses = (408, 429, 500, 502, 503, 504)
    circuit_enabled = bool(core.get("external_circuit_enabled", True))
    circuit_fail_threshold = int(core.get("external_circuit_fail_threshold", 4) or 4)
    circuit_open_s = float(core.get("external_circuit_open_s", 20.0) or 20.0)
    cfg = ExternalEngineConfig(
        engine=str(engine),
        base_url=base_url,
        model=str(model) if model else None,
        api_key=str(api_key) if api_key else None,
        timeout_s=float(timeout_s),
        autostart=bool(autostart),
        start_command=str(start_command) if start_command else None,
        start_workdir=str(workdir) if workdir else None,
        startup_wait_s=float(startup_wait_s),
        retry_max_attempts=max(1, int(retry_max_attempts)),
        retry_backoff_base_s=max(0.0, float(retry_backoff_base_s)),
        retry_backoff_max_s=max(0.0, float(retry_backoff_max_s)),
        retry_on_statuses=retry_on_statuses,
        circuit_enabled=bool(circuit_enabled),
        circuit_fail_threshold=max(1, int(circuit_fail_threshold)),
        circuit_open_s=max(0.0, float(circuit_open_s)),
    )
    return ExternalEngineModel(cfg)
