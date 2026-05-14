from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml

from .agent.sandbox import run_sandbox_command


DEFAULT_TRACK = "python_fastapi_react"
REQUIRED_LOCAL_MODELS = [
    "qwen2.5-coder:14b-instruct-q4_K_S",
    "qwen3:14b",
    "nomic-embed-text",
]


def _windows_host_paths() -> dict[str, str]:
    return {
        "ollama": r"D:\AI\ollama",
        "openwebui": r"D:\AI\openwebui",
        "workspaces": r"D:\AI\workspaces",
        "vault_learning": r"D:\Vault\learning",
        "cyber_range": r"D:\Labs\cyber-range",
    }


def _posix_host_paths() -> dict[str, str]:
    home = Path.home()
    return {
        "ollama": str(home / "AI" / "ollama"),
        "openwebui": str(home / "AI" / "openwebui"),
        "workspaces": str(home / "AI" / "workspaces"),
        "vault_learning": str(home / "Vault" / "learning"),
        "cyber_range": str(home / "Labs" / "cyber-range"),
    }


def _default_host_paths() -> dict[str, str]:
    return _windows_host_paths() if os.name == "nt" else _posix_host_paths()


def _path_from_setting(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def resolve_local_lab_settings(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(settings.get("local_lab", {}) or {})
    host_paths = _default_host_paths()
    raw_paths = cfg.get("host_paths", {}) or {}
    for key, value in raw_paths.items():
        if value:
            host_paths[str(key)] = str(value)
    cfg["host_paths"] = host_paths
    cfg.setdefault("enabled", False)
    cfg.setdefault("track", DEFAULT_TRACK)
    cfg.setdefault("curriculum_path", "config/local_lab_curriculum.yaml")
    cfg.setdefault(
        "progress_path", str(Path(host_paths["vault_learning"]) / "progress.json")
    )
    cfg.setdefault(
        "lessons_path", str(Path(host_paths["vault_learning"]) / "lessons")
    )
    cfg.setdefault("workspaces_path", host_paths["workspaces"])
    cfg.setdefault(
        "continue_config_path", str((base_dir / ".continue" / "config.yaml").resolve())
    )
    cfg.setdefault("sandbox_root", str((base_dir / "data" / "workspaces").resolve()))
    cfg.setdefault("guardrails_enabled", True)
    cfg.setdefault("lab_confirmation_token", "LAB_CONFIRMED")
    cfg.setdefault("devbox_image", "vortex-ai-devbox:latest")
    cfg.setdefault("devbox_workspace_mount", "/workspace")
    return cfg


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)
    if not isinstance(data, dict):
        return copy.deepcopy(default)
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _check_port(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _run_command(cmd: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        stdout = completed.stdout.replace("\x00", "").strip()
        stderr = completed.stderr.replace("\x00", "").strip()
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _windows_ollama_candidates() -> list[Path]:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    candidates: list[Path] = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    candidates.append(Path(r"C:\Program Files\Ollama\ollama.exe"))
    return [candidate for candidate in candidates if candidate.exists()]


def _detect_ollama_command() -> list[str] | None:
    resolved = shutil.which("ollama")
    if resolved:
        return [resolved]
    if os.name == "nt":
        candidates = _windows_ollama_candidates()
        if candidates:
            return [str(candidates[0])]
    return None


def _detect_ollama_runtime() -> str:
    command = _detect_ollama_command()
    if command:
        binary = str(command[0]).lower()
        if os.name == "nt" and binary.endswith("ollama.exe"):
            return "windows_host"
        return "shell_path"
    if _check_port(11434):
        return "api_only"
    return "unavailable"


def _ollama_tags_payload() -> dict[str, Any]:
    try:
        with urllib_request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib_error.URLError) as exc:
        return {"ok": False, "models": [], "error": str(exc)}
    models = payload.get("models", []) if isinstance(payload, dict) else []
    normalized: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("model") or model.get("name") or "").strip()
        if name:
            normalized.append(name)
    return {"ok": True, "models": normalized}


def _normalize_model_name(name: str) -> str:
    value = str(name or "").strip()
    return value[:-7] if value.endswith(":latest") else value


def _required_models_status() -> dict[str, Any]:
    payload = _ollama_tags_payload()
    available = payload.get("models", []) if isinstance(payload.get("models"), list) else []
    available_set = {_normalize_model_name(str(name)) for name in available}
    return {
        "ok": bool(payload.get("ok")),
        "required": list(REQUIRED_LOCAL_MODELS),
        "available": available,
        "missing": [
            model for model in REQUIRED_LOCAL_MODELS if _normalize_model_name(model) not in available_set
        ],
        "ready": all(_normalize_model_name(model) in available_set for model in REQUIRED_LOCAL_MODELS),
        "error": payload.get("error"),
    }


def _compose_path(base_dir: Path) -> Path:
    candidates = [
        (base_dir / "infra" / "local-lab" / "docker-compose.yml").resolve(),
        (base_dir.parent / "infra" / "local-lab" / "docker-compose.yml").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _repo_root(base_dir: Path) -> Path:
    candidates = [base_dir.parent.resolve(), base_dir.resolve()]
    for candidate in candidates:
        if (candidate / "infra" / "local-lab").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "infra").exists() or (candidate / "scripts").exists():
            return candidate
    return base_dir.resolve()


def _vault_root(cfg: dict[str, Any], base_dir: Path) -> Path:
    return Path(cfg["host_paths"]["vault_learning"]).resolve()


def _roadmap_path(cfg: dict[str, Any], base_dir: Path) -> Path:
    return _vault_root(cfg, base_dir) / "ROADMAP.md"


def _bootstrap_plan_path(cfg: dict[str, Any], base_dir: Path) -> Path:
    return _vault_root(cfg, base_dir) / "BOOTSTRAP_PLAN.md"


def _rag_sources_path(cfg: dict[str, Any], base_dir: Path) -> Path:
    return _vault_root(cfg, base_dir) / "rag_sources.json"


def ensure_host_layout(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    host_paths = cfg["host_paths"]
    created: list[str] = []
    touched: list[str] = []

    for key in ("ollama", "openwebui", "workspaces", "vault_learning", "cyber_range"):
        path = Path(host_paths[key])
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))

    openwebui_root = Path(host_paths["openwebui"])
    for child in ("data", "backups", "logs"):
        target = openwebui_root / child
        target.mkdir(parents=True, exist_ok=True)
        touched.append(str(target))

    lessons_root = Path(cfg["lessons_path"])
    lessons_root.mkdir(parents=True, exist_ok=True)
    touched.append(str(lessons_root))

    progress_path = _path_from_setting(cfg["progress_path"], base_dir=base_dir)
    if not progress_path.exists():
        _write_json(
            progress_path,
            {
                "version": 1,
                "track": cfg["track"],
                "updated_at": time.time(),
                "lessons": [],
            },
        )
        touched.append(str(progress_path))

    range_policy = Path(host_paths["cyber_range"]) / "LAB_POLICY.md"
    if not range_policy.exists():
        range_policy.write_text(
            "\n".join(
                [
                    "# Cyber Range Policy",
                    "",
                    "- This environment is for isolated, lab-owned targets only.",
                    "- No production, third-party, or public targets.",
                    "- Offensive exercises require explicit lab confirmation.",
                    "- Every exercise must finish with remediation notes.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        touched.append(str(range_policy))

    continue_path = _path_from_setting(cfg["continue_config_path"], base_dir=base_dir)
    write_continue_config(settings, base_dir, continue_path=continue_path)
    touched.append(str(continue_path))
    roadmap = write_roadmap(settings, base_dir)
    rag_sources = write_rag_sources_manifest(settings, base_dir)
    bootstrap = write_bootstrap_plan(settings, base_dir)
    touched.extend([str(roadmap["path"]), str(rag_sources["path"]), str(bootstrap["path"])])

    return {
        "ok": True,
        "created": created,
        "touched": touched,
        "progress_path": str(progress_path),
        "continue_config_path": str(continue_path),
    }


def collect_local_lab_status(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    host_paths = cfg["host_paths"]
    dirs = {
        name: {"path": path, "exists": Path(path).exists()}
        for name, path in host_paths.items()
    }

    docker_info = _run_command(["docker", "info", "--format", "{{json .ServerVersion}}"])
    wsl_info = (
        _run_command(["wsl", "-l", "-v"])
        if os.name == "nt"
        else {"ok": False, "stderr": "not_windows"}
    )
    ollama_command = _detect_ollama_command()
    ollama_cmd = (
        _run_command([*ollama_command, "--version"])
        if ollama_command
        else {"ok": False, "returncode": -1, "stdout": "", "stderr": "ollama_not_found"}
    )
    code_cmd = _run_command(["code", "--version"])
    compose_path = _compose_path(base_dir)
    ollama_models = _required_models_status()

    continue_path = _path_from_setting(cfg["continue_config_path"], base_dir=base_dir)
    progress_path = _path_from_setting(cfg["progress_path"], base_dir=base_dir)

    return {
        "ok": True,
        "track": cfg["track"],
        "host_paths": dirs,
        "next_module": next_module(settings, base_dir),
        "commands": {
            "docker": docker_info,
            "wsl": wsl_info,
            "ollama": {
                **ollama_cmd,
                "command": ollama_command[0] if ollama_command else None,
                "runtime": _detect_ollama_runtime(),
            },
            "code": code_cmd,
        },
        "services": {
            "ollama_11434": _check_port(11434),
            "openwebui_3000": _check_port(3000),
        },
        "ollama_models": ollama_models,
        "continue_config": {
            "path": str(continue_path),
            "exists": continue_path.exists(),
        },
        "progress": {
            "path": str(progress_path),
            "exists": progress_path.exists(),
        },
        "docker_compose": {
            "path": str(compose_path),
            "exists": compose_path.exists(),
        },
    }


def write_continue_config(
    settings: dict,
    base_dir: Path,
    *,
    continue_path: Path | None = None,
) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    path = continue_path or _path_from_setting(cfg["continue_config_path"], base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "name: Vortex Local Lab",
            "version: 0.0.1",
            "schema: v1",
            "",
            "models:",
            "  - name: Llama 2 Local",
            "    provider: openai",
            "    model: llama-2-7b-chat.Q4_K_M.gguf",
            "    apiBase: http://127.0.0.1:8000/v1",
            "    roles: [chat, edit, apply]",
            "    capabilities: [tool_use]",
            "  - name: Nomic Embed Text",
            "    provider: ollama",
            "    model: nomic-embed-text",
            "    apiBase: http://127.0.0.1:11434",
            "    roles: [embed]",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(path)}


def _curriculum_path(settings: dict, base_dir: Path) -> Path:
    cfg = resolve_local_lab_settings(settings, base_dir)
    return _path_from_setting(cfg["curriculum_path"], base_dir=base_dir)


def load_curriculum(settings: dict, base_dir: Path) -> dict[str, Any]:
    path = _curriculum_path(settings, base_dir)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("curriculum_invalid")
    return raw


def list_modules(settings: dict, base_dir: Path) -> list[dict[str, Any]]:
    curriculum = load_curriculum(settings, base_dir)
    modules = curriculum.get("modules", []) or []
    return [module for module in modules if isinstance(module, dict)]


def next_module(settings: dict, base_dir: Path) -> dict[str, Any]:
    modules = list_modules(settings, base_dir)
    progress = load_progress(settings, base_dir)
    lessons = progress.get("lessons", []) if isinstance(progress.get("lessons"), list) else []
    status_by_module: dict[str, dict[str, Any]] = {}
    for lesson in lessons:
        if not isinstance(lesson, dict):
            continue
        module_id = str(lesson.get("module_id") or "").strip()
        if not module_id:
            continue
        current = status_by_module.get(module_id)
        if current is None or float(lesson.get("updated_at") or 0) >= float(current.get("updated_at") or 0):
            status_by_module[module_id] = lesson

    for module in modules:
        module_id = str(module.get("id") or "").strip()
        latest = status_by_module.get(module_id)
        status = str((latest or {}).get("status") or "not_started")
        if status != "passed":
            return {
                "ok": True,
                "module": module,
                "status": status,
                "workspace": (latest or {}).get("workspace"),
                "reason": "resume_existing" if latest else "start_next_in_sequence",
            }
    return {
        "ok": True,
        "module": None,
        "status": "completed",
        "workspace": None,
        "reason": "all_modules_passed",
    }


def _phase_name(weeks: str) -> str:
    value = str(weeks or "").strip()
    try:
        start = int(value.split("-")[0])
    except Exception:
        start = 999
    if start < 8:
        return "Foundations"
    if start < 18:
        return "Full-Stack"
    return "Security"


def write_roadmap(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    modules = list_modules(settings, base_dir)
    progress = load_progress(settings, base_dir)
    next_item = next_module(settings, base_dir)
    latest_by_module: dict[str, dict[str, Any]] = {}
    for lesson in progress.get("lessons", []) if isinstance(progress.get("lessons"), list) else []:
        if not isinstance(lesson, dict):
            continue
        module_id = str(lesson.get("module_id") or "").strip()
        if not module_id:
            continue
        current = latest_by_module.get(module_id)
        if current is None or float(lesson.get("updated_at") or 0) >= float(current.get("updated_at") or 0):
            latest_by_module[module_id] = lesson

    lines = [
        "# Local Learning Roadmap",
        "",
        f"- Track: `{cfg['track']}`",
        f"- Updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    next_module_id = str(((next_item.get("module") or {}) if isinstance(next_item.get("module"), dict) else {}).get("id") or "")
    if next_module_id:
        lines.append(f"- Next module: `{next_module_id}`")
        lines.append("")

    current_phase = None
    for module in modules:
        phase = _phase_name(str(module.get("weeks") or ""))
        if phase != current_phase:
            current_phase = phase
            lines.extend([f"## {phase}", ""])
        module_id = str(module.get("id") or "")
        latest = latest_by_module.get(module_id, {})
        status = str(latest.get("status") or "not_started")
        title = str(module.get("title") or module_id)
        lines.append(f"- `{module_id}` [{status}] {title}")
    path = _roadmap_path(cfg, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"ok": True, "path": str(path), "next_module": next_item}


def write_rag_sources_manifest(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    payload = {
        "version": 1,
        "track": cfg["track"],
        "sources": [
            {"topic": "python", "url": "https://docs.python.org/3/tutorial/"},
            {"topic": "fastapi", "url": "https://fastapi.tiangolo.com/tutorial/"},
            {"topic": "react", "url": "https://react.dev/learn"},
            {"topic": "http", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview"},
            {"topic": "sql", "url": "https://www.postgresql.org/docs/current/tutorial-start.html"},
            {"topic": "git", "url": "https://git-scm.com/doc"},
            {"topic": "linux", "url": "https://www.gnu.org/software/bash/manual/bash.html"},
            {"topic": "owasp", "url": "https://cheatsheetseries.owasp.org/"},
            {"topic": "nist", "url": "https://www.nist.gov/cyberframework"},
        ],
    }
    path = _rag_sources_path(cfg, base_dir)
    _write_json(path, payload)
    return {"ok": True, "path": str(path), "count": len(payload["sources"])}


def write_bootstrap_plan(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    status = collect_local_lab_status(settings, base_dir)
    repo_root = _repo_root(base_dir)
    bootstrap_script = repo_root / "scripts" / "bootstrap_ollama_wsl.ps1"
    bootstrap_windows_script = repo_root / "scripts" / "bootstrap_ollama_windows.ps1"
    pull_models_script = repo_root / "scripts" / "pull_local_lab_models.ps1"
    start_stack_script = repo_root / "scripts" / "start_local_stack.ps1"
    next_item = next_module(settings, base_dir)
    wsl_stdout = str(((status.get("commands") or {}).get("wsl") or {}).get("stdout") or "").lower()
    ubuntu_present = "ubuntu" in wsl_stdout
    ollama_runtime = str((((status.get("commands") or {}).get("ollama") or {}).get("runtime")) or "unavailable")
    ollama_ready = bool(((status.get("commands") or {}).get("ollama") or {}).get("ok")) or bool(
        status.get("services", {}).get("ollama_11434")
    )
    models_ready = bool((status.get("ollama_models") or {}).get("ready"))

    tasks = [
        {
            "id": "wsl_ubuntu",
            "title": "Install Ubuntu in WSL2",
            "status": "done" if ubuntu_present else "pending",
            "command": f"powershell -ExecutionPolicy Bypass -File {bootstrap_script} -InstallUbuntu",
            "reason": "Required host for Ollama on 127.0.0.1:11434.",
        },
        {
            "id": "ollama_install",
            "title": "Install Ollama runtime",
            "status": "done" if ollama_ready else "pending",
            "command": (
                f"powershell -ExecutionPolicy Bypass -File {bootstrap_windows_script} -StartOllama"
                if ollama_runtime == "windows_host"
                else f"powershell -ExecutionPolicy Bypass -File {bootstrap_script} -InstallOllama"
            ),
            "reason": f"Required local model runtime for tutor, builder, and embeddings. Current mode: {ollama_runtime}.",
        },
        {
            "id": "ollama_models",
            "title": "Pull required local models",
            "status": "done" if models_ready else "pending",
            "command": f"powershell -ExecutionPolicy Bypass -File {pull_models_script}",
            "reason": "Provision qwen2.5-coder, qwen3, and nomic-embed-text for the stack.",
        },
        {
            "id": "docker_desktop",
            "title": "Start Docker Desktop",
            "status": "done" if bool(((status.get('commands') or {}).get('docker') or {}).get('ok')) else "pending",
            "command": "Start Docker Desktop and wait until `docker info` succeeds.",
            "reason": "Required for Open WebUI and ai-devbox.",
        },
        {
            "id": "open_webui",
            "title": "Start Open WebUI on 127.0.0.1:3000",
            "status": "done" if bool(status.get("services", {}).get("openwebui_3000")) else "pending",
            "command": f"powershell -ExecutionPolicy Bypass -File {start_stack_script}",
            "reason": "Primary chat UI for local tutoring, RAG, and session memory.",
        },
        {
            "id": "next_module",
            "title": "Continue the next learning module",
            "status": "done" if str(next_item.get("status") or "") == "completed" else "pending",
            "command": f"python -m c3rnt2.cli local-lab lesson {str(((next_item.get('module') or {}) if isinstance(next_item.get('module'), dict) else {}).get('id') or 'python-basics')}",
            "reason": "Keep the curriculum moving with a single active lesson workspace.",
        },
    ]

    lines = [
        "# Local Stack Bootstrap Plan",
        "",
        f"- Track: `{cfg['track']}`",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    for task in tasks:
        lines.append(f"- [{task['status']}] `{task['id']}` {task['title']}")
        lines.append(f"  Command: `{task['command']}`")
        lines.append(f"  Why: {task['reason']}")
    path = _bootstrap_plan_path(cfg, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"ok": True, "path": str(path), "tasks": tasks}


def _template_dir(base_dir: Path, template_name: str) -> Path:
    return base_dir / "templates" / "local_lab" / template_name


def _render_lesson_markdown(module: dict[str, Any], lesson_id: str, workspace_root: Path) -> str:
    concepts = module.get("concepts", []) or []
    checkpoints = module.get("checkpoints", []) or []
    validation = module.get("validation", []) or []
    lines = [
        f"# {module.get('title', lesson_id)}",
        "",
        f"- Lesson ID: `{lesson_id}`",
        f"- Weeks: `{module.get('weeks', 'n/a')}`",
        f"- Workspace: `{workspace_root}`",
        "",
        str(module.get("summary", "")).strip(),
        "",
        "## Concepts",
    ]
    for concept in concepts:
        lines.append(f"- {concept}")
    if checkpoints:
        lines.extend(["", "## Checkpoints"])
        for item in checkpoints:
            lines.append(f"- {item}")
    if validation:
        lines.extend(["", "## Validation"])
        for item in validation:
            lines.append(f"- `{item}`")
    return "\n".join(lines).strip() + "\n"


def _update_progress(progress_path: Path, entry: dict[str, Any]) -> None:
    payload = _read_json(
        progress_path,
        {"version": 1, "track": DEFAULT_TRACK, "updated_at": time.time(), "lessons": []},
    )
    lessons = payload.get("lessons")
    if not isinstance(lessons, list):
        lessons = []
    replaced = False
    for idx, item in enumerate(lessons):
        if isinstance(item, dict) and item.get("lesson_id") == entry.get("lesson_id"):
            lessons[idx] = entry
            replaced = True
            break
    if not replaced:
        lessons.append(entry)
    payload["lessons"] = lessons
    payload["updated_at"] = time.time()
    _write_json(progress_path, payload)


def create_lesson(
    settings: dict,
    base_dir: Path,
    *,
    module_id: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    module = next(
        (item for item in list_modules(settings, base_dir) if item.get("id") == module_id),
        None,
    )
    if module is None:
        raise KeyError(f"module_not_found:{module_id}")

    lesson_id = f"{module_id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    workspace_base = (
        _path_from_setting(workspace_root, base_dir=base_dir)
        if workspace_root
        else _path_from_setting(cfg["workspaces_path"], base_dir=base_dir)
    )
    target_root = workspace_base / module_id / lesson_id
    target_root.parent.mkdir(parents=True, exist_ok=True)

    exercise = module.get("exercise", {}) or {}
    template_name = exercise.get("template")
    manifest: dict[str, Any] = {
        "version": 1,
        "lesson_id": lesson_id,
        "module_id": module_id,
        "track": cfg["track"],
        "created_at": time.time(),
        "status": "created",
        "workspace": str(target_root),
    }
    if template_name:
        source = _template_dir(base_dir, str(template_name))
        if source.exists():
            shutil.copytree(source, target_root)
            template_manifest_path = target_root / ".vortex-template.json"
            if template_manifest_path.exists():
                template_manifest = _read_json(template_manifest_path, {})
                manifest.update(template_manifest)
                template_manifest_path.unlink(missing_ok=True)
        else:
            target_root.mkdir(parents=True, exist_ok=True)
    else:
        target_root.mkdir(parents=True, exist_ok=True)

    lesson_markdown = _render_lesson_markdown(module, lesson_id, target_root)
    (target_root / "LESSON.md").write_text(lesson_markdown, encoding="utf-8")

    task_lines = [
        f"# Task: {module.get('title', module_id)}",
        "",
        str(exercise.get("objective") or module.get("summary") or "").strip(),
    ]
    acceptance = exercise.get("acceptance", []) or []
    if acceptance:
        task_lines.extend(["", "## Acceptance"])
        for item in acceptance:
            task_lines.append(f"- {item}")
    (target_root / "TASK.md").write_text("\n".join(task_lines).strip() + "\n", encoding="utf-8")

    manifest_path = target_root / ".vortex-lab.json"
    _write_json(manifest_path, manifest)

    lessons_path = _path_from_setting(cfg["lessons_path"], base_dir=base_dir)
    lesson_archive = lessons_path / f"{lesson_id}.json"
    _write_json(
        lesson_archive,
        {
            "lesson_id": lesson_id,
            "module": module,
            "workspace": str(target_root),
            "manifest_path": str(manifest_path),
        },
    )

    progress_path = _path_from_setting(cfg["progress_path"], base_dir=base_dir)
    _update_progress(
        progress_path,
        {
            "lesson_id": lesson_id,
            "module_id": module_id,
            "title": module.get("title"),
            "workspace": str(target_root),
            "status": "created",
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )

    return {
        "ok": True,
        "lesson_id": lesson_id,
        "module_id": module_id,
        "workspace": str(target_root),
        "manifest_path": str(manifest_path),
    }


def load_progress(settings: dict, base_dir: Path) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    progress_path = _path_from_setting(cfg["progress_path"], base_dir=base_dir)
    return _read_json(
        progress_path,
        {"version": 1, "track": cfg["track"], "updated_at": None, "lessons": []},
    )


def _review_markdown(result: dict[str, Any]) -> str:
    status = "PASS" if result.get("ok") else "FAIL"
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    lines = [f"# Review: {status}", ""]
    if stdout:
        lines.extend(["## stdout", "```text", stdout[:4000], "```", ""])
    if stderr:
        lines.extend(["## stderr", "```text", stderr[:4000], "```", ""])
    if not stdout and not stderr:
        lines.append("No test output was produced.")
    return "\n".join(lines).strip() + "\n"


def check_lesson(
    settings: dict,
    base_dir: Path,
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    cfg = resolve_local_lab_settings(settings, base_dir)
    workspace_root = _path_from_setting(workspace, base_dir=base_dir)
    manifest_path = workspace_root / ".vortex-lab.json"
    if not manifest_path.exists():
        raise FileNotFoundError("lesson_manifest_missing")
    manifest = _read_json(manifest_path, {})
    check_command = manifest.get("check_command") or ["pytest", "-q"]
    if not isinstance(check_command, list) or not check_command:
        raise ValueError("check_command_invalid")
    normalized_command = [str(item) for item in check_command]
    if normalized_command and str(normalized_command[0]).lower() == "pytest":
        normalized_command = [sys.executable, "-m", "pytest", *normalized_command[1:]]

    result = run_sandbox_command(
        workspace_root,
        normalized_command,
        _path_from_setting(cfg["sandbox_root"], base_dir=base_dir),
        timeout_s=300,
    )
    review_path = workspace_root / "REVIEW.md"
    review_path.write_text(_review_markdown(result), encoding="utf-8")
    review_json_path = workspace_root / "review.json"
    _write_json(
        review_json_path,
        {
            "ok": bool(result.get("ok")),
            "returncode": result.get("returncode"),
            "stdout": str(result.get("stdout") or "")[:8000],
            "stderr": str(result.get("stderr") or "")[:8000],
            "checked_at": time.time(),
        },
    )

    manifest["status"] = "passed" if result.get("ok") else "failed"
    manifest["checked_at"] = time.time()
    _write_json(manifest_path, manifest)

    progress_path = _path_from_setting(cfg["progress_path"], base_dir=base_dir)
    _update_progress(
        progress_path,
        {
            "lesson_id": manifest.get("lesson_id"),
            "module_id": manifest.get("module_id"),
            "title": manifest.get("module_id"),
            "workspace": str(workspace_root),
            "status": manifest["status"],
            "created_at": manifest.get("created_at"),
            "updated_at": time.time(),
            "last_check_ok": bool(result.get("ok")),
            "review_path": str(review_path),
        },
    )

    return {
        "ok": bool(result.get("ok")),
        "workspace": str(workspace_root),
        "review_path": str(review_path),
        "sandbox": result.get("sandbox"),
        "stdout": str(result.get("stdout") or "")[:4000],
        "stderr": str(result.get("stderr") or "")[:4000],
    }
