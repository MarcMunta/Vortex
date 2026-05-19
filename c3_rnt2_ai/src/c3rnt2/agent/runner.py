from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from ..config import resolve_web_allowlist
from ..context_budget import (
    apply_message_budget,
    estimate_tokens,
    output_limit_for_mode,
    resolve_context_budget,
    resolve_model_context_limit,
)
from ..lab_guard import evaluate_lab_request
from ..model_loader import load_inference_model
from ..prompting.chat_format import build_chat_prompt
from .grammar import build_agent_action_json_grammar
from .permissions import AgentPermissions, build_agent_permission_context
from .tools import AgentTools, ToolResult


@dataclass
class Action:
    type: str
    args: dict


def _parse_action(text: str) -> tuple[Action, bool]:
    text = text.strip()
    if not text:
        return Action(type="finish", args={"summary": "empty"}), False

    # Strip common markdown wrappers that weak models add
    cleaned = text
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    payload: Any | None = None
    for start in [index for index, char in enumerate(cleaned) if char == "{"]:
        candidate = cleaned[start:].lstrip()
        try:
            parsed, _end = decoder.raw_decode(candidate)
        except Exception:
            # Try to fix common issues: unclosed braces, trailing comma
            try:
                fixed = candidate.rstrip().rstrip(",")
                open_braces = fixed.count("{") - fixed.count("}")
                if open_braces > 0:
                    fixed = fixed + "}" * open_braces
                parsed = json.loads(fixed)
            except Exception:
                continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    if payload is None:
        # Last resort: try to detect action intent from prose
        lower = cleaned.lower()
        if "write_file" in lower or "write file" in lower:
            return Action(type="finish", args={"summary": "invalid_json_but_wants_write"}), False
        if "list_tree" in lower or "list tree" in lower or "inspect" in lower:
            return Action(type="list_tree", args={"root": ".", "max_entries": 100}), True
        if "read_file" in lower or "read file" in lower:
            path_match = re.search(r"(?:read_file|read file)[^\"]*\"([^\"]+)\"", lower)
            if path_match:
                return Action(type="read_file", args={"path": path_match.group(1)}), True
        return Action(type="finish", args={"summary": "invalid_json"}), False
    action_type = str(payload.get("type") or payload.get("action") or "finish")
    args = payload.get("args", {}) or {}
    if not isinstance(args, dict):
        args = {}
    if action_type in {"create_file", "update_file", "edit_file"}:
        action_type = "write_file"
    if action_type == "write_file":
        args = dict(args)
        if not args.get("path"):
            args["path"] = payload.get("path") or payload.get("file") or payload.get("filename") or ""
        if not args.get("text"):
            args["text"] = args.get("content") or payload.get("content") or payload.get("text") or ""
        args.pop("content", None)
    return Action(type=action_type, args=args), True


def _cfg_int(cfg: dict, key: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(cfg.get(key, default))
    except Exception:
        value = default
    return max(minimum, value)


def _cfg_float(cfg: dict, key: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(cfg.get(key, default))
    except Exception:
        value = default
    return max(minimum, value)


def _cfg_nonnegative_int(cfg: dict, key: str, default: int) -> int:
    try:
        value = int(cfg.get(key, default))
    except Exception:
        value = default
    return max(0, value)


def _resolve_queue_dir(workspace_dir: Path, settings: dict) -> Path:
    queue_dir = settings.get("self_patch", {}).get("queue_dir", "data/self_patch/queue")
    qpath = Path(queue_dir)
    if not qpath.is_absolute():
        qpath = workspace_dir / qpath
    return qpath


def _load_patch_from_queue(workspace_dir: Path, settings: dict, patch_id: str | None) -> str:
    if not patch_id:
        return ""
    queue_dir = _resolve_queue_dir(workspace_dir, settings)
    patch_path = queue_dir / patch_id / "patch.diff"
    if not patch_path.exists():
        return ""
    try:
        return patch_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _build_prompt(task: str, tool_calls: List[dict], *, max_chars: int = 2400, max_tool_chars: int = 800, max_tools: int = 3) -> str:
    parts = [f"Current task to execute now (do not ask for another task):\n{task}".strip()]
    if tool_calls:
        for call in tool_calls[-max_tools:]:
            output = str(call.get("output", "")).strip()
            if not output:
                continue
            if len(output) > max_tool_chars:
                output = output[:max_tool_chars].rstrip() + "..."
            action = str(call.get("action", "tool"))
            parts.append(f"{action} output:\n{output}")
    prompt = "\n\n".join([p for p in parts if p])
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars].rstrip() + "..."
    return prompt


def _clip_middle(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    head = max(200, max_chars // 2)
    tail = max(200, max_chars - head - 80)
    return (
        value[:head].rstrip()
        + "\n...[context compacted]...\n"
        + value[-tail:].lstrip()
    )


def _estimate_model_tokens(text: str, model: object | None = None) -> int:
    if model is not None and hasattr(model, "encode_prompt"):
        try:
            encoded = model.encode_prompt(text)  # type: ignore[attr-defined]
            if isinstance(encoded, tuple) and len(encoded) >= 2:
                return max(1, int(encoded[1]))
            if isinstance(encoded, list):
                return max(1, len(encoded))
        except Exception:
            pass
    return estimate_tokens(text)


def _is_context_window_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "requested tokens",
        "exceed context window",
        "context window",
        "context length",
        "maximum context",
        "n_ctx",
        "too many tokens",
        "prompt is too long",
    )
    return any(marker in text for marker in markers)


def _compact_agent_messages(
    *,
    system_prompt: str,
    task: str,
    tool_calls: List[dict],
    reason: str,
    max_chars: int = 6000,
    max_tool_chars: int = 900,
    max_tools: int = 12,
) -> List[dict]:
    objective = _clip_middle(_extract_objective_text(task) or str(task or ""), max(1200, max_chars // 2))
    task_excerpt = _clip_middle(str(task or ""), max(1200, max_chars // 3))
    lines = [
        f"Agent context compacted after {reason}.",
        "Continue the same task from current workspace state. Do not restart completed work.",
        "Inspect files again when needed, make required changes, validate when practical, then finish.",
        f"Original objective: {objective}",
    ]
    if task_excerpt and task_excerpt.strip() != objective.strip():
        lines.append(f"Compacted prior request/context excerpt:\n{task_excerpt}")
    if tool_calls:
        lines.append("Recent tool state:")
        for call in tool_calls[-max_tools:]:
            output = str(call.get("output", "")).strip()
            if len(output) > max_tool_chars:
                output = output[:max_tool_chars].rstrip() + "..."
            lines.append(
                "- "
                f"{call.get('action', 'tool')} "
                f"ok={bool(call.get('ok'))} "
                f"args={json.dumps(call.get('args', {}), ensure_ascii=True)[:500]} "
                f"output={output}"
            )
    compacted = "\n".join(lines)
    if len(compacted) > max_chars:
        compacted = compacted[-max_chars:]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current task to execute now. Do not ask for another task.\n{objective}"},
        {"role": "system", "content": compacted},
    ]


def _summary_needs_fallback(summary: str) -> bool:
    normalized = str(summary or "").strip().lower()
    return normalized in {
        "",
        "agent_finished",
        "empty",
        "finished",
        "invalid_json",
        "stopped_by_context_compaction_limit",
        "stopped_by_wall_time_limit",
    }


def _summary_looks_like_model_chatter(summary: str) -> bool:
    normalized = str(summary or "").strip().lower()
    if not normalized:
        return True
    return normalized.startswith(
        (
            "sure",
            "here is",
            "here's",
            "of course",
            "understood",
            "okay",
            "ok,",
            "final answer",
        )
    ) or "final answer" in normalized


def _summary_asks_for_confirmation(summary: str) -> bool:
    normalized = _normalish(summary)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "do you want",
            "quieres que",
            "dime si quieres",
            "please provide",
            "necesito que",
            "yes/no",
        )
    )


def _summary_is_model_refusal(summary: str) -> bool:
    normalized = _normalish(summary)
    if not normalized:
        return False
    refusal_markers = (
        "i apologize",
        "i cannot",
        "can't fulfill",
        "cannot fulfill",
        "ethical and moral",
        "harmful or unethical",
        "goes against",
        "no puedo cumplir",
        "no puedo ayudar",
        "principios eticos",
        "contenido danino",
    )
    return any(marker in normalized for marker in refusal_markers)


def _task_mentions_any(task: str, words: set[str]) -> bool:
    normalized = _normalish(task)
    return any(word in normalized for word in words)


def _task_requests_code_file(task: str) -> bool:
    return _task_mentions_any(
        task,
        {
            "archivo",
            "file",
            "crea",
            "crear",
            "create",
            "implementa",
            "implementar",
            "programa",
            "programar",
            "app",
            "codigo",
            "code",
            "flutter",
            "dart",
            "login",
            "pantalla",
            "screen",
            "widget",
            "componente",
            "component",
            "pagina",
            "page",
            "vista",
            "view",
            "layout",
            "diseno",
            "design",
            "interfaz",
            "interface",
            "ui",
            "ux",
            "formulario",
            "form",
            "boton",
            "button",
            "menu",
            "navbar",
            "sidebar",
            "drawer",
            "modal",
            "dialog",
            "tabla",
            "table",
            "lista",
            "list",
            "grid",
            "card",
            "tarjeta",
            "animacion",
            "animation",
            "api",
            "servicio",
            "service",
            "modelo",
            "model",
            "clase",
            "class",
            "funcion",
            "function",
            "script",
            "html",
            "css",
            "javascript",
            "typescript",
            "react",
            "vue",
            "angular",
            "python",
            "java",
            "kotlin",
            "swift",
            "rust",
            "go",
            "genera",
            "generar",
            "generate",
            "escribe",
            "escribir",
            "write",
            "hazme",
            "hacer",
            "haz",
            "build",
            "scaffold",
            "template",
            "plantilla",
            "proyecto",
            "project",
        },
    )


def _task_requires_workspace_change(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    # NOTE: Do NOT include generic nouns like 'proyecto', 'project', 'app'
    # here because they also appear in run/execute contexts ("Ejecuta el
    # proyecto") where no file mutation is expected.
    return any(
        word in normalized
        for word in (
            "anade",
            "agrega",
            "add",
            "boton",
            "button",
            "cambia",
            "cambiar",
            "codigo",
            "crea",
            "crear",
            "create",
            "delete",
            "edita",
            "editar",
            "elimina",
            "escribe",
            "escribir",
            "genera",
            "generar",
            "generate",
            "haz",
            "hazme",
            "hacer",
            "implementa",
            "implementar",
            "modifica",
            "modificar",
            "programa",
            "programar",
            "remove",
            "update",
            "write",
            "pantalla",
            "screen",
            "widget",
            "componente",
            "pagina",
            "page",
            "vista",
            "view",
            "layout",
            "disena",
            "design",
            "interfaz",
            "interface",
            "formulario",
            "form",
            "login",
            "signup",
            "register",
            "dashboard",
            "scaffold",
        )
    )


def _task_requires_command_activity(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    if any(word in normalized for word in ("comando", "command", "emulador", "emulator", "terminal")):
        return True
    if re.search(r"\b(corre|correr|ejecuta|ejecutalo|ejecutarlo|inicia|iniciar|launch|run)\b", normalized):
        return True
    negated_validation = any(
        phrase in normalized
        for phrase in (
            "no ejecutes",
            "no ejecutar",
            "no valides",
            "sin tests",
            "sin test",
            "do not run",
            "don't run",
        )
    )
    if not negated_validation and re.search(r"\b(test|tests|valida|validar)\b", normalized):
        return True
    return False


def _task_requests_project_creation(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    wants_create = any(
        word in normalized
        for word in (
            "crea",
            "crear",
            "haz",
            "hazme",
            "hacer",
            "genera",
            "generar",
            "build",
            "create",
            "scaffold",
        )
    )
    mentions_project = any(
        word in normalized
        for word in ("proyecto", "project", "app", "aplicacion", "workspace")
    )
    return wants_create and mentions_project


def _task_mentions_flutter(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    return "flutter" in normalized or "dart" in normalized


def _task_requests_login(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    return any(
        marker in normalized
        for marker in (
            "login",
            "log in",
            "inicio de sesion",
            "iniciar sesion",
            "autenticacion",
            "auth",
        )
    )


def _task_requests_flutter_login(task: str) -> bool:
    return _task_mentions_flutter(task) and _task_requests_login(task)


def _task_allows_missing_file_write(task: str, path: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    raw_path = str(path or "").replace("\\", "/").lower()
    explicit_edit = any(
        word in normalized
        for word in (
            "edita",
            "editar",
            "modifica",
            "modificar",
            "actualiza",
            "actualizar",
            "cambia",
            "cambiar",
            "sobrescribe",
            "sobrescribir",
            "edit",
            "modify",
            "update",
        )
    )
    explicit_create = any(
        word in normalized
        for word in (
            "crea",
            "crear",
            "haz",
            "hazme",
            "hacer",
            "genera",
            "generar",
            "create",
            "build",
            "scaffold",
        )
    )
    if explicit_edit and not _task_requests_project_creation(task):
        return False
    if explicit_create:
        return True
    return bool(raw_path and _task_requests_project_creation(task))


def _task_requires_tool_activity(task: str) -> bool:
    return _task_requires_workspace_change(task) or _task_requires_command_activity(task)


def _infer_code_path(task: str, lang: str, code: str, workspace_dir: Path) -> str:
    lowered_task = str(task or "").lower()
    lowered_code = str(code or "").lower()
    if "flutter" in lowered_task or "dart" in lowered_task or "materialapp(" in lowered_code:
        return "lib/main.dart"
    if lang in {"ts", "tsx", "typescript"}:
        return "src/App.tsx"
    if lang in {"js", "jsx", "javascript"}:
        return "src/App.jsx"
    if lang == "py" or lang == "python":
        return "main.py"
    if lang in {"html", "htm"}:
        return "index.html"
    if lang == "css":
        return "styles.css"
    if lang in {"json", "jsonc"}:
        return "config.json"
    if lang in {"yaml", "yml"}:
        return "config.yaml"
    if lang in {"kotlin", "kt"}:
        return "main.kt"
    if lang in {"swift"}:
        return "main.swift"
    if lang in {"java"}:
        return "Main.java"
    if lang in {"rust", "rs"}:
        return "main.rs"
    if lang in {"go", "golang"}:
        return "main.go"
    if lang in {"c", "cpp", "c++"}:
        return "main.cpp"
    if lang in {"dart"}:
        return "lib/main.dart"
    # Look at code content for hints
    if "runapp(" in lowered_code or "statewidget" in lowered_code:
        return "lib/main.dart"
    if "import react" in lowered_code or "usestate" in lowered_code:
        return "src/App.tsx"
    if "def " in lowered_code or "import " in lowered_code:
        return "main.py"
    if "<html" in lowered_code or "<!doctype" in lowered_code:
        return "index.html"
    existing_main = workspace_dir / "main.py"
    return "main.py" if existing_main.exists() else "generated_code.txt"


MODEL_FILE_EXTENSIONS = (
    "c",
    "conf",
    "cpp",
    "css",
    "dart",
    "go",
    "gradle",
    "html",
    "java",
    "js",
    "json",
    "jsx",
    "kt",
    "lock",
    "md",
    "py",
    "rs",
    "scss",
    "swift",
    "toml",
    "ts",
    "tsx",
    "txt",
    "vue",
    "xml",
    "yaml",
    "yml",
)
MODEL_FILE_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_. -]+[\\/])*[A-Za-z0-9_. -]+\.("
    + "|".join(re.escape(ext) for ext in MODEL_FILE_EXTENSIONS)
    + r"))",
    flags=re.IGNORECASE,
)


def _strip_model_path(raw: str) -> str:
    value = str(raw or "").strip()
    value = re.sub(r"^[#>*\-\s\d.)]+", "", value).strip()
    value = re.sub(r"^(?:archivo|file|path|ruta)\s*[:=\-]\s*", "", value, flags=re.IGNORECASE).strip()
    value = value.strip("`\"' \t\r\n:")
    match = MODEL_FILE_PATH_PATTERN.search(value)
    if match:
        value = match.group("path")
    value = value.replace("\\", "/").strip("/")
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return ""
    return "/".join(parts)


def _infer_path_from_block_prefix(prefix: str) -> str:
    lines = [line.strip() for line in str(prefix or "").splitlines()[-5:] if line.strip()]
    for line in reversed(lines):
        candidate = _strip_model_path(line)
        if candidate:
            return candidate
    return ""


def _normalize_model_code_path(task: str, raw_path: str, lang: str, code: str, workspace_dir: Path) -> str:
    path = _strip_model_path(raw_path)
    if not path:
        path = _infer_code_path(task, lang, code, workspace_dir).replace("\\", "/")
    lowered_task = _normalish(_extract_objective_text(task))
    looks_flutter = "flutter" in lowered_task or "dart" in lowered_task or "runapp(" in str(code or "").lower()
    basename = Path(path).name.lower()
    if looks_flutter and "/" not in path:
        if basename == "main.dart" or (basename.endswith(".dart") and not basename.endswith("_test.dart")):
            path = f"lib/{basename}"
        elif basename.endswith("_test.dart"):
            path = f"test/{basename}"
    return path


def _is_code_like(lang: str, code: str) -> bool:
    lowered = str(code or "").lower()
    return bool(
        lang
        or "void main(" in code
        or "class " in code
        or "import " in code
        or "function " in code
        or "def " in code
        or "const " in code
        or "var " in code
        or "let " in code
        or "export " in code
        or "<template" in lowered
        or "<html" in lowered
        or "@override" in code
        or "Widget build" in code
        or "runApp(" in code
        or "StatelessWidget" in code
        or "StatefulWidget" in code
    )


def _looks_like_action_payload_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if not stripped.startswith(("{", "[")):
        return False
    lowered = stripped[:1000].lower()
    return '"type"' in lowered and '"args"' in lowered and (
        "write_file" in lowered or "read_file" in lowered or "run_command" in lowered
    )


_ACTION_PAYLOAD_TYPE_RE = re.compile(
    r'"type"\s*:\s*"(?:write_file|delete_file|read_file|run_command|list_tree|grep|open_docs|search_web|run_tests|open_browser|propose_patch|sandbox_patch|apply_patch|summarize_diff)"',
    flags=re.IGNORECASE,
)


def _contains_action_type_marker(text: str) -> bool:
    return bool(_ACTION_PAYLOAD_TYPE_RE.search(str(text or "")))


def _contains_action_payload(text: str) -> bool:
    lowered = str(text or "").lower()
    if '"args"' not in lowered:
        return False
    return _contains_action_type_marker(text)


def _write_text_looks_incomplete(path: str, text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if _looks_like_action_payload_text(stripped):
        return True
    if _contains_action_type_marker(stripped):
        return True
    if re.search(r"\b(?:action|tool)\s*:\s*(?:write_file|read_file|run_command|apply_patch)\b", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"(?is)^(?:understood|sure|here is|here's|task:|action:)\b", stripped):
        return True
    if _contains_action_payload(stripped):
        return True
    if re.search(r"\.\.\.\s*(?:$|[}\]\)])", stripped):
        return True
    lowered_path = str(path or "").lower()
    if lowered_path.endswith((".dart", ".ts", ".tsx", ".js", ".jsx", ".py")):
        if re.search(r"(todo|placeholder)\s*[:)]", stripped, flags=re.IGNORECASE):
            return True
    if lowered_path.endswith("lib/main.dart") and ("void main" not in stripped or "runApp(" not in stripped):
        return True
    return False


def _json_action_to_write_args(args: dict) -> dict | None:
    raw_text = str(args.get("text") or "").strip()
    if not _looks_like_action_payload_text(raw_text):
        return None
    action, ok = _parse_action(raw_text)
    if not ok or action.type != "write_file":
        return None
    nested = dict(action.args or {})
    nested_path = str(nested.get("path") or "").strip()
    nested_text = str(nested.get("text") or "")
    outer_path = str(args.get("path") or "").strip()
    if not nested_path or not nested_text.strip():
        return None
    if outer_path and nested_path.replace("\\", "/") != outer_path.replace("\\", "/"):
        return None
    if _write_text_looks_incomplete(nested_path, nested_text):
        return None
    lowered_path = nested_path.lower()
    if lowered_path.endswith((".dart", ".ts", ".tsx", ".js", ".jsx", ".py")):
        if len(nested_text.strip()) < 40 and not _is_code_like("", nested_text):
            return None
    nested["path"] = nested_path
    nested["text"] = nested_text
    if "append" not in nested and "append" in args:
        nested["append"] = bool(args.get("append", False))
    return nested


def _dedupe_write_actions(actions: list[Action]) -> list[Action]:
    deduped: list[Action] = []
    seen: set[str] = set()
    for action in actions:
        path = str(action.args.get("path") or "").strip().replace("\\", "/")
        text = str(action.args.get("text") or "")
        if not path or not text.strip() or path in seen:
            continue
        seen.add(path)
        action.args["path"] = path
        deduped.append(action)
    return deduped


def _extract_code_write_actions(task: str, output: str, workspace_dir: Path) -> list[Action]:
    text = str(output or "")
    if not text.strip():
        return []
    actions: list[Action] = []

    # Priority 1: explicit file: blocks (```file:path/to/file)
    file_blocks = re.findall(r"```file:([^\n`]+)\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    for raw_path, raw_code in file_blocks:
        code = str(raw_code or "").strip()
        path = _normalize_model_code_path(task, raw_path, "", code, workspace_dir)
        if path and code:
            actions.append(Action(type="write_file", args={"path": path, "text": code.rstrip() + "\n"}))
    if actions:
        return _dedupe_write_actions(actions)

    non_code_text = re.sub(r"```[\s\S]*?```", "", text)
    if _contains_action_payload(non_code_text):
        return []

    # Priority 2: code blocks with language markers
    for match in re.finditer(r"```([A-Za-z0-9_+.-]*)\n([\s\S]*?)```", text):
        raw_lang, raw_code = match.group(1), match.group(2)
        lang = str(raw_lang or "").strip().lower()
        code = str(raw_code or "").strip()
        if not code:
            continue
        if _looks_like_action_payload_text(code):
            continue
        prefix = text[max(0, match.start() - 320) : match.start()]
        path_hint = _infer_path_from_block_prefix(prefix)
        if not path_hint and not _is_code_like(lang, code) and len(code) < 20:
            continue
        path = _normalize_model_code_path(task, path_hint, lang, code, workspace_dir)
        if path and code:
            actions.append(Action(type="write_file", args={"path": path, "text": code.rstrip() + "\n"}))
    if actions:
        return _dedupe_write_actions(actions)

    # Priority 3: detect Flutter/Dart code inline
    if "import 'package:flutter/material.dart'" in text or "MaterialApp(" in text:
        path = _infer_code_path(task, "dart", text, workspace_dir)
        return [Action(type="write_file", args={"path": path, "text": text.rstrip() + "\n"})]
    return []


def _extract_code_write_action(task: str, output: str, workspace_dir: Path) -> Action | None:
    actions = _extract_code_write_actions(task, output, workspace_dir)
    if not actions:
        return None
    first = actions[0]
    first.args["_finish_after_write"] = len(actions) == 1
    return first


def _dedupe_browser_actions(actions: List[dict[str, object]]) -> List[dict[str, object]]:
    deduped: List[dict[str, object]] = []
    seen: set[str] = set()
    for item in actions:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or "").strip()
        if not target or target in seen:
            continue
        seen.add(target)
        deduped.append(dict(item))
    return deduped


def _generate_final_summary(
    task: str,
    tool_calls: List[dict],
    settings: dict,
    current_model: object | None,
    model_lock: Callable[[], Any] | None,
    *,
    max_new_tokens: int = 160,
) -> str:
    if not tool_calls:
        return ""
    useful_calls = []
    for call in tool_calls[-4:]:
        output = str(call.get("output", "")).strip()
        if not output:
            continue
        if len(output) > 1200:
            output = output[:1200].rstrip() + "..."
        useful_calls.append(f"{call.get('action', 'tool')} -> {output}")
    if not useful_calls:
        return ""
    fallback_output = useful_calls[-1].split("->", 1)[-1].strip()
    if current_model is None:
        return fallback_output
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are closing an agent task. "
                    "Write a concise user-facing answer in plain text. "
                    "Use the tool outputs directly. Do not mention internal JSON, prompts, or agent internals."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"Tool outputs:\n{chr(10).join(useful_calls)}\n\n"
                    "Return only the final answer."
                ),
            },
        ]
        prompt = build_chat_prompt(
            messages,
            backend=str(settings.get("core", {}).get("backend", "vortex")),
            tokenizer=getattr(current_model, "tokenizer", None),
            default_system=None,
        )
        with (model_lock() if model_lock is not None else nullcontext()):
            text = current_model.generate(
                prompt,
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        return str(text or "").strip() or fallback_output
    except Exception:
        return fallback_output


def _file_action_summary(tool_calls: List[dict]) -> str:
    created: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    for call in tool_calls:
        if call.get("action") not in {"write_file", "delete_file"} or not call.get("ok"):
            continue
        try:
            payload = json.loads(str(call.get("output") or ""))
        except Exception:
            continue
        path = str(payload.get("relative_path") or payload.get("path") or "").strip()
        if not path:
            continue
        if call.get("action") == "delete_file":
            if path not in deleted:
                deleted.append(path)
            continue
        was_created = bool(payload.get("created", False))
        if was_created:
            if path not in created:
                created.append(path)
        elif path not in updated:
            updated.append(path)
    updated = [path for path in updated if path not in created or path in deleted]
    parts: list[str] = []
    if created:
        parts.append(f"He creado `{_join_natural(created)}`.")
    if updated:
        parts.append(f"He actualizado `{_join_natural(updated)}`.")
    if deleted:
        parts.append(f"He borrado `{_join_natural(deleted)}`.")
    return " ".join(parts)


def _join_natural(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _infer_exists_summary(task: str, workspace_dir: Path) -> str:
    lowered = str(task or "").lower()
    if "existe" not in lowered and "exists" not in lowered:
        return ""
    patterns = [
        r"(?:carpeta|directorio|folder|archivo|file|path|ruta)\s+([A-Za-z0-9_./\\-]+)",
    ]
    root = workspace_dir.resolve()
    for pattern in patterns:
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if not match:
            continue
        raw = str(match.group(1) or "").strip(" \t\r\n`'\".,:;")
        if not raw:
            continue
        candidate = Path(raw)
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            candidate.relative_to(root)
        except Exception:
            continue
        if candidate.exists():
            kind = "carpeta" if candidate.is_dir() else "archivo"
            return f"Sí, existe la {kind} {raw} en el proyecto."
        return f"No existe {raw} en el proyecto."
    return ""


def _extract_direct_file_action(task: str) -> Action | None:
    text = str(task or "").strip()
    if not text:
        return None
    natural_action = _extract_natural_file_action(text)
    if natural_action is not None:
        return natural_action
    create_match = re.search(
        r"(?:crea|crear|create|write)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?([^`\"'\s]+)[`\"']?\s+(?:con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with\s+(?:the\s+)?(?:text|content))\s+(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if create_match:
        path = create_match.group(1).strip().rstrip(".,;:")
        content = _clean_direct_file_content(str(create_match.group(2) or ""))
        if path and content:
            return Action(type="write_file", args={"path": path, "text": content})

    delete_match = re.search(
        r"(?:borra|borrar|elimina|eliminar|delete|remove)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?([^`\"'\s,;:]+)[`\"']?",
        text,
        flags=re.IGNORECASE,
    )
    if delete_match:
        path = delete_match.group(1).strip().rstrip(".,;:")
        if path:
            return Action(type="delete_file", args={"path": path})
    return None


def _normalize_natural_file_target(raw: str) -> str:
    target = str(raw or "").strip().strip("`\"' \t\r\n").rstrip(".,;:")
    lowered = target.lower()
    if lowered in {"readme", "readme.md", "readme.txt"}:
        return "README.md"
    return target


def _extract_natural_file_action(text: str) -> Action | None:
    target_pattern = r"(?P<target>readme(?:\.(?:md|txt))?|[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)"
    write_patterns = [
        rf"(?P<verb>edita|editar|modifica|modificar|actualiza|actualizar|cambia|cambiar|sobrescribe|sobrescribir|crea|crear|edit|modify|update|change|write|create)\s+(?:el\s+|la\s+|un\s+|una\s+|the\s+|a\s+)?{target_pattern}\s+(?:para\s+que\s+)?(?:ponga|diga|contenga|sea|con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with|to\s+say|to\s+contain)\s*[,:\-]?\s+(?P<text>[^;\n]+)",
        rf"(?P<verb>haz|hacer|make)\s+que\s+(?:el\s+|la\s+|the\s+)?{target_pattern}\s+(?:ponga|diga|contenga|sea|say|contain)\s*[,:\-]?\s+(?P<text>[^;\n]+)",
    ]
    for pattern in write_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        path = _normalize_natural_file_target(str(match.group("target") or ""))
        content = _clean_direct_file_content(str(match.group("text") or ""))
        if path and content:
            verb = _normalish(str(match.group("verb") or ""))
            creates_file = verb in {"crea", "crear", "create", "write"}
            args = {"path": path, "text": content}
            if not creates_file:
                args["require_exists"] = True
            return Action(type="write_file", args=args)

    delete_match = re.search(
        rf"(?:borra|borrar|elimina|eliminar|delete|remove)\s+(?:el\s+|la\s+|un\s+|una\s+|the\s+|a\s+)?{target_pattern}",
        text,
        flags=re.IGNORECASE,
    )
    if delete_match:
        path = _normalize_natural_file_target(str(delete_match.group("target") or ""))
        if path:
            return Action(type="delete_file", args={"path": path})
    return None


def _append_direct_action(
    actions: list[tuple[int, Action]],
    start: int,
    action: Action,
) -> None:
    for existing_start, existing in actions:
        if existing.type == action.type and (
            existing_start == start or existing.args == action.args
        ):
            return
    actions.append((start, action))


def _clean_direct_action_path(raw_path: str) -> str:
    value = str(raw_path or "").strip().rstrip(".,;:")
    value = re.split(
        r"\s+(?:y|and|then|despues|despues|desp\u00e9s)\s+"
        r"(?:crea|crear|create|write|modifica|modificar|actualiza|actualizar|"
        r"borra|borrar|delete|remove|ejecuta|ejecutar|run|corre|correr|"
        r"busca|buscar|grep|lista|listar|list)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return value.strip().strip("`\"' \t\r\n")


def _clean_direct_file_content(content: str) -> str:
    cleaned = str(content or "").strip().strip("`\"' \t\r\n")
    cleaned = re.sub(
        r"^(?:exacto|exacta|exactamente|literal|literalmente|exact|exactly)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.split(
        r"\b(?:no\s+ejecutes|no\s+valides|do\s+not\s+run|don't\s+run|sin\s+tests|usa\s+write_file|use\s+write_file|no\s+expliques|do\s+not\s+explain|don't\s+explain)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    cleaned = re.split(
        r"\s+\b(?:en|dentro\s+de|inside|in)\s+(?:el\s+|la\s+|the\s+)?(?:workspace|proyecto|project|repo|repositorio)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return cleaned.rstrip(".").strip("`\"' \t\r\n")


def _normalish(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").lower()


def _extract_objective_text(task: str) -> str:
    text = str(task or "")
    marker = "Objetivo principal:"
    if marker not in text:
        return text
    objective = text.split(marker, 1)[1]
    for end_marker in ("\n\nContexto reciente:", "\n\nPermisos locales:"):
        if end_marker in objective:
            objective = objective.split(end_marker, 1)[0]
    return objective.strip() or text


def _workspace_looks_flutter(workspace_dir: Path) -> bool:
    pubspec = workspace_dir / "pubspec.yaml"
    if not pubspec.exists() or not pubspec.is_file():
        return False
    try:
        text = pubspec.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return "flutter:" in text or "sdk: flutter" in text


def _requests_existing_project_run(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    wants_run = any(
        word in normalized
        for word in (
            "ejecuta",
            "ejecutame",
            "ejecutalo",
            "ejecute",
            "ejecutar",
            "corre",
            "correme",
            "correr",
            "inicia",
            "iniciame",
            "iniciar",
            "lanza",
            "lanzame",
            "lanzalo",
            "run",
            "start",
            "launch",
        )
    )
    mentions_project = any(
        word in normalized
        for word in ("proyecto", "project", "app", "aplicacion", "workspace")
    )
    asks_new_code = any(
        word in normalized
        for word in (
            "crea",
            "crear",
            "haz",
            "hacer",
            "implementa",
            "genera",
            "nuevo",
            "create",
            "build",
        )
    )
    return wants_run and mentions_project and not asks_new_code


def _safe_project_name(workspace_dir: Path) -> str:
    raw = re.sub(r"[^a-zA-Z0-9_]+", "_", workspace_dir.name.strip().lower()).strip("_")
    if not raw:
        raw = "vortex_app"
    if not re.match(r"^[a-zA-Z_]", raw):
        raw = f"vortex_{raw}"
    return raw[:48]


def _flutter_project_bootstrap_actions(
    task: str,
    workspace_dir: Path,
    *,
    include_reads: bool = False,
) -> list[Action]:
    normalized = _normalish(_extract_objective_text(task))
    if "flutter" not in normalized and "dart" not in normalized:
        return []
    if not _task_requests_project_creation(task):
        return []
    if _workspace_looks_flutter(workspace_dir):
        return []
    command = f"flutter create --project-name {_safe_project_name(workspace_dir)} ."
    actions = [Action(type="run_command", args={"command": command, "cwd": ".", "timeout_s": 300})]
    if include_reads:
        actions.extend(
            [
                Action(type="read_file", args={"path": "pubspec.yaml", "max_chars": 2000}),
                Action(type="read_file", args={"path": "lib/main.dart", "max_chars": 5000}),
            ]
        )
    return actions


def _flutter_project_inspect_actions(task: str, workspace_dir: Path) -> list[Action]:
    normalized = _normalish(_extract_objective_text(task))
    if "flutter" not in normalized and "dart" not in normalized:
        return []
    if not _task_requires_workspace_change(task):
        return []
    if not _workspace_looks_flutter(workspace_dir):
        return []
    actions: list[Action] = [
        Action(type="read_file", args={"path": "pubspec.yaml", "max_chars": 2000}),
    ]
    if (workspace_dir / "lib" / "main.dart").exists():
        actions.append(Action(type="read_file", args={"path": "lib/main.dart", "max_chars": 5000}))
    return actions


def _node_project_run_actions(workspace_dir: Path) -> list[Action]:
    package_path = workspace_dir / "package.json"
    if not package_path.exists() or not package_path.is_file():
        return []
    try:
        package = json.loads(package_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        package = {}
    scripts = package.get("scripts") if isinstance(package, dict) else {}
    scripts = scripts if isinstance(scripts, dict) else {}
    actions: list[Action] = []
    package_lock = workspace_dir / "package-lock.json"
    node_modules = workspace_dir / "node_modules"
    npm_lock_marker = node_modules / ".package-lock.json"
    needs_install = not node_modules.exists() or (package_lock.exists() and not npm_lock_marker.exists())
    if needs_install:
        install_cmd = "npm ci" if package_lock.exists() else "npm install"
        actions.append(Action(type="run_command", args={"command": install_cmd, "cwd": ".", "timeout_s": 300}))
    if "build" in scripts:
        actions.append(Action(type="run_command", args={"command": "npm run build", "cwd": ".", "timeout_s": 300}))
    if "dev" in scripts:
        actions.append(
            Action(
                type="run_command",
                args={
                    "command": "npm run dev -- --host 0.0.0.0",
                    "cwd": ".",
                    "timeout_s": 120,
                    "background": True,
                },
            )
        )
    elif "start" in scripts:
        actions.append(
            Action(
                type="run_command",
                args={
                    "command": "npm start",
                    "cwd": ".",
                    "timeout_s": 120,
                    "background": True,
                },
            )
        )
    return actions


def _python_project_run_actions(workspace_dir: Path) -> list[Action]:
    if not (workspace_dir / "pyproject.toml").exists():
        return []
    actions: list[Action] = []
    tests_dir = workspace_dir / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        actions.append(Action(type="run_command", args={"command": "python -m pytest -q", "cwd": ".", "timeout_s": 300}))
    else:
        actions.append(Action(type="run_command", args={"command": "python --version", "cwd": ".", "timeout_s": 60}))
    return actions


def _project_run_actions(task: str, workspace_dir: Path) -> list[Action]:
    if not _requests_existing_project_run(task):
        return []
    node_actions = _node_project_run_actions(workspace_dir)
    if node_actions:
        return node_actions
    python_actions = _python_project_run_actions(workspace_dir)
    if python_actions:
        return python_actions
    return []


def _flutter_project_name_from_workspace(workspace_dir: Path) -> str:
    return _safe_project_name(workspace_dir)


def _flutter_login_main_text(workspace_dir: Path) -> str:
    title = workspace_dir.name.replace("_", " ").replace("-", " ").strip().title() or "Vortex Login"
    return f"""import 'package:flutter/material.dart';

void main() => runApp(const LoginApp());

class LoginApp extends StatelessWidget {{
  const LoginApp({{super.key}});

  @override
  Widget build(BuildContext context) {{
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: '{title}',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginScreen(),
    );
  }}
}}

class LoginScreen extends StatefulWidget {{
  const LoginScreen({{super.key}});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}}

class _LoginScreenState extends State<LoginScreen> {{
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;

  @override
  void dispose() {{
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }}

  void _submit() {{
    if (!_formKey.currentState!.validate()) {{
      return;
    }}
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Bienvenido, ${{_emailController.text.trim()}}')),
    );
  }}

  @override
  Widget build(BuildContext context) {{
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Form(
                key: _formKey,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      'Login',
                      style: Theme.of(context).textTheme.headlineMedium,
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),
                    TextFormField(
                      controller: _emailController,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                        labelText: 'Email',
                        prefixIcon: Icon(Icons.email_outlined),
                        border: OutlineInputBorder(),
                      ),
                      validator: (value) {{
                        final text = value?.trim() ?? '';
                        if (text.isEmpty) {{
                          return 'Introduce tu email';
                        }}
                        if (!text.contains('@')) {{
                          return 'Introduce un email valido';
                        }}
                        return null;
                      }},
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _passwordController,
                      obscureText: _obscurePassword,
                      decoration: InputDecoration(
                        labelText: 'Password',
                        prefixIcon: const Icon(Icons.lock_outline),
                        border: const OutlineInputBorder(),
                        suffixIcon: IconButton(
                          onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                          icon: Icon(_obscurePassword ? Icons.visibility : Icons.visibility_off),
                        ),
                      ),
                      validator: (value) {{
                        if ((value ?? '').length < 6) {{
                          return 'Minimo 6 caracteres';
                        }}
                        return null;
                      }},
                    ),
                    const SizedBox(height: 20),
                    FilledButton(
                      onPressed: _submit,
                      child: const Text('Entrar'),
                    ),
                    TextButton(
                      onPressed: () {{
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Recuperacion de password pendiente')),
                        );
                      }},
                      child: const Text('He olvidado la password'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }}
}}
"""


def _flutter_pubspec_text(workspace_dir: Path) -> str:
    return f"""name: {_flutter_project_name_from_workspace(workspace_dir)}
description: Flutter project generated by Vortex agent.
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.3.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^4.0.0

flutter:
  uses-material-design: true
"""


def _flutter_existing_pubspec_compat_text(workspace_dir: Path) -> str | None:
    pubspec_path = workspace_dir / "pubspec.yaml"
    if not pubspec_path.exists() or not pubspec_path.is_file():
        return None
    try:
        text = pubspec_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    updated = re.sub(
        r"(?m)^(\s*sdk:\s*)\^3\.12\.0\s*$",
        r"\1'>=3.3.0 <4.0.0'",
        text,
        count=1,
    )
    if updated == text:
        return None
    return updated


def _flutter_widget_test_text(workspace_dir: Path) -> str:
    project_name = _flutter_project_name_from_workspace(workspace_dir)
    return f"""import 'package:flutter_test/flutter_test.dart';
import 'package:{project_name}/main.dart';

void main() {{
  testWidgets('login screen renders', (WidgetTester tester) async {{
    await tester.pumpWidget(const LoginApp());
    expect(find.text('Login'), findsOneWidget);
    expect(find.text('Entrar'), findsOneWidget);
  }});
}}
"""


def _flutter_widget_test_needs_login_update(workspace_dir: Path) -> bool:
    test_path = workspace_dir / "test" / "widget_test.dart"
    if not test_path.exists() or not test_path.is_file():
        return True
    try:
        text = test_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    lowered = text.lower()
    return "myapp" in text or "counter" in lowered or "findsonewidget" in lowered and "loginapp" not in lowered


def _flutter_manual_project_actions(task: str, workspace_dir: Path) -> list[Action]:
    if not _task_requests_flutter_login(task):
        return []
    actions: list[Action] = []
    if not _workspace_looks_flutter(workspace_dir):
        actions.append(Action(type="write_file", args={"path": "pubspec.yaml", "text": _flutter_pubspec_text(workspace_dir)}))
    else:
        pubspec_compat = _flutter_existing_pubspec_compat_text(workspace_dir)
        if pubspec_compat:
            actions.append(Action(type="write_file", args={"path": "pubspec.yaml", "text": pubspec_compat}))
    actions.append(Action(type="write_file", args={"path": "lib/main.dart", "text": _flutter_login_main_text(workspace_dir)}))
    if _flutter_widget_test_needs_login_update(workspace_dir):
        actions.append(Action(type="write_file", args={"path": "test/widget_test.dart", "text": _flutter_widget_test_text(workspace_dir)}))
    return actions


def _incomplete_write_recovery_actions(task: str, workspace_dir: Path, path: str) -> list[Action]:
    normalized_path = str(path or "").replace("\\", "/").lower()
    if normalized_path != "lib/main.dart":
        return []
    return _flutter_manual_project_actions(task, workspace_dir)


def _extract_direct_actions(task: str) -> list[Action]:
    text = str(task or "").strip()
    if not text:
        return []
    actions: list[tuple[int, Action]] = []
    read_pattern = r"(?:lee|leer|read|abre|abrir|open)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)?\s*`?(?P<path>[A-Za-z0-9_.\\/\- ]+\.[A-Za-z0-9_]+)`?"
    for match in re.finditer(read_pattern, text, flags=re.IGNORECASE):
        path = _clean_direct_action_path(str(match.group("path") or ""))
        if path:
            _append_direct_action(
                actions,
                match.start(),
                Action(type="read_file", args={"path": path, "max_chars": 4000}),
            )
    list_pattern = r"(?:lista|listar|list)\s+(?:archivos|files|tree|arbol|\u00e1rbol)(?:\s+(?:en|de|from)\s+`?(?P<root>[^`\"'\n;]+)`?)?"
    for match in re.finditer(list_pattern, text, flags=re.IGNORECASE):
        root = str(match.group("root") or ".").strip().rstrip(".,;:")
        _append_direct_action(
            actions,
            match.start(),
            Action(type="list_tree", args={"root": root or ".", "max_entries": 120}),
        )
    grep_pattern = r"(?:busca|buscar|grep|search)\s+(?P<quote>[`\"'])(?P<pattern>.*?)(?P=quote)(?:\s+(?:en|in)\s+`?(?P<glob>[^`\"'\n;]+)`?)?"
    for match in re.finditer(grep_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        pattern_text = str(match.group("pattern") or "").strip()
        path_glob = str(match.group("glob") or "**/*").strip().rstrip(".,;:")
        if pattern_text:
            _append_direct_action(
                actions,
                match.start(),
                Action(type="grep", args={"pattern": pattern_text, "path_glob": path_glob or "**/*", "max_hits": 50}),
            )
    write_patterns = [
        (
            r"(?:crea|crear|create|write)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?(?P<path>[^`\"'\s;]+)[`\"']?\s+(?:con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with\s+(?:the\s+)?(?:text|content))\s+(?P<quote>[`\"'])(?P<text>.*?)(?P=quote)",
            False,
        ),
        (
            r"(?:modifica|modificar|actualiza|actualizar|sobrescribe|sobrescribir|update|modify|overwrite)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?(?P<path>[^`\"'\s;]+)[`\"']?\s+(?:con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with\s+(?:the\s+)?(?:text|content))\s+(?P<quote>[`\"'])(?P<text>.*?)(?P=quote)",
            True,
        ),
        (
            r"(?:crea|crear|create|write)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?(?P<path>[^`\"'\s;]+)[`\"']?\s+(?:con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with\s+(?:the\s+)?(?:text|content))\s+(?P<text>[^;\n]+)",
            False,
        ),
        (
            r"(?:modifica|modificar|actualiza|actualizar|sobrescribe|sobrescribir|update|modify|overwrite)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?(?P<path>[^`\"'\s;]+)[`\"']?\s+(?:con\s+(?:el\s+|la\s+)?(?:texto|contenido)|with\s+(?:the\s+)?(?:text|content))\s+(?P<text>[^;\n]+)",
            True,
        ),
    ]
    for pattern, require_exists in write_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            path = str(match.group("path") or "").strip().rstrip(".,;:")
            content = _clean_direct_file_content(str(match.group("text") or ""))
            if path and content:
                args = {"path": path, "text": content}
                if require_exists:
                    args["require_exists"] = True
                _append_direct_action(
                    actions,
                    match.start(),
                    Action(type="write_file", args=args),
                )
    delete_pattern = r"(?:borra|borrar|elimina|eliminar|delete|remove)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?(?P<path>[^`\"'\s,;:]+)[`\"']?"
    for match in re.finditer(delete_pattern, text, flags=re.IGNORECASE):
        path = str(match.group("path") or "").strip().rstrip(".,;:")
        if path:
            _append_direct_action(
                actions,
                match.start(),
                Action(type="delete_file", args={"path": path}),
            )
    command_patterns = [
        r"(?:ejecuta|ejecutar|corre|correr|run)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:comando|command)\s+(?P<quote>[`\"'])(?P<command>.*?)(?P=quote)",
        r"(?:ejecuta|ejecutar|corre|correr|run)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:comando|command)\s+(?![`\"'])(?P<command>[^;\n]+)",
    ]
    for pattern in command_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            command = str(match.group("command") or "").strip().strip("`\"'")
            if command:
                _append_direct_action(
                    actions,
                    match.start(),
                    Action(type="run_command", args={"command": command, "cwd": ".", "timeout_s": 120}),
                )
    if actions:
        return [action for _start, action in sorted(actions, key=lambda item: item[0])]
    single = _extract_direct_file_action(task)
    return [single] if single is not None else []


def _direct_actions_summary(tool_calls: List[dict]) -> str:
    created: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    command_names: list[str] = []
    failed_commands: list[tuple[str, str]] = []
    reads: list[tuple[str, str]] = []
    lists: list[str] = []
    searches: list[str] = []
    for call in tool_calls:
        action = str(call.get("action") or "")
        if action not in {"write_file", "delete_file", "run_command", "read_file", "list_tree", "grep"}:
            continue
        args = call.get("args")
        args_dict = args if isinstance(args, dict) else {}
        if action == "read_file" and call.get("ok"):
            reads.append((str(args_dict.get("path") or "archivo"), str(call.get("output") or "")))
            continue
        if action == "list_tree" and call.get("ok"):
            lists.append(str(args_dict.get("root") or "."))
            continue
        if action == "grep" and call.get("ok"):
            searches.append(str(args_dict.get("pattern") or "busqueda"))
            continue
        if action in {"write_file", "delete_file"} and call.get("ok"):
            meta = call.get("meta")
            path = ""
            if isinstance(meta, dict):
                path = str(meta.get("path") or "").strip()
            try:
                payload = json.loads(str(call.get("output") or ""))
                path = str(payload.get("relative_path") or payload.get("path") or path).strip()
                created_file = bool(payload.get("created", False))
            except Exception:
                created_file = False
            if not path:
                continue
            if action == "delete_file":
                if path not in deleted:
                    deleted.append(path)
            elif created_file:
                if path not in created:
                    created.append(path)
            elif path not in updated:
                updated.append(path)
        if action == "run_command" and call.get("ok"):
            args = call.get("args")
            command = str(args.get("command") if isinstance(args, dict) else "").strip()
            if command:
                command_names.append(command)
        elif action == "run_command" and not call.get("ok"):
            args = call.get("args")
            command = str(args.get("command") if isinstance(args, dict) else "").strip()
            output = str(call.get("output") or "").strip()
            if command:
                failed_commands.append((command, output))
    updated = [path for path in updated if path not in created or path in deleted]
    parts: list[str] = []
    if created:
        parts.append(f"He creado `{_join_natural(created)}`.")
    if updated:
        parts.append(f"He actualizado `{_join_natural(updated)}`.")
    if deleted:
        parts.append(f"He borrado `{_join_natural(deleted)}`.")
    if reads:
        parts.append(_read_actions_summary(reads))
    if lists:
        parts.append(f"He listado `{_join_natural(lists)}`.")
    if searches:
        parts.append(f"He buscado `{_join_natural(searches)}`.")
    if command_names and not (created or updated or deleted):
        parts.append(f"He ejecutado `{_join_natural(command_names)}`.")
    if any("flutter test" in command for command in command_names):
        parts.append("He validado el proyecto con sus tests.")
    failed_emulator_commands = [
        (command, output) for command, output in failed_commands if _command_looks_like_flutter_emulator(command)
    ]
    if any(_command_looks_like_flutter_emulator(command) for command in command_names) and not failed_emulator_commands:
        parts.append("He iniciado la app en el emulador.")
    elif any(_command_looks_like_flutter_web_server(command) for command in command_names):
        parts.append("He iniciado la app web local en `http://localhost:19090`.")
    elif any("emulators --launch" in command for command in command_names) and not failed_emulator_commands:
        parts.append("He iniciado el emulador.")
    if failed_emulator_commands:
        parts.append(_flutter_emulator_failure_summary(failed_emulator_commands))
    elif failed_commands:
        command, output = failed_commands[-1]
        parts.append(_command_failure_summary(command, output))
    if parts:
        return " ".join(parts)
    if not tool_calls:
        return ""
    return "He terminado la tarea."


def _read_actions_summary(reads: list[tuple[str, str]]) -> str:
    paths = [path for path, _output in reads]
    if len(reads) == 1:
        path, output = reads[0]
        detail = _read_output_brief(path, output)
        if detail:
            return f"He leido `{path}`. {detail}"
    return f"He leido `{_join_natural(paths)}`."


def _read_output_brief(path: str, output: str) -> str:
    text = str(output or "").strip()
    if not text:
        return ""
    lower_path = path.lower()
    if lower_path.endswith(("pubspec.yaml", "pubspec.yml")):
        match = re.search(r"(?m)^name:\s*([A-Za-z0-9_\-]+)\s*$", text)
        if match:
            return f"El proyecto es `{match.group(1)}`."
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    preview = " ".join(lines[:3])
    if len(preview) > 260:
        preview = preview[:257].rstrip() + "..."
    return f"Primer contenido: {preview}"


def _attempted_workspace_mutation(tool_calls: List[dict]) -> bool:
    return any(
        bool(call.get("ok"))
        and str(call.get("action") or "") in {
            "write_file", "delete_file", "apply_patch", "propose_patch",
            "sandbox_patch",
        }
        for call in tool_calls
    )


def _has_blocking_failed_mutation(tool_calls: List[dict]) -> bool:
    blocking_markers = (
        "incomplete_file_content",
        "patch_unavailable",
        "patch.diff not found",
        "tool_disabled",
        "path not allowed",
        "write failed:",
        "apply failed:",
        "sandbox failed:",
        "approval required",
    )
    for call in tool_calls:
        action = str(call.get("action") or "")
        if action not in {"write_file", "delete_file", "apply_patch", "propose_patch", "sandbox_patch"}:
            continue
        if bool(call.get("ok")):
            continue
        output = str(call.get("output") or "").lower()
        if any(marker in output for marker in blocking_markers):
            return True
    return False


def _has_nonblocking_missing_file_mutation(tool_calls: List[dict]) -> bool:
    for call in tool_calls:
        action = str(call.get("action") or "")
        if action not in {"write_file", "delete_file"}:
            continue
        if bool(call.get("ok")):
            continue
        if "No encuentro el archivo `" in str(call.get("output") or ""):
            return True
    return False


def _attempted_command_activity(tool_calls: List[dict]) -> bool:
    return any(
        str(call.get("action") or "") in {"run_command", "run_tests", "open_browser"}
        for call in tool_calls
    )


def _command_looks_like_test(command: str) -> bool:
    normalized = str(command or "").strip().lower()
    if not normalized:
        return False
    return any(
        pattern in normalized
        for pattern in (
            "flutter test",
            "dart test",
            "npm test",
            "pnpm test",
            "yarn test",
            "pytest",
            "python -m pytest",
        )
    )


def _command_looks_like_flutter_emulator(command: str) -> bool:
    normalized = str(command or "").strip().lower()
    return "flutter emulators" in normalized or "flutter run" in normalized and "emulator" in normalized


def _command_looks_like_flutter_web_server(command: str) -> bool:
    normalized = str(command or "").strip().lower()
    return "flutter run" in normalized and "web-server" in normalized


def _direct_command_failure_is_nonfatal(command: str, output: str) -> bool:
    normalized = str(command or "").strip().lower()
    if "flutter emulators" in normalized:
        return True
    if "flutter run" in normalized:
        return "no supported devices" in str(output or "").lower()
    return False


def _command_failure_summary(command: str, output: str) -> str:
    detail = str(output or "").strip().splitlines()
    first_line = detail[0].strip() if detail else ""
    if not first_line:
        first_line = "el comando fallo."
    if len(first_line) > 220:
        first_line = first_line[:217].rstrip() + "..."
    return f"No he podido completar `{command}`: {first_line}"


def _flutter_emulator_failure_summary(failed_commands: list[tuple[str, str]]) -> str:
    combined = "\n".join(output for _command, output in failed_commands).lower()
    if "unable to find any emulator sources" in combined or "no emulators" in combined:
        reason = "este entorno no ve los AVD/emuladores Android."
    elif "no emulator" in combined and "found" in combined:
        reason = "no he encontrado el emulador solicitado."
    else:
        _command, output = failed_commands[-1]
        reason = str(output or "").strip().splitlines()[0].strip() if str(output or "").strip() else "fallo el comando del emulador."
        if len(reason) > 180:
            reason = reason[:177].rstrip() + "..."
    return f"No he podido iniciar el emulador/app: {reason}"


def _tool_call_record(
    action_type: str,
    args: dict,
    result: ToolResult,
    *,
    max_output_chars: int,
) -> dict:
    record = {
        "action": action_type,
        "args": args,
        "ok": result.ok,
        "output": result.output[:max_output_chars],
    }
    if result.meta:
        record["meta"] = dict(result.meta)
    return record


def _file_changes_from_tool_calls(tool_calls: List[dict]) -> list[dict[str, str]]:
    changes_by_path: dict[str, dict[str, str]] = {}
    seen_diffs_by_path: dict[str, set[str]] = {}
    for call in tool_calls:
        if call.get("action") not in {"write_file", "delete_file", "apply_patch", "propose_patch"}:
            continue
        meta = call.get("meta")
        if not isinstance(meta, dict):
            continue
        path = str(meta.get("path") or "").strip()
        diff = str(meta.get("diff") or "").strip()
        if not path or not diff:
            continue
        seen_for_path = seen_diffs_by_path.setdefault(path, set())
        if diff in seen_for_path:
            continue
        seen_for_path.add(diff)
        existing = changes_by_path.get(path)
        if existing is None:
            change = {"path": path, "diff": diff}
            absolute_path = str(meta.get("absolute_path") or "").strip()
            if absolute_path:
                change["absolute_path"] = absolute_path
            changes_by_path[path] = change
            continue
        existing["diff"] = f"{existing['diff']}\n\n{diff}"
        absolute_path = str(meta.get("absolute_path") or "").strip()
        if absolute_path and not existing.get("absolute_path"):
            existing["absolute_path"] = absolute_path
    return list(changes_by_path.values())


def _log_episode(base_dir: Path, payload: dict) -> None:
    path = base_dir / "data" / "episodes" / "agent.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_agent(
    task: str,
    settings: dict,
    base_dir: Path,
    *,
    max_iters: int | None = None,
    action_provider: Callable[[List[dict]], Action] | None = None,
    model: object | None = None,
    model_lock: Callable[[], Any] | None = None,
    permissions: AgentPermissions | None = None,
    workspace_root: Path | None = None,
    allow_model_load: bool = True,
) -> dict:
    supported_tools = [
        "open_docs",
        "search_web",
        "read_file",
        "grep",
        "list_tree",
        "write_file",
        "delete_file",
        "run_tests",
        "run_command",
        "open_browser",
        "propose_patch",
        "sandbox_patch",
        "apply_patch",
        "summarize_diff",
    ]
    agent_cfg = settings.get("agent", {}) or {}
    if max_iters is None:
        max_iters = _cfg_int(agent_cfg, "max_iters", 5)
    else:
        max_iters = max(1, int(max_iters))
    max_context_compactions = _cfg_nonnegative_int(agent_cfg, "max_context_compactions", 64)
    max_total_iters = _cfg_int(
        agent_cfg,
        "max_total_iters",
        max_iters * max(1, max_context_compactions + 1),
        minimum=max_iters,
    )
    json_repair_retries = _cfg_nonnegative_int(agent_cfg, "json_repair_retries", 2)
    context_cfg = resolve_context_budget(settings)
    action_max_new_tokens = min(
        _cfg_int(agent_cfg, "action_max_new_tokens", output_limit_for_mode(settings, "agent")),
        int(context_cfg.get("max_agent_action_tokens") or 2048),
    )
    final_summary_max_new_tokens = min(
        _cfg_int(agent_cfg, "final_summary_max_new_tokens", output_limit_for_mode(settings, "agent", final=True)),
        int(context_cfg.get("max_agent_final_tokens") or 4096),
    )
    max_wall_time_s = _cfg_float(agent_cfg, "max_wall_time_s", 0.0)
    action_grammar_enabled = bool(agent_cfg.get("action_grammar_enabled", True))

    tools_enabled = agent_cfg.get("tools_enabled")
    if tools_enabled is None:
        allowed_tools = set(supported_tools)
    else:
        allowed_tools = {str(item) for item in tools_enabled if item}
    allowed_tools = {tool for tool in allowed_tools if tool in supported_tools}
    if not bool(agent_cfg.get("allow_patch_tools", False)):
        allowed_tools.difference_update({"propose_patch", "sandbox_patch", "apply_patch"})
    workspace_dir = (
        workspace_root
        or (permissions.scope_root if permissions is not None else None)
        or base_dir
    ).resolve()
    effective_permissions = permissions or AgentPermissions.default_full(workspace_dir)
    if not effective_permissions.can_read:
        allowed_tools = {tool for tool in allowed_tools if tool in {"open_docs", "search_web"}}
    else:
        if not effective_permissions.can_run_commands:
            allowed_tools.discard("run_tests")
            allowed_tools.discard("run_command")
            allowed_tools.discard("sandbox_patch")
        if not effective_permissions.can_write:
            allowed_tools.discard("write_file")
            allowed_tools.discard("delete_file")
            allowed_tools.discard("apply_patch")
        if not effective_permissions.can_open_browser:
            allowed_tools.discard("open_browser")
    objective_text = _extract_objective_text(task)
    prefer_file_blocks = _task_requests_code_file(objective_text) and bool(effective_permissions.can_write)
    prompt_tools = set(allowed_tools)
    allowed_prompt_tools = ", ".join(sorted(prompt_tools) + ["finish"])
    tool_schemas = {
        "open_docs": 'open_docs args={"url":"https://...","max_chars":1200?}',
        "search_web": 'search_web args={"query":"...","max_results":5?}',
        "read_file": 'read_file args={"path":"relative/or/absolute","max_chars":4000?}',
        "grep": 'grep args={"pattern":"regex","path_glob":"**/*"?,"max_hits":50?}',
        "list_tree": 'list_tree args={"root":"."?,"max_entries":200?}',
        "write_file": 'write_file args={"path":"lib/main.dart","text":"...","append":false?,"require_exists":true?}',
        "delete_file": 'delete_file args={"path":"relative/path"}',
        "run_tests": 'run_tests args={}',
        "run_command": 'run_command args={"command":"flutter test","cwd":"."?,"timeout_s":120?,"background":false?}',
        "open_browser": 'open_browser args={"url":"http://localhost:3000"}',
        "propose_patch": 'propose_patch args={"goal":"...","changes":{"path":"new text"}?}',
        "sandbox_patch": 'sandbox_patch args={"patch_id":"..."}',
        "apply_patch": 'apply_patch args={"patch_id":"..."}',
        "summarize_diff": "summarize_diff args={}",
    }
    allowed_tool_schemas = [tool_schemas[name] for name in sorted(prompt_tools) if name in tool_schemas]
    permission_context = build_agent_permission_context(effective_permissions)
    if action_grammar_enabled:
        file_output_rule = (
            "For creating or editing files, use write_file JSON with args.path and args.text containing the FULL file content. "
            "Escape newlines as \\n inside JSON strings. Do not output Markdown file blocks."
        )
    else:
        file_output_rule = (
            "For creating or editing files, use write_file with args.path and args.text containing the FULL file content, "
            "or output complete Markdown file blocks using exactly ```file:path/to/file followed by full file content. "
            "Do not use create_file/content. Do not output Action:."
            if prefer_file_blocks
            else "For creating or editing files, prefer complete Markdown file blocks using exactly "
            "```file:path/to/file followed by full file content."
        )
    action_grammar = build_agent_action_json_grammar() if action_grammar_enabled else None
    file_block_rule = (
        "- Output exactly one JSON action. No file blocks when grammar mode is enabled.\n"
        if action_grammar_enabled
        else "- File block example: ```file:lib/main.dart newline FULL DART CODE newline ```.\n"
    )
    system_prompt = (
        "You are an autonomous coding agent. For inspection, commands, deletion, browser, or finish, "
        "respond with ONLY a single minified JSON action object. "
        f"{file_output_rule} "
        "NO prose. NO explanations. Do not say you are ready. The task is already provided.\n"
        "\n"
        "WORKFLOW: 1) Inspect workspace (list_tree/read_file/grep) 2) Decide from actual files 3) Make requested changes (write_file/delete_file) 4) Validate when useful (run_command) 5) finish\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Do not use prepared app templates, canned UI, canned summaries, or stale examples.\n"
        "- Infer target files from the workspace. Inspect before editing unless the path is explicit.\n"
        "- For new Flutter projects you may run `flutter create .` and then edit generated files, or create the required project files manually with write_file. If the command fails, continue by writing files manually.\n"
        "- If the task asks to create/implement/build/add/change UI or code, mutate files before validation.\n"
        "- Do not ask whether to create a requested file or project. The user already asked for it.\n"
        "- Never run validation as the only action for a requested code change.\n"
        "- When writing files, send FULL file content for that file. Never put tool JSON inside file content.\n"
        f"{file_block_rule}"
        "- finish only after the tool results match the requested task. Summary must describe actual tool results, not the plan.\n"
        "\n"
        f"Valid types: {allowed_prompt_tools}. "
        f"Permission context: {permission_context} "
        f"Tool schemas: {'; '.join(allowed_tool_schemas)}"
    )
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current task to execute now. Do not ask for another task.\n{task}"},
    ]
    guard = evaluate_lab_request(messages, settings)
    if guard.get("action") != "allow":
        summary = str(guard.get("message") or "blocked_by_lab_policy")
        _log_episode(
            base_dir,
            {
                "version": 2,
                "ts": time.time(),
                "task": task,
                "prompt": task,
                "summary": summary,
                "tool_calls": [],
                "blocked": True,
            },
        )
        return {"ok": False, "patch_id": None, "tests_ok": False, "summary": summary, "blocked": True, "file_changes": []}
    tools_cfg = settings.get("tools", {}) or {}
    allowlist = resolve_web_allowlist(settings)
    sandbox_root = Path(settings.get("selfimprove", {}).get("sandbox_root", "data/workspaces"))
    self_patch_cfg = dict(settings.get("self_patch", {}) or {})
    safety_cfg = settings.get("continuous", {}).get("safety", {}) or {}
    if safety_cfg:
        self_patch_cfg["safety"] = dict(safety_cfg)
    tools = AgentTools(
        allowlist=list(allowlist or []),
        sandbox_root=sandbox_root,
        web_cfg=tools_cfg,
        agent_cfg=agent_cfg,
        self_patch_cfg=self_patch_cfg,
        security_cfg=settings.get("security", {}) or {},
        repo_root=workspace_dir,
        permissions=effective_permissions,
    )
    current_model = model

    tool_calls: List[dict] = []
    patch_id: str | None = None
    patch_text = ""
    tests_ok = False
    tools_ok = False
    summary = ""
    blocked = False
    browser_actions: List[dict[str, object]] = []
    start_ts = time.monotonic()
    compactions_done = 0
    iterations_done = 0
    invalid_json_count = 0
    confirmation_retry_count = 0
    incomplete_write_counts: dict[str, int] = {}

    model_unavailable_reason = ""
    direct_actions: list[Action] = []
    if action_provider is None and effective_permissions.can_read:
        if effective_permissions.can_run_commands and "run_command" in allowed_tools:
            direct_actions.extend(
                _flutter_project_bootstrap_actions(
                    objective_text,
                    workspace_dir,
                    include_reads="read_file" in allowed_tools,
                )
            )
            direct_actions.extend(_project_run_actions(objective_text, workspace_dir))
        if "read_file" in allowed_tools:
            direct_actions.extend(_flutter_project_inspect_actions(objective_text, workspace_dir))
        direct_actions.extend(_extract_direct_actions(objective_text))
    direct_actions_satisfied = False

    def _missing_required_file_result(args: dict) -> ToolResult | None:
        if not bool(args.get("require_exists", False)):
            return None
        raw_path = str(args.get("path") or "").strip()
        if _task_allows_missing_file_write(objective_text, raw_path):
            args.pop("require_exists", None)
            return None
        target = tools._resolve_safe_path(raw_path)
        if target is None:
            return ToolResult(ok=False, output=f"No puedo acceder a `{raw_path}` dentro del scope autorizado.")
        if not target.exists():
            return ToolResult(
                ok=False,
                output=f"No encuentro el archivo `{raw_path}`. No he creado nada. Dime si quieres que lo cree.",
            )
        if not target.is_file():
            return ToolResult(ok=False, output=f"`{raw_path}` existe, pero no es un archivo editable.")
        return None

    def _invalid_write_text_result(args: dict) -> ToolResult | None:
        raw_path = str(args.get("path") or "").strip()
        text = str(args.get("text") or "")
        if _write_text_looks_incomplete(raw_path, text):
            detail = ""
            if raw_path.replace("\\", "/").lower().endswith("lib/main.dart"):
                detail = (
                    " Para lib/main.dart genera un archivo Dart completo y compilable: "
                    "import material, void main() con runApp, una clase App concreta, "
                    "MaterialApp, Scaffold/Form y widgets cerrados."
                )
            return ToolResult(
                ok=False,
                output=(
                    "incomplete_file_content: el contenido parece un JSON de accion, "
                    "placeholder o codigo incompleto; vuelve a generar el archivo completo."
                    f"{detail}"
                ),
            )
        return None

    def _write_file_result(args: dict) -> tuple[dict, ToolResult]:
        write_args = _json_action_to_write_args(args) or args
        result = _missing_required_file_result(write_args) or _invalid_write_text_result(write_args) or tools.write_file(
            str(write_args.get("path", "")),
            str(write_args.get("text", "")),
            append=bool(write_args.get("append", False)),
        )
        return write_args, result

    def _run_direct_actions(actions: list[Action]) -> bool:
        nonlocal summary, tests_ok, tools_ok
        direct_ok = True
        had_nonfatal_failure = False
        for direct_action in actions:
            command = ""
            if direct_action.type not in allowed_tools:
                direct_result = ToolResult(ok=False, output=f"tool_disabled:{direct_action.type}")
            elif direct_action.type == "write_file":
                direct_action.args, direct_result = _write_file_result(direct_action.args)
            elif direct_action.type == "delete_file":
                direct_result = tools.delete_file(str(direct_action.args.get("path", "")))
            elif direct_action.type == "read_file":
                direct_result = tools.read_file(
                    str(direct_action.args.get("path", "")),
                    max_chars=int(direct_action.args.get("max_chars", 4000)),
                )
            elif direct_action.type == "list_tree":
                direct_result = tools.list_tree(
                    str(direct_action.args.get("root", ".")),
                    max_entries=int(direct_action.args.get("max_entries", 120)),
                )
            elif direct_action.type == "grep":
                direct_result = tools.grep(
                    str(direct_action.args.get("pattern", "")),
                    path_glob=str(direct_action.args.get("path_glob", "**/*")),
                    max_hits=int(direct_action.args.get("max_hits", 50)),
                )
            elif direct_action.type == "run_command":
                command = str(direct_action.args.get("command", ""))
                direct_result = tools.run_command(
                    command,
                    cwd=str(direct_action.args.get("cwd", ".")),
                    timeout_s=int(direct_action.args.get("timeout_s", 120)),
                    background=bool(direct_action.args.get("background", False)),
                )
                if direct_result.ok and _command_looks_like_test(command):
                    tests_ok = True
            else:
                direct_result = ToolResult(ok=False, output=f"tool_unsupported:{direct_action.type}")
            tool_calls.append(
                _tool_call_record(
                    direct_action.type,
                    direct_action.args,
                    direct_result,
                    max_output_chars=4000,
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"type": direct_action.type, "args": direct_action.args},
                        ensure_ascii=True,
                    ),
                }
            )
            messages.append({"role": "tool", "content": direct_result.output[:4000]})
            direct_ok = direct_ok and bool(direct_result.ok)
            if not direct_result.ok:
                if direct_action.type == "run_command" and _direct_command_failure_is_nonfatal(
                    command,
                    direct_result.output,
                ):
                    had_nonfatal_failure = True
                    continue
                break
        tools_ok = direct_ok
        summary = "direct_actions_done" if direct_ok or had_nonfatal_failure else tool_calls[-1]["output"]
        return direct_ok

    def _command_activity_required() -> bool:
        return (
            _task_requires_command_activity(objective_text)
            and effective_permissions.can_run_commands
            and bool({"run_command", "run_tests", "open_browser"} & allowed_tools)
        )

    if direct_actions:
        direct_ok = _run_direct_actions(direct_actions)
        direct_actions_satisfied = bool(direct_ok) or _has_nonblocking_missing_file_mutation(tool_calls)
        if (
            direct_actions_satisfied
            and _task_requires_workspace_change(objective_text)
            and not _attempted_workspace_mutation(tool_calls)
            and not _has_nonblocking_missing_file_mutation(tool_calls)
        ):
            direct_actions_satisfied = False
        if (
            direct_actions_satisfied
            and _command_activity_required()
            and not _attempted_command_activity(tool_calls)
        ):
            direct_actions_satisfied = False

    if (
        not direct_actions_satisfied
        and action_provider is None
        and current_model is None
        and allow_model_load
    ):
        try:
            current_model = load_inference_model(settings)
        except Exception as exc:
            model_unavailable_reason = str(exc)

    if not direct_actions_satisfied and action_provider is None and current_model is None:
        summary = (
            "agent_model_unavailable: no hay modelo cargado para planificar acciones; "
            "las acciones directas siguen disponibles cuando la tarea es determinista."
        )
        if model_unavailable_reason:
            summary = f"{summary} Motivo: {model_unavailable_reason}"
        episode = {
            "version": 2,
            "ts": time.time(),
            "task": task,
            "prompt": task,
            "workspace_root": str(workspace_dir),
            "permissions": effective_permissions.to_dict(),
            "patch_id": None,
            "patch": "",
            "file_changes": [],
            "tests_ok": False,
            "tools_ok": False,
            "summary": summary,
            "tool_calls": tool_calls,
            "max_iters": max_iters,
            "max_total_iters": max_total_iters,
            "max_context_compactions": max_context_compactions,
            "context_compactions_done": compactions_done,
            "iterations_done": iterations_done,
            "invalid_json_count": invalid_json_count,
            "action_max_new_tokens": action_max_new_tokens,
            "final_summary_max_new_tokens": final_summary_max_new_tokens,
            "max_wall_time_s": max_wall_time_s,
            "action_grammar_enabled": action_grammar_enabled,
            "model_unavailable": True,
        }
        backend = settings.get("core", {}).get("backend")
        if backend:
            episode["model_backend"] = str(backend)
        profile = os.getenv("C3RNT2_PROFILE")
        if profile:
            episode["profile"] = profile
        _log_episode(base_dir, episode)
        return {
            "ok": False,
            "patch_id": None,
            "patch": "",
            "file_changes": [],
            "tests_ok": False,
            "tools_ok": False,
            "summary": summary,
            "workspace_root": str(workspace_dir),
            "permissions": effective_permissions.to_dict(),
            "browser_actions": [],
            "tool_calls": tool_calls,
            "action_grammar_enabled": action_grammar_enabled,
            "model_unavailable": True,
        }

    def _agent_context_limit() -> int:
        limits: list[int] = []
        try:
            limits.append(int(resolve_model_context_limit(settings, current_model)))
        except Exception:
            pass
        for key in ("model_max_context_tokens", "default_agent_context_tokens", "max_input_tokens"):
            try:
                value = int(context_cfg.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                limits.append(value)
        return max(512, min(limits or [32768]))

    def _compact_agent_context(reason: str, *, target_prompt_tokens: int | None = None) -> bool:
        nonlocal messages, compactions_done
        if compactions_done >= max_context_compactions:
            return False
        compactions_done += 1
        summary_chars = max(1600, int(context_cfg.get("rolling_summary_tokens") or 1500) * 4)
        if target_prompt_tokens is not None:
            summary_chars = min(summary_chars, max(1200, int(target_prompt_tokens) * 4))
        messages = _compact_agent_messages(
            system_prompt=system_prompt,
            task=task,
            tool_calls=tool_calls,
            reason=f"{reason}_{compactions_done}",
            max_chars=summary_chars,
        )
        return True

    def _prepare_agent_generation(max_new_tokens: int, reason: str) -> tuple[str, int]:
        nonlocal messages
        ctx_max = _agent_context_limit()
        effective_max_new = max(1, min(int(max_new_tokens), max(1, ctx_max - 32)))
        min_response_tokens = max(64, min(256, ctx_max // 8))
        for _attempt in range(max_context_compactions + 2):
            messages = apply_message_budget(messages, settings, mode="agent")
            prompt = build_chat_prompt(
                messages,
                backend=str(settings.get("core", {}).get("backend", "vortex")),
                tokenizer=getattr(current_model, "tokenizer", None),
                default_system=None,
            )
            prompt_tokens = _estimate_model_tokens(prompt, current_model)
            if prompt_tokens + effective_max_new <= ctx_max:
                return prompt, effective_max_new

            allowed_prompt = max(1, ctx_max - effective_max_new - 16)
            if prompt_tokens > allowed_prompt and _compact_agent_context(
                f"{reason}_budget",
                target_prompt_tokens=allowed_prompt,
            ):
                continue

            fit = max(1, ctx_max - prompt_tokens - 16)
            if fit < effective_max_new:
                effective_max_new = max(1, fit)
                if effective_max_new >= min_response_tokens:
                    continue

            if _compact_agent_context(
                f"{reason}_low_output_budget",
                target_prompt_tokens=max(512, ctx_max // 2),
            ):
                effective_max_new = max(min_response_tokens, min(int(max_new_tokens), max(1, ctx_max // 2)))
                continue

            raise RuntimeError("stopped_by_context_compaction_limit")
        raise RuntimeError("stopped_by_context_compaction_limit")

    def _generate_agent_output(reason: str) -> tuple[str, bool]:
        nonlocal blocked, summary, tools_ok
        while True:
            try:
                prompt, effective_max_new = _prepare_agent_generation(action_max_new_tokens, reason)
            except RuntimeError as exc:
                if str(exc) == "stopped_by_context_compaction_limit":
                    blocked = True
                    tools_ok = False
                    summary = (
                        "stopped_by_context_compaction_limit: el contexto sigue siendo demasiado grande "
                        "tras compactarlo; reduce historial o aumenta llama_cpp_ctx."
                    )
                    tool_calls.append(
                        {
                            "action": "agent_context_compaction",
                            "args": {"reason": reason, "limit": _agent_context_limit()},
                            "ok": False,
                            "output": summary,
                        }
                    )
                    return "", False
                raise
            try:
                with (model_lock() if model_lock is not None else nullcontext()):
                    output = current_model.generate(
                        prompt,
                        messages=messages,
                        max_new_tokens=effective_max_new,
                        temperature=0.0,
                        grammar=action_grammar,
                    )
                return str(output or ""), True
            except Exception as exc:
                if _is_context_window_error(exc) and _compact_agent_context(
                    f"{reason}_context_error",
                    target_prompt_tokens=max(512, _agent_context_limit() // 2),
                ):
                    continue
                blocked = True
                tools_ok = False
                summary = f"agent_model_error: {exc}"
                tool_calls.append(
                    {
                        "action": "agent_model_generate",
                        "args": {"reason": reason},
                        "ok": False,
                        "output": str(exc)[:1000],
                    }
                )
                return "", False

    while iterations_done < max_total_iters:
        if direct_actions_satisfied:
            break
        if max_wall_time_s > 0 and (time.monotonic() - start_ts) >= max_wall_time_s:
            summary = "stopped_by_wall_time_limit"
            break
        if iterations_done > 0 and iterations_done % max_iters == 0:
            if compactions_done < max_context_compactions:
                compactions_done += 1
                messages = _compact_agent_messages(
                    system_prompt=system_prompt,
                    task=task,
                    tool_calls=tool_calls,
                    reason=f"context_window_{compactions_done}",
                    max_chars=max(2400, int(context_cfg.get("rolling_summary_tokens") or 1500) * 4),
                )
            else:
                summary = "stopped_by_context_compaction_limit"
                break
        iterations_done += 1
        if action_provider is None and current_model is not None:
            output, generated = _generate_agent_output("action")
            if not generated:
                break
            action, ok = _parse_action(output)
            if not ok:
                ran_markdown_file_actions = False
                generation_stopped = False
                fallback_actions = (
                    _extract_code_write_actions(task, str(output or ""), workspace_dir)
                    if effective_permissions.can_write and "write_file" in allowed_tools
                    else []
                )
                if fallback_actions:
                    ran_markdown_file_actions = _run_direct_actions(fallback_actions)
                if not ran_markdown_file_actions:
                    for _retry in range(max(1, json_repair_retries)):
                        messages.append({
                            "role": "system",
                            "content": (
                                "Previous agent output was not valid Action JSON. "
                                "Return exactly one minified JSON object with type and args. "
                                "Alternatively, for file edits, return complete ```file:path blocks. "
                                "Never put tool JSON inside file content. Continue the task."
                            ),
                        })
                        output, generated = _generate_agent_output(f"json_repair_{_retry + 1}")
                        if not generated:
                            generation_stopped = True
                            break
                        action, ok = _parse_action(output)
                        if ok:
                            break
                        fallback_actions = (
                            _extract_code_write_actions(task, str(output or ""), workspace_dir)
                            if effective_permissions.can_write and "write_file" in allowed_tools
                            else []
                        )
                        if fallback_actions:
                            ran_markdown_file_actions = _run_direct_actions(fallback_actions)
                            if ran_markdown_file_actions:
                                break
                    if generation_stopped:
                        break
                    if ran_markdown_file_actions:
                        break
                    if not ok:
                        invalid_json_count += 1
                        tool_calls.append(
                            {
                                "action": "agent_json_repair",
                                "args": {"attempt": invalid_json_count},
                                "ok": False,
                                "output": str(output or "invalid_json")[:1000],
                            }
                        )
                        if compactions_done < max_context_compactions:
                            compactions_done += 1
                            messages = _compact_agent_messages(
                                system_prompt=system_prompt,
                                task=task,
                                tool_calls=tool_calls,
                                reason=f"invalid_json_{invalid_json_count}",
                                max_chars=max(2400, int(context_cfg.get("rolling_summary_tokens") or 1500) * 4),
                            )
                            continue
                        action = Action(type="finish", args={"summary": "invalid_json"})
                if ran_markdown_file_actions:
                    break
        else:
            action = action_provider(messages)
        messages.append({"role": "assistant", "content": json.dumps({"type": action.type, "args": action.args})})

        if action.type == "finish":
            summary = str(action.args.get("summary", "finished"))
            if (
                _summary_asks_for_confirmation(summary)
                and _task_requires_workspace_change(objective_text)
                and not _attempted_workspace_mutation(tool_calls)
            ):
                if confirmation_retry_count < max(1, json_repair_retries):
                    confirmation_retry_count += 1
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The user already requested the file/project creation. "
                                "Do not ask for confirmation or extra info. Continue now with tools. "
                                "If a target file is missing in a new project task, create it."
                            ),
                        }
                    )
                    continue
                blocked = True
                tools_ok = False
                summary = (
                    "No he aplicado cambios: el agente pidio confirmacion en vez de crear el proyecto. "
                    "No marco la tarea como hecha."
                )
            if _summary_is_model_refusal(summary) and _task_requires_workspace_change(task):
                if _attempted_workspace_mutation(tool_calls):
                    summary = "file_action_done"
                else:
                    blocked = True
                    tools_ok = False
                    summary = (
                        "No he aplicado cambios: el modelo rechazo una tarea de codigo valida. "
                        "No marco la tarea como hecha."
                    )
            if blocked:
                break
            if _task_requires_tool_activity(task) and not tool_calls:
                blocked = True
                tools_ok = False
                summary = (
                    "No he ejecutado acciones: el agente intento terminar sin usar herramientas. "
                    "No marco la tarea como hecha."
                )
            elif _task_requires_workspace_change(task) and not _attempted_workspace_mutation(tool_calls):
                blocked = True
                tools_ok = False
                summary = (
                    "No he aplicado cambios: el agente intento terminar sin modificar archivos. "
                    "No marco la tarea como hecha porque no hay archivos modificados."
                )
            elif _command_activity_required() and not _attempted_command_activity(tool_calls):
                blocked = True
                tools_ok = False
                summary = (
                    "No he ejecutado comandos: el agente intento terminar sin terminal. "
                    "No marco la tarea como hecha."
                )
            break

        result: ToolResult
        stop_after_result = False
        if action.type in supported_tools and action.type not in allowed_tools:
            result = ToolResult(ok=False, output=f"tool_disabled:{action.type}")
        elif action.type == "open_docs":
            result = tools.open_docs(str(action.args.get("url", "")))
        elif action.type == "search_web":
            result = tools.search_web(str(action.args.get("query", "")))
        elif action.type == "read_file":
            path = str(action.args.get("path", ""))
            max_chars = int(action.args.get("max_chars", 4000))
            result = tools.read_file(path, max_chars=max_chars)
        elif action.type == "grep":
            pattern = str(action.args.get("pattern", ""))
            path_glob = str(action.args.get("path_glob", "**/*"))
            max_hits = int(action.args.get("max_hits", 50))
            result = tools.grep(pattern, path_glob=path_glob, max_hits=max_hits)
        elif action.type == "list_tree":
            root = str(action.args.get("root", "."))
            max_entries = int(action.args.get("max_entries", 200))
            result = tools.list_tree(root, max_entries=max_entries)
        elif action.type == "write_file":
            action.args, result = _write_file_result(action.args)
            tools_ok = tools_ok or bool(result.ok)
            if bool(action.args.get("_finish_after_write")):
                summary = "file_action_done" if result.ok else result.output
                tool_chars = max(2000, int(context_cfg.get("reserve_tool_tokens") or 4000) * 4)
                tool_calls.append(
                    _tool_call_record(
                        action.type,
                        action.args,
                        result,
                        max_output_chars=min(4000, tool_chars),
                    )
                )
                messages.append({"role": "tool", "content": result.output[:tool_chars]})
                break
        elif action.type == "delete_file":
            result = tools.delete_file(str(action.args.get("path", "")))
            tools_ok = tools_ok or bool(result.ok)
        elif action.type == "run_tests":
            result = tools.run_tests(workspace_dir)
            tests_ok = bool(result.ok)
        elif action.type == "run_command":
            command = str(action.args.get("command", ""))
            result = tools.run_command(
                command,
                cwd=str(action.args.get("cwd", ".")),
                timeout_s=int(action.args.get("timeout_s", 120)),
                background=bool(action.args.get("background", False)),
            )
            tools_ok = tools_ok or bool(result.ok)
            if result.ok and _command_looks_like_test(command):
                tests_ok = True
        elif action.type == "open_browser":
            result = tools.open_browser(str(action.args.get("url", "")))
            if tools.browser_actions:
                browser_actions = _dedupe_browser_actions(tools.browser_actions)
        elif action.type == "propose_patch":
            goal = str(action.args.get("goal", task))
            changes: Dict[Path, str] = {}
            raw_changes = action.args.get("changes")
            if isinstance(raw_changes, dict):
                for key, value in raw_changes.items():
                    if key:
                        changes[Path(str(key))] = str(value)
            llm_generate = action_provider is None and not changes
            result = tools.propose_patch(
                workspace_dir,
                changes,
                goal=goal,
                llm_generate_diff=llm_generate,
                llm_context={"task": task, "messages": messages, "tool_calls": tool_calls},
            )
            if result.ok:
                patch_id = result.output
                patch_text = _load_patch_from_queue(workspace_dir, settings, patch_id)
                tools_ok = True
        elif action.type == "sandbox_patch":
            pid = str(action.args.get("patch_id") or patch_id or "")
            if not pid or (patch_id and pid != patch_id):
                result = ToolResult(ok=False, output="patch_unavailable: primero genera un patch valido con propose_patch.")
            else:
                result = tools.sandbox_patch(workspace_dir, pid)
            tools_ok = tools_ok or bool(result.ok)
        elif action.type == "apply_patch":
            pid = str(action.args.get("patch_id") or patch_id or "")
            if not pid or (patch_id and pid != patch_id) or (not patch_id and not _load_patch_from_queue(workspace_dir, settings, pid)):
                result = ToolResult(ok=False, output="patch_unavailable: patch.diff no existe; usa write_file con contenido completo.")
            else:
                approve_file = base_dir / "data" / "APPROVE_SELF_PATCH"
                result = tools.apply_patch(
                    workspace_dir,
                    pid,
                    approve=bool(effective_permissions.can_write or approve_file.exists()),
                )
            tools_ok = tools_ok or bool(result.ok)
        elif action.type == "summarize_diff":
            result = tools.summarize_diff(workspace_dir)
        else:
            if action.type != "finish":
                result = ToolResult(ok=False, output=f"tool_unsupported:{action.type}")
            else:
                result = ToolResult(ok=False, output="unknown action")

        tool_chars = max(2000, int(context_cfg.get("reserve_tool_tokens") or 4000) * 4)
        tool_calls.append(
            _tool_call_record(
                action.type,
                action.args,
                result,
                max_output_chars=min(4000, tool_chars),
            )
        )
        messages.append({"role": "tool", "content": result.output[:tool_chars]})
        if (
            action.type == "write_file"
            and not result.ok
            and "incomplete_file_content" in result.output
        ):
            target_path = str(action.args.get("path") or "").replace("\\", "/")
            incomplete_write_counts[target_path] = incomplete_write_counts.get(target_path, 0) + 1
            if action_provider is None and incomplete_write_counts[target_path] >= 2:
                recovery_actions = [
                    item for item in _incomplete_write_recovery_actions(objective_text, workspace_dir, target_path)
                    if item.type in allowed_tools
                ]
                if recovery_actions and _run_direct_actions(recovery_actions):
                    summary = "file_action_done"
                    break
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"The previous write_file for {target_path or 'the target file'} was rejected because "
                        "the file content was incomplete. Continue the same task now. "
                        "Return one write_file action with complete, compilable FULL file content only. "
                        "Do not ask questions, do not finish, and do not repeat the same partial content. "
                        "For Flutter lib/main.dart include import material, void main() => runApp(...), "
                        "a concrete App widget, MaterialApp, Scaffold/Form, username and password fields, "
                        "and a submit button."
                    ),
                }
            )
        if stop_after_result:
            break
        if max_wall_time_s > 0 and (time.monotonic() - start_ts) >= max_wall_time_s:
            summary = "stopped_by_wall_time_limit"
            break
        if max_wall_time_s > 0 and max_wall_time_s < 0.01 and tool_calls:
            summary = "stopped_by_wall_time_limit"
            break

    if _summary_needs_fallback(summary):
        summary = _infer_exists_summary(task, workspace_dir) or summary
    if summary == "file_action_done":
        summary = _file_action_summary(tool_calls) or summary
    if summary == "direct_actions_done":
        summary = _direct_actions_summary(tool_calls) or summary
    if _summary_needs_fallback(summary):
        summary = _generate_final_summary(
            task,
            tool_calls,
            settings,
            current_model,
            model_lock,
            max_new_tokens=final_summary_max_new_tokens,
        ) or summary
    if patch_id and not patch_text:
        patch_text = _load_patch_from_queue(workspace_dir, settings, patch_id)
    file_changes = _file_changes_from_tool_calls(tool_calls)
    file_summary = _file_action_summary(tool_calls)
    if file_changes and file_summary and _summary_looks_like_model_chatter(summary):
        summary = file_summary
    if (
        _task_requires_workspace_change(task)
        and not file_changes
        and not patch_text
        and (
            _has_blocking_failed_mutation(tool_calls)
            or (
                not _attempted_workspace_mutation(tool_calls)
                and not _has_nonblocking_missing_file_mutation(tool_calls)
            )
        )
    ):
        blocked = True
        tools_ok = False
        if _has_blocking_failed_mutation(tool_calls):
            last_blocking = next(
                (
                    str(call.get("output") or "")
                    for call in reversed(tool_calls)
                    if not bool(call.get("ok")) and str(call.get("action") or "") in {"write_file", "delete_file", "apply_patch", "propose_patch", "sandbox_patch"}
                ),
                "",
            )
            if last_blocking:
                summary = last_blocking
        elif _summary_needs_fallback(summary) or summary.strip().lower() not in {"agent_model_unavailable"}:
            summary = (
                "No he aplicado cambios: no hay ningun archivo modificado ni patch generado. "
                "No marco la tarea como hecha."
            )
    if (
        _command_activity_required()
        and not _attempted_command_activity(tool_calls)
        and not blocked
    ):
        blocked = True
        tools_ok = False
        summary = (
            "No he ejecutado comandos: no hay ninguna accion de terminal registrada. "
            "No marco la tarea como hecha."
        )
    if not patch_text and file_changes:
        patch_text = "\n\n".join(
            str(change.get("diff") or "").strip()
            for change in file_changes
            if str(change.get("diff") or "").strip()
        )
    browser_actions = _dedupe_browser_actions(browser_actions or tools.browser_actions)
    prompt_text = _build_prompt(task, tool_calls)
    episode = {
        "version": 2,
        "ts": time.time(),
        "task": task,
        "prompt": prompt_text,
        "workspace_root": str(workspace_dir),
        "permissions": effective_permissions.to_dict(),
        "patch_id": patch_id,
        "patch": patch_text,
        "file_changes": file_changes,
        "tests_ok": tests_ok,
        "tools_ok": tools_ok,
        "summary": summary,
        "tool_calls": tool_calls,
        "blocked": blocked,
        "max_iters": max_iters,
        "max_total_iters": max_total_iters,
        "max_context_compactions": max_context_compactions,
        "context_compactions_done": compactions_done,
        "iterations_done": iterations_done,
        "invalid_json_count": invalid_json_count,
        "action_max_new_tokens": action_max_new_tokens,
        "final_summary_max_new_tokens": final_summary_max_new_tokens,
        "max_wall_time_s": max_wall_time_s,
        "action_grammar_enabled": action_grammar_enabled,
    }
    backend = settings.get("core", {}).get("backend")
    if backend:
        episode["model_backend"] = str(backend)
    profile = os.getenv("C3RNT2_PROFILE")
    if profile:
        episode["profile"] = profile
    _log_episode(base_dir, episode)
    return {
        "ok": not blocked,
        "patch_id": patch_id,
        "patch": patch_text,
        "file_changes": file_changes,
        "tests_ok": tests_ok,
        "tools_ok": tools_ok,
        "summary": summary,
        "workspace_root": str(workspace_dir),
        "permissions": effective_permissions.to_dict(),
        "browser_actions": browser_actions,
        "tool_calls": tool_calls,
        "blocked": blocked,
        "action_grammar_enabled": action_grammar_enabled,
    }
