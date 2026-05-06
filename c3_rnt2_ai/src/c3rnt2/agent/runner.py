from __future__ import annotations

import json
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from ..config import resolve_web_allowlist
from ..context_budget import apply_message_budget, output_limit_for_mode, resolve_context_budget
from ..lab_guard import evaluate_lab_request
from ..model_loader import load_inference_model
from ..multimodal.obsidian_sync import ObsidianSyncService
from ..prompting.chat_format import build_chat_prompt
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
    decoder = json.JSONDecoder()
    payload: Any | None = None
    for start in [index for index, char in enumerate(text) if char == "{"]:
        candidate = text[start:].lstrip()
        try:
            parsed, _end = decoder.raw_decode(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    if payload is None:
        return Action(type="finish", args={"summary": "invalid_json"}), False
    action_type = str(payload.get("type", "finish"))
    args = payload.get("args", {}) or {}
    if not isinstance(args, dict):
        args = {}
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
    parts = [f"Task: {task}".strip()]
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
    lines = [
        f"Agent context compacted after {reason}.",
        "Continue the same task from current workspace state. Do not restart completed work.",
        "Inspect files again when needed, make required changes, validate when practical, then finish.",
        f"Original task: {task}",
    ]
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
        {"role": "user", "content": task},
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


def _task_mentions_any(task: str, words: set[str]) -> bool:
    normalized = str(task or "").lower()
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
            "programa",
            "app",
            "codigo",
            "código",
            "code",
            "flutter",
            "dart",
            "login",
        },
    )


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
    existing_main = workspace_dir / "main.py"
    return "main.py" if existing_main.exists() else "generated_code.txt"


def _extract_code_write_action(task: str, output: str, workspace_dir: Path) -> Action | None:
    if not _task_requests_code_file(task):
        return None
    text = str(output or "")
    if not text.strip():
        return None
    file_block = re.search(r"```file:([^\n`]+)\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if file_block:
        path = file_block.group(1).strip()
        code = file_block.group(2).strip()
        if path and code:
            return Action(type="write_file", args={"path": path, "text": code.rstrip() + "\n", "_finish_after_write": True})

    blocks = re.findall(r"```([A-Za-z0-9_+.-]*)\n([\s\S]*?)```", text)
    for raw_lang, raw_code in blocks:
        lang = str(raw_lang or "").strip().lower()
        code = str(raw_code or "").strip()
        if not code:
            continue
        code_signal = bool(
            lang
            or "void main(" in code
            or "class " in code
            or "import " in code
            or "function " in code
        )
        if not code_signal:
            continue
        path = _infer_code_path(task, lang, code, workspace_dir)
        return Action(type="write_file", args={"path": path, "text": code.rstrip() + "\n", "_finish_after_write": True})

    if "import 'package:flutter/material.dart'" in text or "MaterialApp(" in text:
        path = _infer_code_path(task, "dart", text, workspace_dir)
        return Action(type="write_file", args={"path": path, "text": text.rstrip() + "\n", "_finish_after_write": True})
    return None


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
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        return str(text or "").strip() or fallback_output
    except Exception:
        return fallback_output


def _file_action_summary(tool_calls: List[dict]) -> str:
    for call in reversed(tool_calls):
        if call.get("action") not in {"write_file", "delete_file"} or not call.get("ok"):
            continue
        try:
            payload = json.loads(str(call.get("output") or ""))
        except Exception:
            continue
        path = str(payload.get("path") or "").strip()
        if not path:
            continue
        if call.get("action") == "delete_file":
            return f"Archivo eliminado: {path}"
        return f"Archivo escrito: {path}"
    return ""


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
    create_match = re.search(
        r"(?:crea|crear|create|write)\s+(?:el\s+|un\s+|the\s+|a\s+)?(?:archivo|file)\s+[`\"']?([^`\"'\s]+)[`\"']?\s+(?:con\s+(?:texto|contenido)|with\s+(?:text|content))\s+(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if create_match:
        path = create_match.group(1).strip().rstrip(".,;:")
        content = create_match.group(2).strip()
        content = re.split(
            r"\b(?:no\s+ejecutes|no\s+valides|do\s+not\s+run|don't\s+run|sin\s+tests)\b",
            content,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        content = content.strip("`\"' \t\r\n")
        content = content.rstrip(".")
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

    tools_enabled = agent_cfg.get("tools_enabled")
    if tools_enabled is None:
        allowed_tools = set(supported_tools)
    else:
        allowed_tools = {str(item) for item in tools_enabled if item}
    allowed_tools = {tool for tool in allowed_tools if tool in supported_tools}
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
    allowed_prompt_tools = ", ".join(sorted(allowed_tools) + ["finish"])
    tool_schemas = {
        "open_docs": 'open_docs args={"url":"https://...","max_chars":1200?}',
        "search_web": 'search_web args={"query":"...","max_results":5?}',
        "read_file": 'read_file args={"path":"relative/or/absolute","max_chars":4000?}',
        "grep": 'grep args={"pattern":"regex","path_glob":"**/*"?,"max_hits":50?}',
        "list_tree": 'list_tree args={"root":"."?,"max_entries":200?}',
        "write_file": 'write_file args={"path":"lib/main.dart","text":"...","append":false?}',
        "delete_file": 'delete_file args={"path":"relative/path"}',
        "run_tests": 'run_tests args={}',
        "run_command": 'run_command args={"command":"flutter test","cwd":"."?,"timeout_s":120?,"background":false?}',
        "open_browser": 'open_browser args={"url":"http://localhost:3000"}',
        "propose_patch": 'propose_patch args={"goal":"...","changes":{"path":"new text"}?}',
        "sandbox_patch": 'sandbox_patch args={"patch_id":"..."}',
        "apply_patch": 'apply_patch args={"patch_id":"..."}',
        "summarize_diff": "summarize_diff args={}",
    }
    allowed_tool_schemas = [tool_schemas[name] for name in sorted(allowed_tools) if name in tool_schemas]
    permission_context = build_agent_permission_context(effective_permissions)
    system_prompt = (
        "You are an autonomous coding agent working like Codex. "
        "Inspect the workspace, edit files, run useful validation, and keep going until the user task is complete. "
        "You must respond with a single minified JSON object Action{type,args}. "
        "Do not use markdown. Do not add prose. "
        "If the task asks to create, modify, or delete files, use write_file, apply_patch, or delete_file before finish. "
        "If full permissions allow commands, run the relevant validation command before finish when practical. "
        "Finish only when the requested work is done or a real blocker remains. "
        f"Valid types: {allowed_prompt_tools}. "
        f"Permission context: {permission_context} "
        f"Tool schemas: {'; '.join(allowed_tool_schemas)}"
    )
    try:
        obsidian_budget = int(context_cfg.get("obsidian_tokens") or 0)
        if obsidian_budget > 0:
            obsidian = ObsidianSyncService(settings=settings, base_dir=base_dir)
            obsidian_context = obsidian.build_context(task, max_tokens=obsidian_budget, top_k=6)
            obsidian_text = str(obsidian_context.get("text") or "").strip()
            if obsidian_text:
                system_prompt += f" Curated Obsidian memory follows. Use when relevant and keep paths traceable. {obsidian_text}"
    except Exception:
        pass
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
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
        return {"ok": False, "patch_id": None, "tests_ok": False, "summary": summary, "blocked": True}
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
    if action_provider is None and current_model is None:
        current_model = load_inference_model(settings)

    tool_calls: List[dict] = []
    patch_id: str | None = None
    patch_text = ""
    tests_ok = False
    tools_ok = False
    summary = ""
    browser_actions: List[dict[str, object]] = []
    start_ts = time.monotonic()
    compactions_done = 0
    iterations_done = 0
    invalid_json_count = 0

    direct_action = (
        _extract_direct_file_action(task)
        if action_provider is None and effective_permissions.can_write
        else None
    )
    if direct_action is not None and direct_action.type in allowed_tools:
        if direct_action.type == "write_file":
            direct_result = tools.write_file(
                str(direct_action.args.get("path", "")),
                str(direct_action.args.get("text", "")),
            )
        elif direct_action.type == "delete_file":
            direct_result = tools.delete_file(str(direct_action.args.get("path", "")))
        else:
            direct_result = ToolResult(ok=False, output=f"tool_unsupported:{direct_action.type}")
        tool_calls.append(
            {
                "action": direct_action.type,
                "args": direct_action.args,
                "ok": direct_result.ok,
                "output": direct_result.output[:4000],
            }
        )
        tools_ok = bool(direct_result.ok)
        summary = "file_action_done" if direct_result.ok else direct_result.output

    while iterations_done < max_total_iters:
        if direct_action is not None:
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
            messages = apply_message_budget(messages, settings, mode="agent")
            prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(current_model, "tokenizer", None), default_system=None)
            with (model_lock() if model_lock is not None else nullcontext()):
                output = current_model.generate(prompt, max_new_tokens=action_max_new_tokens, temperature=0.0)
            action, ok = _parse_action(output)
            if not ok:
                fallback_action = (
                    _extract_code_write_action(task, str(output or ""), workspace_dir)
                    if effective_permissions.can_write and "write_file" in allowed_tools
                    else None
                )
                if fallback_action is not None:
                    action = fallback_action
                    ok = True
                else:
                    for _retry in range(max(1, json_repair_retries)):
                        messages.append({
                            "role": "system",
                            "content": (
                                "Previous agent output was not valid Action JSON. "
                                "Return exactly one minified JSON object with type and args. Continue the task."
                            ),
                        })
                        messages = apply_message_budget(messages, settings, mode="agent")
                        prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(current_model, "tokenizer", None), default_system=None)
                        with (model_lock() if model_lock is not None else nullcontext()):
                            output = current_model.generate(prompt, max_new_tokens=action_max_new_tokens, temperature=0.0)
                        action, ok = _parse_action(output)
                        if ok:
                            break
                        fallback_action = (
                            _extract_code_write_action(task, str(output or ""), workspace_dir)
                            if effective_permissions.can_write and "write_file" in allowed_tools
                            else None
                        )
                        if fallback_action is not None:
                            action = fallback_action
                            ok = True
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
        else:
            action = action_provider(messages)
        messages.append({"role": "assistant", "content": json.dumps({"type": action.type, "args": action.args})})

        if action.type == "finish":
            summary = str(action.args.get("summary", "finished"))
            break

        result: ToolResult
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
            result = tools.write_file(
                str(action.args.get("path", "")),
                str(action.args.get("text", "")),
                append=bool(action.args.get("append", False)),
            )
            tools_ok = tools_ok or bool(result.ok)
            if bool(action.args.get("_finish_after_write")):
                summary = "file_action_done" if result.ok else result.output
                tool_chars = max(2000, int(context_cfg.get("reserve_tool_tokens") or 4000) * 4)
                tool_calls.append({"action": action.type, "args": action.args, "ok": result.ok, "output": result.output[: min(4000, tool_chars)]})
                messages.append({"role": "tool", "content": result.output[:tool_chars]})
                break
        elif action.type == "delete_file":
            result = tools.delete_file(str(action.args.get("path", "")))
            tools_ok = tools_ok or bool(result.ok)
        elif action.type == "run_tests":
            result = tools.run_tests(workspace_dir)
            tests_ok = bool(result.ok)
        elif action.type == "run_command":
            result = tools.run_command(
                str(action.args.get("command", "")),
                cwd=str(action.args.get("cwd", ".")),
                timeout_s=int(action.args.get("timeout_s", 120)),
                background=bool(action.args.get("background", False)),
            )
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
            pid = str(action.args.get("patch_id", patch_id or ""))
            if pid and not patch_id:
                patch_id = pid
            result = tools.sandbox_patch(workspace_dir, pid)
            tools_ok = tools_ok or bool(result.ok)
        elif action.type == "apply_patch":
            pid = str(action.args.get("patch_id", patch_id or ""))
            if pid and not patch_id:
                patch_id = pid
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
        tool_calls.append({"action": action.type, "args": action.args, "ok": result.ok, "output": result.output[: min(4000, tool_chars)]})
        messages.append({"role": "tool", "content": result.output[:tool_chars]})
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
        "tests_ok": tests_ok,
        "tools_ok": tools_ok,
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
    }
    backend = settings.get("core", {}).get("backend")
    if backend:
        episode["model_backend"] = str(backend)
    profile = os.getenv("C3RNT2_PROFILE")
    if profile:
        episode["profile"] = profile
    _log_episode(base_dir, episode)
    return {
        "ok": True,
        "patch_id": patch_id,
        "patch": patch_text,
        "tests_ok": tests_ok,
        "tools_ok": tools_ok,
        "summary": summary,
        "workspace_root": str(workspace_dir),
        "permissions": effective_permissions.to_dict(),
        "browser_actions": browser_actions,
        "tool_calls": tool_calls,
    }
