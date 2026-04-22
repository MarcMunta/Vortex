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
from ..lab_guard import evaluate_lab_request
from ..model_loader import load_inference_model
from ..prompting.chat_format import build_chat_prompt
from .permissions import AgentPermissions, build_agent_permission_context
from .tools import AgentTools, ToolResult


@dataclass
class Action:
    type: str
    args: dict


def _parse_action(text: str) -> Action:
    text = text.strip()
    if not text:
        return Action(type="finish", args={"summary": "empty"}), False
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return Action(type="finish", args={"summary": "invalid_json"}), False
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return Action(type="finish", args={"summary": "invalid_json"}), False
    action_type = str(payload.get("type", "finish"))
    args = payload.get("args", {}) or {}
    return Action(type=action_type, args=args), True


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


def _summary_needs_fallback(summary: str) -> bool:
    normalized = str(summary or "").strip().lower()
    return normalized in {"", "agent_finished", "done", "empty", "finished", "invalid_json"}


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
                max_new_tokens=160,
                temperature=0.0,
            )
        return str(text or "").strip() or fallback_output
    except Exception:
        return fallback_output


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
    max_iters: int = 5,
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
        "run_tests",
        "run_command",
        "open_browser",
        "propose_patch",
        "sandbox_patch",
        "apply_patch",
        "summarize_diff",
    ]
    agent_cfg = settings.get("agent", {}) or {}
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
        "You are an autonomous coding agent. "
        "You must respond with a single minified JSON object Action{type,args}. "
        "Do not use markdown. Do not add prose. "
        f"Valid types: {allowed_prompt_tools}. "
        f"Permission context: {permission_context} "
        f"Tool schemas: {'; '.join(allowed_tool_schemas)}"
    )
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

    for _ in range(max_iters):
        if action_provider is None and current_model is not None:
            prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(current_model, "tokenizer", None), default_system=None)
            with (model_lock() if model_lock is not None else nullcontext()):
                output = current_model.generate(prompt, max_new_tokens=256, temperature=0.0)
            action, ok = _parse_action(output)
            if not ok:
                messages.append({"role": "system", "content": "JSON ONLY. No markdown."})
                prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(current_model, "tokenizer", None), default_system=None)
                with (model_lock() if model_lock is not None else nullcontext()):
                    output = current_model.generate(prompt, max_new_tokens=256, temperature=0.0)
                action, ok = _parse_action(output)
                if not ok:
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

        tool_calls.append({"action": action.type, "args": action.args, "ok": result.ok, "output": result.output[:1000]})
        messages.append({"role": "tool", "content": result.output[:2000]})

    if _summary_needs_fallback(summary):
        summary = _infer_exists_summary(task, workspace_dir) or summary
    if _summary_needs_fallback(summary):
        summary = _generate_final_summary(
            task,
            tool_calls,
            settings,
            current_model,
            model_lock,
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
        "summary": summary,
        "workspace_root": str(workspace_dir),
        "permissions": effective_permissions.to_dict(),
        "browser_actions": browser_actions,
        "tool_calls": tool_calls,
    }
