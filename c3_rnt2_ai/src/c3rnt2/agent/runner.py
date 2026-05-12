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


def _task_requires_workspace_change(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    return any(
        word in normalized
        for word in (
            "anade",
            "añade",
            "agrega",
            "add",
            "boton",
            "button",
            "build",
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
            "genera",
            "generar",
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
        path = str(payload.get("path") or "").strip()
        if not path:
            continue
        if call.get("action") == "delete_file":
            if path not in deleted:
                deleted.append(path)
            continue
        was_created = bool(payload.get("created", False))
        target = created if was_created else updated
        if path not in target:
            target.append(path)
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


def _clean_direct_file_content(content: str) -> str:
    cleaned = str(content or "").strip().strip("`\"' \t\r\n")
    cleaned = re.split(
        r"\b(?:no\s+ejecutes|no\s+valides|do\s+not\s+run|don't\s+run|sin\s+tests)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return cleaned.rstrip(".").strip("`\"' \t\r\n")


def _normalish_legacy(text: str) -> str:
    replacements = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return str(text or "").translate(replacements).lower()


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


def _requests_flutter_login_project(task: str, *, assume_flutter: bool = False) -> bool:
    normalized_objective = _normalish(_extract_objective_text(task))
    wants_project = any(word in normalized_objective for word in ("proyecto", "project", "app", "crea", "crear", "haz", "hacer", "implementa", "genera", "create", "build"))
    wants_login = "login" in normalized_objective or "inicio de sesion" in normalized_objective
    mentions_flutter = assume_flutter or "flutter" in normalized_objective or "dart" in normalized_objective
    return wants_project and wants_login and mentions_flutter


def _requests_flutter_basic_project(task: str, *, assume_flutter: bool = False) -> bool:
    normalized_objective = _normalish(_extract_objective_text(task))
    mentions_flutter = assume_flutter or "flutter" in normalized_objective or "dart" in normalized_objective
    wants_project = any(word in normalized_objective for word in ("proyecto", "project", "app", "crea", "crear", "haz", "hacer", "genera", "create", "build"))
    wants_code = any(word in normalized_objective for word in ("codigo", "code", "basico", "basic"))
    wants_runnable = any(
        word in normalized_objective
        for word in (
            "ejecuta",
            "ejecutar",
            "emulador",
            "emulator",
            "run",
            "runnable",
            "funcione",
            "funcionar",
        )
    )
    return mentions_flutter and wants_project and wants_code and wants_runnable


def _flutter_basic_project_actions() -> list[Action]:
    pubspec = """name: vortex_flutter_app
description: Basic runnable Flutter project generated by Vortex.
publish_to: "none"
version: 1.0.0+1

environment:
  sdk: ">=3.3.0 <4.0.0"

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
    main_dart = """import 'package:flutter/material.dart';

void main() {
  runApp(const VortexFlutterApp());
}

class VortexFlutterApp extends StatelessWidget {
  const VortexFlutterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Vortex Flutter App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _count = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vortex Flutter App')),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Proyecto Flutter listo para ejecutar'),
            const SizedBox(height: 12),
            Text('Clicks: $_count', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 20),
            FilledButton(
              onPressed: () => setState(() => _count++),
              child: const Text('Sumar'),
            ),
          ],
        ),
      ),
    );
  }
}
"""
    widget_test = """import 'package:flutter_test/flutter_test.dart';
import 'package:vortex_flutter_app/main.dart';

void main() {
  testWidgets('counter increments', (WidgetTester tester) async {
    await tester.pumpWidget(const VortexFlutterApp());
    expect(find.text('Clicks: 0'), findsOneWidget);
    await tester.tap(find.text('Sumar'));
    await tester.pump();
    expect(find.text('Clicks: 1'), findsOneWidget);
  });
}
"""
    readme = """# Vortex Flutter App

Basic runnable Flutter project.

Run:

```bash
flutter pub get
flutter run
```

Validate:

```bash
flutter analyze
flutter test
```
"""
    return [
        Action(type="write_file", args={"path": "pubspec.yaml", "text": pubspec}),
        Action(type="write_file", args={"path": "lib/main.dart", "text": main_dart}),
        Action(type="write_file", args={"path": "test/widget_test.dart", "text": widget_test}),
        Action(type="write_file", args={"path": "README.md", "text": readme}),
    ]


def _flutter_login_project_actions() -> list[Action]:
    pubspec = """name: vortex_login_app
description: Basic Flutter login project generated by Vortex.
publish_to: "none"
version: 1.0.0+1

environment:
  sdk: ">=3.3.0 <4.0.0"

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
    main_dart = """import 'package:flutter/material.dart';

void main() {
  runApp(const VortexLoginApp());
}

class VortexLoginApp extends StatelessWidget {
  const VortexLoginApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vortex Login',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginPage(),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;
  bool _loggedIn = false;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _submit() {
    final isValid = _formKey.currentState?.validate() ?? false;
    if (!isValid) return;
    setState(() => _loggedIn = true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: _loggedIn ? _buildHomeCard(context) : _buildLoginCard(context),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoginCard(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Login', style: Theme.of(context).textTheme.headlineMedium),
              const SizedBox(height: 24),
              TextFormField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: const InputDecoration(
                  labelText: 'Email',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  final text = value?.trim() ?? '';
                  if (text.isEmpty) return 'Enter your email';
                  if (!text.contains('@')) return 'Enter a valid email';
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _passwordController,
                obscureText: _obscurePassword,
                decoration: InputDecoration(
                  labelText: 'Password',
                  border: const OutlineInputBorder(),
                  suffixIcon: IconButton(
                    tooltip: _obscurePassword ? 'Show password' : 'Hide password',
                    onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                    icon: Icon(_obscurePassword ? Icons.visibility : Icons.visibility_off),
                  ),
                ),
                validator: (value) {
                  final text = value ?? '';
                  if (text.length < 6) return 'Use at least 6 characters';
                  return null;
                },
              ),
              const SizedBox(height: 20),
              FilledButton(
                onPressed: _submit,
                child: const Text('Sign in'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHomeCard(BuildContext context) {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Welcome', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 8),
            Text('Signed in as ${_emailController.text.trim()}'),
            const SizedBox(height: 20),
            OutlinedButton(
              onPressed: () {
                _passwordController.clear();
                setState(() => _loggedIn = false);
              },
              child: const Text('Sign out'),
            ),
          ],
        ),
      ),
    );
  }
}
"""
    widget_test = """import 'package:flutter_test/flutter_test.dart';
import 'package:vortex_login_app/main.dart' as app;

void main() {
  testWidgets('app starts', (WidgetTester tester) async {
    app.main();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
"""
    readme = """# Vortex Login App

Basic Flutter login app.

Run:

```bash
flutter pub get
flutter run
```

Validate:

```bash
flutter analyze
flutter test
```
"""
    return [
        Action(type="write_file", args={"path": "pubspec.yaml", "text": pubspec}),
        Action(type="write_file", args={"path": "lib/main.dart", "text": main_dart}),
        Action(type="write_file", args={"path": "test/widget_test.dart", "text": widget_test}),
        Action(type="write_file", args={"path": "README.md", "text": readme}),
    ]


def _flutter_login_project_support_actions(workspace_dir: Path) -> list[Action]:
    actions: list[Action] = []
    for action in _flutter_login_project_actions():
        path = str(action.args.get("path") or "")
        if not path or path == "lib/main.dart":
            continue
        if (workspace_dir / path).exists():
            continue
        actions.append(action)
    return actions


def _requests_flutter_forgot_password_button(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    wants_button = "boton" in normalized or "button" in normalized
    mentions_password = "password" in normalized or "contras" in normalized
    wants_recovery = any(
        word in normalized
        for word in ("olvid", "recuerda", "acuerda", "recuper", "forgot")
    )
    return wants_button and mentions_password and wants_recovery


def _flutter_forgot_password_button_actions(
    task: str,
    workspace_dir: Path,
    *,
    include_commands: bool = False,
) -> list[Action]:
    if not _requests_flutter_forgot_password_button(task):
        return []
    main_path = workspace_dir / "lib" / "main.dart"
    if not main_path.exists() or not main_path.is_file():
        return []
    try:
        main_text = main_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    button_text = "Olvidaste la contrasena?"
    actions: list[Action] = []
    if button_text not in main_text:
        sign_in_button = """              FilledButton(
                onPressed: _submit,
                child: const Text('Sign in'),
              ),"""
        forgot_button = sign_in_button + """
              const SizedBox(height: 8),
              TextButton(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Recuperacion de contrasena pendiente')),
                  );
                },
                child: const Text('Olvidaste la contrasena?'),
              ),"""
        if sign_in_button not in main_text:
            return []
        actions.append(
            Action(
                type="write_file",
                args={
                    "path": "lib/main.dart",
                    "text": main_text.replace(sign_in_button, forgot_button, 1),
                    "require_exists": True,
                },
            )
        )
    test_path = workspace_dir / "test" / "widget_test.dart"
    if test_path.exists() and test_path.is_file():
        try:
            test_text = test_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            test_text = ""
        if button_text not in test_text and "expect(tester.takeException(), isNull);" in test_text:
            actions.append(
                Action(
                    type="write_file",
                    args={
                        "path": "test/widget_test.dart",
                        "text": test_text.replace(
                            "    expect(tester.takeException(), isNull);",
                            "    expect(tester.takeException(), isNull);\n"
                            "    expect(find.text('Olvidaste la contrasena?'), findsOneWidget);",
                            1,
                        ),
                        "require_exists": True,
                    },
                )
            )
    if actions and include_commands:
        actions.append(Action(type="run_command", args={"command": "flutter test", "cwd": ".", "timeout_s": 240}))
    return actions


def _requests_flutter_dark_mode_button(task: str) -> bool:
    normalized = _normalish(_extract_objective_text(task))
    wants_button = "boton" in normalized or "button" in normalized
    wants_dark = "modo oscuro" in normalized or "dark mode" in normalized or "tema oscuro" in normalized
    wants_animation = "animacion" in normalized or "animado" in normalized or "transition" in normalized or "transicion" in normalized
    return wants_button and wants_dark and (wants_animation or "color" in normalized)


def _flutter_dark_mode_button_actions(
    task: str,
    workspace_dir: Path,
    *,
    include_commands: bool = False,
) -> list[Action]:
    if not _requests_flutter_dark_mode_button(task):
        return []
    main_path = workspace_dir / "lib" / "main.dart"
    if not main_path.exists() or not main_path.is_file():
        return []
    try:
        main_text = main_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    actions: list[Action] = []
    new_main = main_text
    if "themeMode:" not in new_main:
        old_app = """class VortexLoginApp extends StatelessWidget {
  const VortexLoginApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vortex Login',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const LoginPage(),
    );
  }
}
"""
        new_app = """class VortexLoginApp extends StatefulWidget {
  const VortexLoginApp({super.key});

  @override
  State<VortexLoginApp> createState() => _VortexLoginAppState();
}

class _VortexLoginAppState extends State<VortexLoginApp> {
  bool _isDarkMode = false;

  void _toggleTheme() {
    setState(() => _isDarkMode = !_isDarkMode);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vortex Login',
      debugShowCheckedModeBanner: false,
      themeMode: _isDarkMode ? ThemeMode.dark : ThemeMode.light,
      themeAnimationDuration: const Duration(milliseconds: 450),
      themeAnimationCurve: Curves.easeInOutCubic,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo, brightness: Brightness.light),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: LoginPage(
        isDarkMode: _isDarkMode,
        onToggleTheme: _toggleTheme,
      ),
    );
  }
}
"""
        if old_app not in new_main:
            return []
        new_main = new_main.replace(old_app, new_app, 1)
    if "final bool isDarkMode;" not in new_main:
        old_login = """class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}
"""
        new_login = """class LoginPage extends StatefulWidget {
  const LoginPage({
    super.key,
    required this.isDarkMode,
    required this.onToggleTheme,
  });

  final bool isDarkMode;
  final VoidCallback onToggleTheme;

  @override
  State<LoginPage> createState() => _LoginPageState();
}
"""
        if old_login not in new_main:
            return []
        new_main = new_main.replace(old_login, new_login, 1)
    if "AnimatedSwitcher" not in new_main:
        old_title = "              Text('Login', style: Theme.of(context).textTheme.headlineMedium),"
        theme_button = """              Align(
                alignment: Alignment.centerRight,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 350),
                  curve: Curves.easeInOut,
                  decoration: BoxDecoration(
                    color: widget.isDarkMode ? Colors.white10 : Colors.black12,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: IconButton(
                    tooltip: widget.isDarkMode ? 'Modo claro' : 'Modo oscuro',
                    onPressed: widget.onToggleTheme,
                    icon: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 250),
                      transitionBuilder: (child, animation) {
                        return RotationTransition(
                          turns: animation,
                          child: FadeTransition(opacity: animation, child: child),
                        );
                      },
                      child: Icon(
                        widget.isDarkMode ? Icons.light_mode : Icons.dark_mode,
                        key: ValueKey(widget.isDarkMode),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Text('Login', style: Theme.of(context).textTheme.headlineMedium),"""
        if old_title not in new_main:
            return []
        new_main = new_main.replace(old_title, theme_button, 1)
    if new_main != main_text:
        actions.append(
            Action(
                type="write_file",
                args={"path": "lib/main.dart", "text": new_main, "require_exists": True},
            )
        )

    test_path = workspace_dir / "test" / "widget_test.dart"
    if test_path.exists() and test_path.is_file():
        try:
            test_text = test_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            test_text = ""
        if "find.byTooltip('Modo oscuro')" not in test_text and "expect(tester.takeException(), isNull);" in test_text:
            actions.append(
                Action(
                    type="write_file",
                    args={
                        "path": "test/widget_test.dart",
                        "text": test_text.replace(
                            "    expect(tester.takeException(), isNull);",
                            "    expect(tester.takeException(), isNull);\n"
                            "    expect(find.byTooltip('Modo oscuro'), findsOneWidget);\n"
                            "    await tester.tap(find.byTooltip('Modo oscuro'));\n"
                            "    await tester.pumpAndSettle();\n"
                            "    expect(find.byTooltip('Modo claro'), findsOneWidget);",
                            1,
                        ),
                        "require_exists": True,
                    },
                )
            )
    if actions and include_commands:
        actions.append(Action(type="run_command", args={"command": "flutter test", "cwd": ".", "timeout_s": 240}))
    return actions


def _requests_flutter_terminal_actions(task: str) -> bool:
    normalized = _normalish(task)
    return any(
        word in normalized
        for word in (
            "terminal",
            "comando",
            "comandos",
            "validar",
            "prueba",
            "probar",
            "test",
            "emulador",
            "emulator",
        )
    )


def _requests_flutter_emulator_run(task: str) -> bool:
    normalized = _normalish(task)
    return "emulador" in normalized or "emulator" in normalized


def _flutter_android_platform_missing(workspace_dir: Path | None) -> bool:
    if workspace_dir is None:
        return False
    android_dir = workspace_dir / "android"
    manifest_candidates = [
        android_dir / "app" / "src" / "main" / "AndroidManifest.xml",
        android_dir / "AndroidManifest.xml",
    ]
    return not android_dir.exists() or not any(path.exists() for path in manifest_candidates)


def _flutter_android_platform_actions(workspace_dir: Path | None) -> list[Action]:
    if not _flutter_android_platform_missing(workspace_dir):
        return []
    return [
        Action(
            type="run_command",
            args={
                "command": "flutter create --platforms=android --no-overwrite .",
                "cwd": ".",
                "timeout_s": 240,
            },
        )
    ]


def _requests_flutter_existing_project_run(task: str, *, assume_flutter: bool = False) -> bool:
    normalized_objective = _normalish(_extract_objective_text(task))
    mentions_flutter = assume_flutter or "flutter" in normalized_objective or "dart" in normalized_objective
    wants_run = any(
        word in normalized_objective
        for word in (
            "ejecuta",
            "ejecute",
            "ejecutes",
            "ejecutar",
            "corre",
            "corra",
            "corras",
            "correr",
            "inicia",
            "inicie",
            "inicies",
            "iniciar",
            "run",
            "start",
            "launch",
        )
    )
    mentions_project = any(word in normalized_objective for word in ("proyecto", "project", "app", "aplicacion"))
    asks_for_new_code = any(
        word in normalized_objective
        for word in (
            "crea",
            "crear",
            "haz",
            "hacer",
            "implementa",
            "genera",
            "nuevo",
            "login",
            "basico",
            "basic",
            "create",
            "build",
        )
    )
    return mentions_flutter and wants_run and mentions_project and not asks_for_new_code


def _flutter_project_command_actions(task: str, *, workspace_dir: Path | None = None) -> list[Action]:
    if not _requests_flutter_terminal_actions(task):
        return []
    actions = [
        Action(type="run_command", args={"command": "flutter --version", "cwd": ".", "timeout_s": 60}),
        *_flutter_android_platform_actions(workspace_dir),
        Action(type="run_command", args={"command": "flutter devices", "cwd": ".", "timeout_s": 60}),
        Action(type="run_command", args={"command": "flutter pub get", "cwd": ".", "timeout_s": 180}),
        Action(type="run_command", args={"command": "flutter test", "cwd": ".", "timeout_s": 240}),
    ]
    if _requests_flutter_emulator_run(task):
        actions.extend(
            [
                Action(
                    type="run_command",
                    args={
                        "command": "flutter run -d emulator-5554 --debug",
                        "cwd": ".",
                        "timeout_s": 120,
                        "background": True,
                    },
                ),
            ]
        )
    return actions


def _flutter_existing_project_run_actions(
    task: str,
    *,
    assume_flutter: bool = False,
    workspace_dir: Path | None = None,
) -> list[Action]:
    if not _requests_flutter_existing_project_run(task, assume_flutter=assume_flutter):
        return []
    return [
        Action(type="run_command", args={"command": "flutter --version", "cwd": ".", "timeout_s": 60}),
        *_flutter_android_platform_actions(workspace_dir),
        Action(type="run_command", args={"command": "flutter devices", "cwd": ".", "timeout_s": 60}),
        Action(type="run_command", args={"command": "flutter pub get", "cwd": ".", "timeout_s": 180}),
        Action(type="run_command", args={"command": "flutter test", "cwd": ".", "timeout_s": 240}),
        Action(
            type="run_command",
            args={
                "command": "flutter run -d emulator-5554 --debug",
                "cwd": ".",
                "timeout_s": 120,
                "background": True,
            },
        ),
    ]


def _flutter_project_fallback_actions(
    task: str,
    *,
    include_commands: bool = False,
    assume_flutter: bool = False,
    workspace_dir: Path | None = None,
) -> list[Action]:
    actions: list[Action]
    if _requests_flutter_login_project(task, assume_flutter=assume_flutter):
        actions = _flutter_login_project_actions()
    elif _requests_flutter_basic_project(task, assume_flutter=assume_flutter):
        actions = _flutter_basic_project_actions()
    else:
        return []
    if include_commands:
        actions.extend(_flutter_project_command_actions(task, workspace_dir=workspace_dir))
    return actions


def _extract_direct_actions(task: str) -> list[Action]:
    text = str(task or "").strip()
    if not text:
        return []
    actions: list[tuple[int, Action]] = []
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
    for call in tool_calls:
        action = str(call.get("action") or "")
        if action not in {"write_file", "delete_file", "run_command"}:
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
            else:
                if path not in updated:
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
    parts: list[str] = []
    if created:
        parts.append(f"He creado `{_join_natural(created)}`.")
    if updated:
        parts.append(f"He actualizado `{_join_natural(updated)}`.")
    if deleted:
        parts.append(f"He borrado `{_join_natural(deleted)}`.")
    if any("flutter test" in command for command in command_names):
        parts.append("He validado el proyecto con sus tests.")
    failed_emulator_commands = [
        (command, output)
        for command, output in failed_commands
        if _command_looks_like_flutter_emulator(command)
    ]
    if any("flutter run" in command for command in command_names) and not failed_emulator_commands:
        parts.append("He iniciado la app en el emulador.")
    elif any("emulators --launch" in command for command in command_names) and not failed_emulator_commands:
        parts.append("He iniciado el emulador.")
    if failed_emulator_commands:
        parts.append(_flutter_emulator_failure_summary(failed_emulator_commands))
    elif failed_commands and not parts:
        command, output = failed_commands[-1]
        parts.append(_command_failure_summary(command, output))
    if parts:
        return " ".join(parts)
    if not tool_calls:
        return ""
    return "He terminado la tarea."


def _attempted_workspace_mutation(tool_calls: List[dict]) -> bool:
    return any(
        str(call.get("action") or "") in {"write_file", "delete_file", "apply_patch", "propose_patch", "sandbox_patch"}
        for call in tool_calls
    )


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
    return "flutter emulators" in normalized or "flutter run" in normalized


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
    changes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
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
        key = (path, diff)
        if key in seen:
            continue
        seen.add(key)
        change = {"path": path, "diff": diff}
        absolute_path = str(meta.get("absolute_path") or "").strip()
        if absolute_path:
            change["absolute_path"] = absolute_path
        changes.append(change)
    return changes


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
    allowed_tool_schemas = [tool_schemas[name] for name in sorted(allowed_tools) if name in tool_schemas]
    permission_context = build_agent_permission_context(effective_permissions)
    system_prompt = (
        "You are an autonomous coding agent working like Codex. "
        "Inspect the workspace, edit files, run useful validation, and keep going until the user task is complete. "
        "You must respond with a single minified JSON object Action{type,args}. "
        "Do not use markdown. Do not add prose. "
        "If the task asks to create, modify, or delete files, use write_file, apply_patch, or delete_file before finish. "
        "For edits to an existing file, check that the target exists first or set require_exists=true; if it is missing, finish with a clear blocker and do not create it unless the user asked to create it. "
        "For deletes, if the target is missing, report that clearly instead of doing nothing. "
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

    model_unavailable_reason = ""
    objective_text = _extract_objective_text(task)
    direct_actions = (
        _extract_direct_actions(objective_text)
        if action_provider is None and effective_permissions.can_write
        else []
    )
    assume_flutter_workspace = _workspace_looks_flutter(workspace_dir)
    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and assume_flutter_workspace
        and effective_permissions.can_write
        and "write_file" in allowed_tools
    ):
        direct_actions = _flutter_dark_mode_button_actions(
            objective_text,
            workspace_dir,
            include_commands=effective_permissions.can_run_commands and "run_command" in allowed_tools,
        )
    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and assume_flutter_workspace
        and effective_permissions.can_write
        and "write_file" in allowed_tools
    ):
        direct_actions = _flutter_forgot_password_button_actions(
            objective_text,
            workspace_dir,
            include_commands=effective_permissions.can_run_commands and "run_command" in allowed_tools,
        )
    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and effective_permissions.can_run_commands
        and "run_command" in allowed_tools
    ):
        direct_actions = _flutter_existing_project_run_actions(
            task,
            assume_flutter=assume_flutter_workspace,
            workspace_dir=workspace_dir,
        )
    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and effective_permissions.can_write
        and "write_file" in allowed_tools
        and _requests_flutter_terminal_actions(task)
    ):
        direct_actions = _flutter_project_fallback_actions(
            task,
            include_commands=effective_permissions.can_run_commands and "run_command" in allowed_tools,
            assume_flutter=assume_flutter_workspace,
            workspace_dir=workspace_dir,
        )

    def _missing_required_file_result(args: dict) -> ToolResult | None:
        if not bool(args.get("require_exists", False)):
            return None
        raw_path = str(args.get("path") or "").strip()
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

    def _run_direct_actions(actions: list[Action]) -> bool:
        nonlocal summary, tests_ok, tools_ok
        direct_ok = True
        had_nonfatal_failure = False
        for direct_action in actions:
            command = ""
            if direct_action.type not in allowed_tools:
                direct_result = ToolResult(ok=False, output=f"tool_disabled:{direct_action.type}")
            elif direct_action.type == "write_file":
                direct_result = _missing_required_file_result(direct_action.args) or tools.write_file(
                    str(direct_action.args.get("path", "")),
                    str(direct_action.args.get("text", "")),
                    append=bool(direct_action.args.get("append", False)),
                )
            elif direct_action.type == "delete_file":
                direct_result = tools.delete_file(str(direct_action.args.get("path", "")))
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
            _task_requires_command_activity(task)
            and effective_permissions.can_run_commands
            and bool({"run_command", "run_tests", "open_browser"} & allowed_tools)
        )

    if direct_actions:
        _run_direct_actions(direct_actions)

    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and allow_model_load
    ):
        try:
            current_model = load_inference_model(settings)
        except Exception as exc:
            model_unavailable_reason = str(exc)

    if (
        not direct_actions
        and action_provider is None
        and current_model is None
        and effective_permissions.can_write
        and "write_file" in allowed_tools
    ):
        direct_actions = _flutter_project_fallback_actions(
            task,
            include_commands=effective_permissions.can_run_commands and "run_command" in allowed_tools,
            assume_flutter=assume_flutter_workspace,
            workspace_dir=workspace_dir,
        )
    if direct_actions and not tools_ok and not tool_calls:
        _run_direct_actions(direct_actions)

    if not direct_actions and action_provider is None and current_model is None:
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
            "model_unavailable": True,
        }

    while iterations_done < max_total_iters:
        if direct_actions:
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
            result = _missing_required_file_result(action.args) or tools.write_file(
                str(action.args.get("path", "")),
                str(action.args.get("text", "")),
                append=bool(action.args.get("append", False)),
            )
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
                if (
                    result.ok
                    and _requests_flutter_login_project(task)
                    and str(action.args.get("path") or "").replace("\\", "/") == "lib/main.dart"
                ):
                    for support_action in _flutter_login_project_support_actions(workspace_dir):
                        support_result = tools.write_file(
                            str(support_action.args.get("path", "")),
                            str(support_action.args.get("text", "")),
                            append=bool(support_action.args.get("append", False)),
                        )
                        tool_calls.append(
                            _tool_call_record(
                                support_action.type,
                                support_action.args,
                                support_result,
                                max_output_chars=min(4000, tool_chars),
                            )
                        )
                        tools_ok = tools_ok and bool(support_result.ok)
                        if not support_result.ok:
                            summary = support_result.output
                            break
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
        tool_calls.append(
            _tool_call_record(
                action.type,
                action.args,
                result,
                max_output_chars=min(4000, tool_chars),
            )
        )
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
    if (
        _task_requires_workspace_change(task)
        and not file_changes
        and not patch_id
        and not _attempted_workspace_mutation(tool_calls)
    ):
        blocked = True
        tools_ok = False
        if _summary_needs_fallback(summary) or summary.strip().lower() not in {"agent_model_unavailable"}:
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
    }
