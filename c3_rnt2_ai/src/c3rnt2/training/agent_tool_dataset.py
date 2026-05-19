from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


AGENT_TOOL_SYSTEM = (
    "You are Vortex agent mode. Return exactly one minified JSON action with keys type and args. "
    "Use list_tree/read_file/grep before editing unknown projects. Use write_file with full file content. "
    "Use run_command for validation when useful. Use finish only after tool results prove the task is done."
)


def _action(action_type: str, args: dict[str, Any]) -> str:
    return json.dumps({"type": action_type, "args": args}, ensure_ascii=True, separators=(",", ":"))


def _sample(user: str, action_type: str, args: dict[str, Any], *, ref: str, quality: float = 0.98) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": AGENT_TOOL_SYSTEM},
            {"role": "user", "content": user.strip()},
        ],
        "response": _action(action_type, args),
        "source_kind": "agent_tool_use_sft",
        "source_ref": ref,
        "quality": quality,
        "ts": 0.0,
    }


def _write_text_samples() -> list[dict[str, Any]]:
    files = [
        (
            "Crea notes/todo.txt con tres tareas cortas.",
            "notes/todo.txt",
            "1. Revisar requisitos\n2. Implementar cambio\n3. Ejecutar pruebas\n",
        ),
        (
            "Create src/math_utils.py with an add function.",
            "src/math_utils.py",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        (
            "Crea web/index.html con una pagina minima.",
            "web/index.html",
            "<!doctype html>\n<html lang=\"es\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Vortex</title>\n</head>\n<body>\n  <main>Hola Vortex</main>\n</body>\n</html>\n",
        ),
        (
            "Create package.json for a minimal Vite app.",
            "package.json",
            "{\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"test\": \"vitest run\"\n  },\n  \"dependencies\": {\n    \"@vitejs/plugin-react\": \"latest\",\n    \"vite\": \"latest\",\n    \"react\": \"latest\",\n    \"react-dom\": \"latest\"\n  },\n  \"devDependencies\": {\n    \"vitest\": \"latest\"\n  }\n}\n",
        ),
        (
            "Crea README.md con instrucciones de ejecucion.",
            "README.md",
            "# Proyecto\n\n## Uso\n\n```bash\npython -m pytest -q\n```\n\n## Validacion\n\nEl proyecto debe pasar los tests locales.\n",
        ),
        (
            "Crea lib/main.dart con una app Flutter minima.",
            "lib/main.dart",
            "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(home: Scaffold(body: Center(child: Text('Hola')))));\n",
        ),
    ]
    return [
        _sample(task, "write_file", {"path": path, "text": text}, ref=f"write:{path}")
        for task, path, text in files
    ]


def _inspection_samples() -> list[dict[str, Any]]:
    tasks = [
        ("Haz un proyecto sencillo de Flutter con login.", "list_tree", {"root": ".", "max_entries": 200}),
        ("Anade modo oscuro al proyecto actual.", "list_tree", {"root": ".", "max_entries": 200}),
        ("Add a forgot-password button to the login screen.", "grep", {"pattern": "Login|password|TextFormField", "path_glob": "**/*.{dart,tsx,ts,jsx,js}", "max_hits": 50}),
        ("Lee pubspec.yaml y dime el nombre del proyecto.", "read_file", {"path": "pubspec.yaml", "max_chars": 4000}),
        ("Find where the API client is defined.", "grep", {"pattern": "fetch\\(|axios|baseURL|API", "path_glob": "**/*.{ts,tsx,js,jsx,py}", "max_hits": 50}),
        ("Mira la estructura antes de editar.", "list_tree", {"root": ".", "max_entries": 200}),
    ]
    return [_sample(task, action, args, ref=f"inspect:{i}") for i, (task, action, args) in enumerate(tasks)]


def _command_samples() -> list[dict[str, Any]]:
    tasks = [
        ("Ejecuta los tests de Python del proyecto.", "run_command", {"command": "python -m pytest -q", "cwd": ".", "timeout_s": 120}),
        ("Valida el proyecto Flutter.", "run_command", {"command": "flutter analyze", "cwd": ".", "timeout_s": 180}),
        ("Run TypeScript checks.", "run_command", {"command": "npm run typecheck", "cwd": ".", "timeout_s": 120}),
        ("Arranca el servidor Vite en background.", "run_command", {"command": "npm run dev -- --host 0.0.0.0", "cwd": ".", "timeout_s": 120, "background": True}),
    ]
    return [_sample(task, action, args, ref=f"command:{i}") for i, (task, action, args) in enumerate(tasks)]


def _iterative_samples() -> list[dict[str, Any]]:
    return [
        _sample(
            "Task: Haz un proyecto Flutter con login.\nTool result from list_tree: . is empty.",
            "write_file",
            {
                "path": "pubspec.yaml",
                "text": "name: vortex_login\npublish_to: 'none'\nversion: 1.0.0+1\nenvironment:\n  sdk: '>=3.4.0 <4.0.0'\ndependencies:\n  flutter:\n    sdk: flutter\ndev_dependencies:\n  flutter_test:\n    sdk: flutter\n  flutter_lints: ^4.0.0\nflutter:\n  uses-material-design: true\n",
            },
            ref="iter:flutter_pubspec",
        ),
        _sample(
            "Task: Haz un proyecto Flutter con login.\nPrevious action wrote pubspec.yaml successfully.",
            "write_file",
            {
                "path": "lib/main.dart",
                "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const LoginApp());\n\nclass LoginApp extends StatelessWidget {\n  const LoginApp({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return const MaterialApp(home: LoginScreen());\n  }\n}\n\nclass LoginScreen extends StatefulWidget {\n  const LoginScreen({super.key});\n\n  @override\n  State<LoginScreen> createState() => _LoginScreenState();\n}\n\nclass _LoginScreenState extends State<LoginScreen> {\n  final formKey = GlobalKey<FormState>();\n\n  @override\n  Widget build(BuildContext context) {\n    return Scaffold(\n      body: Center(\n        child: SizedBox(\n          width: 360,\n          child: Form(\n            key: formKey,\n            child: Column(\n              mainAxisSize: MainAxisSize.min,\n              children: [\n                TextFormField(decoration: const InputDecoration(labelText: 'Email')),\n                TextFormField(decoration: const InputDecoration(labelText: 'Password'), obscureText: true),\n                const SizedBox(height: 16),\n                FilledButton(onPressed: () => formKey.currentState?.validate(), child: const Text('Entrar')),\n              ],\n            ),\n          ),\n        ),\n      ),\n    );\n  }\n}\n",
            },
            ref="iter:flutter_main",
        ),
        _sample(
            "Task: Haz un proyecto Flutter con login.\nTool result: wrote pubspec.yaml and lib/main.dart.",
            "run_command",
            {"command": "flutter analyze", "cwd": ".", "timeout_s": 180},
            ref="iter:flutter_analyze",
        ),
        _sample(
            "Task: Haz un proyecto Flutter con login.\nTool result from flutter analyze: No issues found!",
            "finish",
            {"summary": "He creado el proyecto Flutter con login y flutter analyze no encontro errores."},
            ref="iter:flutter_finish",
        ),
        _sample(
            "Task: Anade modo oscuro.\nTool result from read_file lib/main.dart: contains MaterialApp without themeMode.",
            "write_file",
            {
                "path": "lib/main.dart",
                "text": "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const MaterialApp(themeMode: ThemeMode.system, home: Scaffold(body: Center(child: Text('Tema listo')))));\n",
                "require_exists": True,
            },
            ref="iter:dark_mode_edit",
        ),
        _sample(
            "Task: Borra notes/todo.txt.\nTool result from read_file notes/todo.txt: file exists.",
            "delete_file",
            {"path": "notes/todo.txt"},
            ref="iter:delete_file",
        ),
    ]


def build_agent_tool_samples(*, seed: int = 23, target_count: int = 96) -> list[dict[str, Any]]:
    base = _inspection_samples() + _write_text_samples() + _command_samples() + _iterative_samples()
    rng = random.Random(seed)
    variants: list[dict[str, Any]] = []
    intents = [
        ("Create a small Python CLI and then test it.", "list_tree", {"root": ".", "max_entries": 200}),
        ("Modifica el README existente, primero leelo.", "read_file", {"path": "README.md", "max_chars": 4000}),
        ("Busca donde esta definido LoginScreen.", "grep", {"pattern": "class\\s+LoginScreen|LoginScreen", "path_glob": "**/*.dart", "max_hits": 50}),
        ("Valida que no haya errores de lint.", "run_tests", {}),
        ("Open the local preview in the browser.", "open_browser", {"url": "http://localhost:5173"}),
    ]
    for idx in range(target_count):
        task, action, args = intents[idx % len(intents)]
        suffix = f" Case {idx // len(intents) + 1}."
        variants.append(_sample(task + suffix, action, dict(args), ref=f"variant:{idx}", quality=0.94))
    rows = base + variants
    rng.shuffle(rows)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode("utf-8") + row["response"].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(row)
    return out


def write_agent_tool_dataset(output_path: Path, *, target_count: int = 96) -> dict[str, Any]:
    rows = build_agent_tool_samples(target_count=target_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return {"ok": True, "path": str(output_path), "samples": len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SFT data for Vortex agent tool use.")
    parser.add_argument("--out", default="config/datasets/agent_tool_use_sft.jsonl")
    parser.add_argument("--target-count", type=int, default=96)
    args = parser.parse_args(argv)
    result = write_agent_tool_dataset(Path(args.out), target_count=max(16, int(args.target_count)))
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
