from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SYSTEM_PROMPT = (
    "You are Vortex, a local master programming assistant specialized in "
    "Flutter, Dart, Python, FastAPI, and the Vortex codebase. Answer directly, "
    "with practical code and validation steps."
)

TRAIN_NAME = "gemma4_flutter_python_sft.jsonl"
EVAL_NAME = "gemma4_flutter_python_eval.jsonl"

EXCLUDE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".dart_tool",
    "hf-cache",
    "hf_offload",
    "logs",
    "output",
    "test-results",
}

SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|cookie|authorization)\s*[:=]"
)

DOMAIN_PATTERNS = {
    "flutter": re.compile(r"(?i)flutter|widget|buildcontext|scaffold|layout|overflow"),
    "dart": re.compile(r"(?i)\bdart\b|future<|stream<|async|await|state management"),
    "python": re.compile(r"(?i)\bpython\b|def\s+\w+\(|pytest|typing|pydantic"),
    "fastapi": re.compile(r"(?i)fastapi|APIRouter|@app\.|Depends\(|HTTPException"),
    "vortex": re.compile(r"(?i)vortex|c3rnt2|hf_train|adapter|rtx4080|gemma"),
}


@dataclass(frozen=True)
class Sample:
    messages: list[dict[str, str]]
    source: str
    domain: str
    source_ref: str

    def to_json(self) -> dict:
        return {
            "messages": self.messages,
            "source": self.source,
            "domain": self.domain,
            "source_ref": self.source_ref,
        }


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def is_safe_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_PARTS:
        return False
    if path.name.startswith(".env"):
        return False
    if path.suffix.lower() in {".safetensors", ".pt", ".bin", ".db", ".sqlite", ".zip"}:
        return False
    return True


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def read_text(path: Path) -> str | None:
    if not is_safe_path(path):
        return None
    try:
        if path.stat().st_size > 96_000:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if SECRET_RE.search(text):
        return None
    return text.strip()


def clip(text: str, limit: int = 2600) -> str:
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n..."


def chat(user: str, assistant: str, *, source: str, domain: str, source_ref: str) -> Sample:
    return Sample(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        source=source,
        domain=domain,
        source_ref=source_ref,
    )


def detect_domain(path: Path, text: str) -> str:
    joined = f"{path.as_posix()}\n{text}"
    for name in ("fastapi", "flutter", "dart", "python", "vortex"):
        if DOMAIN_PATTERNS[name].search(joined):
            return name
    if path.suffix == ".py":
        return "python"
    if path.suffix in {".md", ".yaml", ".yml", ".json", ".toml"}:
        return "vortex"
    return "vortex"


def code_samples(path: Path, text: str, root: Path) -> list[Sample]:
    r = rel(path, root)
    snippet = clip(text)
    domain = detect_domain(path, text)
    out: list[Sample] = []
    if path.suffix == ".py":
        out.append(
            chat(
                f"Explain this Python file from Vortex. Focus on purpose, key functions, and risks.\n\nPath: {r}\n```python\n{snippet}\n```",
                "This file should be reviewed by purpose first, then by boundaries.\n\n- Identify public entry points and side effects.\n- Check path, subprocess, network, and serialization handling.\n- Keep changes scoped to the owning module.\n- Validate with the nearest pytest file or a focused smoke test.\n\nFor Vortex, preserve local-first defaults, adapter gating, and explicit eval before promotion.",
                source="vortex_repo",
                domain=domain,
                source_ref=r,
            )
        )
        if "FastAPI" in text or "APIRouter" in text or "@app." in text:
            out.append(
                chat(
                    f"Using this Vortex FastAPI code, propose a safe endpoint change and test plan.\n\nPath: {r}\n```python\n{snippet}\n```",
                    "Use request/response models where possible, fail closed on invalid input, and avoid leaking internal paths or secrets. Add a test with `TestClient` that covers success and one bad request. If the endpoint mutates runtime state, guard it behind existing server locks or settings gates.",
                    source="vortex_repo",
                    domain="fastapi",
                    source_ref=r,
                )
            )
    elif path.suffix in {".ts", ".tsx"}:
        out.append(
            chat(
                f"Explain this Vortex frontend component/service and how it talks to the backend.\n\nPath: {r}\n```tsx\n{snippet}\n```",
                "Read it as UI state plus API contract. Identify component state, service calls, error handling, and user-visible states. Keep future changes consistent with existing TS types, avoid hidden network assumptions, and validate with the nearest Playwright or unit smoke path.",
                source="vortex_repo",
                domain="vortex",
                source_ref=r,
            )
        )
    elif path.suffix in {".md", ".yaml", ".yml", ".json", ".toml"}:
        out.append(
            chat(
                f"Summarize this Vortex config/doc for a maintainer. Extract constraints that matter for training or serving.\n\nPath: {r}\n```text\n{snippet}\n```",
                "Treat config as contract. Preserve runtime defaults, web-disabled settings, adapter registry paths, eval gates, and rollback instructions. Do not infer that training is active unless the profile explicitly enables it.",
                source="vortex_repo",
                domain=domain,
                source_ref=r,
            )
        )
    return out


CURATED: list[tuple[str, str, str]] = [
    (
        "flutter",
        "Fix a Flutter Row overflow caused by long text next to an icon.",
        "Wrap the text side with `Expanded` or `Flexible`, keep the icon at fixed size, and set `overflow: TextOverflow.ellipsis` when truncation is acceptable.\n\n```dart\nRow(\n  children: [\n    const Icon(Icons.info_outline),\n    const SizedBox(width: 8),\n    Expanded(\n      child: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis),\n    ),\n  ],\n)\n```\n\nValidate on narrow width and with long localized strings.",
    ),
    (
        "flutter",
        "Create a clean reusable Flutter card widget for a training status row.",
        "Keep widget inputs explicit and UI stable.\n\n```dart\nclass TrainingStatusTile extends StatelessWidget {\n  const TrainingStatusTile({super.key, required this.title, required this.status});\n\n  final String title;\n  final String status;\n\n  @override\n  Widget build(BuildContext context) {\n    return ListTile(\n      title: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis),\n      subtitle: Text(status),\n      leading: const Icon(Icons.model_training),\n    );\n  }\n}\n```\n\nAvoid business logic inside `build`; pass derived state in.",
    ),
    (
        "flutter",
        "Explain how to make Flutter layout responsive without magic pixel checks.",
        "Use constraints from `LayoutBuilder`, not raw device names. Pick layout from available width, keep breakpoints named, and test narrow/wide cases.\n\n```dart\nLayoutBuilder(\n  builder: (context, constraints) {\n    final compact = constraints.maxWidth < 640;\n    return compact ? const CompactView() : const WideView();\n  },\n)\n```",
    ),
    (
        "flutter",
        "Fix a Flutter Column where a ListView throws an unbounded height error.",
        "Put the scrollable child inside `Expanded` when the parent is a `Column`, or give it a bounded height.\n\n```dart\nColumn(\n  children: [\n    const Header(),\n    Expanded(\n      child: ListView.builder(\n        itemCount: items.length,\n        itemBuilder: (context, index) => ItemTile(item: items[index]),\n      ),\n    ),\n  ],\n)\n```\n\nDo not set `shrinkWrap: true` by default; it can cost perf on long lists.",
    ),
    (
        "flutter",
        "Create a Flutter loading/error/data pattern for a local model status panel.",
        "Model UI state explicitly and keep rendering branches small.\n\n```dart\nswitch (state) {\n  case StatusLoading():\n    return const Center(child: CircularProgressIndicator());\n  case StatusError(:final message):\n    return Text(message, style: TextStyle(color: Theme.of(context).colorScheme.error));\n  case StatusReady(:final data):\n    return StatusBody(data: data);\n}\n```\n\nEach branch should be testable without network calls.",
    ),
    (
        "flutter",
        "Improve Flutter widget architecture for a chat input with send/stop states.",
        "Split text editing, command buttons, and transport state. Keep the input widget controlled by explicit props: `controller`, `isSending`, `onSend`, `onStop`. Disable send for blank text and show stop only while a request is active.",
    ),
    (
        "flutter",
        "Explain when to use `const` in Flutter widgets.",
        "Use `const` when constructor args are compile-time constants. It reduces rebuild work and makes intent clear. Do not force `const` when values depend on theme, localization, provider state, or runtime data.",
    ),
    (
        "flutter",
        "Write a Flutter widget test for a status tile.",
        "Pump the widget, find stable text/icons, and assert state-specific UI.\n\n```dart\ntestWidgets('shows ready status', (tester) async {\n  await tester.pumpWidget(const MaterialApp(\n    home: TrainingStatusTile(title: 'Gemma', status: 'ready'),\n  ));\n  expect(find.text('Gemma'), findsOneWidget);\n  expect(find.text('ready'), findsOneWidget);\n});\n```",
    ),
    (
        "dart",
        "Write a small Dart service method with typed errors for loading adapter metadata.",
        "Return typed data, keep HTTP errors explicit, and avoid dynamic maps crossing the boundary.\n\n```dart\nFuture<AdapterMeta> loadAdapterMeta(Uri uri, http.Client client) async {\n  final res = await client.get(uri);\n  if (res.statusCode != 200) {\n    throw AdapterLoadException('HTTP ${res.statusCode}');\n  }\n  return AdapterMeta.fromJson(jsonDecode(res.body) as Map<String, Object?>);\n}\n```",
    ),
    (
        "dart",
        "Explain basic Dart state management for a chat screen.",
        "Keep source of truth small: messages, pending request, and error. Use immutable updates, expose derived UI state, and isolate API calls in a service. For larger apps, use Riverpod/Bloc; for simple local screens, `ValueNotifier` or `ChangeNotifier` can be enough.",
    ),
    (
        "dart",
        "Convert a nullable dynamic JSON field into a typed Dart value safely.",
        "Check type before casting and pick a clear default.\n\n```dart\nfinal raw = json['maxSteps'];\nfinal maxSteps = raw is int ? raw : int.tryParse(raw?.toString() ?? '') ?? 0;\n```\n\nAvoid `as int` on API data unless the contract is already validated.",
    ),
    (
        "dart",
        "Explain `Future` vs `Stream` in Dart for Vortex UI state.",
        "`Future` is one async result, good for loading adapter metadata once. `Stream` is many results over time, good for logs, token streaming, or live status. Cancel streams in `dispose` or use framework helpers that own cancellation.",
    ),
    (
        "dart",
        "Write a Dart model class for adapter status JSON.",
        "Keep parsing strict at the boundary.\n\n```dart\nclass AdapterStatus {\n  const AdapterStatus({required this.active, required this.path});\n\n  final bool active;\n  final String? path;\n\n  factory AdapterStatus.fromJson(Map<String, Object?> json) {\n    return AdapterStatus(\n      active: json['active'] == true,\n      path: json['path'] as String?,\n    );\n  }\n}\n```",
    ),
    (
        "dart",
        "Handle cancellation for an in-flight Dart HTTP request in a UI service.",
        "Keep a cancellable operation or client lifecycle outside the widget. When user presses stop, cancel the operation, clear pending UI state, and avoid writing stale results into state after cancellation.",
    ),
    (
        "fastapi",
        "Create a FastAPI endpoint that queues an adapter reload request safely.",
        "Validate input, write a small JSON request, and return a predictable response.\n\n```python\nfrom pydantic import BaseModel\nfrom fastapi import APIRouter, HTTPException\n\nrouter = APIRouter()\n\nclass ReloadRequest(BaseModel):\n    adapter_path: str\n\n@router.post('/v1/reload_adapter_request')\ndef reload_adapter_request(payload: ReloadRequest) -> dict[str, object]:\n    if '..' in payload.adapter_path:\n        raise HTTPException(status_code=400, detail='invalid adapter_path')\n    # write request under data/state in real impl\n    return {'ok': True, 'queued': True}\n```",
    ),
    (
        "python",
        "Refactor a Python function that mixes parsing, validation, and file writes.",
        "Split into pure parse/validate functions plus one side-effect function. Unit-test pure functions first. Keep path resolution in one place and reject paths outside the intended root before writing.",
    ),
    (
        "python",
        "Write a pytest for a function that returns JSON metadata.",
        "Assert exact stable fields and avoid depending on wall-clock values unless injected.\n\n```python\ndef test_build_meta_has_required_fields():\n    meta = build_meta(run_id='r1', steps=10, loss=0.5)\n    assert meta['run_id'] == 'r1'\n    assert meta['steps'] == 10\n    assert 'loss' in meta\n```",
    ),
    (
        "vortex",
        "Explain Vortex adapter promotion policy in one direct answer.",
        "Training creates a candidate adapter. It must pass eval/bench gates before promotion. For Colab imports, keep manual promotion by default: copy adapter, inspect report, run local doctor/bench, then update registry or call reload only after approval.",
    ),
    (
        "vortex",
        "Generate instructions for Codex to make a small Vortex backend patch.",
        "Tell Codex the target file, expected behavior, tests to run, and constraints. Example: `Modify c3_rnt2_ai/src/c3rnt2/server.py to validate reload input. Do not change runtime profiles. Add/adjust tests in c3_rnt2_ai/tests/test_server_reload_request.py. Run focused pytest only.`",
    ),
]


def curated_samples() -> list[Sample]:
    return [
        chat(user, answer, source="curated_static", domain=domain, source_ref=f"curated:{idx}")
        for idx, (domain, user, answer) in enumerate(CURATED, 1)
    ]


def load_seed_samples(paths: Iterable[Path], root: Path) -> list[Sample]:
    out: list[Sample] = []
    for path in paths:
        if not path.exists() or not is_safe_path(path):
            continue
        # External bulk corpora require separate human review. Skip by default.
        if rel(path, root) == "data/registry/hf_train/sft_samples.jsonl":
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, 1):
                if idx > 300:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                text = json.dumps(payload, ensure_ascii=False)
                if SECRET_RE.search(text):
                    continue
                messages = payload.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
                if len(user) < 20 or len(assistant) < 20:
                    continue
                if assistant in user and len(assistant) > 200:
                    continue
                domain = "vortex"
                for name, pattern in DOMAIN_PATTERNS.items():
                    if pattern.search(text):
                        domain = name
                        break
                out.append(
                    chat(
                        user,
                        assistant,
                        source=str(payload.get("source_kind") or "seed"),
                        domain=domain,
                        source_ref=f"{rel(path, root)}:{idx}",
                    )
                )
    return out


def collect_repo_files(root: Path) -> list[Path]:
    include_roots = [
        root / "README.md",
        root / "docs",
        root / "c3_rnt2_ai" / "README.md",
        root / "c3_rnt2_ai" / "src",
        root / "c3_rnt2_ai" / "tests",
        root / "c3_rnt2_ai" / "templates",
        root / "c3_rnt2_ai" / "config",
        root / "vortex-chat" / "app",
        root / "vortex-chat" / "components",
        root / "vortex-chat" / "services",
        root / "vortex-chat" / "tests",
        root / "vortex-chat" / "README.md",
        root / "vortex-chat" / "package.json",
    ]
    suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".ts", ".tsx"}
    files: list[Path] = []
    for base in include_roots:
        if not base.exists():
            continue
        if base.is_file():
            candidates = [base]
        else:
            candidates = [p for p in base.rglob("*") if p.is_file()]
        for path in candidates:
            if path.suffix.lower() in suffixes and is_safe_path(path):
                files.append(path)
    return sorted(set(files))


def dedup(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[str] = set()
    out: list[Sample] = []
    for sample in samples:
        payload = json.dumps(sample.to_json()["messages"], sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(sample)
    return out


def quality(samples: list[Sample]) -> dict[str, object]:
    domains = {name: 0 for name in DOMAIN_PATTERNS}
    invalid = 0
    secret_hits = 0
    lengths: list[int] = []
    hashes: set[str] = set()
    duplicates = 0
    for sample in samples:
        obj = sample.to_json()
        text = json.dumps(obj, ensure_ascii=False)
        if SECRET_RE.search(text):
            secret_hits += 1
        messages = obj.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            invalid += 1
        roles = [m.get("role") for m in messages] if isinstance(messages, list) else []
        if roles != ["system", "user", "assistant"]:
            invalid += 1
        domains[sample.domain] = domains.get(sample.domain, 0) + 1
        lengths.append(len(text))
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in hashes:
            duplicates += 1
        hashes.add(h)
    avg_chars = sum(lengths) / max(1, len(lengths))
    return {
        "samples": len(samples),
        "invalid": invalid,
        "duplicates": duplicates,
        "secret_hits": secret_hits,
        "avg_chars": round(avg_chars, 1),
        "domains": domains,
    }


def gate(report: dict[str, object], *, min_train: int) -> tuple[bool, list[str]]:
    domains = report.get("domains", {})
    assert isinstance(domains, dict)
    reasons: list[str] = []
    if int(report.get("samples", 0)) < min_train:
        reasons.append(f"too_small: samples < {min_train}")
    if int(report.get("invalid", 0)) > 0:
        reasons.append("invalid_chat_format")
    if int(report.get("duplicates", 0)) > 0:
        reasons.append("duplicates_present")
    if int(report.get("secret_hits", 0)) > 0:
        reasons.append("secret_like_text_present")
    minimum_domains = {"flutter": 12, "dart": 8, "python": 40, "fastapi": 8, "vortex": 40}
    for domain, minimum in minimum_domains.items():
        if int(domains.get(domain, 0)) < minimum:
            reasons.append(f"domain_{domain}_underrepresented")
    return not reasons, reasons


def write_jsonl(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    domains = payload.get("domains", {})
    lines = [
        "# Gemma 4 Vortex Dataset Quality Report",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Train samples: {payload.get('train_samples')}",
        f"Eval samples: {payload.get('eval_samples')}",
        f"Gate ok: {payload.get('gate_ok')}",
        f"Gate reasons: {', '.join(payload.get('gate_reasons', []) or ['none'])}",
        "",
        "## Domain Counts",
    ]
    if isinstance(domains, dict):
        for key, value in domains.items():
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Bulk external `data/registry/hf_train/sft_samples.jsonl` skipped. It contains OpenOrca-style web data and needs separate review.",
            "- Missing Flutter/Dart seed files are compensated only by curated static examples. Add real Flutter/Dart project code before serious training.",
            "- No adapter was trained by this script. It only prepares data and quality signals.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def build_dataset(root: Path, output_root: Path, *, max_train: int, eval_size: int, min_train: int) -> dict[str, object]:
    for sub in (
        "datasets/raw",
        "datasets/processed",
        "datasets/eval",
        "notebooks",
        "outputs/adapters",
        "outputs/logs",
        "outputs/reports",
        "exports",
    ):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    samples.extend(curated_samples())
    seed_paths = [
        root / "config/datasets/flutter_dart_seed.jsonl",
        root / "config/datasets/flutter_dart_eval.jsonl",
        root / "data/registry/hf_train/programming_eval.jsonl",
        root / "c3_rnt2_ai/data/registry/hf_train/sft_samples.jsonl",
        root / "c3_rnt2_ai/data/registry/hf_train/gemma4_e4b_sft_samples.jsonl",
    ]
    samples.extend(load_seed_samples(seed_paths, root))
    for path in collect_repo_files(root):
        text = read_text(path)
        if not text:
            continue
        samples.extend(code_samples(path, text, root))

    samples = dedup(samples)
    # Stable split: every 7th sample is eval until eval_size is reached.
    eval_samples: list[Sample] = []
    train_samples: list[Sample] = []
    for idx, sample in enumerate(samples):
        if len(eval_samples) < eval_size and idx % 7 == 0:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)
    train_samples = train_samples[:max_train]

    train_report = quality(train_samples)
    ok, reasons = gate(train_report, min_train=min_train)

    train_path = output_root / "datasets/processed" / TRAIN_NAME
    eval_path = output_root / "datasets/eval" / EVAL_NAME
    write_jsonl(train_path, train_samples)
    write_jsonl(eval_path, eval_samples)

    notebook_src = root / "c3_rnt2_ai/notebooks/gemma4_flutter_python_lora.ipynb"
    notebook_dst = output_root / "notebooks/gemma4_flutter_python_lora.ipynb"
    if notebook_src.exists():
        shutil.copy2(notebook_src, notebook_dst)

    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "output_root": str(output_root),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "gate_ok": ok,
        "gate_reasons": reasons,
        **train_report,
    }
    write_report(output_root / "outputs/reports/dataset_quality_report.md", payload)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed Vortex SFT JSONL for Gemma 4 LoRA/QLoRA.")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()))
    parser.add_argument(
        "--output-root",
        default="c3_rnt2_ai/data/registry/hf_train/gemma4_e4b/colab_export",
        help="Root that will contain VortexTraining-style folders.",
    )
    parser.add_argument("--max-train", type=int, default=360)
    parser.add_argument("--eval-size", type=int, default=48)
    parser.add_argument("--min-train", type=int, default=180)
    parser.add_argument("--allow-poor", action="store_true", help="Exit 0 even when quality gate fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    report = build_dataset(
        root,
        output_root,
        max_train=max(1, int(args.max_train)),
        eval_size=max(1, int(args.eval_size)),
        min_train=max(1, int(args.min_train)),
    )
    print(json.dumps(report, ensure_ascii=True))
    if not bool(report.get("gate_ok")) and not args.allow_poor:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
