from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


SYSTEM = (
    "You are Vortex, a local master programming assistant specialized in "
    "Flutter, Dart, Python, FastAPI and robust app architecture."
)

DATASET_FILES = {
    "general": "flutter_official_docs_sft.jsonl",
    "code": "flutter_official_docs_code_sft.jsonl",
    "debugging": "flutter_official_docs_debugging_sft.jsonl",
    "architecture": "flutter_official_docs_architecture_sft.jsonl",
}

DEBUG_TOPICS = {"constraints", "layout", "debugging", "rendering", "performance", "error_handling"}
CODE_TOPICS = {
    "widgets",
    "forms_validation",
    "navigation",
    "async_futures_streams",
    "networking",
    "assets_images",
    "animations",
    "gestures",
    "testing_unit_widget_integration",
    "golden_tests",
    "adaptive_responsive",
}
ARCH_TOPICS = {"architecture", "clean_architecture", "state_management", "persistence", "security", "packages"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def compact(text: str, limit: int = 900) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def sample_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def make_sample(user: str, response: str, *, source_kind: str, source_ref: str, quality: float, topic: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
        ],
        "response": response.strip(),
        "source_kind": source_kind,
        "quality": quality,
        "source_ref": source_ref,
        "topic": topic,
    }


def explanation_sample(chunk: dict) -> dict:
    topic = str(chunk.get("topic") or "flutter_basics")
    title = str(chunk.get("title") or topic)
    text = compact(str(chunk.get("text") or ""))
    user = f"Explica el tema Flutter `{topic}` aplicado a una app real. Usa la fuente `{title}` como base."
    response = (
        f"En Flutter, `{topic}` se debe aplicar con una regla práctica: entiende el contrato del widget/API, "
        f"elige una solución concreta, y valida en el caso de UI correcto.\n\n"
        f"Puntos clave derivados de la documentación oficial:\n- {text}\n\n"
        "Respuesta útil para Vortex: da causa, solución, código mínimo cuando aplique, y prueba de validación."
    )
    return make_sample(user, response, source_kind="flutter_official_docs_sft", source_ref=str(chunk.get("url") or ""), quality=0.95, topic=topic)


def debugging_sample(chunk: dict) -> dict:
    topic = str(chunk.get("topic") or "layout")
    text = compact(str(chunk.get("text") or ""), 700)
    if topic == "constraints":
        user = "Tengo `RenderBox was not laid out` en Flutter. Analiza causas probables y dame una corrección robusta."
        response = (
            "Causa probable: un render object recibió constraints no acotadas o un hijo no pudo calcular tamaño. "
            "Revisa scrollables dentro de `Column`, `Expanded/Flexible`, `SizedBox`, y widgets que necesitan bounds.\n\n"
            "Fix típico:\n```dart\nColumn(\n  children: [\n    const Header(),\n    Expanded(child: ListView.builder(itemBuilder: buildItem)),\n  ],\n)\n```\n\n"
            f"Base oficial: {text}\n\nValida en ancho/alto pequeño y mira el primer widget propio en el stack trace."
        )
    elif topic == "layout":
        user = "Tengo `RenderFlex overflowed by 42 pixels` en Flutter. Dame diagnóstico y fix sin parches frágiles."
        response = (
            "Overflow en `Row`/`Column` significa que los hijos piden más espacio que el eje principal disponible. "
            "No lo tapes con padding arbitrario. Acota el hijo flexible.\n\n"
            "Fix típico:\n```dart\nRow(\n  children: [\n    const Icon(Icons.info),\n    const SizedBox(width: 8),\n    Expanded(child: Text(title, overflow: TextOverflow.ellipsis)),\n  ],\n)\n```\n\n"
            f"Base oficial: {text}\n\nPrueba con texto largo y viewport estrecho."
        )
    else:
        user = f"Depura un problema Flutter relacionado con `{topic}` y da pasos concretos."
        response = (
            f"Empieza por reproducir mínimo, identifica el widget/API que rompe el contrato, y cambia estructura antes que síntomas.\n\n"
            f"Base oficial: {text}\n\nDa un fix pequeño, luego añade test o caso manual para evitar regresión."
        )
    return make_sample(user, response, source_kind="flutter_official_docs_debugging_sft", source_ref=str(chunk.get("url") or ""), quality=0.97, topic=topic)


def code_sample(chunk: dict) -> dict:
    topic = str(chunk.get("topic") or "widgets")
    code_blocks = chunk.get("code_blocks") if isinstance(chunk.get("code_blocks"), list) else []
    code_hint = str(code_blocks[0])[:900] if code_blocks else ""
    text = compact(str(chunk.get("text") or ""), 650)
    user = f"Genera una implementación Flutter/Dart limpia para `{topic}` y explica cómo validarla."
    response = (
        f"Implementa `{topic}` con API pequeña, estado explícito, y límites de layout claros.\n\n"
        "```dart\nclass VortexExampleWidget extends StatelessWidget {\n  const VortexExampleWidget({super.key, required this.title});\n\n  final String title;\n\n  @override\n  Widget build(BuildContext context) {\n    return LayoutBuilder(\n      builder: (context, constraints) {\n        final compact = constraints.maxWidth < 640;\n        return compact\n            ? Text(title, maxLines: 2, overflow: TextOverflow.ellipsis)\n            : Row(children: [Expanded(child: Text(title))]);\n      },\n    );\n  }\n}\n```\n\n"
        f"Base oficial: {text}\n"
        + (f"\nReferencia de código oficial revisada:\n```dart\n{code_hint}\n```\n" if code_hint else "")
        + "\nValida con widget test, texto largo, y viewport compacto/ancho."
    )
    return make_sample(user, response, source_kind="flutter_official_docs_code_sft", source_ref=str(chunk.get("url") or ""), quality=0.96, topic=topic)


def architecture_sample(chunk: dict) -> dict:
    topic = str(chunk.get("topic") or "architecture")
    text = compact(str(chunk.get("text") or ""), 700)
    user = f"Diseña una estructura Flutter mantenible para `{topic}`. Incluye capas, responsabilidades y tests."
    response = (
        "Usa una feature vertical y separa responsabilidades:\n\n"
        "```text\nfeature/\n  data/          # DTO, datasource, repository impl\n  domain/        # entity, repository contract, use cases\n  presentation/  # widgets, controllers/state, routing glue\n  test/\n```\n\n"
        "Reglas:\n- UI no llama HTTP directo.\n- DTO no cruza a widgets.\n- State expone loading/error/data.\n- Errores de infra se transforman en fallos de dominio.\n- Tests: unit para use cases, widget para estados, integration para flujo crítico.\n\n"
        f"Base oficial: {text}"
    )
    return make_sample(user, response, source_kind="flutter_official_docs_architecture_sft", source_ref=str(chunk.get("url") or ""), quality=0.96, topic=topic)


def codex_prompt_sample(chunk: dict) -> dict:
    topic = str(chunk.get("topic") or "flutter_basics")
    user = f"Dame un prompt para que Codex modifique un repo Flutter y resuelva `{topic}` sin romper comportamiento."
    response = (
        f"Prompt para Codex:\n\n"
        f"`Lee primero los widgets y tests relacionados con {topic}. Implementa el cambio mínimo. "
        "No cambies arquitectura global ni dependencias sin necesidad. Añade o ajusta widget tests. "
        "Valida con `flutter test` y, si toca layout responsive, prueba viewports compacto/ancho. "
        "Explica archivos cambiados y riesgo residual.`"
    )
    return make_sample(user, response, source_kind="flutter_official_docs_sft", source_ref=str(chunk.get("url") or ""), quality=0.94, topic=topic)


def build_datasets(chunks: list[dict], *, max_per_topic: int = 24) -> dict[str, list[dict]]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_topic[str(chunk.get("topic") or "flutter_basics")].append(chunk)
    rng = random.Random(17)
    datasets: dict[str, list[dict]] = {key: [] for key in DATASET_FILES}
    seen: set[str] = set()

    def add(kind: str, sample: dict) -> None:
        h = sample_hash(sample)
        if h in seen:
            return
        seen.add(h)
        datasets[kind].append(sample)

    for topic, topic_chunks in sorted(by_topic.items()):
        rng.shuffle(topic_chunks)
        for chunk in topic_chunks[:max_per_topic]:
            add("general", explanation_sample(chunk))
            if topic in DEBUG_TOPICS:
                add("debugging", debugging_sample(chunk))
            if topic in CODE_TOPICS or chunk.get("code_blocks"):
                add("code", code_sample(chunk))
            if topic in ARCH_TOPICS:
                add("architecture", architecture_sample(chunk))
            if len(topic_chunks) > 0:
                add("general", codex_prompt_sample(chunk))
    return datasets


HARD_EVAL_TEMPLATES = [
    ("Mi `ListView` dentro de `Column` lanza `RenderBox was not laid out`. Dame causa, fix y test.", ["constraints", "layout", "RenderBox"], ["mentions bounded constraints", "uses Expanded/Flexible or SizedBox", "warns about scrollables"]),
    ("Tengo `RenderFlex overflowed by 42 pixels` en un Row con texto largo. Corrige robusto.", ["layout", "RenderFlex"], ["explains main-axis overflow", "uses Expanded/Flexible", "handles long text"]),
    ("Disena pantalla responsive Flutter mobile/desktop sin ifs fragiles por plataforma.", ["adaptive_responsive", "layout"], ["uses LayoutBuilder/MediaQuery", "separates compact/wide UI", "tests both widths"]),
    ("Crea navegacion con auth guard y deep link seguro.", ["navigation", "security"], ["mentions router guard", "keeps auth state separate", "handles redirect loops"]),
    ("Implementa formulario con validacion async y submit idempotente.", ["forms_validation", "async_futures_streams"], ["validates fields", "handles loading/error", "prevents double submit"]),
    ("Explica Future vs Stream para una UI de chat Flutter.", ["async_futures_streams"], ["distinguishes one-shot vs multiple events", "mentions cancellation", "maps to UI state"]),
    ("Disena state management para loading/error/data sin acoplar UI a HTTP.", ["state_management", "architecture"], ["separates service/repository", "explicit states", "testable"]),
    ("Escribe widget test para pantalla loading/error/data.", ["testing_unit_widget_integration"], ["uses pumpWidget", "asserts states", "mocks dependencies"]),
    ("Como haria golden tests utiles sin snapshots fragiles?", ["golden_tests", "testing_unit_widget_integration"], ["stable fonts/theme", "multiple sizes", "review workflow"]),
    ("Optimiza lista con imagenes que hace jank al scrollear.", ["performance", "assets_images"], ["mentions image sizing/cache", "ListView.builder", "DevTools/profile"]),
    ("Diagnostica rebuilds innecesarios en un widget gigante.", ["performance", "widgets"], ["uses const/extract widgets", "profiles rebuilds", "preserves behavior"]),
    ("Haz prompt para Codex que refactorice un widget Flutter gigante.", ["architecture", "widgets"], ["specific files/tests", "small steps", "no behavior change"]),
    ("Explica Slivers cuando una pantalla necesita header colapsable y lista grande.", ["layout", "performance"], ["CustomScrollView", "SliverAppBar/SliverList", "bounded scroll model"]),
    ("Implementa accesibilidad minima en un boton icon-only y una tarjeta.", ["accessibility", "widgets"], ["semantics/tooltip/labels", "focus/tap targets", "screen reader"]),
    ("Prepara build release Android/iOS y riesgos previos.", ["build_release", "android_ios_deploy"], ["signing", "permissions", "flavors/config"]),
    ("Integra plugin nativo con platform channel y manejo de errores.", ["platform_integration", "plugins"], ["MethodChannel", "typed errors", "platform tests"]),
]


def build_hard_eval(chunks: list[dict], *, count: int = 80) -> list[dict]:
    source_by_topic: dict[str, str] = {}
    for chunk in chunks:
        topic = str(chunk.get("topic") or "")
        if topic and topic not in source_by_topic:
            source_by_topic[topic] = str(chunk.get("url") or "")
    rows: list[dict] = []
    idx = 0
    while len(rows) < count:
        prompt, topics, rubric = HARD_EVAL_TEMPLATES[idx % len(HARD_EVAL_TEMPLATES)]
        topic_ref = next((source_by_topic.get(t) for t in topics if source_by_topic.get(t)), "")
        rows.append(
            {
                "prompt": prompt if idx < len(HARD_EVAL_TEMPLATES) else f"{prompt} Caso variante #{idx // len(HARD_EVAL_TEMPLATES) + 1}.",
                "expected_topics": topics,
                "rubric": rubric,
                "source_kind": "flutter_official_hard_eval",
                "source_ref": topic_ref,
            }
        )
        idx += 1
    return rows


def write_datasets(chunks_path: Path, out_dir: Path) -> dict:
    chunks = read_jsonl(chunks_path)
    datasets = build_datasets(chunks)
    counts = {}
    for key, filename in DATASET_FILES.items():
        path = out_dir / filename
        write_jsonl(path, datasets[key])
        counts[filename] = len(datasets[key])
    eval_rows = build_hard_eval(chunks)
    eval_path = out_dir / "flutter_official_hard_eval.jsonl"
    write_jsonl(eval_path, eval_rows)
    counts[eval_path.name] = len(eval_rows)
    return {"chunks": len(chunks), "out_dir": str(out_dir), "counts": counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Flutter official docs SFT/eval datasets.")
    parser.add_argument("--chunks", default="data/flutter_docs/processed/chunks.jsonl")
    parser.add_argument("--out", default="config/datasets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_datasets(Path(args.chunks), Path(args.out))
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
