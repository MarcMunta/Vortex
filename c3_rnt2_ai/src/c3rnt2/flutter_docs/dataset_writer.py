from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


class DomainConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.system_prompt = data.get("system_prompt", "You are Vortex, a master programming assistant.")
        self.domain_name = data.get("domain_name", "generic")
        self.dataset_files = data.get("dataset_files", {
            "general": f"{self.domain_name}_sft.jsonl",
            "code": f"{self.domain_name}_code_sft.jsonl",
            "debugging": f"{self.domain_name}_debugging_sft.jsonl",
            "architecture": f"{self.domain_name}_architecture_sft.jsonl"
        })
        
        topics = data.get("topics", {})
        self.debug_topics = set(topics.get("debugging", []))
        self.code_topics = set(topics.get("code", []))
        self.arch_topics = set(topics.get("architecture", []))
        
        self.hard_eval_templates = [
            (t["prompt"], t["topics"], t["rubric"]) 
            for t in data.get("hard_eval_templates", [])
        ]


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


class DatasetFactory:
    def __init__(self, config: DomainConfig):
        self.config = config

    def make_sample(self, user: str, response: str, *, source_kind: str, source_ref: str, quality: float, topic: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user.strip()},
            ],
            "response": response.strip(),
            "source_kind": source_kind,
            "quality": quality,
            "source_ref": source_ref,
            "topic": topic,
        }

    def explanation_sample(self, chunk: dict) -> dict:
        topic = str(chunk.get("topic") or f"{self.config.domain_name}_basics")
        title = str(chunk.get("title") or topic)
        text = compact(str(chunk.get("text") or ""))
        user = f"Explica el tema `{topic}` aplicado a un caso real. Usa la fuente `{title}` como base."
        response = (
            f"En {self.config.domain_name}, `{topic}` se debe aplicar con una regla práctica basada en el contrato principal.\n\n"
            f"Puntos clave derivados de la documentación oficial:\n- {text}\n\n"
            "Respuesta útil para Vortex: da causa, solución y código mínimo."
        )
        return self.make_sample(user, response, source_kind=f"{self.config.domain_name}_official_docs_sft", source_ref=str(chunk.get("url") or ""), quality=0.95, topic=topic)

    def debugging_sample(self, chunk: dict) -> dict:
        topic = str(chunk.get("topic") or "debugging")
        text = compact(str(chunk.get("text") or ""), 700)
        user = f"Depura un problema de {self.config.domain_name} relacionado con `{topic}` y da pasos concretos."
        response = (
            f"Empieza por reproducir el error mínimo, identifica el fallo en el contrato y soluciona la estructura base.\n\n"
            f"Base oficial: {text}\n\nProvee un fix pequeño y validación posterior."
        )
        return self.make_sample(user, response, source_kind=f"{self.config.domain_name}_official_docs_debugging_sft", source_ref=str(chunk.get("url") or ""), quality=0.97, topic=topic)

    def code_sample(self, chunk: dict) -> dict:
        topic = str(chunk.get("topic") or "coding")
        code_blocks = chunk.get("code_blocks") if isinstance(chunk.get("code_blocks"), list) else []
        code_hint = str(code_blocks[0])[:900] if code_blocks else ""
        text = compact(str(chunk.get("text") or ""), 650)
        user = f"Genera una implementación limpia para `{topic}` en {self.config.domain_name} y explica cómo validarla."
        response = (
            f"Implementa `{topic}` con un estado explícito y claro.\n\n"
            f"Base oficial: {text}\n"
            + (f"\nReferencia de código oficial revisada:\n```\n{code_hint}\n```\n" if code_hint else "")
        )
        return self.make_sample(user, response, source_kind=f"{self.config.domain_name}_official_docs_code_sft", source_ref=str(chunk.get("url") or ""), quality=0.96, topic=topic)

    def architecture_sample(self, chunk: dict) -> dict:
        topic = str(chunk.get("topic") or "architecture")
        text = compact(str(chunk.get("text") or ""), 700)
        user = f"Diseña una estructura mantenible para `{topic}` en {self.config.domain_name}. Incluye capas, responsabilidades y tests."
        response = (
            "Usa una arquitectura que separe claramente responsabilidades (e.g. dominio, infraestructura, presentación).\n\n"
            f"Base oficial: {text}"
        )
        return self.make_sample(user, response, source_kind=f"{self.config.domain_name}_official_docs_architecture_sft", source_ref=str(chunk.get("url") or ""), quality=0.96, topic=topic)


def build_datasets(chunks: list[dict], factory: DatasetFactory, *, max_per_topic: int = 24) -> dict[str, list[dict]]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_topic[str(chunk.get("topic") or f"{factory.config.domain_name}_basics")].append(chunk)
    
    rng = random.Random(17)
    datasets: dict[str, list[dict]] = {key: [] for key in factory.config.dataset_files}
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
            add("general", factory.explanation_sample(chunk))
            if topic in factory.config.debug_topics:
                add("debugging", factory.debugging_sample(chunk))
            if topic in factory.config.code_topics or chunk.get("code_blocks"):
                add("code", factory.code_sample(chunk))
            if topic in factory.config.arch_topics:
                add("architecture", factory.architecture_sample(chunk))
    return datasets


def build_hard_eval(chunks: list[dict], factory: DatasetFactory, *, count: int = 80) -> list[dict]:
    source_by_topic: dict[str, str] = {}
    for chunk in chunks:
        topic = str(chunk.get("topic") or "")
        if topic and topic not in source_by_topic:
            source_by_topic[topic] = str(chunk.get("url") or "")
            
    rows: list[dict] = []
    idx = 0
    templates = factory.config.hard_eval_templates
    if not templates:
        return rows
        
    while len(rows) < count:
        prompt, topics, rubric = templates[idx % len(templates)]
        topic_ref = next((source_by_topic.get(t) for t in topics if source_by_topic.get(t)), "")
        rows.append(
            {
                "prompt": prompt if idx < len(templates) else f"{prompt} Caso variante #{idx // len(templates) + 1}.",
                "expected_topics": topics,
                "rubric": rubric,
                "source_kind": f"{factory.config.domain_name}_official_hard_eval",
                "source_ref": topic_ref,
            }
        )
        idx += 1
    return rows


def write_datasets(chunks_path: Path, out_dir: Path, config_path: Path) -> dict:
    config = DomainConfig(config_path)
    factory = DatasetFactory(config)
    
    chunks = read_jsonl(chunks_path)
    datasets = build_datasets(chunks, factory)
    
    counts = {}
    for key, filename in config.dataset_files.items():
        path = out_dir / filename
        write_jsonl(path, datasets[key])
        counts[filename] = len(datasets[key])
        
    eval_rows = build_hard_eval(chunks, factory)
    if eval_rows:
        eval_path = out_dir / f"{config.domain_name}_official_hard_eval.jsonl"
        write_jsonl(eval_path, eval_rows)
        counts[eval_path.name] = len(eval_rows)
        
    return {"chunks": len(chunks), "out_dir": str(out_dir), "counts": counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Domain-Agnostic SFT/eval datasets.")
    parser.add_argument("--chunks", default="data/flutter_docs/processed/chunks.jsonl")
    parser.add_argument("--out", default="config/datasets")
    parser.add_argument("--config", default="domain_config.json", help="Path to domain configuration JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_datasets(Path(args.chunks), Path(args.out), Path(args.config))
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
