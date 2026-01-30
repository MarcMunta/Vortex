from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..agent.memory import MemoryStore
from ..agent.tools import AgentTools
from .knowledge_store import KnowledgeStore, embed_text
from .replay_buffer import ReplayBuffer, ReplayItem
from .types import Sample


@dataclass
class CollectStats:
    new_docs: int
    novelty_avg: float
    successes: int
    filtered: int
    total_candidates: int


@dataclass
class CollectedSamples:
    samples: List[Sample]
    stats: CollectStats
    gold_samples: List[Sample]


def _load_logs(data_dir: Path) -> Iterable[str]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".txt"}:
            yield path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield json.dumps(payload)
                except Exception:
                    continue


def _load_episodes(path: Path) -> List[dict]:
    if not path.exists():
        return []
    episodes = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            episodes.append(payload)
    return episodes


def _cosine_sim(a: List[float], b: List[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def _novelty_score(text: str, recent_vecs: List[List[float]]) -> float:
    if not recent_vecs:
        return 1.0
    vec = embed_text(text)
    sims = [_cosine_sim(vec, other) for other in recent_vecs]
    return max(0.0, 1.0 - max(sims))


def _quality_score(text: str, source_kind: str, max_repeat_ratio: float) -> float:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return 0.0
    words = cleaned.split()
    length = len(words)
    if length < 5:
        return 0.1
    unique_ratio = len(set(words)) / max(1, length)
    repeat_ratio = 1.0 - unique_ratio
    base = 0.7
    if source_kind == "logs":
        base *= 0.3
    if source_kind == "memory":
        base *= 0.8
    if source_kind == "web":
        base *= 0.9
    if source_kind == "episode":
        base *= 1.2
    if repeat_ratio > max_repeat_ratio:
        base *= 0.4
    return max(0.0, min(1.0, base))


def ingest_sources(base_dir: Path, allowlist: List[str], settings: dict) -> int:
    continuous = settings.get("continuous", {})
    knowledge_path = Path(continuous.get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    store = KnowledgeStore(knowledge_path)
    new_docs = 0

    # Memory store
    memory_path = base_dir / "data" / "memory" / "agent_memory.sqlite"
    if memory_path.exists():
        mem = MemoryStore(memory_path)
        for item in mem.query("summary", top_k=50):
            new_docs += store.ingest_text("memory", str(memory_path), item.text, quality=0.6)

    # Logs
    for text in _load_logs(base_dir / "data"):
        new_docs += store.ingest_text("logs", "local", text, quality=0.2)

    # Web docs (best effort)
    if bool(continuous.get("ingest_web", True)) and allowlist:
        urls = continuous.get("ingest_urls", ["https://docs.python.org/3/", "https://pytorch.org/docs/stable/"])
        tools = AgentTools(allowlist=allowlist)
        for url in urls:
            doc = tools.open_docs(url)
            if doc.ok:
                new_docs += store.ingest_text("web", url, doc.output, quality=0.7)

    # Episodes
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    for ep in _load_episodes(episodes_path):
        task = str(ep.get("task", "")).strip()
        context = str(ep.get("prompt", "")).strip()
        diff = str(ep.get("patch", "")).strip()
        if task or diff:
            text = f"{task}\n{context}\n{diff}".strip()
            new_docs += store.ingest_text("episode", episodes_path.as_posix(), text, quality=0.9)
    return new_docs


def collect_samples(base_dir: Path, allowlist: List[str], settings: dict) -> CollectedSamples:
    continuous = settings.get("continuous", {})
    replay_cfg = continuous.get("replay", {})
    filter_cfg = continuous.get("filter", {})
    min_quality = float(filter_cfg.get("min_quality", 0.35))
    max_repeat_ratio = float(filter_cfg.get("max_repeat_ratio", 0.8))
    knowledge_path = Path(continuous.get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    replay_path = Path(replay_cfg.get("path", base_dir / "data" / "continuous" / "replay.sqlite"))
    sample_size = int(replay_cfg.get("sample_size", 64))
    top_frac = float(replay_cfg.get("top_frac", 0.7))
    random_frac = float(replay_cfg.get("random_frac", 0.3))
    seed_chunks = int(replay_cfg.get("seed_chunks", 40))
    max_items = replay_cfg.get("max_items")
    max_items = int(max_items) if max_items is not None else None

    store = KnowledgeStore(knowledge_path)
    replay = ReplayBuffer(replay_path)
    new_docs = ingest_sources(base_dir, allowlist, settings)

    recent_vecs = [embed_text(t) for t in replay.recent_texts(limit=50)]
    total_candidates = 0
    filtered = 0
    novelty_scores: List[float] = []
    successes = 0
    gold_samples: List[Sample] = []

    # Episodes -> gold samples
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    for ep in _load_episodes(episodes_path):
        if not ep.get("tests_ok"):
            continue
        task = str(ep.get("task", "")).strip()
        context = str(ep.get("prompt", "")).strip()
        diff = str(ep.get("patch", "")).strip()
        if not diff:
            continue
        prompt = f"Fix the bug: {task}".strip()
        if context:
            prompt = f"{prompt}\nContext:\n{context}"
        sample = Sample(prompt=prompt, response=diff, source_kind="episode")
        event_id = hashlib.sha256(f"{task}\n{context}\n{diff}\ntests_ok".encode("utf-8")).hexdigest()
        gold_samples.append(sample)
        quality = _quality_score(diff, "episode", max_repeat_ratio)
        novelty = _novelty_score(diff, recent_vecs)
        total_candidates += 1
        successes += 1
        if quality >= min_quality:
            inserted = replay.add(
                ReplayItem(
                    sample=sample,
                    source_kind="episode",
                    quality_score=quality,
                    novelty_score=novelty,
                    success_count=1,
                ),
                max_items=max_items,
            )
            if inserted:
                novelty_scores.append(novelty)
            digest = replay.hash_sample(sample.prompt, sample.response)
            replay.bump_success_once(digest, event_id, delta=1)
        else:
            filtered += 1

    # Knowledge chunks -> seed replay
    chunks = store.sample_chunks(limit=seed_chunks, min_quality=min_quality)
    for chunk in chunks:
        prompt = "Continue"
        if chunk.source_kind == "memory":
            prompt = "Summarize"
        elif chunk.source_kind == "web":
            prompt = "Read docs"
        elif chunk.source_kind == "episode":
            prompt = "Review"
        sample = Sample(prompt=prompt, response=chunk.text, source_kind=chunk.source_kind)
        quality = _quality_score(chunk.text, chunk.source_kind, max_repeat_ratio)
        novelty = _novelty_score(chunk.text, recent_vecs)
        total_candidates += 1
        if quality >= min_quality:
            inserted = replay.add(
                ReplayItem(
                    sample=sample,
                    source_kind=chunk.source_kind,
                    quality_score=quality,
                    novelty_score=novelty,
                    success_count=0,
                ),
                max_items=max_items,
            )
            if inserted:
                novelty_scores.append(novelty)
        else:
            filtered += 1

    samples = replay.sample(sample_size, top_frac=top_frac, random_frac=random_frac)
    novelty_avg = sum(novelty_scores) / max(1, len(novelty_scores))
    stats = CollectStats(
        new_docs=new_docs,
        novelty_avg=novelty_avg,
        successes=successes,
        filtered=filtered,
        total_candidates=total_candidates,
    )
    return CollectedSamples(samples=samples, stats=stats, gold_samples=gold_samples)


def retrieve_context(base_dir: Path, query: str, settings: dict, top_k: int = 3) -> str:
    rag_cfg = settings.get("rag", {})
    max_chars = int(rag_cfg.get("max_chars", 1200))
    knowledge_path = Path(settings.get("continuous", {}).get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    store = KnowledgeStore(knowledge_path)
    chunks = store.retrieve(query, top_k=top_k, min_quality=0.0)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined
