from __future__ import annotations

import json
import sqlite3
import time
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..agent.memory import MemoryStore
from ..agent.tools import AgentTools
from .knowledge_store import KnowledgeStore, IngestPolicy, EmbeddingBackend, embed_text
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


class IngestState:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS ingest_state (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
            conn.commit()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO ingest_state (key, value, ts) VALUES (?, ?, ?)", (key, value, time.time()))
            conn.commit()

    def get_json(self, key: str, default: dict) -> dict:
        raw = self.get(key)
        if not raw:
            return dict(default)
        try:
            return json.loads(raw)
        except Exception:
            return dict(default)

    def set_json(self, key: str, value: dict) -> None:
        self.set(key, json.dumps(value))


def _iter_log_files(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "agent.jsonl" and "episodes" in path.parts:
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".txt", ".jsonl"}:
            yield path


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


def _resolve_queue_dir(base_dir: Path, settings: dict) -> Path:
    queue_dir = settings.get("self_patch", {}).get("queue_dir", "data/self_patch/queue")
    qpath = Path(queue_dir)
    if not qpath.is_absolute():
        qpath = base_dir / qpath
    return qpath


def _load_patch_from_queue(queue_dir: Path, patch_id: str | None) -> str:
    if not patch_id:
        return ""
    patch_path = queue_dir / patch_id / "patch.diff"
    if not patch_path.exists():
        return ""
    try:
        return patch_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _episode_prompt(payload: dict) -> str:
    prompt = str(payload.get("prompt", "") or payload.get("context", "")).strip()
    if prompt:
        return prompt
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg.get("content")).strip()
    return ""


def _episode_hash(task: str, prompt: str, patch: str, tests_ok: bool, tools_ok: bool) -> str:
    signals = []
    if tests_ok:
        signals.append("tests_ok")
    if tools_ok:
        signals.append("tools_ok")
    signal_label = "+".join(signals)
    payload = f"{task}\n{prompt}\n{patch}\n{signal_label}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


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


_INSTRUCTION_KEYWORDS = {
    "system",
    "assistant",
    "developer",
    "user",
    "instruction",
    "instructions",
    "ignore",
    "follow",
    "must",
    "policy",
    "prompt",
    "role",
    "override",
    "jailbreak",
}


def _instruction_density(text: str) -> float:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", lowered)
    if not tokens:
        return 0.0
    keyword_hits = sum(1 for tok in tokens if tok in _INSTRUCTION_KEYWORDS)
    pattern_hits = 0
    patterns = [
        r"ignore (all|previous) instructions",
        r"do not follow",
        r"system prompt",
        r"role:\s*system",
        r"developer message",
        r"you are (an|a) (assistant|system)",
    ]
    for pat in patterns:
        if re.search(pat, lowered):
            pattern_hits += 1
    return (keyword_hits + pattern_hits * 3) / max(1, len(tokens))


def _sanitize_web_text(
    text: str,
    *,
    max_chars: int,
    max_instruction_density: float,
    max_repeat_lines: int,
) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if max_repeat_lines > 0 and lines:
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for line in lines:
            key = line.lower()
            count = seen.get(key, 0)
            if count >= max_repeat_lines:
                continue
            seen[key] = count + 1
            deduped.append(line)
        lines = deduped
    cleaned = " ".join(" ".join(lines).split())
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    if _instruction_density(cleaned) > max_instruction_density:
        return ""
    return cleaned.strip()


def _promote_web_quarantine(
    store: KnowledgeStore,
    recent_vecs: List[List[float]],
    filter_cfg: dict,
    max_repeat_ratio: float,
) -> int:
    min_quality = float(filter_cfg.get("min_quality", 0.35))
    min_chars = int(filter_cfg.get("min_chars", 200))
    min_novelty = float(filter_cfg.get("min_novelty", 0.2))
    limit = int(filter_cfg.get("quarantine_limit", 200))
    promoted = 0
    for chunk in store.sample_chunks(limit=limit, min_quality=0.0, source_kinds=["web"]):
        if chunk.score >= min_quality:
            continue
        text = chunk.text
        if len(text) < min_chars:
            continue
        quality = _quality_score(text, "web", max_repeat_ratio)
        novelty = _novelty_score(text, recent_vecs)
        if quality >= min_quality and novelty >= min_novelty:
            if store.update_quality(text, max(min_quality, quality)):
                promoted += 1
    return promoted


def ingest_sources(base_dir: Path, allowlist: List[str], settings: dict) -> int:
    continuous = settings.get("continuous", {})
    ingest_cfg = continuous.get("ingest", {}) or {}
    max_files_per_tick = int(ingest_cfg.get("max_files_per_tick", 200))
    max_bytes_per_file = int(ingest_cfg.get("max_bytes_per_file", 2_000_000))
    max_total_bytes_per_tick = int(ingest_cfg.get("max_total_bytes_per_tick", 10_000_000))
    web_cfg = ingest_cfg.get("web", {}) or {}
    web_cooldown = float(web_cfg.get("cooldown_minutes", 60))
    knowledge_path = Path(continuous.get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    knowledge_cfg = settings.get("knowledge", {}) or {}
    policy_cfg = knowledge_cfg.get("policy", {}) or {}
    policy = IngestPolicy(
        min_quality=float(policy_cfg.get("min_quality", 0.0)),
        max_age_days=policy_cfg.get("max_age_days"),
        allow_domains=policy_cfg.get("allow_domains"),
        deny_domains=policy_cfg.get("deny_domains"),
        allow_source_kinds=policy_cfg.get("allow_source_kinds"),
        deny_source_kinds=policy_cfg.get("deny_source_kinds"),
    )
    embed_backend = knowledge_cfg.get("embedding_backend", "auto")
    embed_model = knowledge_cfg.get("embedding_model")
    embedder = EmbeddingBackend(backend=str(embed_backend), model_name=embed_model) if embed_model else embed_backend
    store = KnowledgeStore(
        knowledge_path,
        embedding_backend=embedder,
        index_backend=knowledge_cfg.get("index_backend", "auto"),
        policy=policy,
    )
    state = IngestState(knowledge_path)
    new_docs = 0
    files_used = 0
    bytes_used = 0

    # Memory store (small; dedup via hash)
    memory_path = base_dir / "data" / "memory" / "agent_memory.sqlite"
    if memory_path.exists():
        mem = MemoryStore(memory_path)
        for item in mem.query("summary", top_k=50):
            new_docs += store.ingest_text("memory", str(memory_path), item.text, quality=0.6)

    # Logs (incremental)
    for path in _iter_log_files(base_dir / "data"):
        if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
            break
        try:
            stat = path.stat()
        except Exception:
            continue
        key = f"log:{path.as_posix()}"
        meta = state.get_json(key, {"mtime": 0.0, "size": 0, "offset": 0})
        prev_mtime = float(meta.get("mtime", 0.0))
        prev_size = int(meta.get("size", 0))
        offset = int(meta.get("offset", 0))
        if stat.st_size < offset:
            offset = 0
        if stat.st_mtime == prev_mtime and stat.st_size == prev_size and offset >= stat.st_size:
            continue
        remaining = stat.st_size - offset
        if remaining <= 0:
            state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size, "offset": offset})
            continue
        budget_left = max_total_bytes_per_tick - bytes_used
        if budget_left <= 0:
            break
        max_bytes = min(max_bytes_per_file, budget_left, remaining)
        if max_bytes <= 0:
            break
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(max_bytes)
        except Exception:
            continue
        if not data:
            continue
        if path.suffix.lower() == ".jsonl":
            last_nl = data.rfind(b"\n")
            if last_nl == -1:
                continue
            data = data[: last_nl + 1]
        text = data.decode("utf-8", errors="ignore")
        if text:
            new_docs += store.ingest_text("logs", path.as_posix(), text, quality=0.2)
        bytes_used += len(data)
        files_used += 1
        offset += len(data)
        state.set_json(key, {"mtime": stat.st_mtime, "size": stat.st_size, "offset": offset})

    # Web docs (cache + cooldown)
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    web_enabled = bool(web_cfg.get("enabled", False))
    if bool(continuous.get("ingest_web", True)) and allowlist and web_enabled:
        urls = continuous.get("ingest_urls", ["https://docs.python.org/3/", "https://pytorch.org/docs/stable/"])
        tools = AgentTools(allowlist=allowlist, web_cfg=tools_cfg)
        for url in urls:
            if files_used >= max_files_per_tick or bytes_used >= max_total_bytes_per_tick:
                break
            key = f"web:{url}"
            meta = state.get_json(key, {})
            last_ts = float(meta.get("ts", 0.0))
            last_hash = meta.get("hash")
            if web_cooldown > 0 and (time.time() - last_ts) < web_cooldown * 60.0:
                continue
            doc = tools.open_docs(url)
            if not doc.ok:
                continue
            content = doc.output or ""
            sanitize_cfg = web_cfg.get("sanitize", {}) or {}
            max_chars = int(sanitize_cfg.get("max_chars", 2000))
            max_instr = float(sanitize_cfg.get("max_instruction_density", 0.04))
            max_repeat_lines = int(sanitize_cfg.get("max_repeat_lines", 2))
            content = _sanitize_web_text(
                content,
                max_chars=max_chars,
                max_instruction_density=max_instr,
                max_repeat_lines=max_repeat_lines,
            )
            if not content:
                continue
            content_bytes = content.encode("utf-8", errors="ignore")
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            if content_hash == last_hash:
                state.set_json(key, {"ts": time.time(), "hash": last_hash})
                continue
            budget_left = max_total_bytes_per_tick - bytes_used
            if budget_left <= 0:
                break
            max_bytes = min(max_bytes_per_file, budget_left)
            if max_bytes > 0 and len(content_bytes) > max_bytes:
                content_bytes = content_bytes[:max_bytes]
                content = content_bytes.decode("utf-8", errors="ignore")
            new_docs += store.ingest_text("web", url, content, quality=0.0)
            bytes_used += len(content_bytes)
            files_used += 1
            state.set_json(key, {"ts": time.time(), "hash": content_hash})

    # Episodes (incremental)
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    if episodes_path.exists() and files_used < max_files_per_tick and bytes_used < max_total_bytes_per_tick:
        key = f"episodes:{episodes_path.as_posix()}"
        meta = state.get_json(key, {"size": 0, "offset": 0})
        offset = int(meta.get("offset", 0))
        queue_dir = _resolve_queue_dir(base_dir, settings)
        try:
            stat = episodes_path.stat()
        except Exception:
            stat = None
        if stat is not None:
            if stat.st_size < offset:
                offset = 0
            if not (stat.st_size == int(meta.get("size", 0)) and offset >= stat.st_size):
                remaining = stat.st_size - offset
                if remaining > 0:
                    budget_left = max_total_bytes_per_tick - bytes_used
                    max_bytes = min(max_bytes_per_file, budget_left, remaining)
                    if max_bytes > 0:
                        with episodes_path.open("rb") as handle:
                            handle.seek(offset)
                            data = handle.read(max_bytes)
                        if data:
                            last_nl = data.rfind(b"\n")
                            if last_nl != -1:
                                data = data[: last_nl + 1]
                                offset += len(data)
                                text = data.decode("utf-8", errors="ignore")
                                for line in text.splitlines():
                                    try:
                                        payload = json.loads(line)
                                    except Exception:
                                        continue
                                    if not isinstance(payload, dict):
                                        continue
                                    tests_ok = bool(payload.get("tests_ok"))
                                    tools_ok = bool(payload.get("tools_ok"))
                                    if not (tests_ok or tools_ok):
                                        continue
                                    task = str(payload.get("task", "")).strip()
                                    context = _episode_prompt(payload)
                                    diff = str(payload.get("patch", "")).strip()
                                    if not diff:
                                        raw_patch_id = payload.get("patch_id")
                                        patch_id = str(raw_patch_id).strip() if raw_patch_id else ""
                                        diff = _load_patch_from_queue(queue_dir, patch_id)
                                    if task or diff:
                                        doc_text = f"{task}\n{context}\n{diff}".strip()
                                        new_docs += store.ingest_text("episode", episodes_path.as_posix(), doc_text, quality=0.9)
                                bytes_used += len(data)
                                files_used += 1
                        state.set_json(key, {"size": stat.st_size, "offset": offset})

    return new_docs


def collect_samples(base_dir: Path, allowlist: List[str], settings: dict, ingest: bool = True) -> CollectedSamples:
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
    new_docs = ingest_sources(base_dir, allowlist, settings) if ingest else 0

    recent_vecs = [embed_text(t) for t in replay.recent_texts(limit=50)]
    _promote_web_quarantine(store, recent_vecs, filter_cfg, max_repeat_ratio)
    total_candidates = 0
    filtered = 0
    novelty_scores: List[float] = []
    successes = 0
    gold_samples: List[Sample] = []

    # Episodes -> gold samples
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    queue_dir = _resolve_queue_dir(base_dir, settings)
    seen_episode_hashes: set[str] = set()
    for ep in _load_episodes(episodes_path):
        tests_ok = bool(ep.get("tests_ok"))
        tools_ok = bool(ep.get("tools_ok"))
        if not (tests_ok or tools_ok):
            continue
        task = str(ep.get("task", "")).strip()
        context = _episode_prompt(ep)
        diff = str(ep.get("patch", "")).strip()
        if not diff:
            raw_patch_id = ep.get("patch_id")
            patch_id = str(raw_patch_id).strip() if raw_patch_id else ""
            diff = _load_patch_from_queue(queue_dir, patch_id)
        if not diff:
            continue
        prompt = f"Task: {task}".strip()
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        sample = Sample(prompt=prompt, response=diff, source_kind="episode")
        event_id = _episode_hash(task, context, diff, tests_ok, tools_ok)
        if event_id in seen_episode_hashes:
            continue
        seen_episode_hashes.add(event_id)
        gold_samples.append(sample)
        quality = _quality_score(diff, "episode", max_repeat_ratio)
        novelty = _novelty_score(diff, recent_vecs)
        total_candidates += 1
        successes += 1
        digest = replay.hash_sample(sample.prompt, sample.response)
        replay.bump_success_once(digest, event_id, delta=1)
        if quality >= min_quality:
            inserted = replay.add(
                ReplayItem(
                    sample=sample,
                    source_kind="episode",
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


def retrieve_context_details(base_dir: Path, query: str, settings: dict, top_k: int = 3) -> tuple[str, list[str]]:
    rag_cfg = settings.get("rag", {})
    max_chars = int(rag_cfg.get("max_chars", 1200))
    knowledge_path = Path(settings.get("continuous", {}).get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    store = KnowledgeStore(knowledge_path)
    chunks = store.retrieve(query, top_k=top_k, min_quality=0.0)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars]
    refs = [chunk.source_ref for chunk in chunks if chunk.source_ref]
    return joined, refs


def retrieve_context(base_dir: Path, query: str, settings: dict, top_k: int = 3) -> str:
    context, _refs = retrieve_context_details(base_dir, query, settings, top_k=top_k)
    return context

