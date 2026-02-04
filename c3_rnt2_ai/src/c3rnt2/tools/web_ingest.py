from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from ..config import resolve_web_strict
from .web_access import web_fetch


@dataclass
class WebIngestItem:
    url: str
    source_ref: str
    text: str
    content_hash: str | None
    etag: str | None
    last_modified: str | None
    chunk_hash: str
    ts: float


_INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"do not follow",
    r"system prompt",
    r"role:\s*system",
    r"developer message",
    r"you are (an|a) (assistant|system)",
    r"override (the|all) instructions",
]
_IMPERATIVE_TOKENS = {
    "ignore",
    "follow",
    "must",
    "instruction",
    "instructions",
    "system",
    "assistant",
    "developer",
    "user",
    "prompt",
    "override",
    "jailbreak",
    "execute",
    "run",
    "always",
    "never",
}


def canonicalize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
        if not path:
            path = "/"
    return urlunsplit((scheme, netloc, path, parts.query or "", ""))


def _instruction_density(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if tok in _IMPERATIVE_TOKENS)
    pattern_hits = 0
    lowered = text.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, lowered):
            pattern_hits += 1
    return (hits + pattern_hits * 3) / max(1, len(tokens))


def sanitize_text(
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
    if lines:
        filtered: list[str] = []
        for line in lines:
            lowered = line.lower()
            if any(re.search(pat, lowered) for pat in _INJECTION_PATTERNS):
                continue
            filtered.append(line)
        lines = filtered
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


def _chunk_text(text: str, max_chars: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return [cleaned]
    chunk_size = min(800, max_chars) if max_chars > 0 else 800
    if chunk_size <= 0:
        chunk_size = 800
    overlap = 120 if chunk_size > 120 else max(0, chunk_size // 6)
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _simhash(text: str) -> int:
    weights = [0] * 64
    for tok in re.findall(r"[a-zA-Z0-9']+", text.lower()):
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        val = int(digest, 16)
        for i in range(64):
            weights[i] += 1 if (val >> i) & 1 else -1
    out = 0
    for i, w in enumerate(weights):
        if w > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ingest_urls(
    urls: Iterable[str],
    allow_domains: list[str],
    *,
    base_dir: Path,
    settings: dict,
    state: dict | None = None,
) -> list[WebIngestItem]:
    if not allow_domains:
        raise ValueError("allow_domains required")
    now = time.time()
    tools_web = settings.get("tools", {}).get("web", {}) or {}
    ingest_web = settings.get("continuous", {}).get("ingest", {}).get("web", {}) or {}
    max_chars = int((ingest_web.get("sanitize", {}) or {}).get("max_chars", 2000))
    max_instr = float((ingest_web.get("sanitize", {}) or {}).get("max_instruction_density", 0.04))
    max_repeat = int((ingest_web.get("sanitize", {}) or {}).get("max_repeat_lines", 2))
    cooldown_min = float(ingest_web.get("cooldown_minutes", 60))
    max_bytes = int(tools_web.get("max_bytes", 512000))
    timeout_s = int(tools_web.get("timeout_s", 10))
    rate_limit = int(tools_web.get("rate_limit_per_min", 30))
    cache_dir = Path(tools_web.get("cache_dir", base_dir / "data" / "web_cache"))
    cache_ttl_s = tools_web.get("cache_ttl_s", 3600)
    allow_types = tools_web.get("allow_content_types", ["text/", "application/json"])

    state = state or {}
    web_state = state.setdefault("web", {})

    seen_hashes: set[str] = set()
    seen_sim: list[int] = []
    items: list[WebIngestItem] = []
    for raw_url in urls:
        url = canonicalize_url(str(raw_url))
        key = f"url:{url}"
        meta = web_state.get(key, {})
        last_ts = float(meta.get("ts", 0.0))
        if cooldown_min > 0 and (now - last_ts) < cooldown_min * 60.0:
            continue
        fetch = web_fetch(
            url,
            allow_domains,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            cache_dir=cache_dir,
            rate_limit_per_min=rate_limit,
            cache_ttl_s=cache_ttl_s,
            allow_content_types=allow_types,
            strict=resolve_web_strict(settings),
        )
        web_state[key] = {"ts": now, "etag": fetch.etag, "last_modified": fetch.last_modified, "hash": fetch.content_hash}
        if not fetch.ok:
            continue
        cleaned = sanitize_text(fetch.text, max_chars=max_chars, max_instruction_density=max_instr, max_repeat_lines=max_repeat)
        if not cleaned:
            continue
        chunks = _chunk_text(cleaned, max_chars=max_chars)
        for idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_hash in seen_hashes:
                continue
            sim = _simhash(chunk)
            if any(_hamming(sim, other) <= 3 for other in seen_sim):
                continue
            seen_hashes.add(chunk_hash)
            seen_sim.append(sim)
            source_ref = url if len(chunks) == 1 else f"{url}#chunk={idx}"
            items.append(
                WebIngestItem(
                    url=url,
                    source_ref=source_ref,
                    text=chunk,
                    content_hash=fetch.content_hash,
                    etag=fetch.etag,
                    last_modified=fetch.last_modified,
                    chunk_hash=chunk_hash,
                    ts=now,
                )
            )

    if items:
        out_dir = base_dir / "data" / "ingest"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "web.jsonl"
        with out_path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(
                    json.dumps(
                        {
                            "url": item.url,
                            "source_ref": item.source_ref,
                            "text": item.text,
                            "content_hash": item.content_hash,
                            "etag": item.etag,
                            "last_modified": item.last_modified,
                            "chunk_hash": item.chunk_hash,
                            "ts": item.ts,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    return items
