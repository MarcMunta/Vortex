from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


@dataclass
class WebFetchResult:
    ok: bool
    url: str
    status: int | None
    text: str
    from_cache: bool
    error: str | None
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str | None = None


class TokenBucket:
    def __init__(self, rate_per_min: int):
        self.capacity = max(1, int(rate_per_min))
        self.tokens = float(self.capacity)
        self.rate_per_sec = self.capacity / 60.0
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            elapsed = max(0.0, now - self.last)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


_BUCKETS: dict[str, TokenBucket] = {}
_BUCKET_LOCK = threading.Lock()


def reset_rate_limits() -> None:
    with _BUCKET_LOCK:
        _BUCKETS.clear()


def _bucket(cache_dir: Path, rate_limit_per_min: int) -> TokenBucket:
    key = str(cache_dir.resolve())
    with _BUCKET_LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = TokenBucket(rate_limit_per_min)
            _BUCKETS[key] = bucket
        return bucket


def _allow_url(url: str, allowlist: list[str]) -> bool:
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    domain = domain.split("@")[-1]
    if domain.startswith("[") and domain.endswith("]"):
        domain = domain[1:-1]
    domain = domain.split(":")[0]
    allowlist = [a.lower().strip() for a in allowlist]
    return any(domain == a or domain.endswith("." + a) for a in allowlist)


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_cache(cache_dir: Path, url: str, ttl_s: Optional[int]) -> Optional[dict]:
    if ttl_s is not None and ttl_s <= 0:
        return None
    path = _cache_path(cache_dir, url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if ttl_s is not None:
        ts = float(payload.get("ts", 0.0))
        if time.time() - ts > ttl_s:
            return None
    return payload


def _write_cache(cache_dir: Path, url: str, payload: dict) -> None:
    path = _cache_path(cache_dir, url)
    payload = dict(payload)
    payload.setdefault("url", url)
    payload["ts"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _content_type_allowed(content_type: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return False
    ct = (content_type or "").split(";")[0].strip().lower()
    if not ct:
        return True
    for item in allowlist:
        val = item.strip().lower()
        if not val:
            continue
        if val.endswith("/*"):
            prefix = val[:-1]
            if ct.startswith(prefix):
                return True
        elif val.endswith("/"):
            if ct.startswith(val):
                return True
        elif ct == val or ct.startswith(val):
            return True
    return False


def _log_event(base_dir: Path, payload: dict) -> None:
    log_path = base_dir / "data" / "logs" / "web_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def web_fetch(
    url: str,
    allowlist: list[str],
    max_bytes: int = 512_000,
    timeout_s: int = 10,
    cache_dir: Path | str = Path("data") / "web_cache",
    rate_limit_per_min: int = 30,
    cache_ttl_s: Optional[int] = None,
    allow_content_types: Optional[list[str]] = None,
) -> WebFetchResult:
    cache_dir = Path(cache_dir)
    base_dir = Path(".")
    allow_content_types = allow_content_types or ["text/", "application/json"]
    use_cache = cache_ttl_s is None or cache_ttl_s > 0

    if not _allow_url(url, allowlist):
        _log_event(base_dir, {"url": url, "ok": False, "error": "allowlist_blocked"})
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="allowlist_blocked")

    bucket = _bucket(cache_dir, rate_limit_per_min)
    if not bucket.allow():
        _log_event(base_dir, {"url": url, "ok": False, "error": "rate_limited"})
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="rate_limited")

    cache_hit = _read_cache(cache_dir, url, ttl_s=cache_ttl_s) if use_cache else None
    raw_cache = _read_cache(cache_dir, url, ttl_s=None) if use_cache else None

    no_net = os.getenv("C3RNT2_NO_NET", "").strip().lower() in {"1", "true", "yes"}
    if no_net:
        cached = cache_hit or raw_cache
        if cached is not None:
            text = str(cached.get("text", ""))
            status = int(cached.get("status", 200))
            _log_event(base_dir, {"url": url, "ok": True, "cached": True, "offline": True, "bytes": len(text), "status": status})
            return WebFetchResult(ok=True, url=url, status=status, text=text, from_cache=True, error=None, etag=cached.get("etag"), last_modified=cached.get("last_modified"), content_hash=cached.get("content_hash"))
        _log_event(base_dir, {"url": url, "ok": False, "error": "network_disabled"})
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error="network_disabled")

    if cache_hit is not None:
        text = str(cache_hit.get("text", ""))
        status = int(cache_hit.get("status", 200))
        _log_event(base_dir, {"url": url, "ok": True, "cached": True, "bytes": len(text), "status": status})
        return WebFetchResult(ok=True, url=url, status=status, text=text, from_cache=True, error=None)

    headers: dict[str, str] = {}
    if raw_cache:
        etag = raw_cache.get("etag")
        last_modified = raw_cache.get("last_modified")
        if etag:
            headers["If-None-Match"] = str(etag)
        if last_modified:
            headers["If-Modified-Since"] = str(last_modified)

    try:
        resp = requests.get(url, timeout=timeout_s, stream=True, headers=headers)
    except Exception as exc:
        _log_event(base_dir, {"url": url, "ok": False, "error": str(exc)})
        return WebFetchResult(ok=False, url=url, status=None, text="", from_cache=False, error=str(exc))

    status = resp.status_code
    if status == 304 and raw_cache is not None:
        text = str(raw_cache.get("text", ""))
        payload = dict(raw_cache)
        payload["status"] = int(raw_cache.get("status", 200))
        if use_cache:
            _write_cache(cache_dir, url, payload)
        _log_event(base_dir, {"url": url, "ok": True, "cached": True, "status": 304, "bytes": len(text)})
        return WebFetchResult(
            ok=True,
            url=url,
            status=int(payload.get("status", 200)),
            text=text,
            from_cache=True,
            error=None,
            etag=payload.get("etag"),
            last_modified=payload.get("last_modified"),
            content_hash=payload.get("content_hash"),
        )
    if status != 200:
        _log_event(base_dir, {"url": url, "ok": False, "status": status})
        return WebFetchResult(ok=False, url=url, status=status, text="", from_cache=False, error=f"http {status}")

    content_type = resp.headers.get("Content-Type", "")
    if not _content_type_allowed(content_type, allow_content_types):
        _log_event(base_dir, {"url": url, "ok": False, "status": status, "error": "unsupported_content_type", "content_type": content_type})
        return WebFetchResult(ok=False, url=url, status=status, text="", from_cache=False, error="unsupported_content_type")

    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                truncated = True
                break
            chunks.append(chunk)
    finally:
        try:
            resp.close()
        except Exception:
            pass
    text = b"".join(chunks).decode("utf-8", errors="ignore")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = {
        "text": text,
        "status": status,
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "content_type": content_type,
        "content_hash": content_hash,
    }
    if use_cache:
        _write_cache(cache_dir, url, payload)
    _log_event(base_dir, {"url": url, "ok": True, "status": status, "bytes": len(text), "truncated": truncated})
    return WebFetchResult(
        ok=True,
        url=url,
        status=status,
        text=text,
        from_cache=False,
        error=None,
        etag=resp.headers.get("ETag"),
        last_modified=resp.headers.get("Last-Modified"),
        content_hash=content_hash,
    )
