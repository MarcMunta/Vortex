from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse

from .web_access import web_fetch
from .web_ingest import canonicalize_url


def _allow_url(url: str, allow_domains: list[str]) -> bool:
    domain = urlparse(url).netloc.lower()
    if not domain:
        return False
    domain = domain.split("@")[-1]
    if domain.startswith("[") and domain.endswith("]"):
        domain = domain[1:-1]
    domain = domain.split(":")[0]
    allow_domains = [a.lower().strip() for a in allow_domains if a]
    return any(domain == a or domain.endswith("." + a) for a in allow_domains)


def _extract_urls_from_ddg_html(html: str) -> list[str]:
    if not html:
        return []
    text = html.replace("&amp;", "&")
    urls: list[str] = []
    for m in re.finditer(r"uddg=([^&\"'>]+)", text):
        raw = m.group(1)
        try:
            urls.append(unquote(raw))
        except Exception:
            continue
    for m in re.finditer(r"href=[\"'](https?://[^\"'<>\s]+)", text):
        urls.append(m.group(1))
    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def discover_urls(settings: dict, *, base_dir: Path, state: dict) -> dict[str, Any]:
    cont = settings.get("continuous", {}) or {}
    cfg = cont.get("web_discovery", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return {"ok": True, "skipped": "disabled"}
    seed_queries = list(cfg.get("seed_queries") or [])
    if not seed_queries:
        return {"ok": True, "skipped": "no_seed_queries"}
    max_urls = int(cfg.get("max_urls_per_tick", 10))
    if max_urls <= 0:
        return {"ok": True, "skipped": "max_urls_per_tick<=0"}

    tools_web = settings.get("tools", {}).get("web", {}) or {}
    allow_domains = list(tools_web.get("allow_domains") or [])
    if not allow_domains:
        return {"ok": False, "error": "allow_domains required"}

    max_bytes = int(tools_web.get("max_bytes", 512000))
    timeout_s = int(tools_web.get("timeout_s", 10))
    rate_limit = int(tools_web.get("rate_limit_per_min", 30))
    cache_dir = Path(tools_web.get("cache_dir", base_dir / "data" / "web_cache"))
    cache_ttl_s = tools_web.get("cache_ttl_s", 3600)
    allow_types = list(tools_web.get("allow_content_types", ["text/", "application/json"]))

    found: list[str] = []
    for query in seed_queries:
        if len(found) >= max_urls:
            break
        q = str(query).strip()
        if not q:
            continue
        url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
        fetch = web_fetch(
            url,
            allow_domains,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            cache_dir=cache_dir,
            rate_limit_per_min=rate_limit,
            cache_ttl_s=cache_ttl_s,
            allow_content_types=allow_types,
        )
        if not fetch.ok:
            continue
        for raw_url in _extract_urls_from_ddg_html(fetch.text):
            if len(found) >= max_urls:
                break
            canon = canonicalize_url(str(raw_url))
            if not _allow_url(canon, allow_domains):
                continue
            found.append(canon)

    existing = list(state.get("discovered_urls", []) or [])
    existing_seen = set(existing)
    added = 0
    for u in found:
        if u in existing_seen:
            continue
        existing_seen.add(u)
        existing.append(u)
        added += 1

    # Keep a bounded list to avoid unbounded state growth.
    cap = int(cfg.get("max_total_urls", 200))
    if cap > 0 and len(existing) > cap:
        existing = existing[-cap:]
    state["discovered_urls"] = existing

    return {"ok": True, "added": added, "total": len(existing)}
