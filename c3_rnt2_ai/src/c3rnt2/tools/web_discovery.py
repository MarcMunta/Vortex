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


def _extract_links(html: str, base_url: str) -> list[str]:
    if not html:
        return []
    base = urlparse(base_url)
    links: list[str] = []
    for m in re.finditer(r"href=[\"']([^\"']+)[\"']", html):
        href = m.group(1).strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        if href.startswith("//"):
            href = f"{base.scheme}:{href}"
        if href.startswith("/"):
            href = f"{base.scheme}://{base.netloc}{href}"
        if not (href.startswith("http://") or href.startswith("https://")):
            continue
        links.append(href)
    # Dedup preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _frontier(state: dict) -> dict:
    wd = state.setdefault("web_discovery", {})
    wd.setdefault("queue", [])
    wd.setdefault("seen", {})
    wd.setdefault("last_sitemap_ts", {})
    return wd


def _prune_seen(front: dict, *, now: float, ttl_s: float) -> None:
    seen = front.get("seen", {}) or {}
    if not isinstance(seen, dict):
        front["seen"] = {}
        return
    if ttl_s <= 0:
        return
    keep: dict[str, float] = {}
    for k, v in seen.items():
        try:
            ts = float(v)
        except Exception:
            continue
        if now - ts <= ttl_s:
            keep[str(k)] = ts
    front["seen"] = keep


def _push(front: dict, url: str, *, now: float, cap: int) -> bool:
    url = canonicalize_url(url)
    seen = front.get("seen", {}) or {}
    if url in seen:
        return False
    queue = front.get("queue", [])
    if not isinstance(queue, list):
        queue = []
    queue.append({"url": url, "ts": now})
    # Trim oldest.
    if cap > 0 and len(queue) > cap:
        queue = queue[-cap:]
    front["queue"] = queue
    return True


def _pop_batch(front: dict, *, now: float, max_urls: int) -> list[str]:
    queue = front.get("queue", [])
    if not isinstance(queue, list) or not queue:
        return []
    batch: list[str] = []
    rest: list[dict] = []
    for item in queue:
        if len(batch) >= max_urls:
            rest.append(item)
            continue
        url = str((item or {}).get("url", "")).strip()
        if not url:
            continue
        batch.append(url)
    front["queue"] = rest
    seen = front.get("seen", {}) or {}
    for u in batch:
        seen[canonicalize_url(u)] = now
    front["seen"] = seen
    return batch


def _sitemap_urls(domain: str) -> list[str]:
    scheme = "https"
    return [f"{scheme}://{domain}/sitemap.xml"]


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
    ttl_hours = float(cfg.get("ttl_hours", 72))
    ttl_s = ttl_hours * 3600.0 if ttl_hours and ttl_hours > 0 else 0.0
    queue_cap = int(cfg.get("max_queue", 500))
    max_crawl_pages = int(cfg.get("max_crawl_pages_per_tick", 2))
    max_links_per_page = int(cfg.get("max_links_per_page", 50))
    max_sitemap = int(cfg.get("max_sitemap_urls", 200))

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

    now = __import__("time").time()
    front = _frontier(state)
    _prune_seen(front, now=now, ttl_s=ttl_s)

    added = 0
    # Seed frontier from DDG results if empty.
    if not front.get("queue"):
        for query in seed_queries:
            q = str(query).strip()
            if not q:
                continue
            ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
            fetch = web_fetch(
                ddg_url,
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
                canon = canonicalize_url(str(raw_url))
                if not _allow_url(canon, allow_domains):
                    continue
                if _push(front, canon, now=now, cap=queue_cap):
                    added += 1
                if queue_cap > 0 and len(front.get("queue", [])) >= queue_cap:
                    break
            if queue_cap > 0 and len(front.get("queue", [])) >= queue_cap:
                break

    # Seed sitemaps (best effort; cached by web_fetch).
    last_sitemap = front.get("last_sitemap_ts", {}) or {}
    for domain in allow_domains:
        if max_sitemap <= 0:
            break
        last = float(last_sitemap.get(domain, 0.0) or 0.0)
        if ttl_s > 0 and now - last < ttl_s:
            continue
        for sm_url in _sitemap_urls(domain):
            fetch = web_fetch(
                sm_url,
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
            locs = re.findall(r"<loc>([^<]+)</loc>", fetch.text)
            for raw_url in locs[:max_sitemap]:
                canon = canonicalize_url(str(raw_url))
                if not _allow_url(canon, allow_domains):
                    continue
                if _push(front, canon, now=now, cap=queue_cap):
                    added += 1
        last_sitemap[domain] = now
    front["last_sitemap_ts"] = last_sitemap

    batch = _pop_batch(front, now=now, max_urls=max_urls)

    # Crawl a few pages in the batch to expand the frontier with internal links.
    crawled = 0
    for url in batch[:max_crawl_pages]:
        crawled += 1
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
        for link in _extract_links(fetch.text, url)[:max_links_per_page]:
            canon = canonicalize_url(link)
            if not _allow_url(canon, allow_domains):
                continue
            if _push(front, canon, now=now, cap=queue_cap):
                added += 1

    state["discovered_urls"] = batch
    return {"ok": True, "added": added, "emitted": len(batch), "crawled": crawled, "queue": len(front.get("queue", []))}
