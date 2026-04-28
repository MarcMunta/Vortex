from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCES = ["docs.flutter.dev", "api.flutter.dev"]
OPTIONAL_SOURCES = ["dart.dev"]
USER_AGENT = "VortexFlutterDocsIngest/1.0 (+local research; contact: local)"
ALLOWED_SCHEMES = {"https"}
EXCLUDE_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".xml",
    ".rss",
    ".atom",
    ".zip",
    ".gz",
    ".pdf",
    ".wasm",
}


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    html_path: str | None
    title: str
    section: str
    source_domain: str
    fetched_at: str
    content_hash: str | None
    etag: str | None
    last_modified: str | None
    license_checked: bool
    error: str | None = None

    def to_json(self) -> dict:
        return self.__dict__.copy()


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(part.strip() for part in self.title_parts if part.strip()).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str, allowed_domains: set[str]) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.scheme not in ALLOWED_SCHEMES:
        return None
    if parsed.netloc and parsed.netloc not in allowed_domains:
        return None
    path = parsed.path or "/"
    suffix = Path(path).suffix.lower()
    if suffix in EXCLUDE_EXT:
        return None
    clean = parsed._replace(fragment="", query="")
    normalized = urllib.parse.urlunparse(clean)
    if normalized.endswith("/index.html"):
        normalized = normalized[: -len("index.html")]
    return normalized.rstrip("/") if normalized != f"{parsed.scheme}://{parsed.netloc}/" else normalized


def source_url(domain: str) -> str:
    return f"https://{domain}/"


def source_domain(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def request_url(url: str, headers: dict[str, str] | None = None, timeout: float = 20.0) -> tuple[int, dict[str, str], bytes]:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.1"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        data = resp.read()
        out_headers = {k.lower(): v for k, v in resp.headers.items()}
        if out_headers.get("content-encoding") == "gzip":
            data = gzip.decompress(data)
        return status, out_headers, data


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def robots_for(domain: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"https://{domain}/robots.txt")
    try:
        rp.read()
    except Exception:
        # Fail closed: no crawler should use this parser as allowed unless can_fetch says so.
        rp.parse(["User-agent: *", "Disallow: /"])
    return rp


def robots_sitemaps(domain: str) -> list[str]:
    try:
        _status, _headers, data = request_url(f"https://{domain}/robots.txt", timeout=10)
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return []
    sitemaps = []
    for line in text.splitlines():
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":", 1)[1].strip())
    if not sitemaps:
        sitemaps.append(f"https://{domain}/sitemap.xml")
    return sitemaps


def check_license(domain: str) -> dict:
    license_urls = {
        "docs.flutter.dev": "https://docs.flutter.dev/tos",
        "api.flutter.dev": "https://docs.flutter.dev/tos",
        "dart.dev": "https://dart.dev/terms",
    }
    url = license_urls.get(domain)
    if not url:
        return {"ok": False, "domain": domain, "url": None, "reason": "license_url_unknown"}
    try:
        _status, _headers, data = request_url(url, timeout=15)
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "domain": domain, "url": url, "reason": str(exc)}
    ok = "Creative Commons Attribution 4.0" in text and "3-Clause BSD License" in text
    return {
        "ok": bool(ok),
        "domain": domain,
        "url": url,
        "doc_license": "CC-BY-4.0" if ok else None,
        "code_license": "BSD-3-Clause" if ok else None,
    }


def parse_sitemap(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    urls: list[str] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return urls
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}", 1)[0] + "}"
    for loc in root.findall(f".//{ns}loc"):
        if loc.text:
            urls.append(loc.text.strip())
    return urls


PRIORITY_PATTERNS = [
    "ui/layout",
    "constraints",
    "ui/widgets",
    "cookbook/forms",
    "cookbook/networking",
    "cookbook/navigation",
    "data-and-backend/state-mgmt",
    "app-architecture",
    "testing",
    "cookbook/testing",
    "performance",
    "tools/devtools",
    "ui/adaptive-responsive",
    "ui/accessibility",
    "ui/assets",
    "ui/animations",
    "deployment",
    "platform-integration",
    "packages-and-plugins",
    "dart.dev/language",
    "dart.dev/libraries/async",
]

CURATED_PRIORITY_URLS = [
    "https://docs.flutter.dev/get-started/fundamentals/dart",
    "https://docs.flutter.dev/get-started/fundamentals",
    "https://docs.flutter.dev/get-started/fundamentals/widgets",
    "https://docs.flutter.dev/get-started/fundamentals/layout",
    "https://docs.flutter.dev/ui/layout",
    "https://docs.flutter.dev/ui/layout/constraints",
    "https://docs.flutter.dev/ui/layout/tutorial",
    "https://docs.flutter.dev/ui/layout/scrolling",
    "https://docs.flutter.dev/ui/widgets",
    "https://docs.flutter.dev/ui/adaptive-responsive",
    "https://docs.flutter.dev/ui/adaptive-responsive/general",
    "https://docs.flutter.dev/ui/accessibility",
    "https://docs.flutter.dev/ui/assets/assets-and-images",
    "https://docs.flutter.dev/ui/assets/images",
    "https://docs.flutter.dev/ui/animations",
    "https://docs.flutter.dev/ui/interactivity/gestures",
    "https://docs.flutter.dev/cookbook/gestures/handling-taps/",
    "https://docs.flutter.dev/cookbook/forms/validation",
    "https://docs.flutter.dev/cookbook/forms/text-input",
    "https://docs.flutter.dev/cookbook/navigation/navigation-basics",
    "https://docs.flutter.dev/cookbook/navigation/named-routes",
    "https://docs.flutter.dev/cookbook/networking/fetch-data",
    "https://docs.flutter.dev/cookbook/networking/send-data",
    "https://docs.flutter.dev/cookbook/persistence/key-value",
    "https://docs.flutter.dev/cookbook/persistence/sqlite",
    "https://docs.flutter.dev/cookbook/persistence/reading-writing-files",
    "https://docs.flutter.dev/data-and-backend/state-mgmt/simple",
    "https://docs.flutter.dev/data-and-backend/state-mgmt/options",
    "https://docs.flutter.dev/app-architecture/guide",
    "https://docs.flutter.dev/app-architecture/concepts",
    "https://docs.flutter.dev/app-architecture/case-study",
    "https://docs.flutter.dev/app-architecture/design-patterns/offline-first",
    "https://docs.flutter.dev/testing/overview",
    "https://docs.flutter.dev/cookbook/testing/unit/introduction",
    "https://docs.flutter.dev/cookbook/testing/widget/introduction",
    "https://docs.flutter.dev/cookbook/testing/integration/introduction",
    "https://docs.flutter.dev/cookbook/testing/golden/introduction",
    "https://api.flutter.dev/flutter/flutter_test/matchesGoldenFile.html",
    "https://docs.flutter.dev/perf",
    "https://docs.flutter.dev/perf/best-practices",
    "https://docs.flutter.dev/tools/devtools/performance",
    "https://docs.flutter.dev/debugging",
    "https://docs.flutter.dev/deployment/android",
    "https://docs.flutter.dev/deployment/ios",
    "https://docs.flutter.dev/deployment/web",
    "https://docs.flutter.dev/platform-integration/platform-channels",
    "https://docs.flutter.dev/platform-integration",
    "https://docs.flutter.dev/packages-and-plugins/using-packages",
    "https://docs.flutter.dev/packages-and-plugins/developing-packages",
    "https://docs.flutter.dev/ui/internationalization",
    "https://docs.flutter.dev/cookbook/design/themes",
    "https://docs.flutter.dev/security",
    "https://docs.flutter.dev/testing/errors",
    "https://docs.flutter.dev/testing/common-errors",
    "https://docs.flutter.dev/app-architecture/design-patterns/result",
    "https://docs.flutter.dev/release/breaking-changes",
    "https://docs.flutter.dev/release/breaking-changes/material-theme-system-updates",
    "https://docs.flutter.dev/deployment/web",
    "https://docs.flutter.dev/platforms/desktop",
    "https://dart.dev/language",
    "https://dart.dev/language/async",
    "https://dart.dev/libraries/dart-async",
    "https://dart.dev/effective-dart",
    "https://api.flutter.dev/flutter/widgets/ListView-class.html",
    "https://api.flutter.dev/flutter/widgets/GridView-class.html",
    "https://api.flutter.dev/flutter/widgets/SliverList-class.html",
    "https://api.flutter.dev/flutter/widgets/Expanded-class.html",
    "https://api.flutter.dev/flutter/widgets/Flexible-class.html",
    "https://api.flutter.dev/flutter/widgets/LayoutBuilder-class.html",
    "https://api.flutter.dev/flutter/rendering/RenderBox-class.html",
    "https://api.flutter.dev/flutter/rendering/RenderFlex-class.html",
]


def priority_score(url: str) -> tuple[int, str]:
    lower = url.lower()
    curated_index = {item.lower(): idx for idx, item in enumerate(CURATED_PRIORITY_URLS)}
    if lower in curated_index:
        return -1000 + curated_index[lower], url
    for idx, pattern in enumerate(PRIORITY_PATTERNS):
        if pattern in lower:
            return idx, url
    if "api.flutter.dev/flutter/widgets" in lower:
        return 2, url
    if "api.flutter.dev/flutter/rendering" in lower:
        return 3, url
    if "api.flutter.dev/flutter/material" in lower:
        return 4, url
    return len(PRIORITY_PATTERNS), url


def discover_urls(domains: list[str], *, max_urls: int | None = None) -> list[str]:
    allowed = set(domains)
    discovered: list[str] = []
    seen: set[str] = set()
    for raw in CURATED_PRIORITY_URLS:
        norm = normalize_url(raw, allowed)
        if norm and norm not in seen:
            seen.add(norm)
            discovered.append(norm)
    for domain in domains:
        for sitemap in robots_sitemaps(domain):
            try:
                _status, _headers, data = request_url(sitemap, timeout=20)
            except Exception:
                continue
            for raw in parse_sitemap(data):
                norm = normalize_url(raw, allowed)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                discovered.append(norm)
    discovered = sorted(discovered, key=priority_score)
    if max_urls:
        return discovered[:max_urls]
    if discovered:
        return discovered
    # Fallback: crawl source roots only; page crawl can expand later.
    for domain in domains:
        norm = normalize_url(source_url(domain), allowed)
        if norm and norm not in seen:
            seen.add(norm)
            discovered.append(norm)
    return discovered


def cache_lookup(rows: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        url = str(row.get("url") or "")
        if url:
            latest[url] = row
    return latest


def fetch_page(url: str, out_dir: Path, previous: dict | None, *, license_checked: bool) -> FetchResult:
    headers: dict[str, str] = {}
    if previous:
        if previous.get("etag"):
            headers["If-None-Match"] = str(previous["etag"])
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = str(previous["last_modified"])
    fetched_at = utc_now()
    domain = source_domain(url)
    try:
        status, response_headers, data = request_url(url, headers=headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and previous:
            return FetchResult(
                url=url,
                status=304,
                html_path=previous.get("html_path"),
                title=str(previous.get("title") or ""),
                section=str(previous.get("section") or ""),
                source_domain=domain,
                fetched_at=fetched_at,
                content_hash=previous.get("content_hash"),
                etag=previous.get("etag"),
                last_modified=previous.get("last_modified"),
                license_checked=license_checked,
            )
        return FetchResult(url, int(exc.code), None, "", "", domain, fetched_at, None, None, None, license_checked, str(exc))
    except Exception as exc:
        return FetchResult(url, 0, None, "", "", domain, fetched_at, None, None, None, license_checked, str(exc))

    content_type = response_headers.get("content-type", "")
    if status != 200 or "html" not in content_type.lower():
        return FetchResult(url, status, None, "", "", domain, fetched_at, None, response_headers.get("etag"), response_headers.get("last-modified"), license_checked, f"ignored_content_type:{content_type}")

    html = data.decode("utf-8", errors="replace")
    parser = LinkParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    content_hash = sha256_text(html)
    html_dir = out_dir / "raw/html"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / f"{content_hash}.html"
    html_path.write_text(html, encoding="utf-8")
    section = urllib.parse.urlparse(url).path.strip("/").split("/", 1)[0] or "root"
    return FetchResult(
        url=url,
        status=status,
        html_path=str(html_path.as_posix()),
        title=parser.title,
        section=section,
        source_domain=domain,
        fetched_at=fetched_at,
        content_hash=content_hash,
        etag=response_headers.get("etag"),
        last_modified=response_headers.get("last-modified"),
        license_checked=license_checked,
    )


def crawl(
    *,
    sources: list[str],
    out_dir: Path,
    rate_limit: float,
    max_pages: int | None,
    dry_run: bool,
) -> dict:
    allowed = set(sources)
    robots = {domain: robots_for(domain) for domain in sources}
    licenses = {domain: check_license(domain) for domain in sources}
    license_ok = all(bool(item.get("ok")) for item in licenses.values())
    urls = discover_urls(sources, max_urls=max_pages)
    allowed_urls = []
    ignored: list[dict] = []
    for url in urls:
        norm = normalize_url(url, allowed)
        if not norm:
            ignored.append({"url": url, "reason": "outside_allowlist_or_asset"})
            continue
        rp = robots.get(source_domain(norm))
        if rp is not None and not rp.can_fetch(USER_AGENT, norm):
            ignored.append({"url": norm, "reason": "robots_disallow"})
            continue
        allowed_urls.append(norm)

    manifest_path = out_dir / "manifest.json"
    pages_path = out_dir / "raw/pages.jsonl"
    existing = cache_lookup(read_jsonl(pages_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[dict] = []
    failed = 0
    if not dry_run and license_ok:
        for idx, url in enumerate(allowed_urls):
            result = fetch_page(url, out_dir, existing.get(url), license_checked=True)
            append_jsonl(pages_path, [result.to_json()])
            fetched.append(result.to_json())
            if result.status not in (200, 304):
                failed += 1
            if idx < len(allowed_urls) - 1 and rate_limit > 0:
                time.sleep(rate_limit)

    manifest = {
        "sources": sources,
        "allowed_domains": sorted(allowed),
        "generated_at": utc_now(),
        "license_ok": license_ok,
        "licenses": licenses,
        "robots_checked": True,
        "dry_run": dry_run,
        "rate_limit_s": rate_limit,
        "discovered_urls": len(urls),
        "allowed_urls": len(allowed_urls),
        "fetched_urls": len(fetched),
        "failed_pages": failed,
        "ignored": ignored[:1000],
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest official Flutter/Dart documentation.")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=DEFAULT_SOURCES + OPTIONAL_SOURCES)
    parser.add_argument("--out", default="data/flutter_docs")
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = crawl(
        sources=list(args.sources),
        out_dir=Path(args.out),
        rate_limit=max(0.0, float(args.rate_limit)),
        max_pages=args.max_pages,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(manifest, ensure_ascii=True))
    return 0 if bool(manifest.get("license_ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
