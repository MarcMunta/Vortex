from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from .classifier import classify_chunk


SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form"}
NAV_CLASSES = re.compile(r"(?i)(search|cookie|banner)")


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0


class CleanHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._skip_depth = 0
        self._nav_depth = 0
        self._current: list[str] = []
        self._current_kind: str | None = None
        self._current_level = 0
        self._code: list[str] = []
        self._in_code = False
        self._title: list[str] = []
        self._in_title = False
        self._links: list[tuple[str, str]] = []
        self._current_link_href: str | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        cls = attrs_dict.get("class", "")
        role = attrs_dict.get("role", "")
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in {"nav", "footer", "header", "aside"} or (tag != "div" and NAV_CLASSES.search(cls)) or role in {"navigation", "search"}:
            self._nav_depth += 1
            return
        if self._skip_depth or self._nav_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._current_kind = "heading"
            self._current_level = int(tag[1])
        elif tag in {"p", "li", "td", "th", "blockquote"}:
            self._flush()
            self._current_kind = "text"
        elif tag == "pre":
            self._flush()
            self._in_code = True
            self._code = []
        elif tag == "br":
            self._push("\n")
        elif tag == "a":
            href = attrs_dict.get("href")
            if href and not href.startswith("#"):
                self._current_link_href = href
                self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._nav_depth and tag in {"nav", "footer", "header", "aside", "div"}:
            self._nav_depth -= 1
            return
        if self._skip_depth or self._nav_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == "pre" and self._in_code:
            code = "\n".join(self._code).strip()
            if code:
                self.blocks.append(Block("code", code))
            self._code = []
            self._in_code = False
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote"}:
            self._flush()
        elif tag == "a" and self._current_link_href:
            text = clean_space(" ".join(self._current_link_text))
            if text:
                self._links.append((text, self._current_link_href))
            self._current_link_href = None
            self._current_link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._nav_depth:
            return
        if self._in_title:
            self._title.append(data)
        if self._in_code:
            self._code.append(data.rstrip("\n"))
            return
        if self._current_link_href:
            self._current_link_text.append(data)
        self._push(data)

    def _push(self, data: str) -> None:
        if self._current_kind:
            self._current.append(data)

    def _flush(self) -> None:
        if not self._current_kind:
            return
        text = clean_space(" ".join(self._current))
        if text:
            self.blocks.append(Block(self._current_kind, text, self._current_level))
        self._current = []
        self._current_kind = None
        self._current_level = 0

    @property
    def title(self) -> str:
        return clean_space(" ".join(self._title))

    @property
    def links(self) -> list[tuple[str, str]]:
        return self._links[:50]


def clean_space(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def content_id(url: str, text: str) -> str:
    return "flutter_docs_" + hashlib.sha256(f"{url}\n{text}".encode("utf-8")).hexdigest()[:24]


def chunk_blocks(page: dict, blocks: list[Block], parser_title: str) -> list[dict]:
    url = str(page.get("url") or "")
    title = str(page.get("title") or parser_title or url)
    content_hash = str(page.get("content_hash") or "")
    chunks: list[dict] = []
    heading_path: list[str] = []
    current: list[str] = []
    code_blocks: list[str] = []

    def flush() -> None:
        nonlocal current, code_blocks
        text = "\n\n".join(part for part in current if part.strip()).strip()
        if len(text) < 40:
            current = []
            code_blocks = []
            return
        row = {
            "id": content_id(url, text),
            "url": url,
            "title": title,
            "heading_path": heading_path[:],
            "text": text[:5000],
            "code_blocks": code_blocks[:10],
            "topic": "flutter_basics",
            "difficulty": "intermediate",
            "content_hash": content_hash,
            "source_kind": "flutter_official_docs",
        }
        chunks.append(classify_chunk(row))
        current = []
        code_blocks = []

    for block in blocks:
        if block.kind == "heading":
            if current and len("\n".join(current)) > 900:
                flush()
            level = max(1, min(6, block.level or 1))
            heading_path[:] = heading_path[: level - 1] + [block.text]
            current.append("#" * min(level, 4) + " " + block.text)
        elif block.kind == "code":
            code_blocks.append(block.text)
            current.append(f"```dart\n{block.text[:1600]}\n```")
        else:
            current.append(block.text)
        if len("\n\n".join(current)) > 3600:
            flush()
    flush()
    return chunks


def clean_page(page: dict) -> list[dict]:
    html_path = page.get("html_path")
    if not html_path:
        return []
    path = Path(str(html_path))
    if not path.exists():
        return []
    html = path.read_text(encoding="utf-8", errors="replace")
    parser = CleanHtmlParser()
    parser.feed(html)
    return chunk_blocks(page, parser.blocks, parser.title)


def clean_pages(pages_path: Path, out_path: Path) -> dict:
    rows = read_jsonl(pages_path)
    chunks: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    ignored = 0
    for page in rows:
        if int(page.get("status") or 0) not in (200, 304):
            ignored += 1
            continue
        for chunk in clean_page(page):
            h = hashlib.sha256(str(chunk.get("text") or "").encode("utf-8")).hexdigest()
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            chunks.append(chunk)
    write_jsonl(out_path, chunks)
    return {"pages": len(rows), "chunks": len(chunks), "duplicates_removed": duplicates, "ignored_pages": ignored, "out": str(out_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean cached Flutter docs HTML into semantic chunks.")
    parser.add_argument("--pages", default="data/flutter_docs/raw/pages.jsonl")
    parser.add_argument("--out", default="data/flutter_docs/processed/chunks.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = clean_pages(Path(args.pages), Path(args.out))
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
