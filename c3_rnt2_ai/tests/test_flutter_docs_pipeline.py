from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.flutter_docs.classifier import classify_text
from c3rnt2.flutter_docs.cleaner import clean_page
from c3rnt2.flutter_docs.coverage import build_coverage_report
from c3rnt2.flutter_docs.crawler import CURATED_PRIORITY_URLS, DEFAULT_SOURCES, OPTIONAL_SOURCES, normalize_url


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_crawler_rejects_urls_outside_allowed_domains() -> None:
    allowed = {"docs.flutter.dev", "api.flutter.dev", "dart.dev"}

    assert normalize_url("https://docs.flutter.dev/ui/layout", allowed) == "https://docs.flutter.dev/ui/layout"
    assert normalize_url("https://api.flutter.dev/flutter/widgets/ListView-class.html", allowed) == "https://api.flutter.dev/flutter/widgets/ListView-class.html"
    assert normalize_url("https://dart.dev/language/async", allowed) == "https://dart.dev/language/async"
    assert normalize_url("https://evil.example/ui/layout", allowed) is None
    assert normalize_url("https://pub.dev/packages/provider", allowed) is None
    assert normalize_url("https://github.com/flutter/flutter/issues/1", allowed) is None
    assert normalize_url("https://docs.flutter.dev/assets/logo.png", allowed) is None
    assert normalize_url("http://docs.flutter.dev/ui/layout", allowed) is None


def test_flutter_ingest_sources_are_official_only() -> None:
    approved = {"docs.flutter.dev", "api.flutter.dev", "dart.dev"}
    assert set(DEFAULT_SOURCES) == {"docs.flutter.dev", "api.flutter.dev"}
    assert set(OPTIONAL_SOURCES) == {"dart.dev"}
    for url in CURATED_PRIORITY_URLS:
        assert normalize_url(url, approved) == url.rstrip("/")


def test_cleaner_removes_navigation_and_preserves_code(tmp_path: Path) -> None:
    html = """
    <html><head><title>Layout docs</title></head>
    <body>
      <nav>Search menu should disappear</nav>
      <main>
        <h1>Constraints</h1>
        <p>Flutter layout passes constraints down and sizes up.</p>
        <pre><code>Expanded(child: ListView())</code></pre>
      </main>
      <footer>Footer disappears</footer>
    </body></html>
    """
    html_path = tmp_path / "page.html"
    html_path.write_text(html, encoding="utf-8")
    chunks = clean_page(
        {
            "url": "https://docs.flutter.dev/ui/layout/constraints",
            "title": "Layout docs",
            "html_path": str(html_path),
            "content_hash": "abc",
            "status": 200,
        }
    )

    assert chunks
    text = chunks[0]["text"]
    assert "Search menu" not in text
    assert "Footer disappears" not in text
    assert "Expanded(child: ListView())" in text
    assert chunks[0]["code_blocks"] == ["Expanded(child: ListView())"]


def test_classifier_assigns_expected_flutter_topics() -> None:
    assert classify_text("RenderBox was not laid out because constraints are unbounded")[0] == "constraints"
    assert classify_text("RenderFlex overflowed by 42 pixels in a Row")[0] == "layout"
    assert classify_text("Use widget tests with flutter_test and pumpWidget")[0] == "testing_unit_widget_integration"
    assert classify_text("Use DevTools timeline to diagnose jank and frame rendering")[0] == "performance"


def test_coverage_report_detects_low_coverage_topics(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    datasets = tmp_path / "datasets"
    reports = tmp_path / "reports"
    manifest = tmp_path / "manifest.json"
    pages = tmp_path / "raw/pages.jsonl"
    manifest.write_text(json.dumps({"discovered_urls": 2, "allowed_urls": 2, "ignored": []}), encoding="utf-8")
    _write_jsonl(pages, [{"url": "https://docs.flutter.dev/ui/layout", "status": 200}])
    _write_jsonl(
        chunks,
        [
            {
                "id": "c1",
                "url": "https://docs.flutter.dev/ui/layout",
                "title": "Layout",
                "text": "constraints layout",
                "topic": "constraints",
            }
        ],
    )
    _write_jsonl(
        datasets / "flutter_official_docs_sft.jsonl",
        [{"topic": "constraints", "source_kind": "flutter_official_docs_sft", "messages": [], "response": "x"}],
    )

    report = build_coverage_report(chunks, datasets, reports, manifest_path=manifest)

    assert report["coverage_ok"] is False
    assert any(item["topic"] == "layout" for item in report["low_coverage_topics"])
    assert (reports / "coverage.md").exists()


def test_training_profile_keeps_general_web_disabled() -> None:
    settings = load_settings("rtx4080_16gb_programming_qwen_coder_train_docker")

    assert settings["tools"]["web"]["enabled"] is False
    assert settings["continuous"]["ingest_web"] is False
    assert settings["autolearn"]["enabled"] is False
    assert settings["hf_train"]["use_weighted_sampling"] is True
    assert settings["hf_train"]["source_kind_weights"]["flutter_official_docs_debugging_sft"] > settings["hf_train"]["source_kind_weights"]["episode"]
