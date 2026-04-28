from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .classifier import TAXONOMY


MIN_CHUNKS_PER_TOPIC = 5
MIN_SAMPLES_PER_TOPIC = 8


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def dataset_files(datasets_dir: Path) -> list[Path]:
    names = [
        "flutter_official_docs_sft.jsonl",
        "flutter_official_docs_code_sft.jsonl",
        "flutter_official_docs_debugging_sft.jsonl",
        "flutter_official_docs_architecture_sft.jsonl",
        "flutter_official_hard_eval.jsonl",
    ]
    return [datasets_dir / name for name in names]


def build_coverage_report(chunks_path: Path, datasets_dir: Path, out_dir: Path, *, manifest_path: Path | None = None) -> dict:
    chunks = read_jsonl(chunks_path)
    manifest = read_json(manifest_path or Path("data/flutter_docs/manifest.json"))
    chunk_topics = Counter(str(row.get("topic") or "unknown") for row in chunks)
    source_counts = Counter(str(row.get("url") or "") for row in chunks)
    samples_by_topic: Counter[str] = Counter()
    eval_by_topic: Counter[str] = Counter()
    dataset_counts: dict[str, int] = {}
    duplicates = 0
    seen: set[str] = set()
    for path in dataset_files(datasets_dir):
        rows = read_jsonl(path)
        dataset_counts[path.name] = len(rows)
        for row in rows:
            text = json.dumps(row, sort_keys=True, ensure_ascii=True)
            if text in seen:
                duplicates += 1
            seen.add(text)
            if row.get("source_kind") == "flutter_official_hard_eval":
                for topic in row.get("expected_topics") or []:
                    eval_by_topic[str(topic)] += 1
            else:
                samples_by_topic[str(row.get("topic") or "unknown")] += 1

    low = []
    for topic in TAXONOMY:
        if chunk_topics.get(topic, 0) < MIN_CHUNKS_PER_TOPIC or samples_by_topic.get(topic, 0) < MIN_SAMPLES_PER_TOPIC:
            low.append(
                {
                    "topic": topic,
                    "chunks": int(chunk_topics.get(topic, 0)),
                    "train_samples": int(samples_by_topic.get(topic, 0)),
                    "eval_samples": int(eval_by_topic.get(topic, 0)),
                }
            )

    pages = read_jsonl((manifest_path or Path("data/flutter_docs/manifest.json")).parent / "raw/pages.jsonl")
    failed = [row for row in pages if int(row.get("status") or 0) not in (200, 304)]
    ignored = manifest.get("ignored") if isinstance(manifest.get("ignored"), list) else []
    report = {
        "total_urls_discovered": int(manifest.get("discovered_urls") or 0),
        "total_urls_allowed": int(manifest.get("allowed_urls") or 0),
        "total_urls_processed": len({str(row.get("url") or "") for row in pages if int(row.get("status") or 0) in (200, 304)}),
        "pages_failed": len(failed),
        "chunks_total": len(chunks),
        "chunks_by_topic": dict(sorted(chunk_topics.items())),
        "train_samples_by_topic": dict(sorted(samples_by_topic.items())),
        "eval_samples_by_topic": dict(sorted(eval_by_topic.items())),
        "dataset_counts": dataset_counts,
        "low_coverage_topics": low,
        "top_sources": source_counts.most_common(20),
        "duplicates_eliminated_or_detected": duplicates,
        "ignored_pages": ignored[:100],
        "coverage_ok": len(low) == 0 and len(chunks) >= 250,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "coverage.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Flutter Official Docs Coverage",
        "",
        f"Coverage OK: {report.get('coverage_ok')}",
        f"Discovered URLs: {report.get('total_urls_discovered')}",
        f"Allowed URLs: {report.get('total_urls_allowed')}",
        f"Processed URLs: {report.get('total_urls_processed')}",
        f"Failed pages: {report.get('pages_failed')}",
        f"Chunks total: {report.get('chunks_total')}",
        f"Duplicates detected: {report.get('duplicates_eliminated_or_detected')}",
        "",
        "## Dataset Counts",
    ]
    for name, count in (report.get("dataset_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Chunks By Topic")
    for topic, count in (report.get("chunks_by_topic") or {}).items():
        lines.append(f"- {topic}: {count}")
    lines.append("")
    lines.append("## Training Samples By Topic")
    for topic, count in (report.get("train_samples_by_topic") or {}).items():
        lines.append(f"- {topic}: {count}")
    lines.append("")
    lines.append("## Eval Samples By Topic")
    for topic, count in (report.get("eval_samples_by_topic") or {}).items():
        lines.append(f"- {topic}: {count}")
    lines.append("")
    lines.append("## Low Coverage Topics")
    low = report.get("low_coverage_topics") or []
    if not low:
        lines.append("- none")
    else:
        for item in low:
            lines.append(
                f"- {item['topic']}: chunks={item['chunks']}, train={item['train_samples']}, eval={item['eval_samples']}"
            )
    lines.append("")
    lines.append("## Top Sources")
    for url, count in report.get("top_sources") or []:
        lines.append(f"- {count}: {url}")
    lines.append("")
    lines.append("## Ignored Pages")
    ignored = report.get("ignored_pages") or []
    if not ignored:
        lines.append("- none")
    else:
        for item in ignored[:50]:
            lines.append(f"- {item.get('reason')}: {item.get('url')}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Flutter docs coverage.")
    parser.add_argument("--chunks", default="data/flutter_docs/processed/chunks.jsonl")
    parser.add_argument("--datasets", default="config/datasets")
    parser.add_argument("--out", default="data/flutter_docs/reports")
    parser.add_argument("--manifest", default="data/flutter_docs/manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_coverage_report(Path(args.chunks), Path(args.datasets), Path(args.out), manifest_path=Path(args.manifest))
    print(json.dumps(report, ensure_ascii=True))
    return 0 if bool(report.get("coverage_ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
