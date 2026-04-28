from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_writer import build_hard_eval, read_jsonl, write_jsonl


def build_eval(chunks_path: Path, out_path: Path, *, count: int = 80) -> dict:
    chunks = read_jsonl(chunks_path)
    rows = build_hard_eval(chunks, count=count)
    write_jsonl(out_path, rows)
    return {"chunks": len(chunks), "eval_samples": len(rows), "out": str(out_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard Flutter eval prompts.")
    parser.add_argument("--chunks", default="data/flutter_docs/processed/chunks.jsonl")
    parser.add_argument("--out", default="config/datasets/flutter_official_hard_eval.jsonl")
    parser.add_argument("--count", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_eval(Path(args.chunks), Path(args.out), count=max(50, min(100, int(args.count))))
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
