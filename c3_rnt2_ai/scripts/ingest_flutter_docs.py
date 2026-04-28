from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.flutter_docs.cleaner import clean_pages
from c3rnt2.flutter_docs.crawler import main as crawler_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    rc = crawler_main(args)
    if rc != 0:
        return rc
    out = "data/flutter_docs"
    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 < len(args):
            out = args[idx + 1]
    if "--dry-run" not in args:
        clean_pages(Path(out) / "raw/pages.jsonl", Path(out) / "processed/chunks.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
