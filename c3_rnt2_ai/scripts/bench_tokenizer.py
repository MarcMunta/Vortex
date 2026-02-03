from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings  # type: ignore[import-not-found]
from c3rnt2.tokenizer import vortex_tok as vt  # type: ignore[import-not-found]


def _sample_text() -> str:
    return (
        "VORTEX-Tok tokenizer benchmark.\n"
        "This is a short-ish sample with repetition repetition repetition.\n"
        "Unicode: ñ é ö 😀\n"
        "Code:\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench_tokenizer.py")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    profile = args.profile
    settings = load_settings(profile)
    tok_cfg = settings.get("tokenizer", {}) or {}
    model_path = tok_cfg.get("vortex_tok_path") or tok_cfg.get("vortex_model_path") or "data/runs/vortex_tok.pt"
    block_size = int(tok_cfg.get("block_size", 64))

    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    model = vt.load_or_create(path, block_size=block_size)

    text = args.text or _sample_text()
    iters = max(1, int(args.iters))

    t0 = time.perf_counter()
    stream = None
    for _ in range(iters):
        stream = vt.encode(text, model)
    encode_ms = (time.perf_counter() - t0) * 1000.0 / iters
    assert stream is not None

    t1 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = vt.decode(stream, model)
    decode_ms = (time.perf_counter() - t1) * 1000.0 / iters
    assert out == text

    m = vt.metrics(stream)
    result = {
        "ts": time.time(),
        "profile": profile or "default",
        "block_size": int(model.patch_codebook.block_size),
        "macro_codebook_size": int(model.macro_codebook.size),
        "encode_ms": round(encode_ms, 4),
        "decode_ms": round(decode_ms, 4),
        **m,
    }

    bench_dir = ROOT / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "tokenizer_latest.json"
    out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

