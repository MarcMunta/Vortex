from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings
from c3rnt2.model.bad_decode import bad_decode
from c3rnt2.model.core_transformer import CoreTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="def f(x):")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    settings = load_settings(args.profile)
    core = CoreTransformer.from_settings(settings)
    decode_cfg = settings.get("decode", {})
    bad_cfg = settings.get("bad", {})

    block_size = int(bad_cfg.get("block_size", decode_cfg.get("draft_block", 8)))
    entropy_threshold = float(bad_cfg.get("entropy_threshold", decode_cfg.get("entropy_threshold", 3.5)))

    temperature = float(decode_cfg.get("temperature", 1.0))
    top_p = float(decode_cfg.get("top_p", 1.0))
    repetition_penalty = float(decode_cfg.get("repetition_penalty", 1.0))
    no_repeat_ngram = int(decode_cfg.get("no_repeat_ngram", 0))
    adaptive_granularity = bool(decode_cfg.get("adaptive_granularity", True))
    entropy_top_k = int(bad_cfg.get("entropy_top_k", 64))
    penalty_window = int(bad_cfg.get("penalty_window", 512))
    top_p_min_k = int(bad_cfg.get("top_p_min_k", 128))
    top_p_max_k = int(bad_cfg.get("top_p_max_k", 512))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    _text, stats = bad_decode(
        core,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_size=block_size,
        entropy_threshold=entropy_threshold,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
        adaptive_granularity=adaptive_granularity,
        entropy_top_k=entropy_top_k,
        penalty_window=penalty_window,
        top_p_min_k=top_p_min_k,
        top_p_max_k=top_p_max_k,
    )
    elapsed = max(1e-6, time.time() - start)
    tokens_per_sec = args.max_new_tokens / elapsed

    result = {
        "tokens_per_second": round(tokens_per_sec, 3),
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "entropy_high": stats.entropy_high,
    }
    if torch.cuda.is_available():
        result["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    print(result)


if __name__ == "__main__":
    main()
