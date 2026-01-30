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

from c3rnt2.config import load_settings  # type: ignore[import-not-found]
from c3rnt2.model.bad_decode import bad_decode, _sample_logits_topk, _RepetitionTracker, _NgramTracker  # type: ignore[import-not-found]
from c3rnt2.model.core_transformer import CoreTransformer  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="def f(x):")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bench-topk", action="store_true")
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
    exact_copy_mode = bool(decode_cfg.get("exact_copy_mode", False))
    escape_restrict = bool(decode_cfg.get("escape_restrict", False))
    use_mtp = bool(decode_cfg.get("use_mtp", True))
    entropy_top_k = int(bad_cfg.get("entropy_top_k", 64))
    penalty_window = int(bad_cfg.get("penalty_window", 512))
    top_p_min_k = int(bad_cfg.get("top_p_min_k", 128))
    top_p_max_k = int(bad_cfg.get("top_p_max_k", 512))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    core.reset_depth_stats()

    tok_start = time.time()
    _ids, _total = core.encode_prompt(args.prompt)
    time_tokenizer_ms = (time.time() - tok_start) * 1000.0

    stream_topk = bool(settings.get("runtime", {}).get("paged_lm_head_stream_topk", False))
    if stream_topk:
        core.runtime_cfg["paged_lm_head_stream_topk"] = False

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
        exact_copy_mode=exact_copy_mode,
        escape_restrict=escape_restrict,
        use_mtp=use_mtp,
    )
    elapsed = max(1e-6, time.time() - start)
    tokens_per_sec = args.max_new_tokens / elapsed

    stream_tps = None
    if stream_topk:
        core.runtime_cfg["paged_lm_head_stream_topk"] = settings.get("runtime", {}).get("paged_lm_head_stream_topk")
        start = time.time()
        _text_s, _stats_s = bad_decode(
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
            exact_copy_mode=exact_copy_mode,
            escape_restrict=escape_restrict,
            use_mtp=use_mtp,
        )
        elapsed_s = max(1e-6, time.time() - start)
        stream_tps = args.max_new_tokens / elapsed_s
        core.runtime_cfg["paged_lm_head_stream_topk"] = False

    bench_topk_ms = None
    if args.bench_topk:
        rep = _RepetitionTracker(128)
        for tok in range(32):
            rep.add(tok)
        ng = _NgramTracker(3)
        for tok in range(3):
            ng.add(tok)
        vals = torch.randn(1, 256, device=core.device)
        idx = torch.arange(256, device=core.device).unsqueeze(0)
        if core.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        iters = 200
        for _ in range(iters):
            _sample_logits_topk(
                vals,
                idx,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.2,
                rep_tracker=rep,
                ngram_tracker=ng,
                top_p_min_k=128,
                top_p_max_k=256,
            )
        if core.device.type == "cuda":
            torch.cuda.synchronize()
        bench_topk_ms = (time.time() - start) * 1000.0 / iters

    depth_stats = core.depth_stats()
    lava_reads = sum(block.lava.stats.reads for block in core.blocks)
    lava_writes = sum(block.lava.stats.writes for block in core.blocks)
    result = {
        "tokens_per_second": round(tokens_per_sec, 3),
        "tokens_per_second_stream_topk": round(stream_tps, 3) if stream_tps else None,
        "topk_sample_ms": round(bench_topk_ms, 4) if bench_topk_ms is not None else None,
        "time_tokenizer_ms": round(time_tokenizer_ms, 3),
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "entropy_high": stats.entropy_high,
        "avg_depth_used": round(depth_stats.get("avg_depth_used", 0.0), 3),
        "lava_reads": lava_reads,
        "lava_writes": lava_writes,
        "bytes_h2d": 0,
        "page_faults": 0,
        "prefetch_hits": 0,
        "bytes_compressed_read": 0,
        "cuda_graphs": bool(settings.get("core", {}).get("cuda_graphs", False)),
        "lava_clusters": int(settings.get("vortex_model", {}).get("lava_clusters", 0)),
        "lava_ann_mode": settings.get("vortex_model", {}).get("lava_ann_mode", "flat"),
    }
    lm_head = getattr(core, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "stats"):
        stats_dict = lm_head.stats()
        result["bytes_h2d"] = stats_dict.get("bytes_h2d", 0)
        result["page_faults"] = stats_dict.get("page_faults", 0)
        result["prefetch_hits"] = stats_dict.get("prefetch_hits", 0)
        result["bytes_compressed_read"] = stats_dict.get("bytes_compressed_read", 0)
    if torch.cuda.is_available():
        result["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
    print(result)
    bench_dir = ROOT / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    latest = bench_dir / "latest.txt"
    latest.write_text(str(result), encoding="utf-8")


if __name__ == "__main__":
    main()
