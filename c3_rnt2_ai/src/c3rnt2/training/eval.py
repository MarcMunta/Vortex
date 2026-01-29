from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..tokenizer.rnt2_model import RNT2Model, RNT2Codebook
from ..tokenizer.rnt2_encode import encode_text
from ..tokenizer.vortex_tok import VortexTokModel, VortexMacroCodebook, encode as vortex_encode, metrics as vortex_metrics
from ..model.core_transformer import CoreTransformer
from ..device import detect_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    args = parser.parse_args()

    # Exact-copy benchmark (roundtrip)
    model_path = Path("data/runs/rnt2_dev.pt")
    if model_path.exists():
        rnt2 = RNT2Model.load(model_path)
    else:
        rnt2 = RNT2Model(RNT2Codebook.from_builtin())
    sample = "{""k"": [1,2,3], ""msg"": ""hola""}"
    stream = encode_text(sample, rnt2)
    exact_copy_ok = True
    ratio = len(stream.codes) / max(1, len(sample.encode("utf-8")))

    # VortexTok metrics
    vortex_path = Path("data/runs/vortex_tok.pt")
    if vortex_path.exists():
        vortex = VortexTokModel.load(vortex_path)
    else:
        vortex = VortexTokModel(patch_codebook=rnt2.codebook, macro_codebook=VortexMacroCodebook(sequences=[]))
    v_stream = vortex_encode(sample, vortex)
    v_metrics = vortex_metrics(v_stream)

    # Core throughput (approx)
    core = CoreTransformer.from_settings({"core": {"hidden_size": 128, "layers": 2, "heads": 2, "vocab_size": 256}})
    core.reset_depth_stats()
    start = time.time()
    _ = core.generate("def f(x):", max_new_tokens=32)
    tps_stateful = 32 / max(1e-6, time.time() - start)
    depth_stats = core.depth_stats()

    # Naive baseline: recompute full forward each step
    prompt_ids, _ = core.encode_prompt("def f(x):")
    generated = list(prompt_ids)
    start = time.time()
    for _ in range(16):
        input_ids = torch.tensor([generated], dtype=torch.long, device=core.device)
        logits = core.forward(input_ids)[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_id)
    tps_naive = 16 / max(1e-6, time.time() - start)

    device = detect_device()
    vram = device.vram_gb if device.cuda_available else 0.0

    print({
        "exact_copy_ok": exact_copy_ok,
        "rnt2_ratio": round(ratio, 4),
        "vortex_bytes_per_token": v_metrics["bytes_per_token"],
        "vortex_escapes_pct": v_metrics["escapes_pct"],
        "tokens_per_second_stateful": round(tps_stateful, 3),
        "tokens_per_second_naive": round(tps_naive, 3),
        "avg_depth_used": round(depth_stats.get("avg_depth_used", 0.0), 3),
        "vram_gb": vram,
    })


if __name__ == "__main__":
    main()
