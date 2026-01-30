from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from ..runtime.router import FEATURES, RouterNet


DEFAULT_BACKEND_LABELS = ["core", "hf"]
DEFAULT_MODE_LABELS = ["full", "stream_topk"]


def _score_event(event: Dict[str, Any], weights: Dict[str, float]) -> float:
    tps = float(event.get("tok_per_s", event.get("tokens_per_sec", 0.0)) or 0.0)
    vram = float(event.get("vram_peak_mb", 0.0) or 0.0)
    err = float(event.get("error_rate", 0.0) or 0.0)
    accept = float(event.get("verify_accept_rate", 0.0) or 0.0)
    return (
        weights.get("speed", 1.0) * tps
        - weights.get("vram", 0.01) * vram
        - weights.get("error", 5.0) * err
        + weights.get("quality", 1.0) * accept
    )


def _load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob("router_events.jsonl"))
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            events.append(payload)
    return events


def _vectorize(events: List[Dict[str, Any]], features: List[str]) -> List[List[float]]:
    vecs = []
    for event in events:
        row = []
        for key in features:
            try:
                row.append(float(event.get(key, 0.0) or 0.0))
            except Exception:
                row.append(0.0)
        vecs.append(row)
    return vecs


def _normalize(vecs: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not vecs:
        return vecs, [], []
    cols = len(vecs[0])
    means = [0.0] * cols
    stds = [1.0] * cols
    for c in range(cols):
        vals = [row[c] for row in vecs]
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = math.sqrt(var)
        means[c] = mean
        stds[c] = std if std > 1e-6 else 1.0
    normed = [[(row[c] - means[c]) / stds[c] for c in range(cols)] for row in vecs]
    return normed, means, stds


def train_router(
    data_path: Path,
    output_path: Path,
    weights: Dict[str, float],
    epochs: int = 200,
    hidden_size: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch not available")
    events = _load_events(data_path)
    if not events:
        raise ValueError("No router events found")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for idx, event in enumerate(events):
        req_id = str(event.get("request_id") or idx)
        grouped.setdefault(req_id, []).append(event)
    selected: List[Dict[str, Any]] = []
    for req_id, items in grouped.items():
        best = max(items, key=lambda e: _score_event(e, weights))
        selected.append(best)

    features = list(FEATURES)
    vecs = _vectorize(selected, features)
    vecs, means, stds = _normalize(vecs)
    backend_labels = list(DEFAULT_BACKEND_LABELS)
    mode_labels = list(DEFAULT_MODE_LABELS)
    y_backend = []
    y_mode = []
    sample_weights = []
    for event in selected:
        backend = str(event.get("backend", "core")).lower()
        if backend not in backend_labels:
            backend = backend_labels[0]
        stream = bool(event.get("stream_topk", False))
        mode = "stream_topk" if stream else "full"
        y_backend.append(backend_labels.index(backend))
        y_mode.append(mode_labels.index(mode))
        score = _score_event(event, weights)
        sample_weights.append(max(0.1, float(score) + 1.0))

    torch.manual_seed(seed)
    x = torch.tensor(vecs, dtype=torch.float32)
    yb = torch.tensor(y_backend, dtype=torch.long)
    ym = torch.tensor(y_mode, dtype=torch.long)
    w = torch.tensor(sample_weights, dtype=torch.float32)
    model = RouterNet(in_dim=len(features), hidden=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits_backend, logits_mode = model(x)
        loss_b = nn.functional.cross_entropy(logits_backend, yb, reduction="none")
        loss_m = nn.functional.cross_entropy(logits_mode, ym, reduction="none")
        loss = (loss_b + loss_m) * w
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    manifest = {
        "features": features,
        "backend_labels": backend_labels,
        "mode_labels": mode_labels,
        "means": means,
        "stds": stds,
        "hidden_size": hidden_size,
        "trained_samples": len(selected),
        "weights": weights,
        "ts": time.time(),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    return {"ok": True, "samples": len(selected), "output": str(output_path), "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/logs")
    parser.add_argument("--output", type=str, default="data/runs/router.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--w-vram", type=float, default=0.01)
    parser.add_argument("--w-error", type=float, default=5.0)
    parser.add_argument("--w-quality", type=float, default=1.0)
    args = parser.parse_args()

    weights = {
        "speed": args.w_speed,
        "vram": args.w_vram,
        "error": args.w_error,
        "quality": args.w_quality,
    }
    result = train_router(
        Path(args.data),
        Path(args.output),
        weights=weights,
        epochs=int(args.epochs),
        hidden_size=int(args.hidden),
        seed=int(args.seed),
    )
    print(result)


if __name__ == "__main__":
    main()
