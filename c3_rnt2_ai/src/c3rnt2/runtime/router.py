from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


FEATURES = [
    "prompt_tokens",
    "max_new_tokens",
    "vram_budget",
    "mem_cost_estimate",
    "vram_peak_mb",
    "tok_per_s",
    "verify_accept_rate",
    "error_rate",
]


@dataclass
class RouterDecision:
    backend: str
    stream_topk: bool
    cuda_graphs: bool
    scores: Dict[str, float]


@dataclass
class RouterConfig:
    features: List[str]
    backend_labels: List[str]
    mode_labels: List[str]
    stability_threshold: float = 0.1


@dataclass
class RouterState:
    last_backend: str | None = None
    last_stream: bool | None = None
    last_scores: Dict[str, float] | None = None


class RouterNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.backend_head = nn.Linear(hidden, 2)
        self.mode_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.backend_head(h), self.mode_head(h)


class Router:
    def __init__(self, config: RouterConfig, model: RouterNet, means: List[float], stds: List[float]):
        self.config = config
        self.model = model
        self.means = means
        self.stds = stds
        self.state = RouterState()

    def _normalize(self, feats: List[float]) -> List[float]:
        out = []
        for v, mean, std in zip(feats, self.means, self.stds):
            denom = std if std > 1e-6 else 1.0
            out.append((v - mean) / denom)
        return out

    def _vectorize(self, features: Dict[str, Any]) -> List[float]:
        vec = []
        for key in self.config.features:
            val = features.get(key, 0.0)
            try:
                vec.append(float(val))
            except Exception:
                vec.append(0.0)
        return vec

    def decide(self, features: Dict[str, Any]) -> RouterDecision:
        if torch is None:
            backend = "core"
            stream_topk = False
            scores = {"backend_core": 1.0, "backend_hf": 0.0, "stream_topk": 0.0}
            return RouterDecision(backend=backend, stream_topk=stream_topk, cuda_graphs=False, scores=scores)
        vec = self._vectorize(features)
        vec = self._normalize(vec)
        x = torch.tensor([vec], dtype=torch.float32)
        with torch.inference_mode():
            backend_logits, mode_logits = self.model(x)
            backend_probs = torch.softmax(backend_logits, dim=-1).squeeze(0)
            mode_probs = torch.softmax(mode_logits, dim=-1).squeeze(0)
        backend_idx = int(torch.argmax(backend_probs).item())
        stream_idx = int(torch.argmax(mode_probs).item())
        backend = self.config.backend_labels[backend_idx]
        stream_topk = bool(self.config.mode_labels[stream_idx] == "stream_topk")
        scores = {
            f"backend_{self.config.backend_labels[0]}": float(backend_probs[0].item()),
            f"backend_{self.config.backend_labels[1]}": float(backend_probs[1].item()),
            "stream_topk": float(mode_probs[1].item()),
        }
        # hysteresis
        if self.state.last_backend is not None and self.state.last_backend != backend:
            prev = self.state.last_scores or {}
            prev_score = prev.get(f"backend_{self.state.last_backend}", 0.0)
            new_score = scores.get(f"backend_{backend}", 0.0)
            if new_score - prev_score < self.config.stability_threshold:
                backend = self.state.last_backend
        if self.state.last_stream is not None and self.state.last_stream != stream_topk:
            prev_stream = prev.get("stream_topk", 0.0) if self.state.last_scores else 0.0
            new_stream = scores.get("stream_topk", 0.0)
            if new_stream - prev_stream < self.config.stability_threshold:
                stream_topk = self.state.last_stream
        self.state.last_backend = backend
        self.state.last_stream = stream_topk
        self.state.last_scores = scores
        return RouterDecision(backend=backend, stream_topk=stream_topk, cuda_graphs=False, scores=scores)


def load_router(model_path: Path, manifest_path: Path | None = None) -> Optional[Router]:
    if torch is None:
        return None
    if not model_path.exists():
        return None
    if manifest_path is None:
        manifest_path = model_path.with_suffix(".json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest.get("features", FEATURES)
    backend_labels = manifest.get("backend_labels", ["core", "hf"])
    mode_labels = manifest.get("mode_labels", ["full", "stream_topk"])
    means = manifest.get("means", [0.0 for _ in features])
    stds = manifest.get("stds", [1.0 for _ in features])
    config = RouterConfig(features=list(features), backend_labels=list(backend_labels), mode_labels=list(mode_labels))
    model = RouterNet(in_dim=len(config.features), hidden=int(manifest.get("hidden_size", 16)))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return Router(config=config, model=model, means=list(means), stds=list(stds))


def build_features(prompt: str, max_new_tokens: int, settings: dict, extra: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    runtime = settings.get("runtime", {}) or {}
    vram_budget = float(runtime.get("cache_vram_budget_mb", 0))
    prompt_tokens = len(prompt.split())
    feats = {
        "prompt_tokens": float(prompt_tokens),
        "max_new_tokens": float(max_new_tokens),
        "vram_budget": float(vram_budget),
    }
    if extra:
        for key, val in extra.items():
            if key in FEATURES:
                try:
                    feats[key] = float(val)
                except Exception:
                    continue
    for key in FEATURES:
        feats.setdefault(key, 0.0)
    return feats


def log_router_event(base_dir: Path, payload: Dict[str, Any]) -> None:
    path = base_dir / "data" / "logs" / "router_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
