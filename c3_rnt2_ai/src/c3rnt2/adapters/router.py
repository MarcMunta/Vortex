from __future__ import annotations

import math
from dataclasses import dataclass

from ..continuous.knowledge_store import EmbeddingBackend


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-9:
        return vec
    return [v / norm for v in vec]


@dataclass(frozen=True)
class RouterDecision:
    selected_adapter: str | None
    reason: str
    score: float | None = None


class AdapterRouter:
    def __init__(
        self,
        *,
        mode: str = "keyword_map",
        keyword_map: dict[str, str] | None = None,
        default_adapter: str | None = None,
        embedding_backend: str = "hash",
        embedding_dim: int = 128,
        embedding_min_score: float = 0.0,
    ):
        self.mode = str(mode or "keyword_map").lower()
        self.keyword_map = {str(k).lower(): str(v) for k, v in (keyword_map or {}).items() if k and v}
        self.default_adapter = str(default_adapter) if default_adapter else None
        self.embedding_min_score = float(embedding_min_score or 0.0)
        self._embedder = EmbeddingBackend(backend=str(embedding_backend or "hash"), dim=int(embedding_dim))

    @classmethod
    def from_settings(cls, settings: dict) -> "AdapterRouter":
        cfg = settings.get("adapters", {}) or {}
        router_cfg = cfg.get("router", {}) or {}
        knowledge = settings.get("knowledge", {}) or {}
        return cls(
            mode=router_cfg.get("mode", "keyword_map"),
            keyword_map=dict(router_cfg.get("keyword_map", {}) or {}),
            default_adapter=router_cfg.get("default") or cfg.get("default"),
            embedding_backend=router_cfg.get("embedding_backend", knowledge.get("embedding_backend", "hash")),
            embedding_dim=int(router_cfg.get("embedding_dim", 128)),
            embedding_min_score=float(router_cfg.get("embedding_min_score", 0.0)),
        )

    def select(self, prompt: str, adapter_names: list[str]) -> RouterDecision:
        if not adapter_names:
            return RouterDecision(None, reason="no_adapters")
        prompt_l = (prompt or "").lower()

        if self.mode == "keyword_map":
            # Prefer longest matches to reduce accidental collisions.
            for kw, adapter in sorted(self.keyword_map.items(), key=lambda kv: len(kv[0]), reverse=True):
                if adapter not in adapter_names:
                    continue
                if kw and kw in prompt_l:
                    return RouterDecision(adapter, reason=f"keyword:{kw}")
            fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
            return RouterDecision(fallback, reason="default")

        if self.mode == "embedding":
            # Embed prompt and per-adapter routing text, then pick highest cosine.
            by_adapter: dict[str, list[str]] = {}
            for kw, adapter in self.keyword_map.items():
                by_adapter.setdefault(adapter, []).append(kw)
            texts = []
            for name in adapter_names:
                kws = " ".join(sorted(set(by_adapter.get(name, []))))
                texts.append(f"{name} {kws}".strip())
            vecs = self._embedder.encode([prompt] + texts)
            qvec = _normalize(vecs[0])
            best_name = adapter_names[0]
            best_score = -1.0
            for name, vec in zip(adapter_names, vecs[1:]):
                score = _dot(qvec, _normalize(vec))
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_score < self.embedding_min_score:
                fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
                return RouterDecision(fallback, reason="embedding_below_threshold", score=float(best_score))
            return RouterDecision(best_name, reason="embedding", score=float(best_score))

        # Unknown mode -> safe fallback.
        fallback = self.default_adapter if self.default_adapter in adapter_names else adapter_names[0]
        return RouterDecision(fallback, reason="unknown_mode_default")

