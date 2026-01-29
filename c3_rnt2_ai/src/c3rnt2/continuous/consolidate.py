from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _registry_runs(base_dir: Path) -> Path:
    return base_dir / "data" / "registry" / "runs"


def _adapter_path(base_dir: Path, run_id: str) -> Path:
    return base_dir / "data" / "registry" / "adapters" / f"{run_id}.pt"


def _load_meta(base_dir: Path, run_id: str) -> dict:
    meta_path = _registry_runs(base_dir) / run_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def select_top_runs(base_dir: Path, top_n: int, weights: dict | None = None) -> List[str]:
    weights = weights or {"loss": 1.0, "anchors": 1.0, "episodes": 0.2}
    runs_dir = _registry_runs(base_dir)
    if not runs_dir.exists():
        return []
    scores: List[Tuple[str, float]] = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        run_id = run_path.name
        meta = _load_meta(base_dir, run_id)
        loss = meta.get("loss")
        anchor_reg = meta.get("anchor_regression")
        episodes = meta.get("episode_successes", 0)
        score = 0.0
        if loss is not None:
            score -= float(weights.get("loss", 1.0)) * float(loss)
        if anchor_reg is not None:
            score -= float(weights.get("anchors", 1.0)) * max(0.0, float(anchor_reg))
        if episodes:
            score += float(weights.get("episodes", 0.2)) * float(episodes)
        if _adapter_path(base_dir, run_id).exists():
            scores.append((run_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [run_id for run_id, _score in scores[:top_n]]


def adapter_soup(base_dir: Path, run_ids: List[str], weights: Dict[str, float] | None = None) -> Dict[str, dict]:
    weights = weights or {run_id: 1.0 for run_id in run_ids}
    total = sum(weights.values()) or 1.0
    merged: Dict[str, dict] = {}
    for run_id in run_ids:
        path = _adapter_path(base_dir, run_id)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        weight = float(weights.get(run_id, 1.0)) / total
        for name, state in payload.items():
            if name not in merged:
                merged[name] = {
                    "A": state["A"].clone() * weight,
                    "B": state["B"].clone() * weight,
                    "rank": state["rank"],
                    "alpha": state["alpha"],
                }
            else:
                merged[name]["A"] += state["A"].clone() * weight
                merged[name]["B"] += state["B"].clone() * weight
    return merged


def write_consolidated_adapter(base_dir: Path, payload: Dict[str, dict]) -> Path:
    path = _adapter_path(base_dir, "consolidated")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_consolidated_meta(base_dir: Path, meta: dict) -> None:
    run_dir = _registry_runs(base_dir) / "consolidated"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_out = meta.copy()
    meta_out["ts"] = time.time()
    (run_dir / "meta.json").write_text(json.dumps(meta_out), encoding="utf-8")

