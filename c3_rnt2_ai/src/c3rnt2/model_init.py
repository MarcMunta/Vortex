from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "llama-2-7b-chat.Q4_K_M.gguf"


def resolve_cache_dir(raw: str | None = None) -> Path:
    candidate = str(raw or "").strip()
    if not candidate:
        candidate = (
            os.getenv("HF_HOME")
            or os.getenv("HF_CACHE_DIR")
            or os.getenv("TRANSFORMERS_CACHE")
            or str(Path.home() / ".cache" / "huggingface")
        )
    return Path(candidate).expanduser().resolve()


def _repo_cache_dirs(cache_dir: Path, model_id: str) -> list[Path]:
    repo_key = model_id.replace("/", "--")
    repo_name = f"models--{repo_key}"
    return [cache_dir / "hub" / repo_name, cache_dir / repo_name]


def model_cache_status(model_id: str, cache_dir: Path) -> dict[str, Any]:
    candidate_dirs = _repo_cache_dirs(cache_dir, model_id)
    repo_dir = next((path for path in candidate_dirs if path.exists()), candidate_dirs[0])
    snapshots_dir = repo_dir / "snapshots"
    snapshot_paths = []
    if snapshots_dir.exists():
        snapshot_paths = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    refs_dir = repo_dir / "refs"
    refs = []
    if refs_dir.exists():
        refs = [path.name for path in refs_dir.iterdir() if path.is_file()]
    return {
        "model_id": model_id,
        "cache_dir": str(cache_dir),
        "repo_dir": str(repo_dir),
        "candidate_repo_dirs": [str(path) for path in candidate_dirs],
        "cached": bool(snapshot_paths),
        "snapshot_count": len(snapshot_paths),
        "snapshots": [path.name for path in snapshot_paths],
        "last_snapshot": snapshot_paths[-1].name if snapshot_paths else None,
        "refs": refs,
    }


def ensure_model(model_id: str, *, cache_dir: Path) -> dict[str, Any]:
    before = model_cache_status(model_id, cache_dir)
    if before.get("cached"):
        return {"ok": True, "downloaded": False, "ts": time.time(), **before}
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "error": f"huggingface_hub_unavailable:{exc}",
            "model_id": model_id,
            "cache_dir": str(cache_dir),
        }
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        resume_download=True,
    )
    after = model_cache_status(model_id, cache_dir)
    return {
        "ok": bool(after.get("cached")),
        "downloaded": True,
        "ts": time.time(),
        "local_path": str(local_path),
        **after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or explicitly populate the local Hugging Face model cache.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Explicitly download the model into the local cache.")
    args = parser.parse_args(argv)

    cache_dir = resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.download):
        payload = ensure_model(str(args.model), cache_dir=cache_dir)
    else:
        status = model_cache_status(str(args.model), cache_dir)
        payload = {
            "ok": bool(status.get("cached", False)),
            "downloaded": False,
            "ts": time.time(),
            **status,
        }
        if not bool(payload.get("ok")):
            payload["error"] = "model_not_cached"
            payload["next_step"] = (
                "Run explicitly: python -m c3rnt2.model_init "
                f"--model {args.model} --cache-dir {cache_dir} --download"
            )
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
