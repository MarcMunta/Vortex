from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


REQUIRED = {"adapter_config.json", "adapter_model.safetensors"}


def repo_c3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_adapter(path: Path) -> list[str]:
    missing = [name for name in sorted(REQUIRED) if not (path / name).exists()]
    return missing


def copy_adapter(source: Path, target: Path, *, force: bool) -> None:
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"adapter source missing: {source}")
    missing = verify_adapter(source)
    if missing:
        raise FileNotFoundError(f"adapter source missing required files: {missing}")
    if target.exists():
        if not force:
            raise FileExistsError(f"target exists; pass --force to replace: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def write_registry(registry_dir: Path, adapter_dir: Path) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_adapter": str(adapter_dir.as_posix()),
        "last_run_id": adapter_dir.name,
        "ts": time.time(),
        "source": "colab_import",
    }
    (registry_dir / "registry.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def write_reload_request(c3_root: Path, adapter_dir: Path) -> Path:
    path = c3_root / "data/state/reload_adapter_request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"adapter_path": str(adapter_dir.as_posix()), "ts": time.time()}, ensure_ascii=True), encoding="utf-8")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a Colab LoRA adapter into Vortex without auto-promotion.")
    parser.add_argument("source", help="Adapter dir copied/downloaded from Drive.")
    parser.add_argument(
        "--target",
        default="data/registry/hf_train/gemma4_e4b/colab_flutter_python_lora",
        help="Target dir relative to c3_rnt2_ai unless absolute.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Update gemma4_e4b/registry.json. Off by default.")
    parser.add_argument("--reload-request", action="store_true", help="Write data/state/reload_adapter_request.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    c3_root = repo_c3_root()
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target)
    if not target.is_absolute():
        target = (c3_root / target).resolve()
    copy_adapter(source, target, force=bool(args.force))
    registry_dir = c3_root / "data/registry/hf_train/gemma4_e4b"
    promoted = False
    if args.promote:
        write_registry(registry_dir, target)
        promoted = True
    reload_path = None
    if args.reload_request:
        reload_path = write_reload_request(c3_root, target)
    result = {
        "ok": True,
        "source": str(source),
        "target": str(target),
        "promoted": promoted,
        "reload_request": str(reload_path) if reload_path else None,
    }
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
