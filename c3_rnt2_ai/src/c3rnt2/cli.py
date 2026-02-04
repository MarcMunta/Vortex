from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
try:
    from rich import print as rprint
except Exception:  # pragma: no cover
    def rprint(*args, **kwargs):
        print(*args, **kwargs)


from .config import load_settings, resolve_profile, validate_profile
from .device import detect_device
from .doctor import check_deps, run_deep_checks
from .logging import setup_logging, get_logger
from .tokenizer.rnt2_encode import encode_text
from .tokenizer.rnt2_decode import decode_stream
from .tokenizer.rnt2_model import RNT2Model, RNT2Codebook
from .model.core_transformer import CoreTransformer
from .agent.agent_loop import run_demo_agent

app = typer.Typer(help="Vortex CLI")
logger = get_logger(__name__)


@app.callback()
def _main(log_level: Optional[str] = None, profile: Optional[str] = None):
    setup_logging(log_level)
    os.environ["C3RNT2_PROFILE"] = resolve_profile(profile)


@app.command()
def doctor(deep: bool = False):
    """Check CUDA, deps, and environment."""
    info = detect_device()
    rprint({
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    })

    modules = [
        "torch",
        "bitsandbytes",
        "faiss",
        "triton",
        "fastapi",
        "zstandard",
        "lz4",
    ]
    status = check_deps(modules)
    rprint({"deps": status})

    base_dir = Path(__file__).resolve().parents[2]
    try:
        settings = load_settings(None)
        validate_profile(settings, base_dir=base_dir)
        rprint({"settings_ok": True, "profile": resolve_profile(None)})
        if deep:
            try:
                deep_result = run_deep_checks(settings, base_dir=base_dir)
                rprint({"deep": deep_result})
            except Exception as exc:
                rprint({"deep": {"deep_ok": False, "error": str(exc)}})
    except Exception as exc:
        rprint({"warning": "settings_invalid", "error": str(exc), "hint": "Update config/settings.yaml to include missing keys"})
        if deep:
            rprint({"deep": {"deep_ok": False}})

    tracked_env = []
    try:
        result = subprocess.run(["git", "ls-files", ".env", ".mypy_cache"], capture_output=True, text=True)
        tracked_env = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        tracked_env = []
    if tracked_env:
        rprint({"warning": "tracked_sensitive_files", "files": tracked_env})

    tracked_cache = []
    tracked_large = []
    try:
        result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        repo_root = Path(__file__).resolve().parents[2]
        cache_markers = [".mypy_cache/", "__pycache__/", ".pytest_cache/", ".ruff_cache/", ".venv/", "data/"]
        for rel in files:
            rel_norm = rel.replace("\\", "/").lower()
            if any(marker in rel_norm for marker in cache_markers) or rel_norm.endswith(".pyc"):
                tracked_cache.append(rel)
            if rel_norm.endswith((".pt", ".bin", ".safetensors")):
                tracked_large.append(rel)
                continue
            try:
                size = (repo_root / rel).stat().st_size
                if size > 100 * 1024 * 1024:
                    tracked_large.append(f"{rel} ({size / (1024 ** 2):.1f} MB)")
            except Exception:
                continue
    except Exception:
        tracked_cache = []
        tracked_large = []
    if tracked_cache:
        rprint({"warning": "tracked_cache_files", "files": tracked_cache})
    if tracked_large:
        rprint({"warning": "tracked_large_files", "files": tracked_large})

@app.command()
def demo_tokenizer(text: Optional[str] = None, profile: Optional[str] = None):
    """Run RNT-2 encode/decode demo and report ratio."""
    settings = load_settings(profile)
    tok_cfg = settings.get("tokenizer", {})
    model_path = Path(tok_cfg.get("rnt2_model_path", "data/runs/rnt2_dev.pt"))
    block_size = int(tok_cfg.get("block_size", 64))

    sample = text or "Hola ?? mundo — código: for i in range(3): print(i) {\"k\": 1}"
    if not model_path.exists():
        logger.info("RNT-2 model not found, creating default codebook: %s", model_path)
        codebook = RNT2Codebook.from_builtin(block_size=block_size)
        model = RNT2Model(codebook=codebook)
        model.save(model_path)
    else:
        model = RNT2Model.load(model_path)

    stream = encode_text(sample, model)
    decoded = decode_stream(stream, model)

    if decoded != sample:
        rprint("[red]Exactness failure[/red]")
        rprint({"decoded": decoded})
        raise typer.Exit(code=1)

    byte_len = len(sample.encode("utf-8"))
    code_len = len(stream.codes)
    escapes = len(stream.escapes)
    ratio = round(code_len / max(1, byte_len), 4)
    escapes_pct = round((escapes / max(1, len(stream.blocks))) * 100.0, 2)
    bpe_like = byte_len  # proxy: 1 token per byte
    bpe_ratio = round(bpe_like / max(1, byte_len), 4)

    rprint({
        "bytes": byte_len,
        "codes": code_len,
        "blocks": len(stream.blocks),
        "ratio_codes_per_byte": ratio,
        "escape_pct": escapes_pct,
        "bpe_proxy_tokens": bpe_like,
        "bpe_proxy_ratio": bpe_ratio,
    })


@app.command()
def demo_core_generate(prompt: Optional[str] = None, max_new_tokens: int = 32, profile: Optional[str] = None):
    """Generate text token-by-token using the core transformer."""
    settings = load_settings(profile)
    model = CoreTransformer.from_settings(settings)
    text = prompt or "def add(a, b):"
    out = model.generate(text, max_new_tokens=max_new_tokens)
    rprint(out)


@app.command()
def demo_agent(profile: Optional[str] = None):
    """Run the agent demo loop."""
    settings = load_settings(profile)
    report = run_demo_agent(settings)
    rprint(report)


if __name__ == "__main__":
    app()
