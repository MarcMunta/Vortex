from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import inspect
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .config import DEFAULT_SETTINGS_PATH, load_settings, validate_profile
from .device import detect_device
from .utils.locks import FileLock, LockUnavailable


def check_deps(modules: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in modules:
        try:
            __import__(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"missing ({exc.__class__.__name__})"
    return status


def _profile_checks(base_dir: Path) -> dict[str, str]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8")) or {}
        profiles = data.get("profiles", {}) or {}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    results: dict[str, str] = {}
    for name in profiles.keys():
        try:
            settings = load_settings(name)
            validate_profile(settings, base_dir=base_dir)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"error: {exc}"
    return results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _tokenizer_roundtrip_strict(settings: dict, base_dir: Path) -> dict[str, Any]:
    from .tokenizer.vortex_tok import decode_from_ids, encode, encode_to_ids, load_or_create, metrics  # type: ignore

    tok_cfg = settings.get("tokenizer", {}) or {}
    model_path = Path(tok_cfg.get("vortex_tok_path", tok_cfg.get("vortex_model_path", base_dir / "data" / "runs" / "vortex_tok.pt")))
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    block_size = int(tok_cfg.get("block_size", 64))
    tok_model = load_or_create(model_path, block_size=block_size)

    cases: list[str] = [
        "",
        "hello world",
        "ASCII: ~!@#$%^&*()_+-=[]{}|;:',.<>/?",
        "utf8: ñ é ö 😀 — 東京",
        "\tTabs\nNewlines\r\nWindows newlines\n\n",
        "code:\n```py\ndef f(x):\n    return x * (x + 1) // 2\n```\n",
        'json: {"ok": true, "text": "ñ 😀 \\\\ \\" \\n", "n": 123}',
        "\x00\x01\t\ncontrol-chars",
        "a" * 2048,
        ("abc" * 600) + " END",
    ]
    rng = random.Random(0)
    alphabet = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t-_=+()[]{}<>/\\'\".,:;!?"
        "ñéö😀—東京"
    )
    while len(cases) < 50:
        n = rng.randint(0, 200)
        cases.append("".join(rng.choice(alphabet) for _ in range(n)))

    failures: list[dict[str, Any]] = []
    start = time.perf_counter()
    for text in cases:
        ids, total_len = encode_to_ids(text, tok_model)
        out = decode_from_ids(ids, tok_model, total_len=total_len)
        if out != text:
            failures.append({"case": text[:120], "error": "mismatch"})
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    stream = encode("\n\n".join(cases), tok_model)
    tok_metrics = metrics(stream)
    tok_metrics["tokens_per_byte"] = round(float(tok_metrics.get("tokens", 0)) / max(1, float(tok_metrics.get("bytes", 1))), 6)

    return {
        "ok": not failures,
        "elapsed_ms": round(elapsed_ms, 3),
        "cases": len(cases),
        "failures": failures[:3] if failures else None,
        **tok_metrics,
        "fallback_pct": tok_metrics.get("escapes_pct"),
    }


def _inference_smoke(settings: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    if torch is None:
        return {"ok": False, "error": "torch not available"}, {"ok": True, "skipped": "torch_missing"}

    from .model.core_transformer import CoreTransformer
    from .nn.paged_linear import PagedLinear

    core = CoreTransformer.from_settings(deepcopy(settings))
    prompt = "def add(a, b):\n    return a + b\n"

    # Paging sanity (if enabled).
    runtime_cfg = getattr(core, "runtime_cfg", {}) or {}
    paging_enabled = bool(runtime_cfg.get("paged_lm_head", False))
    lm_head = getattr(core, "lm_head", None)
    lm_is_paged = isinstance(lm_head, PagedLinear) if lm_head is not None else False
    stats_before = lm_head.stats() if lm_head is not None and hasattr(lm_head, "stats") else None
    try:
        hidden = torch.randn(
            1,
            1,
            int(core.config.hidden_size),
            device=core.device,
            dtype=core.dtype if core.device.type == "cuda" else torch.float32,
        )
        _ = lm_head(hidden) if lm_head is not None else None
    except Exception:
        pass
    stats_after = lm_head.stats() if lm_head is not None and hasattr(lm_head, "stats") else None
    paging_ok = True
    paging_reason = None
    if paging_enabled:
        if not lm_is_paged:
            paging_ok = False
            paging_reason = "paged_lm_head_enabled_but_lm_head_not_paged"
        elif isinstance(stats_before, dict) and isinstance(stats_after, dict):
            before_faults = float(stats_before.get("page_faults", 0.0) or 0.0)
            after_faults = float(stats_after.get("page_faults", 0.0) or 0.0)
            if after_faults < before_faults:
                paging_ok = False
                paging_reason = "paged_stats_not_monotonic"

    paging = {
        "ok": bool(paging_ok),
        "paging_enabled": bool(paging_enabled),
        "lm_head_is_paged": bool(lm_is_paged) if paging_enabled else None,
        "reason": paging_reason,
        "stats_before": stats_before if isinstance(stats_before, dict) else None,
        "stats_after": stats_after if isinstance(stats_after, dict) else None,
    }

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    start = time.perf_counter()
    _text = core.generate(prompt, max_new_tokens=32)
    elapsed = max(1e-6, time.perf_counter() - start)

    vram_alloc = None
    vram_res = None
    if torch.cuda.is_available():
        try:
            vram_alloc = float(torch.cuda.max_memory_allocated() / 1e6)
        except Exception:
            vram_alloc = None
        try:
            vram_res = float(torch.cuda.max_memory_reserved() / 1e6)
        except Exception:
            vram_res = None

    smoke = {
        "ok": True,
        "tokens": 32,
        "tokens_per_sec": round(32.0 / elapsed, 6),
        "latency_ms_total": round(elapsed * 1000.0, 3),
        "vram_peak_mb_allocated": round(vram_alloc, 3) if vram_alloc is not None else None,
        "vram_peak_mb_reserved": round(vram_res, 3) if vram_res is not None else None,
    }

    try:
        del core
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return smoke, paging


def _self_train_mock(settings: dict, base_dir: Path) -> dict[str, Any]:
    if torch is None:
        return {"ok": False, "error": "torch not available"}

    from torch.nn.utils.rnn import pad_sequence

    from .continuous.anchors import DEFAULT_ANCHORS, load_anchors
    from .continuous.formatting import format_chat_sample
    from .continuous.lora import LoRAConfig, LoRALinear, inject_lora, resolve_target_modules, save_lora_state
    from .continuous.types import Sample
    from .device import autocast_context
    from .model.core_transformer import CoreTransformer
    from .training.lora_utils import eval_loss

    cont = settings.get("continuous", {}) or {}
    adapter_cfg = cont.get("adapters", {}) or {}
    anchors_path = Path((cont.get("eval", {}) or {}).get("anchors_path", "data/continuous/anchors.jsonl"))
    if not anchors_path.is_absolute():
        anchors_path = base_dir / anchors_path
    anchors = load_anchors(anchors_path)
    if not anchors:
        anchors = [Sample(prompt=item["prompt"], response=item["response"]) for item in DEFAULT_ANCHORS]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / "data" / "continuous" / "quarantine" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / "adapter.pt"

    steps = int(cont.get("mock_steps", 2) or 2)
    lr = float(cont.get("lr", 1e-4) or 1e-4)
    batch_tokens = int(cont.get("batch_tokens", 2048) or 2048)
    batch_tokens = max(128, min(batch_tokens, 2048))
    grad_accum = int(cont.get("grad_accum", 1) or 1)
    grad_accum = max(1, min(grad_accum, 8))

    model = CoreTransformer.from_settings(deepcopy(settings))
    core_cfg = settings.get("core", {}) or {}
    backend = str(core_cfg.get("backend", "vortex"))
    default_system = core_cfg.get("hf_system_prompt", "You are a helpful coding assistant.")
    tokenizer = getattr(model, "tokenizer", None)

    base_loss = eval_loss(model, anchors, backend=backend, tokenizer=tokenizer, default_system=default_system)

    strict = bool(adapter_cfg.get("strict_target_modules", False))
    targets = resolve_target_modules(adapter_cfg, strict=strict)
    lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
    inject_lora(model, lora_cfg, target_modules=targets)
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
    for block in getattr(model, "blocks", []):
        try:
            block.lava.enable_write = False
        except Exception:
            pass

    trainable = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    use_scaler = model.device.type == "cuda" and model.dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    samples = anchors[:]
    step_idx = 0
    tokens_seen = 0
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while step_idx < steps:
        sequences = []
        token_count = 0
        attempts = 0
        while token_count < batch_tokens and attempts < max(4, len(samples)):
            sample = samples[attempts % len(samples)]
            attempts += 1
            text = format_chat_sample(sample, backend=backend, tokenizer=tokenizer, default_system=default_system)
            ids, _ = model.encode_prompt(text)
            if len(ids) < 2:
                continue
            seq = torch.tensor(ids, dtype=torch.long)
            sequences.append(seq)
            token_count += len(ids)
        if not sequences:
            break
        inputs = [seq[:-1] for seq in sequences]
        targets_ids = [seq[1:] for seq in sequences]
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=0).to(model.device)
        target_ids = pad_sequence(targets_ids, batch_first=True, padding_value=-100).to(model.device)
        tokens_seen += int(target_ids.numel())
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            logits = model.forward(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
            loss = loss / max(1, grad_accum)
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step_idx += 1
        if step_idx % grad_accum == 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if step_idx % grad_accum != 0:
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    save_lora_state(model, adapter_path)
    elapsed = max(1e-6, time.perf_counter() - start)
    tokens_per_sec = float(tokens_seen) / elapsed if tokens_seen else None

    new_loss = eval_loss(model, anchors, backend=backend, tokenizer=tokenizer, default_system=default_system)
    regression = None
    passed_eval = False
    max_regression = float((cont.get("eval", {}) or {}).get("max_regression", 0.2))
    if base_loss is not None and new_loss is not None:
        regression = (float(new_loss) - float(base_loss)) / max(1e-6, float(base_loss))
        passed_eval = float(regression) <= float(max_regression)

    dataset_hash = hashlib.sha256("\n".join(f"{s.prompt}\n{s.response}" for s in anchors).encode("utf-8", errors="ignore")).hexdigest()
    manifest = {
        "run_id": run_id,
        "adapter_path": str(adapter_path),
        "steps": int(step_idx),
        "lr": float(lr),
        "batch_tokens": int(batch_tokens),
        "grad_accum": int(grad_accum),
        "tokens_seen": int(tokens_seen),
        "tokens_per_sec": float(tokens_per_sec) if tokens_per_sec is not None else None,
        "anchor_base_loss": base_loss,
        "anchor_new_loss": new_loss,
        "anchor_regression": regression,
        "passed_eval": bool(passed_eval),
        "max_regression": float(max_regression),
        "dataset_hash": dataset_hash,
        "manual_approval": False,
        "ts": time.time(),
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    promo_req = {
        "kind": "PROMOTION_REQUEST",
        "run_id": run_id,
        "adapter_path": str(adapter_path),
        "manifest_path": str(manifest_path),
        "passed_eval": bool(passed_eval),
        "manual_approval_required": True,
        "ts": time.time(),
    }
    promo_path = out_dir / "PROMOTION_REQUEST.json"
    _write_json(promo_path, promo_req)

    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "ok": True,
        "run_id": run_id,
        "quarantine_dir": str(out_dir),
        "adapter_path": str(adapter_path),
        "manifest_path": str(manifest_path),
        "promotion_request_path": str(promo_path),
        "passed_eval": bool(passed_eval),
    }


def _llama_cpp_backend_check(settings: dict, base_dir: Path) -> dict[str, Any]:
    core = settings.get("core", {}) or {}
    model_path = core.get("llama_cpp_model_path")
    if not model_path:
        return {"ok": False, "error": "core.llama_cpp_model_path missing"}
    path = Path(str(model_path))
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        return {"ok": False, "error": "gguf_missing", "path": str(path)}
    try:
        from .llama_cpp_backend import load_llama_cpp_model

        model = load_llama_cpp_model(settings, base_dir=base_dir)
        text = model.generate(prompt="Hello", max_new_tokens=8, temperature=0.0, top_p=1.0)
        return {"ok": True, "path": str(path), "sample_len": int(len(text))}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": str(path)}


def _promotion_gating_wiring_check(base_dir: Path) -> dict[str, Any]:
    try:
        from .promotion import gating as gating_mod
    except Exception as exc:
        return {"ok": False, "error": f"promotion_gating_import_failed:{exc}"}
    try:
        from .continuous.promotion import promote_quarantine_run

        sig = inspect.signature(promote_quarantine_run)
        has_settings = "settings" in sig.parameters
    except Exception as exc:
        return {"ok": False, "error": f"promotion_import_failed:{exc}"}
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "promotions.jsonl"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"kind": "doctor_promotion_wiring", "ts": time.time(), "gating": bool(getattr(gating_mod, "bench_gate", None))},
                    ensure_ascii=True,
                )
                + "\n"
            )
    except Exception as exc:
        return {"ok": False, "error": f"promotions_log_unwritable:{exc}", "log_path": str(path)}
    if not has_settings:
        return {"ok": False, "error": "promote_quarantine_run_missing_settings_param", "log_path": str(path)}
    return {"ok": True, "log_path": str(path)}


def _security_deep_check(settings: dict, base_dir: Path) -> dict[str, Any]:
    security = settings.get("security", {}) or {}
    web_sec = (security.get("web", {}) or {}) if isinstance(security, dict) else {}
    tools_web = settings.get("tools", {}).get("web", {}) or {}
    cont = settings.get("continuous", {}) or {}

    strict = bool(web_sec.get("strict", True))
    web_enabled = bool(tools_web.get("enabled", False))
    ingest_web = bool(cont.get("ingest_web", False))
    allowlist = tools_web.get("allow_domains")
    if not allowlist:
        allowlist = (settings.get("agent", {}) or {}).get("web_allowlist", [])
    allowlist = list(allowlist) if allowlist else []

    errors: list[str] = []
    if (web_enabled or ingest_web) and not strict:
        errors.append("web_strict_disabled")
    if (web_enabled or ingest_web) and not allowlist:
        errors.append("web_allowlist_empty")

    autopilot_cfg = settings.get("autopilot", {}) or {}
    autopatch_enabled = bool(autopilot_cfg.get("autopatch_enabled", False))
    autopatch_require_approval = bool(autopilot_cfg.get("autopatch_require_approval", False))
    approval_file = autopilot_cfg.get("approval_file")
    self_patch_cfg = settings.get("self_patch", {}) or {}
    self_patch_enabled = bool(self_patch_cfg.get("enabled", False))
    auto_sandbox = bool(self_patch_cfg.get("auto_sandbox", True))

    if autopatch_enabled:
        if not autopatch_require_approval:
            errors.append("autopatch_require_approval_false")
        if not approval_file:
            errors.append("autopatch_approval_file_missing")
        if not self_patch_enabled:
            errors.append("self_patch_disabled")
        if not auto_sandbox:
            errors.append("self_patch_auto_sandbox_disabled")

    return {
        "ok": not errors,
        "errors": errors or None,
        "web": {
            "enabled": web_enabled,
            "ingest_web": ingest_web,
            "strict": strict,
            "allowlist_len": len(allowlist),
        },
        "autopatch": {
            "enabled": autopatch_enabled,
            "require_approval": autopatch_require_approval,
            "approval_file": str(approval_file) if approval_file else None,
            "self_patch_enabled": self_patch_enabled,
            "auto_sandbox": auto_sandbox,
        },
    }


def _bench_minimal_check(settings: dict, base_dir: Path) -> dict[str, Any]:
    from .bench import BenchArgs, run_bench
    from .promotion.gating import DEFAULT_BENCH_PROMPT

    bench_cfg = settings.get("bench", {}) or {}
    required_ctx = bench_cfg.get("required_ctx")
    try:
        ctx = int(required_ctx) if required_ctx is not None else None
    except Exception:
        ctx = None
    profile = str(settings.get("_profile") or "unknown")
    out_path = base_dir / "data" / "bench" / "doctor_minimal.json"
    args = BenchArgs(
        profile=profile,
        prompt=DEFAULT_BENCH_PROMPT,
        prompt_file=None,
        ctx=ctx,
        max_new=64,
        warmup=1,
        repeat=1,
        seed=0,
        json_out=out_path,
        jsonl_out=None,
    )

    backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
    offline_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    old_env = {k: os.environ.get(k) for k in offline_env}
    if backend == "hf":
        for k, v in offline_env.items():
            os.environ[k] = v
    try:
        report = run_bench(settings, base_dir=base_dir, args=args)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "backend": backend, "json_out": str(out_path)}
    finally:
        if backend == "hf":
            for k, prev in old_env.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev
    return {"ok": bool(report.get("ok", False)), "backend": report.get("backend"), "tokens_per_sec": report.get("tokens_per_sec"), "json_out": str(out_path)}


def run_deep_checks(settings: dict, base_dir: Path, *, mock: bool = False) -> dict[str, Any]:
    info = detect_device()
    report: dict[str, Any] = {"deep_ok": True, "ts": time.time()}
    checks: dict[str, Any] = {
        "windows_cuda": {"ok": (not sys.platform.startswith("win")) or bool(info.cuda_available)},
        "profiles": _profile_checks(base_dir),
    }

    if sys.platform.startswith("win") and not bool(info.cuda_available):
        checks["windows_cuda"]["error"] = "CUDA not available on Windows (install CUDA-enabled torch + drivers)."
        report["deep_ok"] = False

    backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
    if backend == "llama_cpp":
        if mock:
            checks["llama_cpp_backend"] = {"ok": True, "skipped": "mock"}
        else:
            checks["llama_cpp_backend"] = _llama_cpp_backend_check(settings, base_dir)
            if not bool(checks["llama_cpp_backend"].get("ok", False)):
                report["deep_ok"] = False

    try:
        checks["tokenizer_roundtrip"] = _tokenizer_roundtrip_strict(settings, base_dir)
        if not bool(checks["tokenizer_roundtrip"].get("ok", False)):
            report["deep_ok"] = False
    except Exception as exc:
        checks["tokenizer_roundtrip"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    try:
        smoke, paging = _inference_smoke(settings)
        checks["inference_smoke"] = smoke
        checks["paging_sanity"] = paging
        if not bool(smoke.get("ok", False)):
            report["deep_ok"] = False
        if isinstance(paging, dict) and paging.get("ok") is False:
            report["deep_ok"] = False
    except Exception as exc:
        checks["inference_smoke"] = {"ok": False, "error": str(exc)}
        checks["paging_sanity"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    # Minimal bench: run only in non-mock mode, and only when bench gating is enabled/required.
    try:
        learning_cfg = settings.get("learning", {}) or {}
        autopilot_cfg = settings.get("autopilot", {}) or {}
        bench_required = bool(learning_cfg.get("require_bench_ok", False))
        bench_enabled = bool(autopilot_cfg.get("bench_enabled", False) or bench_required)
        if not bench_enabled:
            checks["bench_minimal"] = {"ok": True, "skipped": "disabled"}
        elif mock:
            checks["bench_minimal"] = {"ok": True, "skipped": "mock"}
        else:
            checks["bench_minimal"] = _bench_minimal_check(settings, base_dir)
            if not bool(checks["bench_minimal"].get("ok", False)):
                report["deep_ok"] = False
    except Exception as exc:
        checks["bench_minimal"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    # In mock mode we still run the full self-train path (no network); it is already tiny (1–2 steps).
    try:
        checks["self_train_mock"] = _self_train_mock(settings, base_dir)
        if not bool(checks["self_train_mock"].get("ok", False)):
            report["deep_ok"] = False
    except Exception as exc:
        checks["self_train_mock"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    try:
        checks["promotion_gating"] = _promotion_gating_wiring_check(base_dir)
        if not bool(checks["promotion_gating"].get("ok", False)):
            report["deep_ok"] = False
    except Exception as exc:
        checks["promotion_gating"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    try:
        checks["security"] = _security_deep_check(settings, base_dir)
        if not bool(checks["security"].get("ok", False)):
            report["deep_ok"] = False
    except Exception as exc:
        checks["security"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    lock_dir = base_dir / "data" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_status = {}
    for role in ("serve", "train", "self_patch", "gpu"):
        lock_path = lock_dir / f"{role}.lock"
        lock = FileLock(lock_path)
        try:
            lock.acquire(blocking=False)
            lock.release()
            lock_status[role] = "available"
        except LockUnavailable:
            lock_status[role] = "locked"
        except Exception as exc:
            lock_status[role] = f"error: {exc}"
    checks["locks"] = lock_status

    report["checks"] = checks
    out_path = base_dir / "data" / "doctor" / "last.json"
    _write_json(out_path, report)
    report["report_path"] = str(out_path)
    return report


def doctor_report(settings: dict, base_dir: Path, deep: bool = False, *, mock: bool = False) -> dict[str, Any]:
    info = detect_device()
    report: Dict[str, Any] = {
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    }
    validate_profile(settings, base_dir=base_dir)
    if deep:
        report.update(run_deep_checks(settings, base_dir=base_dir, mock=mock))
    return report
