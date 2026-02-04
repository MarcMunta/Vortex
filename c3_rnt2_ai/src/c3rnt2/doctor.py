from __future__ import annotations

import gc
import hashlib
import importlib.util
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
    default_system = core_cfg.get("hf_system_prompt", "You are Vortex, a helpful coding assistant.")
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
    from .config import resolve_web_allowlist, resolve_web_strict

    tools_web = settings.get("tools", {}).get("web", {}) or {}
    cont = settings.get("continuous", {}) or {}

    strict = bool(resolve_web_strict(settings))
    web_enabled = bool(tools_web.get("enabled", False))
    ingest_web = bool(cont.get("ingest_web", False))
    allowlist = resolve_web_allowlist(settings)

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


def _deep_check_120b_like_profile(settings: dict, base_dir: Path, *, mock: bool) -> dict[str, Any]:
    profile = str(settings.get("_profile") or "")
    if profile != "rtx4080_16gb_120b_like":
        return {"ok": True, "skipped": "not_120b_like"}

    errors: list[str] = []
    info: dict[str, Any] = {}

    core_backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
    experts_enabled = bool((settings.get("experts", {}) or {}).get("enabled", False))
    adapters_enabled = bool((settings.get("adapters", {}) or {}).get("enabled", False))
    if core_backend in {"hf", "transformers"} and (experts_enabled or adapters_enabled):
        peft_ok = bool(importlib.util.find_spec("peft") is not None)
        info["peft"] = {
            "ok": peft_ok,
            "required": True,
            "enabled": {"experts": experts_enabled, "adapters": adapters_enabled},
        }
        if not peft_ok:
            info["peft"]["error"] = "peft_missing_for_hf_experts"
            info["peft"]["install"] = 'python -m pip install -e ".[hf,experts]"'
            errors.append("peft_missing_for_hf_experts")

    try:
        from .prepare import prepare_model_state

        prep = prepare_model_state(settings, base_dir=base_dir)
        info["prepare_model"] = {
            "ok": bool(prep.get("ok", False)),
            "backend_requested": prep.get("backend_requested"),
            "backend_resolved": prep.get("backend_resolved"),
            "quant_mode": prep.get("quant_mode"),
            "offload_enabled": prep.get("offload_enabled"),
            "gguf_path": prep.get("gguf_path"),
            "warnings": prep.get("warnings"),
            "errors": prep.get("errors"),
            "next_steps": prep.get("next_steps"),
        }
        if not bool(prep.get("ok", False)):
            errors.append("prepare_model_failed")
    except Exception as exc:
        info["prepare_model"] = {"ok": False, "error": str(exc)}
        errors.append("prepare_model_failed")

    thresholds = settings.get("bench_thresholds", {}) or {}
    required = ("min_tokens_per_sec", "max_regression", "max_vram_peak_mb", "required_ctx")
    missing = [key for key in required if thresholds.get(key) is None]
    if missing:
        errors.append(f"bench_thresholds_missing:{','.join(missing)}")
    else:
        try:
            info["bench_thresholds"] = {
                "min_tokens_per_sec": float(thresholds.get("min_tokens_per_sec")),
                "max_regression": float(thresholds.get("max_regression")),
                "max_vram_peak_mb": float(thresholds.get("max_vram_peak_mb")),
                "required_ctx": int(thresholds.get("required_ctx")),
            }
        except Exception:
            errors.append("bench_thresholds_invalid_types")

    if not mock:
        baseline_path = base_dir / "data" / "bench" / "baseline.json"
        baseline_ok = False
        baseline_backend = None
        try:
            baseline_backend = str((info.get("prepare_model") or {}).get("backend_resolved") or core_backend).strip().lower()
        except Exception:
            baseline_backend = str(core_backend or "").strip().lower()
        baseline_info: dict[str, Any] = {"path": str(baseline_path), "backend": baseline_backend}
        if not baseline_path.exists():
            baseline_info.update({"ok": False, "error": "baseline_missing", "hint": f'Run: python -m vortex bench --profile {profile} --update-baseline'})
        else:
            try:
                payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            entry = None
            if isinstance(payload, dict):
                prof = payload.get(profile)
                if isinstance(prof, dict):
                    entry = prof.get(baseline_backend)
            tps = entry.get("tokens_per_sec") if isinstance(entry, dict) else None
            baseline_ok = tps is not None
            baseline_info.update({"ok": bool(baseline_ok), "tokens_per_sec": tps})
            if not baseline_ok:
                baseline_info.update({"error": "baseline_missing_for_backend", "hint": f'Run: python -m vortex bench --profile {profile} --update-baseline'})
        info["bench_baseline"] = baseline_info
        if not baseline_ok:
            errors.append("bench_baseline_missing")

    def _check_router(section: str) -> dict[str, Any]:
        cfg = settings.get(section, {}) or {}
        enabled = bool(cfg.get("enabled", False))
        router_cfg = cfg.get("router", {}) or {}
        try:
            top_k = int(router_cfg.get("top_k", 1) or 1)
        except Exception:
            top_k = 0
        try:
            max_loaded = int(cfg.get("max_loaded", 0) or 0)
        except Exception:
            max_loaded = 0
        mode = str(router_cfg.get("mode", "") or "").strip().lower()
        mix_mode = str(router_cfg.get("mix_mode", "single") or "single").strip().lower()
        section_errors: list[str] = []
        if not enabled:
            section_errors.append("disabled")
        if top_k < 1:
            section_errors.append("top_k_lt_1")
        if enabled and max_loaded < top_k:
            section_errors.append("max_loaded_lt_top_k")
        if top_k > 1 and mix_mode != "weighted":
            section_errors.append("mix_mode_not_weighted_for_topk")
        if mode and mode != "hybrid":
            section_errors.append("router_mode_not_hybrid")
        return {
            "ok": not section_errors,
            "enabled": enabled,
            "top_k": top_k,
            "max_loaded": max_loaded,
            "mode": mode or None,
            "mix_mode": mix_mode,
            "errors": section_errors or None,
        }

    adapters_chk = _check_router("adapters")
    experts_chk = _check_router("experts")
    info["adapters"] = adapters_chk
    info["experts"] = experts_chk
    if not bool(adapters_chk.get("ok", False)):
        errors.append("adapters_router_invalid")
    if not bool(experts_chk.get("ok", False)):
        errors.append("experts_router_invalid")

    if not mock:
        try:
            from .model_loader import load_inference_model

            offline_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
            old_env = {k: os.environ.get(k) for k in offline_env}
            for k, v in offline_env.items():
                os.environ[k] = v
            model = None
            try:
                model = load_inference_model(settings)
                quant_info: dict[str, Any] = {"ok": True}
                if bool(getattr(model, "is_llama_cpp", False)):
                    quant_info.update({"backend": "llama_cpp", "quant_active": True})
                elif bool(getattr(model, "is_hf", False)):
                    core = settings.get("core", {}) or {}
                    quant_requested = bool(core.get("hf_load_in_4bit") or core.get("hf_load_in_8bit"))
                    quant_fallback = bool(getattr(model, "quant_fallback", False))
                    quant_active = bool(quant_requested and not quant_fallback)
                    quant_info.update(
                        {
                            "backend": "hf",
                            "quant_requested": bool(quant_requested),
                            "quant_fallback": bool(quant_fallback),
                            "quant_active": bool(quant_active),
                        }
                    )
                    if not quant_active:
                        quant_info["ok"] = False
                        quant_info["error"] = "120b_like_requires_quant_backend"
                else:
                    quant_info.update({"ok": False, "error": "120b_like_requires_quant_backend", "backend": "unknown"})
                info["quant_active"] = quant_info
                if not bool(quant_info.get("ok", False)):
                    errors.append(str(quant_info.get("error") or "120b_like_requires_quant_backend"))
            except Exception as exc:
                info["quant_active"] = {"ok": False, "error": "120b_like_requires_quant_backend", "detail": str(exc)}
                errors.append("120b_like_requires_quant_backend")
            finally:
                for k, prev in old_env.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
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
        except Exception as exc:
            info["quant_active"] = {"ok": False, "error": "120b_like_requires_quant_backend", "detail": str(exc)}
            errors.append("120b_like_requires_quant_backend")

    if mock:
        try:
            from .bench import BenchArgs, run_bench
            from .promotion.gating import DEFAULT_BENCH_PROMPT

            required_ctx = (thresholds.get("required_ctx") if isinstance(thresholds, dict) else None) or (settings.get("bench", {}) or {}).get("required_ctx")
            try:
                ctx = int(required_ctx) if required_ctx is not None else None
            except Exception:
                ctx = None
            out_path = base_dir / "data" / "bench" / "doctor_120b_like_mock.json"
            bench_report = run_bench(
                settings,
                base_dir=base_dir,
                args=BenchArgs(
                    profile=profile,
                    prompt=DEFAULT_BENCH_PROMPT,
                    prompt_file=None,
                    ctx=ctx,
                    max_new=16,
                    warmup=0,
                    repeat=1,
                    seed=0,
                    json_out=out_path,
                    jsonl_out=None,
                    mock=True,
                ),
            )
            info["bench_mock"] = {
                "ok": bool(bench_report.get("ok", False)),
                "backend": bench_report.get("backend"),
                "tokens_per_sec": bench_report.get("tokens_per_sec"),
                "json_out": str(out_path),
            }
            if not bool(bench_report.get("ok", False)):
                errors.append("bench_mock_failed")
        except Exception as exc:
            info["bench_mock"] = {"ok": False, "error": str(exc)}
            errors.append("bench_mock_failed")

    return {"ok": not errors, "errors": errors or None, "info": info}


def _wsl_available_deep_check(settings: dict, *, timeout_s: float = 1.5) -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        return {"ok": True, "skipped": "not_windows"}

    profile = str(settings.get("_profile") or "")
    if profile not in {"rtx4080_16gb_120b_like", "rtx4080_16gb_safe_windows_hf"}:
        return {"ok": True, "skipped": "not_target_profile"}

    hf_train_enabled = bool((settings.get("hf_train", {}) or {}).get("enabled", False))
    train_strategy = str((settings.get("server", {}) or {}).get("train_strategy", "") or "").strip().lower()
    if not hf_train_enabled and train_strategy != "wsl_subprocess_unload":
        return {"ok": True, "skipped": "hf_train_disabled"}

    try:
        from .utils.wsl import is_wsl_available
    except Exception as exc:
        return {"ok": False, "error": f"wsl_check_unavailable:{exc}"}

    status = is_wsl_available(timeout_s=float(timeout_s))
    if not bool(status.ok):
        return {"ok": False, "error": status.error or "wsl_unavailable"}
    return {"ok": True}


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
        checks["profile_120b_like"] = _deep_check_120b_like_profile(settings, base_dir, mock=mock)
        if not bool(checks["profile_120b_like"].get("ok", False)):
            report["deep_ok"] = False
    except Exception as exc:
        checks["profile_120b_like"] = {"ok": False, "error": str(exc)}
        report["deep_ok"] = False

    try:
        if mock:
            checks["wsl_available"] = {"ok": True, "skipped": "mock"}
        else:
            checks["wsl_available"] = _wsl_available_deep_check(settings)
            if not bool(checks["wsl_available"].get("ok", False)):
                report["deep_ok"] = False
    except Exception as exc:
        checks["wsl_available"] = {"ok": False, "error": str(exc)}
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
