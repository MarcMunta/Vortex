from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..continuous.anchors import load_anchors, write_default_anchors
from ..continuous.types import Sample
from ..promotion.gating import bench_gate, log_promotion_decision, resolve_bench_thresholds


@dataclass(frozen=True)
class DomainExpertResult:
    ok: bool
    domain: str
    run_id: str
    run_dir: Path
    adapter_dir: Path | None
    manifest_path: Path | None
    error: str | None = None


def _hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return h.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            rel = str(path).replace("\\", "/")
        h.update(rel.encode("utf-8", errors="ignore"))
        try:
            h.update(path.read_bytes())
        except Exception:
            continue
    return h.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def _load_domain_samples(domain_dir: Path, *, domain: str) -> list[Sample]:
    samples: list[Sample] = []
    files = sorted(domain_dir.rglob("*.jsonl"))
    for path in files:
        for payload in _iter_jsonl(path):
            messages = payload.get("messages")
            prompt = str(payload.get("prompt", "") or payload.get("instruction", "") or payload.get("input", "")).strip()
            response = str(payload.get("response", "") or payload.get("output", "") or payload.get("completion", "")).strip()
            if isinstance(messages, list) and messages and not response:
                # Common pattern: last assistant message is the target.
                try:
                    last = messages[-1]
                    if str(last.get("role", "")).lower() == "assistant":
                        response = str(last.get("content", "")).strip()
                        messages = list(messages[:-1])
                except Exception:
                    pass
            if not response:
                continue
            samples.append(Sample(prompt=prompt, response=response, source_kind=str(domain), messages=messages if isinstance(messages, list) else None))
    return samples


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def train_hf_experts(
    settings: dict,
    domains: list[str],
    data_root: Path,
    output_root: Path,
    *,
    mock: bool = False,
    steps: int | None = None,
    lr: float | None = None,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    base_dir = Path(".").resolve()
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = base_dir / data_root
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = base_dir / output_root

    profile = str(settings.get("_profile") or "unknown")
    hf_cfg = settings.get("hf_train", {}) or {}
    max_seq_len_val = int(max_seq_len) if max_seq_len is not None else int(hf_cfg.get("max_seq_len", 1024) or 1024)
    steps_val = int(steps) if steps is not None else int(hf_cfg.get("max_steps", 50) or 50)
    lr_val = float(lr) if lr is not None else float(hf_cfg.get("lr", 2e-4) or 2e-4)

    ts_label = time.strftime("%Y%m%d_%H%M%S")
    results: dict[str, Any] = {"ok": True, "profile": profile, "output_root": str(output_root), "domains": {}}

    for raw_domain in domains:
        domain = str(raw_domain or "").strip()
        if not domain:
            continue
        run_id = f"{domain}_{ts_label}_{uuid.uuid4().hex[:8]}"
        run_dir = output_root / "quarantine" / domain / run_id
        adapter_dir = run_dir / "adapter"
        manifest_path = run_dir / "manifest.json"
        dataset_path = run_dir / "dataset.jsonl"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            adapter_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"mkdir_failed:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        domain_dir = data_root / domain
        dataset_hash = _hash_tree(domain_dir) if domain_dir.exists() else _hash_tree(data_root)
        samples = _load_domain_samples(domain_dir, domain=domain) if domain_dir.exists() else []

        if mock:
            (adapter_dir / "MOCK_ADAPTER.txt").write_text("mock adapter (no weights)\n", encoding="utf-8")
            _atomic_write_json(
                manifest_path,
                {
                    "version": 1,
                    "kind": "hf_expert",
                    "profile": profile,
                    "domain": domain,
                    "run_id": run_id,
                    "ts": time.time(),
                    "dataset_hash": dataset_hash,
                    "samples": len(samples),
                    "steps": int(steps_val),
                    "lr": float(lr_val),
                    "max_seq_len": int(max_seq_len_val),
                    "anchor_eval": None,
                    "regression": 0.0,
                    "passed_eval": True,
                    "bench": {"tokens_per_sec": 12.0, "vram_peak_mb": None, "ctx": int((settings.get("bench_thresholds", {}) or {}).get("required_ctx", 0) or 0) or None},
                    "bench_gate": {"ok": True, "reason": ""},
                    "ready": True,
                    "adapter_dir": str(adapter_dir),
                    "dataset_path": str(dataset_path),
                    "mock": True,
                },
            )
            results["domains"][domain] = {
                "ok": True,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "adapter_dir": str(adapter_dir),
                "manifest": str(manifest_path),
                "ready": True,
                "mock": True,
            }
            continue

        try:
            from . import hf_qlora as hfq  # local import (heavy deps)
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"hf_qlora_unavailable:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        try:
            import torch  # type: ignore
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"torch_missing:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        core = settings.get("core", {}) or {}
        model_name = hf_cfg.get("model_name") or core.get("hf_model")
        if not model_name:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": "model_name_missing", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        # Persist the dataset used for this domain/run for reproducibility.
        try:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(
                        json.dumps(
                            {
                                "prompt": sample.prompt,
                                "response": sample.response,
                                "messages": sample.messages,
                                "source_kind": sample.source_kind,
                                "ts": sample.ts,
                                "quality": sample.quality,
                                "source_ref": sample.source_ref,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
        except Exception:
            pass

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"hf_deps_missing:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        device = str(hf_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        dtype = str(hf_cfg.get("compute_dtype", "bf16"))
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16

        load_in_4bit = bool(hf_cfg.get("load_in_4bit", True))
        load_in_8bit = bool(hf_cfg.get("load_in_8bit", False))
        quant_config = None
        if load_in_4bit or load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=bool(hf_cfg.get("double_quant", True)),
                bnb_4bit_quant_type=str(hf_cfg.get("quant_type", "nf4")),
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(hf_cfg.get("use_fast", True)))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        attn_impl = hf_cfg.get("attn_implementation") or core.get("hf_attn_implementation")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        max_memory = hf_cfg.get("max_memory")
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto" if device.startswith("cuda") else "cpu"

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        anchors_path = Path((settings.get("continuous", {}) or {}).get("eval", {}).get("anchors_path", base_dir / "data" / "continuous" / "anchors.jsonl"))
        if not anchors_path.is_absolute():
            anchors_path = base_dir / anchors_path
        if not anchors_path.exists():
            write_default_anchors(anchors_path)
        anchors = load_anchors(anchors_path)
        anchor_samples = anchors[:8] if anchors else []
        anchor_eval: dict[str, Any] | None = None
        base_loss = None
        adapter_loss = None
        if anchor_samples:
            try:
                base_loss = float(hfq._eval_loss(base_model, tokenizer, anchor_samples, max_length=max_seq_len_val))  # type: ignore[attr-defined]
            except Exception:
                base_loss = None

        if load_in_4bit or load_in_8bit:
            base_model = prepare_model_for_kbit_training(base_model)

        lora_rank = int(hf_cfg.get("lora_rank", 8) or 8)
        lora_alpha = int(hf_cfg.get("lora_alpha", 16) or 16)
        lora_dropout = float(hf_cfg.get("lora_dropout", 0.05) or 0.05)
        target_modules = hf_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(target_modules) if isinstance(target_modules, list) else None,
        )
        model = get_peft_model(base_model, lora_cfg)
        if bool(hf_cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        default_system = hf_cfg.get("default_system") or core.get("hf_system_prompt") or "You are Vortex, a helpful coding assistant."
        texts = hfq.build_sft_texts(samples, tokenizer=tokenizer, default_system=default_system)  # type: ignore[attr-defined]
        weights: list[float] = []
        if bool(hf_cfg.get("use_weighted_sampling", False)):
            try:
                weights = hfq._compute_sample_weights(samples, settings)  # type: ignore[attr-defined]
            except Exception:
                weights = [1.0 for _ in texts]

        # Train.
        try:
            avg_loss, steps_done, _tps_train, _vram_peak_train = hfq._run_training_steps(  # type: ignore[attr-defined]
                model,
                tokenizer,
                texts,
                weights,
                cfg={**hf_cfg, "max_steps": int(steps_val), "lr": float(lr_val), "max_seq_len": int(max_seq_len_val)},
                device=device,
                torch_dtype=torch_dtype,
                micro_batch_size=int(hf_cfg.get("micro_batch_size", 1) or 1),
                grad_accum_steps=int(hf_cfg.get("grad_accum_steps", 8) or 8),
            )
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"train_failed:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        # Save adapter.
        try:
            model.save_pretrained(adapter_dir)
        except Exception as exc:
            results["ok"] = False
            results["domains"][domain] = {"ok": False, "error": f"adapter_save_failed:{exc}", "run_id": run_id, "run_dir": str(run_dir)}
            continue

        # Anchor eval after training.
        if anchor_samples:
            try:
                adapter_loss = float(hfq._eval_loss(model, tokenizer, anchor_samples, max_length=max_seq_len_val))  # type: ignore[attr-defined]
            except Exception:
                adapter_loss = None
        anchor_regression = None
        passed_eval = True
        if base_loss is not None and adapter_loss is not None and base_loss > 0:
            anchor_regression = float((adapter_loss - base_loss) / base_loss)
            max_regress = float((hf_cfg.get("eval", {}) or {}).get("max_regression", (settings.get("continuous", {}) or {}).get("eval", {}).get("max_regression", 0.2)))
            passed_eval = float(anchor_regression) <= float(max_regress)
        if base_loss is not None or adapter_loss is not None:
            anchor_eval = {"base_loss": base_loss, "adapter_loss": adapter_loss, "regression": anchor_regression, "passed": passed_eval}

        # Minimal bench gate (best-effort).
        bench_tps, bench_vram = None, None
        try:
            bench_tps, bench_vram = hfq._bench_short(model, tokenizer, max_new_tokens=64)  # type: ignore[attr-defined]
        except Exception:
            bench_tps, bench_vram = None, None
        ctx_val = None
        try:
            ctx_val = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0) or None
        except Exception:
            ctx_val = None
        thresholds = resolve_bench_thresholds(settings)
        verdict = bench_gate(
            {"tokens_per_sec": bench_tps, "vram_peak_mb": bench_vram, "ctx": ctx_val},
            baseline=None,
            thresholds=thresholds,
        )
        bench_ok = bool(verdict.get("ok", False)) if bench_tps is not None else None
        ready = bool(passed_eval and bench_ok is not False)

        if anchor_eval is None:
            anchor_eval = None

        manifest = {
            "version": 1,
            "kind": "hf_expert",
            "profile": profile,
            "domain": domain,
            "run_id": run_id,
            "ts": time.time(),
            "dataset_hash": dataset_hash,
            "samples": len(samples),
            "steps": int(steps_done),
            "lr": float(lr_val),
            "max_seq_len": int(max_seq_len_val),
            "loss": float(avg_loss) if avg_loss is not None else None,
            "anchor_eval": anchor_eval,
            "regression": anchor_regression,
            "passed_eval": bool(passed_eval),
            "bench": {"tokens_per_sec": bench_tps, "vram_peak_mb": bench_vram, "ctx": ctx_val},
            "bench_gate": verdict,
            "ready": bool(ready),
            "adapter_dir": str(adapter_dir),
            "dataset_path": str(dataset_path),
            "mock": False,
        }
        _atomic_write_json(manifest_path, manifest)
        try:
            log_promotion_decision(
                base_dir,
                {
                    "kind": "expert_gate",
                    "backend": "hf_experts",
                    "profile": profile,
                    "domain": domain,
                    "run_id": run_id,
                    "ready": bool(ready),
                    "passed_eval": bool(passed_eval),
                    "bench_ok": bench_ok,
                    "bench_gate": verdict,
                    "bench_tokens_per_sec": bench_tps,
                    "bench_vram_peak_mb": bench_vram,
                    "ctx": ctx_val,
                    "thresholds": thresholds,
                    "adapter_dir": str(adapter_dir),
                    "manifest": str(manifest_path),
                },
            )
        except Exception:
            pass

        results["domains"][domain] = {
            "ok": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "manifest": str(manifest_path),
            "ready": bool(ready),
            "passed_eval": bool(passed_eval),
            "bench_ok": bench_ok,
            "bench_tokens_per_sec": bench_tps,
            "bench_vram_peak_mb": bench_vram,
            "regression": anchor_regression,
            "ctx": ctx_val,
        }

    return results

