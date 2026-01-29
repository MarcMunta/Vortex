from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from ..model.core_transformer import CoreTransformer
from ..device import autocast_context
from .dataset import collect_samples
from .types import Sample
from .formatting import format_chat_sample
from .lora import LoRAConfig, inject_lora, load_lora_state, save_lora_state, resolve_target_modules
from .registry import begin_run, finalize_run, load_registry, rollback, save_registry
from .anchors import load_anchors
from .consolidate import select_top_runs, adapter_soup, write_consolidated_adapter, write_consolidated_meta


@dataclass
class TrainResult:
    run_id: str
    promoted: bool
    loss: float
    samples: int


class ContinualTrainer:
    def __init__(self, settings: dict, base_dir: Path):
        self.settings = settings
        self.base_dir = base_dir
        self.state = load_registry(base_dir)

    def _build_model(self) -> CoreTransformer:
        return CoreTransformer.from_settings(self.settings)

    def run_tick(self) -> TrainResult:
        run_id, _run_path = begin_run(self.base_dir)
        try:
            allowlist = self.settings.get("agent", {}).get("web_allowlist", ["docs.python.org"])
            collected = collect_samples(self.base_dir, allowlist, self.settings)
            stats = collected.stats
            trigger_cfg = self.settings.get("continuous", {}).get("trigger", {})
            if bool(trigger_cfg.get("enabled", True)):
                min_new_docs = int(trigger_cfg.get("min_new_docs", 1))
                min_novelty = float(trigger_cfg.get("min_novelty", 0.2))
                min_successes = int(trigger_cfg.get("min_successes", 0))
                if stats.new_docs < min_new_docs and stats.novelty_avg < min_novelty and stats.successes <= min_successes:
                    finalize_run(
                        self.base_dir,
                        run_id,
                        promote=False,
                        meta={
                            "loss": None,
                            "samples": len(collected.samples),
                            "reason": "signal_low",
                            "stats": stats.__dict__,
                            "ts": time.time(),
                        },
                    )
                    return TrainResult(run_id=run_id, promoted=False, loss=0.0, samples=len(collected.samples))

            samples = collected.samples
            if len(samples) < 2:
                finalize_run(
                    self.base_dir,
                    run_id,
                    promote=False,
                    meta={"loss": None, "samples": len(samples), "reason": "insufficient_samples", "stats": stats.__dict__, "ts": time.time()},
                )
                return TrainResult(run_id=run_id, promoted=False, loss=0.0, samples=len(samples))

            # split train/holdout
            random.shuffle(samples)
            split_idx = max(1, int(len(samples) * 0.9))
            train_samples = samples[:split_idx]
            holdout = samples[split_idx:]

            model = self._build_model()
            adapter_cfg = self.settings.get("continuous", {}).get("adapters", {})
            lora_cfg = LoRAConfig(
                rank=int(adapter_cfg.get("rank", self.settings.get("continuous", {}).get("adapter_rank", 4))),
                alpha=float(adapter_cfg.get("alpha", 1.0)),
            )
            strict = bool(adapter_cfg.get("strict_target_modules", False))
            target_modules = resolve_target_modules(adapter_cfg, strict=strict)
            inject_lora(model, lora_cfg, target_modules=target_modules)

            # Load current adapter if exists
            current = load_registry(self.base_dir).current_run_id
            if current:
                load_lora_state(model, self._adapter_path(current))

            anchors_path = Path(self.settings.get("continuous", {}).get("eval", {}).get("anchors_path", "data/continuous/anchors.jsonl"))
            anchors = load_anchors(anchors_path)
            gold_samples = collected.gold_samples

            base_loss = self._eval_loss(model, holdout)
            anchor_base_loss = self._eval_loss(model, anchors) if anchors else None
            gold_base_loss = self._eval_loss(model, gold_samples) if gold_samples else None

            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(self.settings.get("continuous", {}).get("lr", 1e-4)))
            max_steps = int(self.settings.get("continuous", {}).get("max_steps_per_tick", self.settings.get("continuous", {}).get("max_steps", 50)))
            batch_tokens = int(self.settings.get("continuous", {}).get("batch_tokens", 2048))

            model.train()
            use_scaler = model.device.type == "cuda" and model.dtype == torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
            loss_val = 0.0
            for _ in range(max_steps):
                sequences = []
                token_count = 0
                attempts = 0
                while token_count < batch_tokens and attempts < max(4, len(train_samples)):
                    sample = random.choice(train_samples)
                    text = format_chat_sample(sample)
                    ids, _ = model.encode_prompt(text)
                    attempts += 1
                    if len(ids) < 2:
                        continue
                    seq = torch.tensor(ids, dtype=torch.long)
                    sequences.append(seq)
                    token_count += len(ids)
                if not sequences:
                    continue
                inputs = [seq[:-1] for seq in sequences]
                targets = [seq[1:] for seq in sequences]
                input_ids = pad_sequence(inputs, batch_first=True, padding_value=0).to(model.device)
                target_ids = pad_sequence(targets, batch_first=True, padding_value=-100).to(model.device)
                for block in model.blocks:
                    block.lava.reset_stats()
                optimizer.zero_grad(set_to_none=True)
                mtp_weight = float(self.settings.get("continuous", {}).get("mtp_loss_weight", 0.1))
                with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
                    logits, _mtp_logits, aux_loss = model.forward_with_aux(
                        input_ids,
                        labels=target_ids,
                        return_aux=True,
                    )
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
                    if aux_loss is not None:
                        loss = loss + mtp_weight * aux_loss
                    mem_cost = sum(block.lava.stats.reads + block.lava.stats.writes for block in model.blocks)
                    loss = loss + 1e-6 * mem_cost
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                loss_val = float(loss.item())

            new_loss = self._eval_loss(model, holdout)
            anchor_new_loss = self._eval_loss(model, anchors) if anchors else None
            gold_new_loss = self._eval_loss(model, gold_samples) if gold_samples else None

            min_improve = float(self.settings.get("continuous", {}).get("eval", {}).get("min_improvement", 0.0))
            max_regression = float(self.settings.get("continuous", {}).get("eval", {}).get("max_regression", 0.2))

            improved = False
            if base_loss is not None and new_loss is not None:
                improved = (base_loss - new_loss) / max(1e-6, base_loss) >= min_improve

            anchor_regression = 0.0
            anchor_improved = False
            if anchor_base_loss is not None and anchor_new_loss is not None:
                anchor_regression = (anchor_new_loss - anchor_base_loss) / max(1e-6, anchor_base_loss)
                anchor_improved = (anchor_base_loss - anchor_new_loss) / max(1e-6, anchor_base_loss) >= min_improve

            gold_regression = 0.0
            if gold_base_loss is not None and gold_new_loss is not None:
                gold_regression = (gold_new_loss - gold_base_loss) / max(1e-6, gold_base_loss)

            anchor_ok = anchor_regression <= max_regression
            gold_ok = gold_regression <= max_regression
            promoted = (improved or anchor_improved) and anchor_ok and gold_ok

            # Save adapter for this run
            save_lora_state(model, self._adapter_path(run_id))
            finalize_run(
                self.base_dir,
                run_id,
                promote=promoted,
                meta={
                    "loss": new_loss,
                    "base_loss": base_loss,
                    "anchor_loss": anchor_new_loss,
                    "anchor_base_loss": anchor_base_loss,
                    "anchor_regression": anchor_regression,
                    "gold_loss": gold_new_loss,
                    "gold_base_loss": gold_base_loss,
                    "gold_regression": gold_regression,
                    "samples": len(samples),
                    "stats": stats.__dict__,
                    "ts": time.time(),
                },
            )

            self._maybe_consolidate(anchors, max_regression)

            loss_out = new_loss if new_loss is not None else loss_val
            return TrainResult(run_id=run_id, promoted=promoted, loss=float(loss_out), samples=len(samples))
        except Exception:
            rollback(self.base_dir)
            raise

    def _adapter_path(self, run_id: str) -> Path:
        return self.base_dir / "data" / "registry" / "adapters" / f"{run_id}.pt"

    def _eval_loss(self, model: CoreTransformer, samples: List[Sample]) -> Optional[float]:
        if not samples:
            return None
        model.eval()
        losses = []
        with torch.inference_mode():
            for sample in samples[: max(1, len(samples))]:
                text = format_chat_sample(sample)
                if not text:
                    continue
                ids, _ = model.encode_prompt(text)
                if len(ids) < 2:
                    continue
                input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
                targets = torch.tensor([ids[1:]], dtype=torch.long, device=model.device)
                with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
                    logits = model.forward(input_ids)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                losses.append(float(loss.item()))
        return sum(losses) / max(1, len(losses))

    def _eval_adapter_loss(self, samples: List[Sample], run_id: Optional[str]) -> Optional[float]:
        model = self._build_model()
        adapter_cfg = self.settings.get("continuous", {}).get("adapters", {})
        lora_cfg = LoRAConfig(
            rank=int(adapter_cfg.get("rank", self.settings.get("continuous", {}).get("adapter_rank", 4))),
            alpha=float(adapter_cfg.get("alpha", 1.0)),
        )
        strict = bool(adapter_cfg.get("strict_target_modules", False))
        target_modules = resolve_target_modules(adapter_cfg, strict=strict)
        inject_lora(model, lora_cfg, target_modules=target_modules)
        if run_id:
            path = self._adapter_path(run_id)
            if path.exists():
                load_lora_state(model, path)
        return self._eval_loss(model, samples)

    def _maybe_consolidate(self, anchors: List[Sample], max_regression: float) -> None:
        cfg = self.settings.get("continuous", {}).get("consolidation", {})
        if not bool(cfg.get("enabled", False)):
            return
        every_n = int(cfg.get("every_n_runs", 10))
        runs_dir = self.base_dir / "data" / "registry" / "runs"
        if not runs_dir.exists():
            return
        run_count = len([p for p in runs_dir.iterdir() if p.is_dir() and p.name != "consolidated"])
        if run_count % every_n != 0:
            return
        top_n = int(cfg.get("top_n", 3))
        weights = cfg.get("weights", {})
        run_ids = select_top_runs(self.base_dir, top_n=top_n, weights=weights)
        if not run_ids:
            return
        payload = adapter_soup(self.base_dir, run_ids)
        if not payload:
            return
        write_consolidated_adapter(self.base_dir, payload)
        write_consolidated_meta(self.base_dir, {"runs": run_ids})

        if anchors:
            base_id = load_registry(self.base_dir).current_run_id
            base_loss = self._eval_adapter_loss(anchors, base_id)
            consolidated_loss = self._eval_adapter_loss(anchors, "consolidated")
            if base_loss is not None and consolidated_loss is not None:
                regression = (consolidated_loss - base_loss) / max(1e-6, base_loss)
                if regression > max_regression:
                    return

        state = load_registry(self.base_dir)
        if state.current_run_id:
            state.history.append(state.current_run_id)
        state.current_run_id = "consolidated"
        save_registry(self.base_dir, state)
