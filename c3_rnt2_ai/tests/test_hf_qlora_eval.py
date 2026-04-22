from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

from c3rnt2.training import hf_qlora


class _DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def __call__(self, text: str, truncation: bool = False, max_length: int | None = None, return_tensors: str | None = None):
        tokens = [1] * max(1, min(len(text.split()) or 1, max_length or 4))
        if return_tensors == "pt":
            input_ids = torch.tensor([tokens], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        return {"input_ids": tokens}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "a a a"


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(use_cache=False)

    @property
    def device(self):
        return self.dummy.device

    def gradient_checkpointing_enable(self) -> None:
        return None

    def forward(self, **kwargs):
        loss = self.dummy.sum() * 0 + torch.tensor(0.5, device=self.dummy.device)
        return types.SimpleNamespace(loss=loss)

    def generate(self, input_ids=None, max_new_tokens: int = 0, do_sample: bool = False, **kwargs):
        if input_ids is None:
            input_ids = torch.tensor([[1]], device=self.dummy.device)
        extra = torch.ones((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "dummy.bin").write_text("ok", encoding="utf-8")


def _install_dummy_hf(monkeypatch) -> None:
    transformers = types.ModuleType("transformers")
    peft = types.ModuleType("peft")

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _DummyModel()

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _DummyTokenizer()

    class BitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, *args, **kwargs):
            return model

    def get_peft_model(model, *args, **kwargs):
        return model

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        return model

    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.AutoTokenizer = AutoTokenizer
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    peft.LoraConfig = LoraConfig
    peft.PeftModel = PeftModel
    peft.get_peft_model = get_peft_model
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)


def _base_settings(tmp_path: Path) -> dict:
    dataset_path = tmp_path / "sft_samples.jsonl"
    payload = {"prompt": "hi", "response": "hello", "source_kind": "web"}
    dataset_path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "core": {"backend": "hf", "hf_model": "dummy"},
        "knowledge": {"embedding_backend": "hash"},
        "continuous": {"knowledge_path": str(tmp_path / "knowledge.sqlite"), "eval": {"anchors_path": str(tmp_path / "anchors.jsonl")}},
        "hf_train": {
            "enabled": True,
            "model_name": "dummy",
            "dataset_path": str(dataset_path),
            "registry_dir": str(tmp_path / "registry"),
            "state_path": str(tmp_path / "registry" / "state.json"),
            "device": "cpu",
            "max_steps": 1,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "max_seq_len": 16,
            "use_fast": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "eval": {
                "enabled": True,
                "max_samples": 1,
                "gen_max_new_tokens": 4,
                "min_improvement": -1.0,
                "max_regression": 1.0,
                "max_repeat_ratio": 0.9,
            },
        },
    }


def test_train_once_eval_ok_fields(tmp_path: Path, monkeypatch) -> None:
    _install_dummy_hf(monkeypatch)
    settings = _base_settings(tmp_path)
    result = hf_qlora.train_once(settings, tmp_path, reuse_dataset=True)
    assert result.ok_train is True
    assert result.ok_eval is True
    assert result.eval_ok is True
    assert result.improvement is not None
    assert result.repeat_ratio is not None
    assert result.ok is True


def test_train_once_eval_failure_sets_ok_false(tmp_path: Path, monkeypatch) -> None:
    _install_dummy_hf(monkeypatch)
    settings = _base_settings(tmp_path)
    settings["hf_train"]["eval"]["max_repeat_ratio"] = 0.1
    result = hf_qlora.train_once(settings, tmp_path, reuse_dataset=True)
    assert result.ok_train is True
    assert result.ok_eval is False
    assert result.eval_ok is False
    assert result.ok is False


def test_resolve_lora_target_modules_filters_out_multimodal_wrappers() -> None:
    class Gemma4ClippableLinear:
        pass

    class _NamedModules:
        def named_modules(self):
            return iter(
                [
                    ("model.language_model.layers.0.self_attn.q_proj", torch.nn.Linear(4, 4, bias=False)),
                    ("model.language_model.layers.0.self_attn.k_proj", torch.nn.Linear(4, 4, bias=False)),
                    ("model.vision_tower.encoder.layers.0.self_attn.q_proj", Gemma4ClippableLinear()),
                    ("model.audio_tower.layers.0.self_attn.q_proj.linear", torch.nn.Linear(4, 4, bias=False)),
                ]
            )

    resolved = hf_qlora._resolve_lora_target_modules(
        _NamedModules(),
        {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    )

    assert resolved == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
    ]
