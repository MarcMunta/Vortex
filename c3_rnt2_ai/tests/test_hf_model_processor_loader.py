from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from c3rnt2.hf_model import load_hf_model


class _DummyTokenizer:
    last_messages = None

    @classmethod
    def from_pretrained(cls, _name, **_kwargs):
        return cls()

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def apply_chat_template(self, messages, tokenize=False, return_tensors=None, add_generation_prompt=False):
        _ = return_tensors, add_generation_prompt
        _DummyTokenizer.last_messages = messages
        if not tokenize:
            return "<tokenizer-prompt>"
        return torch.tensor([[20, 21, 22]])

    def decode(self, _ids, **_kwargs):
        return "decoded-by-tokenizer"


class _DummyProcessor:
    last_kwargs = None
    last_messages = None
    last_text = None

    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    @classmethod
    def from_pretrained(cls, _name, **kwargs):
        cls.last_kwargs = dict(kwargs)
        return cls()

    def apply_chat_template(self, messages, tokenize=False, return_dict=False, return_tensors=None, add_generation_prompt=False):
        _ = return_dict, return_tensors, add_generation_prompt
        _DummyProcessor.last_messages = messages
        if not tokenize:
            return "<prompt>"
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def __call__(self, text=None, **_kwargs):
        _DummyProcessor.last_text = text
        return {
            "input_ids": torch.tensor([[20, 21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, _ids, **_kwargs):
        return "raw-processor-output"

    def parse_response(self, response):
        return {"final_response": f"parsed::{response}"}


class _DummyVisionTextModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, _name, **_kwargs):
        return cls()

    def generate(self, input_ids=None, **_kwargs):
        assert input_ids is not None
        return torch.tensor([[10, 11, 12, 13, 14]])


def test_hf_image_text_loader_uses_processor(monkeypatch) -> None:
    dummy = types.SimpleNamespace(
        AutoModelForCausalLM=_DummyVisionTextModel,
        AutoModelForImageTextToText=_DummyVisionTextModel,
        AutoProcessor=_DummyProcessor,
        AutoTokenizer=_DummyTokenizer,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", dummy)

    settings = {
        "core": {
            "backend": "hf",
            "hf_model": "google/gemma-3-12b-it",
            "hf_model_loader": "image_text_to_text",
            "hf_processor_padding_side": "left",
            "hf_device": "cpu",
            "dtype": "fp16",
        }
    }

    model = load_hf_model(settings)
    out = model.generate(
        messages=[{"role": "user", "content": "Hola"}],
        max_new_tokens=8,
    )

    assert model.model_loader == "image_text_to_text"
    assert out == "parsed::raw-processor-output"
    assert _DummyProcessor.last_kwargs == {"padding_side": "left"}
    assert _DummyProcessor.last_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hola"}]}
    ]


def test_hf_processor_causal_loader_uses_processor_and_parses(monkeypatch) -> None:
    dummy = types.SimpleNamespace(
        AutoModelForCausalLM=_DummyVisionTextModel,
        AutoModelForImageTextToText=_DummyVisionTextModel,
        AutoProcessor=_DummyProcessor,
        AutoTokenizer=_DummyTokenizer,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", dummy)

    settings = {
        "core": {
            "backend": "hf",
            "hf_model": "google/gemma-4-26B-A4B-it",
            "hf_model_loader": "processor_causal_lm",
            "hf_processor_padding_side": "left",
            "hf_enable_thinking": False,
            "hf_device": "cpu",
            "dtype": "fp16",
        }
    }

    model = load_hf_model(settings)
    out = model.generate(
        messages=[{"role": "user", "content": "Hola"}],
        max_new_tokens=8,
    )

    assert model.model_loader == "processor_causal_lm"
    assert out == "decoded-by-tokenizer"
    assert _DummyProcessor.last_kwargs == {"padding_side": "left"}
    assert _DummyTokenizer.last_messages == [
        {"role": "user", "content": "Hola"}
    ]
    assert _DummyProcessor.last_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hola"}]}
    ]
    assert _DummyProcessor.last_text is None
