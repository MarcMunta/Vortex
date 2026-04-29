from __future__ import annotations

from c3rnt2.server import _resolve_decode_args
from c3rnt2.runtime.vram_governor import decide_max_new_tokens


def test_request_without_max_tokens_uses_large_default() -> None:
    args = _resolve_decode_args({"decode": {"max_new_tokens": 2048}}, {})
    assert args["max_new_tokens"] >= 2048
    assert args["max_tokens_effective"] >= 2048


def test_request_max_tokens_4096_respected_without_cuda_governor() -> None:
    settings = {
        "decode": {"max_new_tokens": 2048, "hard_max_new_tokens": 4096},
        "core": {"vram_ceil_tokens": 4096},
    }
    args = _resolve_decode_args(settings, {"max_tokens": 4096})
    decided = decide_max_new_tokens(args["max_new_tokens"], "cpu", "bf16", settings)
    assert decided == 4096


def test_code_request_default_receives_code_budget() -> None:
    args = _resolve_decode_args(
        {"generation": {"default_max_tokens": 2048, "code_max_tokens": 3072, "hard_max_tokens": 4096}},
        {"response_mode": "code"},
    )
    assert args["max_new_tokens"] >= 3072
    assert args["preserve_max_new_tokens"] is True


def test_chat_request_does_not_preserve_vram_budget() -> None:
    args = _resolve_decode_args(
        {"generation": {"default_max_tokens": 2048, "code_max_tokens": 3072, "hard_max_tokens": 4096}},
        {},
    )
    assert args["max_new_tokens"] >= 2048
    assert args["preserve_max_new_tokens"] is False


def test_hard_cap_applies_above_4096() -> None:
    args = _resolve_decode_args({"generation": {"hard_max_tokens": 4096}}, {"max_tokens": 8192})
    assert args["max_tokens_requested"] == 8192
    assert args["max_new_tokens"] == 4096
