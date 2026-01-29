import pytest

from c3rnt2.continuous.lora import resolve_target_modules


def test_lora_target_modules_required() -> None:
    with pytest.raises(ValueError):
        resolve_target_modules({}, strict=True)
    targets = resolve_target_modules({}, strict=False)
    assert targets

