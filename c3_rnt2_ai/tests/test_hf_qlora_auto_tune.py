from __future__ import annotations

from c3rnt2.training import hf_qlora


def test_auto_tune_batch_on_oom() -> None:
    calls = {"count": 0}

    def _train_fn(micro: int, grad: int):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return (0.0, 1, 100.0, None)

    result, micro, grad = hf_qlora._auto_tune_train(_train_fn, 4, 2, max_retries=2, enabled=True)
    assert result[0] == 0.0
    assert micro == 2
    assert grad == 4
