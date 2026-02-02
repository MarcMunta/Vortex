from __future__ import annotations

import random

from c3rnt2.continuous.types import Sample
from c3rnt2.training.hf_qlora import _compute_sample_weights


def test_weighted_sampling_prefers_high_quality() -> None:
    settings = {
        "hf_train": {
            "use_weighted_sampling": True,
            "source_kind_weights": {"high": 2.0, "low": 1.0},
        }
    }
    samples = [
        Sample(prompt="p1", response="r1", source_kind="high", quality=1.0),
        Sample(prompt="p2", response="r2", source_kind="low", quality=0.1),
    ]
    weights = _compute_sample_weights(samples, settings)
    rng = random.Random(7)
    counts = [0, 0]
    for _ in range(1000):
        idx = rng.choices([0, 1], weights=weights, k=1)[0]
        counts[idx] += 1
    assert counts[0] > counts[1]
