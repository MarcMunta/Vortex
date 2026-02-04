from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.training.train_hf_experts import train_hf_experts


def test_train_hf_experts_mock_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    domain_dir = tmp_path / "data" / "corpora" / "code"
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "samples.jsonl").write_text(json.dumps({"prompt": "hi", "response": "hello"}) + "\n", encoding="utf-8")

    settings = {"_profile": "rtx4080_16gb_120b_like", "bench_thresholds": {"required_ctx": 4096}}
    result = train_hf_experts(
        settings,
        ["code"],
        data_root=tmp_path / "data" / "corpora",
        output_root=tmp_path / "data" / "experts_hf",
        mock=True,
    )
    assert result.get("ok") is True
    dom = result["domains"]["code"]
    manifest_path = Path(dom["manifest"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("dataset_hash", "steps", "lr", "max_seq_len", "anchor_eval", "regression", "passed_eval"):
        assert key in payload
    assert payload["domain"] == "code"
    assert payload.get("mock") is True

