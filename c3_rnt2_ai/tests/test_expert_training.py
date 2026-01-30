from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.training.train_experts import train_experts
from c3rnt2.training.finetune_adapters import finetune_adapter


def test_expert_train_and_finetune(tmp_path: Path):
    data_root = tmp_path / "corpora" / "programming"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "sample.txt").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    settings = load_settings("dev_small")
    out_root = tmp_path / "experts"
    result = train_experts(
        settings=settings,
        domains=["programming"],
        data_root=tmp_path / "corpora",
        output_root=out_root,
        steps=1,
        lr=1e-4,
        batch_tokens=64,
        grad_accum=1,
    )
    domain_result = result["domains"]["programming"]
    assert domain_result["ok"]
    adapter_path = Path(domain_result["adapter"])
    assert adapter_path.exists()
    finetuned = finetune_adapter(
        settings=settings,
        adapter_path=adapter_path,
        data_path=data_root,
        output_path=tmp_path / "finetuned.pt",
        steps=1,
        lr=1e-4,
        batch_tokens=64,
        grad_accum=1,
    )
    assert Path(finetuned["adapter"]).exists()
