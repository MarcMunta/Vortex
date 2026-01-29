from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from c3rnt2.config import load_settings  # type: ignore[import-not-found]
from c3rnt2.model.core_transformer import CoreTransformer  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    settings = load_settings(args.profile)
    model = CoreTransformer.from_settings(settings)
    draft = getattr(model, "draft_model", None)
    if draft is None:
        raise SystemExit("Draft model not enabled in settings.")
    model.eval()
    draft.train()

    samples = [
        "def f(x): return x * 2",
        "json {\"a\": 1, \"b\": [2, 3]}",
        "hello world",
        "for i in range(3): print(i)",
    ]
    optimizer = torch.optim.Adam(draft.parameters(), lr=args.lr)

    for step in range(args.steps):
        text = samples[step % len(samples)]
        ids, _ = model.encode_prompt(text)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            full_logits = model.forward(input_ids)
        draft_logits = draft.forward(input_ids)
        loss = torch.nn.functional.mse_loss(draft_logits.float(), full_logits.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("distill_complete")


if __name__ == "__main__":
    main()
