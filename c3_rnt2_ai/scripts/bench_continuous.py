from __future__ import annotations

import json
import time
from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.continuous.dataset import ingest_sources, retrieve_context, collect_samples
from c3rnt2.continuous.trainer import ContinualTrainer


def main() -> None:
    settings = load_settings()
    allowlist = settings.get("agent", {}).get("web_allowlist", [])

    t0 = time.time()
    ingested = ingest_sources(Path("."), allowlist, settings)
    ingest_time = time.time() - t0

    t1 = time.time()
    context = retrieve_context(Path("."), "python function", settings, top_k=3)
    rag_time = time.time() - t1

    t2 = time.time()
    trainer = ContinualTrainer(settings=settings, base_dir=Path("."))
    result = trainer.run_tick()
    train_time = time.time() - t2

    collected = collect_samples(Path("."), allowlist, settings)
    total = collected.stats.total_candidates
    filtered = collected.stats.filtered
    filtered_pct = (filtered / max(1, total)) * 100.0

    report = {
        "ingest_time_sec": round(ingest_time, 4),
        "ingested_docs": ingested,
        "rag_time_sec": round(rag_time, 4),
        "rag_chars": len(context),
        "train_time_sec": round(train_time, 4),
        "samples_used": result.samples,
        "filtered_pct": round(filtered_pct, 2),
        "promoted": result.promoted,
        "loss": result.loss,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

