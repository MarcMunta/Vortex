# Gemma 4 Flutter Official Docs Training (Legacy)

Legacy note: primary Flutter/Dart path moved to `QWEN_CODER_FLUTTER_OFFICIAL_DOCS_TRAINING.md` with `Qwen/Qwen2.5-Coder-7B-Instruct`. Keep this document only for old Gemma adapter reference.

Goal: build a reproducible, auditable Flutter/Dart knowledge pipeline for `google/gemma-4-E4B-it` adapters. This prepares data and eval. It does not train or promote by itself.

## Sources

Allowed:

- `https://docs.flutter.dev/`
- `https://api.flutter.dev/`
- `https://dart.dev/` when Dart language coverage is needed

Legal checks:

- `docs.flutter.dev/robots.txt` advertises `https://docs.flutter.dev/sitemap.xml`.
- `dart.dev/robots.txt` advertises `https://dart.dev/sitemap.xml`.
- Flutter and Dart site terms state docs are Creative Commons Attribution 4.0 unless noted otherwise, and code samples are 3-Clause BSD.

Do not ingest blogs, pub.dev package pages, GitHub issues, StackOverflow, or random web pages in this pipeline.

## Safety

- Crawler is domain allowlisted.
- Assets, JS, CSS, feeds, archives, images, anchors, and non-HTML pages are excluded.
- Rate limit defaults to `1.0` second.
- ETag and Last-Modified are stored when provided.
- Every page record stores URL, title, domain, fetch time, content hash, status, and license flag.
- General Vortex web/autolearn stays disabled.

## Commands

From `c3_rnt2_ai/`:

```powershell
python scripts/ingest_flutter_docs.py `
  --sources docs.flutter.dev api.flutter.dev `
  --out data/flutter_docs `
  --rate-limit 1.0
```

For a safe smoke run:

```powershell
python scripts/ingest_flutter_docs.py `
  --sources docs.flutter.dev api.flutter.dev `
  --out data/flutter_docs `
  --rate-limit 1.0 `
  --max-pages 20
```

Build datasets:

```powershell
python scripts/build_flutter_training_dataset.py `
  --chunks data/flutter_docs/processed/chunks.jsonl `
  --out config/datasets
```

Audit coverage:

```powershell
python scripts/audit_flutter_coverage.py `
  --chunks data/flutter_docs/processed/chunks.jsonl `
  --datasets config/datasets `
  --out data/flutter_docs/reports
```

Expected outputs:

- `data/flutter_docs/raw/pages.jsonl`
- `data/flutter_docs/raw/html/<hash>.html`
- `data/flutter_docs/processed/chunks.jsonl`
- `data/flutter_docs/manifest.json`
- `data/flutter_docs/reports/coverage.md`
- `data/flutter_docs/reports/coverage.json`
- `config/datasets/flutter_official_docs_sft.jsonl`
- `config/datasets/flutter_official_docs_code_sft.jsonl`
- `config/datasets/flutter_official_docs_debugging_sft.jsonl`
- `config/datasets/flutter_official_docs_architecture_sft.jsonl`
- `config/datasets/flutter_official_hard_eval.jsonl`

## Coverage Gate

Do not train final adapter until `coverage.json` has:

- `coverage_ok: true`
- broad topic coverage, not only intro pages
- nonzero samples for layout, constraints, rendering, state management, navigation, forms, async, testing, performance, accessibility, deployment, plugins/platform, and architecture
- low or zero failed pages
- no suspicious duplicate spike

If `low_coverage_topics` contains Flutter-critical topics, run targeted ingestion by adding official URLs or increasing `--max-pages`, then rebuild datasets and coverage.

## Training

Only after coverage passes:

```powershell
docker compose run --rm trainer python -m c3rnt2 train-once `
  --profile rtx4080_16gb_programming_train_docker
```

Training profile uses weighted sampling:

- `flutter_official_docs_debugging_sft`: `5.0`
- `flutter_official_docs_code_sft`: `4.5`
- `flutter_official_docs_architecture_sft`: `4.5`
- `flutter_official_docs_sft`: `4.0`
- `web`: `0.0`

Safe initial training config:

- `max_steps: 80`
- `max_seq_len: 1024`
- `micro_batch_size: 1`
- `grad_accum_steps: 8`
- `lora_rank: 8`
- `lora_alpha: 16`
- `lora_dropout: 0.05`

Do not increase steps until eval shows improvement without overfitting.

## Eval

Run after baseline and after adapter:

```powershell
docker compose run --rm eval python scripts/eval_flutter_adapter.py `
  --profile rtx4080_16gb_programming_gemma4_local `
  --eval config/datasets/flutter_official_hard_eval.jsonl
```

Outputs:

- `data/bench/flutter_eval_outputs.jsonl`
- `data/bench/flutter_eval_report.md`

Review manually for:

- technical accuracy
- concrete fixes
- no generic advice
- constraints/layout understanding
- compilable Flutter code
- mobile/web/desktop distinction
- useful Codex prompts

## Tests

```powershell
pytest -q tests/test_flutter_docs_pipeline.py
pytest -q tests/test_flutter_dataset_writer.py
pytest -q tests/test_settings_normalization.py
pytest -q tests/test_server_multimodal_api.py
```

## Rollback

This pipeline does not promote adapters. If a trained adapter is later promoted and fails:

1. Restore previous `data/registry/hf_train/gemma4_e4b/registry.json`.
2. Restart API or call `/v1/reload_adapter`.
3. Re-run doctor and Flutter eval.

## Expanding Weak Topics

Use `coverage.md` low coverage list. Add targeted official pages for weak topics:

- layout/constraints: layout, constraints, slivers, scrolling
- testing: unit/widget/integration/golden docs
- performance: DevTools, profiling, jank, rendering
- architecture: app architecture guide/case study
- platform: plugins, platform channels, Android/iOS deployment

Then rebuild datasets and re-audit.
