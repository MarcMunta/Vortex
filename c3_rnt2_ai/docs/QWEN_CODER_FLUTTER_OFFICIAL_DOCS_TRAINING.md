# Qwen Coder Flutter Official Docs Training

Goal: build a reproducible, auditable Flutter/Dart pipeline for `Qwen/Qwen2.5-Coder-7B-Instruct` adapters. This prepares data, training, and eval for Qwen Coder. It does not train AWQ models.

## Runtime And Training

- Daily runtime profile: `rtx4080_16gb_programming_qwen_coder_local`
- Training profile: `rtx4080_16gb_programming_qwen_coder_train_docker`
- Base trainable model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Adapter registry: `data/registry/hf_train/qwen_coder_flutter`
- Manual SGLang inference profile: `rtx4080_16gb_programming_qwen_coder_sglang`

Do not train `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`. AWQ is inference-only.

## Sources

Allowed only:

- `https://docs.flutter.dev/`
- `https://api.flutter.dev/`
- `https://dart.dev/` for Dart language and async coverage needed by Flutter

Rejected:

- blogs
- StackOverflow
- pub.dev package pages
- GitHub issues
- random web pages
- autonomous web discovery

The crawler checks robots/sitemaps, rate limits requests, records license metadata, stores content hash, and preserves ETag/Last-Modified when provided.

## Safety

- Domain allowlist is enforced.
- Assets, JS, CSS, feeds, archives, images, anchors, and non-HTML pages are excluded.
- Rate limit defaults to `1.0` second.
- Every page record stores URL, title, domain, fetch time, content hash, status, ETag/Last-Modified, and license flag.
- General Vortex web/autolearn stays disabled. This is controlled ingestion, not autonomous browsing.

## Commands

From `c3_rnt2_ai/`:

```powershell
python -m c3rnt2.model_init --model Qwen/Qwen2.5-Coder-7B-Instruct --cache-dir data/models/hf-cache --status-only
docker compose up -d vortex-api vortex-control vortex-frontend
python -m c3rnt2 prepare-model --profile rtx4080_16gb_programming_qwen_coder_local
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_programming_qwen_coder_local
python -m c3rnt2 bench --profile rtx4080_16gb_programming_qwen_coder_local --max-new-tokens 64 --json-out data/bench/programming_qwen_coder_local.json
```

Download missing base model explicitly:

```powershell
python -m c3rnt2.model_init --model Qwen/Qwen2.5-Coder-7B-Instruct --cache-dir data/models/hf-cache --download
```

Ingest official docs:

```powershell
python scripts/ingest_flutter_docs.py `
  --sources docs.flutter.dev api.flutter.dev `
  --out data/flutter_docs `
  --rate-limit 1.0
```

Add Dart docs only when needed:

```powershell
python scripts/ingest_flutter_docs.py `
  --sources docs.flutter.dev api.flutter.dev dart.dev `
  --out data/flutter_docs `
  --rate-limit 1.0
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

Train:

```powershell
docker compose run --rm trainer python -m c3rnt2 train-once `
  --profile rtx4080_16gb_programming_qwen_coder_train_docker
```

Eval:

```powershell
docker compose run --rm eval python scripts/eval_flutter_adapter.py `
  --profile rtx4080_16gb_programming_qwen_coder_local `
  --eval config/datasets/flutter_official_hard_eval.jsonl
```

## Expected Outputs

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

## Training Config

Initial QLoRA settings:

- `load_in_4bit: true`
- `max_seq_len: 1024`
- `micro_batch_size: 1`
- `grad_accum_steps: 8`
- `max_steps: 80`
- `lora_rank: 8`
- `lora_alpha: 16`
- `lora_dropout: 0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

Weighted sampling priorities:

- `flutter_official_docs_debugging_sft`: `5.0`
- `flutter_official_docs_code_sft`: `4.5`
- `flutter_official_docs_architecture_sft`: `4.5`
- `flutter_official_docs_sft`: `4.0`
- `web`: `0.0`

## Eval Criteria

Review outputs for:

- technical accuracy
- compilable Flutter/Dart code
- layouts/constraints
- navigation
- state
- async
- testing
- performance
- accessibility
- mobile/web/desktop distinctions
- useful Codex responses with files/tests/commands

## Tests

```powershell
pytest -q tests/test_flutter_docs_pipeline.py
pytest -q tests/test_flutter_dataset_writer.py
pytest -q tests/test_settings_normalization.py
pytest -q tests/test_server_multimodal_api.py
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_programming_qwen_coder_local
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_programming_qwen_coder_train_docker
```

## Rollback

1. Restore previous `data/registry/hf_train/qwen_coder_flutter/registry.json`.
2. Restart API or call `/v1/reload_adapter`.
3. Re-run doctor and Flutter eval.

