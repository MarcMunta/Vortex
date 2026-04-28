# Colab Gemma 4 Fine-tune for Vortex

Goal: train a small LoRA/QLoRA adapter for `google/gemma-4-E4B-it`, focused on Flutter, Dart, Python, FastAPI, and Vortex repo patterns. This does not train the base model.

## Security Bounds

- Use only `/MyDrive/VortexTraining/`.
- Do not grant Gmail, Photos, Contacts, or broad Drive reads beyond the mounted folder workflow.
- Do not print HF tokens or other secrets.
- Do not use external web datasets unless separately reviewed.
- Do not auto-promote imported adapters. Run eval first.

## Drive Layout

The notebook creates:

```text
/MyDrive/VortexTraining/
  datasets/raw/
  datasets/processed/
  datasets/eval/
  notebooks/
  outputs/adapters/
  outputs/logs/
  outputs/reports/
  exports/
```

## Prepare Dataset

From repo root:

```powershell
cd D:\GitHub\Vortex
python c3_rnt2_ai\scripts\export_training_dataset.py
```

Outputs:

- `c3_rnt2_ai/data/registry/hf_train/gemma4_e4b/colab_export/datasets/processed/gemma4_flutter_python_sft.jsonl`
- `c3_rnt2_ai/data/registry/hf_train/gemma4_e4b/colab_export/datasets/eval/gemma4_flutter_python_eval.jsonl`
- `c3_rnt2_ai/data/registry/hf_train/gemma4_e4b/colab_export/outputs/reports/dataset_quality_report.md`

If the quality gate fails, do not train. Fix data first.

Current known data caveat: this repo has strong Python/FastAPI/Vortex coverage, but little or no real Flutter/Dart app code. Add real Flutter/Dart source or reviewed seed JSONL before serious training.

## Upload To Drive

Use Colab file upload or Drive UI. Put files under:

```text
/MyDrive/VortexTraining/datasets/processed/gemma4_flutter_python_sft.jsonl
/MyDrive/VortexTraining/datasets/eval/gemma4_flutter_python_eval.jsonl
/MyDrive/VortexTraining/notebooks/gemma4_flutter_python_lora.ipynb
```

No need to upload repo secrets, `.env`, model cache, logs, or existing adapters.

## Run Colab

Open `gemma4_flutter_python_lora.ipynb`.

Steps:

1. Select GPU runtime.
2. Run setup cells.
3. Mount Drive.
4. Authenticate Hugging Face with `notebook_login()`.
5. Run dataset quality cell.
6. Train only if `gate_ok == True`.
7. Review eval report.

The notebook first tries Unsloth if import and model load work. If not, it falls back to `transformers + peft + trl + bitsandbytes + accelerate`.

Initial safe config:

```text
max_seq_length: 1024
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
max_steps: 200
load_in_4bit: true
```

## Outputs

Adapter:

```text
/MyDrive/VortexTraining/outputs/adapters/gemma4_flutter_python_lora/
```

Reports:

```text
/MyDrive/VortexTraining/outputs/reports/eval_report.md
/MyDrive/VortexTraining/outputs/reports/eval_outputs.jsonl
```

Adapter dir should contain:

- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer files if saved
- `training_args.json`
- `README.md`

## Import Adapter Locally

After downloading/copying adapter from Drive:

```powershell
cd D:\GitHub\Vortex\c3_rnt2_ai
python scripts\import_colab_adapter.py <adapter_dir> --target data/registry/hf_train/gemma4_e4b/colab_flutter_python_lora
```

This copies only. It does not promote.

To promote after review:

```powershell
python scripts\import_colab_adapter.py <adapter_dir> --target data/registry/hf_train/gemma4_e4b/colab_flutter_python_lora --force --promote
```

Runtime reload endpoint exists:

```powershell
curl -X POST http://127.0.0.1:8000/v1/reload_adapter
```

Or create a reload request while importing:

```powershell
python scripts\import_colab_adapter.py <adapter_dir> --target data/registry/hf_train/gemma4_e4b/colab_flutter_python_lora --force --promote --reload-request
```

## Local Eval

```powershell
cd D:\GitHub\Vortex\c3_rnt2_ai
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_programming_gemma4_local
python -m c3rnt2 bench --profile rtx4080_16gb_programming_gemma4_local --scenario default
pytest -q tests/test_settings_normalization.py
pytest -q tests/test_server_multimodal_api.py
```

## Rollback

1. Restore previous `data/registry/hf_train/gemma4_e4b/registry.json`.
2. Or point `current_adapter` back to the previous adapter path.
3. Restart API or call `/v1/reload_adapter`.
4. Re-run doctor/bench.

## Overfitting Risks

- Small adapter can memorize exact repo snippets.
- Missing real Flutter/Dart code can make Flutter behavior generic.
- Too many copy/exact-return samples degrade assistant style.
- Eval prompts must be held out from train set.
- Keep `max_steps` small until eval shows improvement without regressions.
