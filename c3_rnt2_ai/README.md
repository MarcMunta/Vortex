# Vortex (C3 + RNT-2 + LAVA + BAD)

Este repo implementa un prototipo modular para una “IA 120B-like” local combinando:
- **C3**: pesos paginados y comprimidos con caché coherente.
- **Vortex-Tok**: tokenización reversible de tasa variable (macro/patch/ESC).
- **V-Blocks**: LocalMixer + SSM + LAVA Memory + GatedMLP.
- **BAD**: Blockwise Adaptive Decoding (draft+verify).
- **Agente**: herramientas, memoria persistente, auto-entrenamiento y auto-mejora segura.

Nota de rol:
- El backend **HF** (Qwen2.5-8B-Instruct) usa el *tokenizer de HuggingFace*.
- **Vortex-Tok** es para los backends Vortex/C3 (no se usa para HF). Bench: `python scripts/bench_tokenizer.py --profile dev_small`.

## Ruta recomendada (RTX 4080 16GB)
Ver `docs/rtx4080.md`, `docs/120b_like.md` y `docs/WINDOWS_SELF_TRAIN_WSL.md` (perfil recomendado: `rtx4080_16gb_120b_like`).

## RTX 4080 16GB Quickstart (Windows, perfil safe)
Comandos recomendados (core backend, sin descargas, web deny-by-default):
```bash
pip install -e .
python -m vortex doctor --profile rtx4080_16gb_safe
python -m vortex doctor --deep --profile rtx4080_16gb_safe
python -m vortex bench --profile rtx4080_16gb_safe --max-new 64 --json-out data/bench/last.json
```

Self-train seguro (quarantine + aprobaciÃ³n manual):
- `doctor --deep` escribe un adapter candidato en `data/continuous/quarantine/<run_id>/`.
- Para promoverlo: crea `APPROVE.txt` dentro del directorio del run y ejecuta:
```bash
python -m vortex promote-quarantine --run-id <run_id>
```

Troubleshooting OOM:
- Baja `kv.window_size`, `runtime.cache_vram_budget_mb`, `continuous.batch_tokens` o `decode.max_new_tokens` en tu perfil.

## Instalacion
```bash
python -m venv .venv
. .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -e .
```

Para API local (serve):
```bash
pip install -e .[api]
```

## HTTP API (OpenAI-compatible)

El servidor expone un contrato estilo OpenAI:
- `GET /healthz`, `GET /readyz`
- `GET /v1/models`
- `POST /v1/chat/completions` (no-stream + streaming SSE con `data: ...` y `data: [DONE]`)
- `GET /metrics`
- `GET|POST /doctor`, `GET /doctor/deep`

Arranque (ejemplo):
```bash
# opcional: auth dev (si se define, exige Authorization: Bearer ...)
export KLIMEAI_API_TOKEN=devtoken

klimeai serve --host 0.0.0.0 --port 8000
```

Smoke test:
```bash
pytest -q tests/test_openai_api_smoke.py
python scripts/smoke_api.py --base-url http://localhost:8000
```

Para HF + entrenamiento QLoRA:
```bash
pip install -e .[hf,train]
```

Embeddings CPU opcionales (RAG, incluye FAISS si esta disponible):
```bash
pip install -e .[rag]
```

## CLI (Vortex)
Comandos principales vía `python -m vortex` (compat: `python -m c3rnt2`):
```bash
python -m c3rnt2 tokenizer-train
python -m c3rnt2 eval
python -m c3rnt2 chat
python -m c3rnt2 chat --profile rtx4080_16gb --backend hf --model Qwen/Qwen2.5-8B-Instruct
python -m c3rnt2 agent-demo
python -m c3rnt2 agent-run --task "Fix failing test"
python -m c3rnt2 doctor --deep --profile rtx4080_16gb_vortexx_next
python -m c3rnt2 serve --profile rtx4080_16gb_vortexx_next
python -m c3rnt2 bootstrap --profile dev_small --checkpoint path.pt
python -m c3rnt2 ingest-once --profile dev_small
python -m c3rnt2 train-once --profile qwen8b_train
python -m c3rnt2 self-train --once --profile qwen8b_train
python -m c3rnt2 self-patch --goal "fix failing test" --dry-run
python -m c3rnt2 apply-patch <id> --approve
python -m c3rnt2 bench --profile rtx4080_16gb_vortexx_next --max-new-tokens 512
python -m c3rnt2 learn ingest --profile qwen8b_train
python -m c3rnt2 learn train --profile qwen8b_train --steps 50
python -m c3rnt2 learn eval --profile qwen8b_train
python -m c3rnt2 learn promote --profile qwen8b_train

Perfil HF (Qwen-8B):

python -m c3rnt2 chat --profile qwen8b_base
python -m c3rnt2 serve --profile qwen8b_base
python -m c3rnt2 chat --profile rtx4080_16gb

Requiere: pip install .[hf]
```

## Dependencias HF / RAG
```bash
pip install -e .[hf,train]
pip install -e .[rag]
```
Si `faiss-cpu` no esta disponible en tu plataforma, el sistema funciona sin FAISS.

## Minimo reproducible Qwen-8B
Prerequisito:
```bash
pip install -e .[hf,train]
```
Flujo minimo:
```bash
python -m c3rnt2 doctor --deep --profile qwen8b_base
python -m c3rnt2 serve --profile qwen8b_base
python -m c3rnt2 bootstrap --teacher Qwen/Qwen2.5-8B-Instruct --teacher-quant 4bit --reuse-dataset --profile qwen8b_base
python -m c3rnt2 ingest-once --profile qwen8b_base
python -m c3rnt2 train-once --profile qwen8b_train
python -m c3rnt2 self-patch --goal "mejorar seguridad de self_patch" --dry-run
```
Aplicar un patch aprobado:
```bash
python -m c3rnt2 apply-patch <id>
```

## Tokenizador Vortex-Tok
Entrenamiento MVP (patch codebook + macro codebook opcional):
```bash
python -m c3rnt2 tokenizer-train --corpus data/corpora
```

## Vortex Core
Usa V-Blocks (LocalMixer + SSM + LAVA + GatedMLP). El contexto largo depende de LAVA + SSM, no de KV infinito.

## C3 Runtime
El runtime usa tiles comprimidos y caché con LRU + estabilidad. En el MVP, la descompresión ocurre en CPU/torch.

## Agente y Auto-mejora
El demo crea un repo mini, abre docs (best effort), edita un bug y ejecuta tests.
```bash
python -m c3rnt2 agent-demo
```

## Benchmarks
```bash
python -m c3rnt2 eval
```
Generación rápida (BAD decode + stats):
```bash
python scripts/bench_generate.py --profile dev_small
```
Salida reciente en `data/bench/latest.txt`.

## Quickstart (Qwen-8B HF)
```bash
python -m c3rnt2 chat --profile rtx4080_16gb --backend hf --model Qwen/Qwen2.5-8B-Instruct --stream
python -m c3rnt2 serve --profile rtx4080_16gb --backend hf --model Qwen/Qwen2.5-8B-Instruct
```
RTX 4080 + Qwen-8B (serve + self-train):
```bash
python -m c3rnt2 doctor --deep --profile rtx4080_16gb
python -m c3rnt2 serve --profile rtx4080_16gb
python -m c3rnt2 train-once --profile safe_selftrain_4080_hf
python -m c3rnt2 serve-self-train --profile safe_selftrain_4080_hf --host 0.0.0.0 --port 8000
```
Comando recomendado (modo continuo seguro):
```bash
python -m c3rnt2 serve-autopilot --profile safe_selftrain_4080_hf --host 0.0.0.0 --port 8000
```

Modo autonomo (internet + self-train + autopatch + reinicio):
```bash
# 1) Smoke test
python -m c3rnt2 doctor --deep --profile autonomous_4080_hf

# 2) Gate de aprobacion para autopatch (sin este archivo NO se aplican/mergean parches)
type NUL > data\\APPROVE_AUTOPATCH

# 3) Runner que reinicia automaticamente si autopilot pide restart (exit code 23)
python scripts\\run_daemon.py --profile autonomous_4080_hf --host 0.0.0.0 --port 8000
```
Para activar RAG en el servidor:
```bash
# en config/settings.yaml
rag:
  enabled: true
  top_k: 3
  max_chars: 1200
```

## Doctor (smoke test de perfil)
```bash
python -m c3rnt2 doctor --profile rtx4080_16gb
python -m c3rnt2 doctor --profile safe_selftrain_4080
python -m c3rnt2 doctor --profile safe_selftrain_4080_hf --deep
python -m c3rnt2 doctor --profile autonomous_4080_hf --deep
```

## Web seguro (allowlist)
```yaml
tools:
  web:
    enabled: true
    allow_domains: ["docs.python.org", "pytorch.org", "github.com"]
```

## Benchmark vortexx_next
```bash
python scripts/bench_generate.py --profile rtx4080_16gb_vortexx_next --max-new-tokens 512
```

## Learning loop (incremental)
```bash
python -m c3rnt2 learn ingest --profile qwen8b_train
python -m c3rnt2 learn train --profile qwen8b_train --steps 50
python -m c3rnt2 learn eval --profile qwen8b_train
python -m c3rnt2 learn promote --profile qwen8b_train
```

## Agent run
```bash
python -m c3rnt2 agent-run --task "Fix failing test" --profile qwen8b_base
```

## Agent safe mode
- Web tool desactivada por defecto. Activar en `config/settings.yaml` (`tools.web.enabled: true`) y allowlist.
- Tests de herramientas en sandbox con `C3RNT2_NO_NET=1` y sin secretos del entorno.

## Web tools allowlist
Para habilitar busqueda y apertura de docs en el agente/servidor:
```yaml
tools:
  web:
    enabled: true
    allow_domains: ["docs.python.org", "pytorch.org", "github.com", "duckduckgo.com"]
```
Web learning seguro (ingest + sanitize):
```yaml
continuous:
  ingest_web: true
  ingest:
    web:
      cooldown_minutes: 60
      sanitize:
        max_chars: 2000
        max_instruction_density: 0.04
        max_repeat_lines: 2
```

## Feedback y entrenamiento
- Cada chat guarda un episodio en `data/episodes/chat.jsonl` con `request_id`.
- Envia feedback con:
```bash
curl -X POST http://localhost:8000/v1/feedback -H "Content-Type: application/json" -d '{"request_id":"<id>","rating":"up","ideal_response":"respuesta ideal"}'
```
- Si hay `ideal_response` y rating="up", se crea un training event en `data/episodes/training.jsonl` y queda disponible para SFT.

## Comandos minimos
```bash
python -m c3rnt2 doctor --profile qwen8b_base
python -m c3rnt2 chat --profile qwen8b_base
python -m c3rnt2 serve --profile qwen8b_base
python -m c3rnt2 ingest-once --profile qwen8b_base
python -m c3rnt2 train-once --profile qwen8b_train
python -m c3rnt2 self-train --once --profile qwen8b_train
```

## Configuración
`config/settings.yaml` define perfiles (`dev_small`, `core_only`, `c3_paged`, `agent`) con parámetros para Vortex, BAD y self-train.

Flags útiles:
- `core.mtp_k`: activa Multi-Token Prediction (MTP).
- `core.compile_step` / `core.compile_local_mixer_step`: compila rutas críticas (experimental).
- `vortex_model.lava_clusters` / `vortex_model.lava_cluster_top`: ANN coarse->fine en LAVA.
- `vortex_model.lava_read_every` / `vortex_model.lava_write_every` / `vortex_model.lava_write_on_surprise`: duty-cycling LAVA.
- `bad.entropy_top_k`: top-k para entropía aproximada.
- `bad.penalty_window`: ventana de repetición.
- `bad.top_p_min_k` / `bad.top_p_max_k`: límites top-k adaptativo para top_p.
- `decode.exact_copy_mode` / `decode.escape_restrict`: control de ESC/bytes.
- `depth_gating.*`: router de profundidad (min/max + hysteresis).
- `c3.pinned_memory` / `c3.prefetch_depth`: transferencia asíncrona.

## Notas
- Este MVP prioriza arquitectura, métricas y exactitud. Rendimiento se optimiza en fases siguientes.
- El repo está diseñado para enchufar un backend HF más grande en el futuro.


## Benchmarks RTX 4080
Generación (BAD decode + cache/prefetch stats):
```bash
python scripts/bench_generate.py --profile rtx4080_16gb_vortexx_next --max-new-tokens 512
```
Microbench de PagedLinear:
```bash
python scripts/bench_paged_linear.py --dtype bf16 --accum-fp32
```

Métricas clave:
- `page_faults`: fallos de caché de tiles (menos es mejor).
- `prefetch_hits`: tiles consumidos desde prefetch sin bloqueo.
- `bytes_h2d`: bytes reales transferidos H2D.
- `bytes_compressed_read`: bytes leídos desde payload comprimido.


Bootstrap con Qwen-8B (teacher 4bit):

python -m c3rnt2 bootstrap --profile rtx4080_16gb_vortexx_next --teacher Qwen/Qwen2.5-8B-Instruct --teacher-quant 4bit --max-prompts 64 --steps 200 --batch-tokens 8192 --reuse-dataset

Qwen-8B HF backend:

python -m c3rnt2 chat --profile qwen8b_base
python -m c3rnt2 serve --profile qwen8b_base


## End-to-End
Router + experts + adapters:
```bash
python -m c3rnt2 train-experts --domains programming
python -m c3rnt2 train-router
python -m c3rnt2 finetune-adapter --adapter data/experts/programming/adapter.pt --data data/corpora/programming
```


## Base 100% (comandos)
```bash
python -m c3rnt2 chat --profile qwen8b_base
python -m c3rnt2 serve --profile qwen8b_base
python -m c3rnt2 bootstrap --profile rtx4080_16gb_vortexx_next --teacher Qwen/Qwen2.5-8B-Instruct --teacher-quant 4bit --max-prompts 64 --steps 200 --batch-tokens 8192 --reuse-dataset
python -m c3rnt2 ingest-once --profile qwen8b_base
python -m c3rnt2 train-once --profile qwen8b_train
python -m c3rnt2 self-patch --goal "Fix failing test" --dry-run
```
