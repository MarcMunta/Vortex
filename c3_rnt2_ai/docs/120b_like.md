# Perfil `rtx4080_16gb_120b_like` (“120B-like”)

<!-- markdownlint-disable MD013 -->

## ¿Qué significa “120B-like” aquí?

Este repo usa el término **“120B-like”** como shorthand práctico:

- **Base pequeña (7B) cuantizada** (HF 4-bit/8-bit cuando aplica o GGUF con `llama.cpp`) para caber en **RTX 4080 16GB**.
- **Banco de expertos LoRA/PEFT** (muchos adapters especializados, entrenados incrementalmente).
- **Router `top-k`** para elegir los *k* mejores expertos por prompt.
- **Mezcla ponderada** (*weighted mix*) cuando `top_k > 1` para aproximar un modelo grande mediante combinación de expertos.
- **Budgets + métricas + gates**: límites duros de VRAM/ctx y umbrales de rendimiento para no degradar con el tiempo.

El objetivo es tener una experiencia “tipo 120B” (más cobertura/competencia por dominio) **sin** cargar un modelo 120B real.

## Seguridad por defecto (deny-by-default)

El perfil está diseñado para ser **offline por defecto**:

- `tools.web.enabled: false`
- `security.web.strict: true`
- `security.web.allowlist_domains: []`

Para habilitar web/ingesta, añade allowlist explícita (y entiende el riesgo).

## Instalación (Windows)

```bash
cd c3_rnt2_ai
python -m venv .venv
.venv\Scripts\activate

python -m pip install -e .
python -m pip install -e ".[api,hf,experts]"
# Opcional (según entorno):
python -m pip install -e ".[train,llama_cpp]"
```

## Doctor (sin bajar pesos)

Mock (CI/local sin descargas):

```bash
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_120b_like
```

Real (requiere que los pesos estén ya disponibles localmente si usas HF):

```bash
python -m c3rnt2 doctor --deep --profile rtx4080_16gb_120b_like
```

## Bench (reproducible + JSON)

Mock (sin pesos):

```bash
python -m c3rnt2 bench --mock --profile rtx4080_16gb_120b_like --max-new-tokens 16 --json-out data\bench\last_120b_like.json
```

Real:

```bash
python -m c3rnt2 bench --profile rtx4080_16gb_120b_like --max-new-tokens 64 --json-out data\bench\last_120b_like.json
```

El JSON incluye `tokens_per_sec`, `prefill_tokens_per_sec`, `decode_tokens_per_sec`, `vram_peak_mb` (o `null`), `backend`, `active_adapters` y `adapter_load_ms`.

### Baseline (regresión cerrada por defecto)

Para crear/actualizar baseline de rendimiento:

```bash
python -m vortex bench --profile rtx4080_16gb_120b_like --max-new-tokens 64 --update-baseline
```

En modo real (sin `--mock`), `doctor --deep` para el perfil 120B-like exige que exista baseline.

## Prepare-model (Windows fail-safe, sin red)

Antes de cargar HF en Windows, valida la ruta segura (cuantización u offload) o fuerza `llama.cpp` si hay GGUF:

```bash
python -m vortex prepare-model --profile rtx4080_16gb_120b_like
```

Escribe `data/models/prepared_rtx4080_16gb_120b_like.json`.

## Serve

```bash
python -m c3rnt2 serve --profile rtx4080_16gb_120b_like --host 0.0.0.0 --port 8000
```

## Serve + self-train (WSL2 / wsl_subprocess_unload)

```bash
python -m c3rnt2 serve-self-train --profile rtx4080_16gb_120b_like --host 0.0.0.0 --port 8000
```

En Windows, el perfil usa por defecto `server.train_strategy: wsl_subprocess_unload` (ver `docs/WINDOWS_SELF_TRAIN_WSL.md`).

## Entrenar expertos HF por dominio (PEFT)

Mock (sin descargar pesos; genera estructura + `manifest.json` por dominio):

```bash
python -m c3rnt2 train-hf-experts --profile rtx4080_16gb_120b_like --data data\\corpora --output data\\experts_hf --domains code,docs --mock
```

## Promoción manual (fail-closed)

Para el perfil 120B-like, la promoción de expertos HF requiere un archivo de aprobación:

- `data/APPROVE_PROMOTION`

La promoción aplica gates (`bench_thresholds`) y deja motivo en `data/logs/promotions.jsonl`.

## Notas de rendimiento (HF vs llama.cpp)

El perfil arranca con `core.backend: hf`, pero:

- si **HF** falla por OOM/carga, puede hacer **fallback** a `llama_cpp` **si** hay GGUF disponible.
- si quieres **preferir** `llama.cpp` cuando exista GGUF, configura `core.llama_cpp_model_path` a un `.gguf` real.
- `doctor --deep` para el perfil 120B-like falla si no hay cuantización real activa (HF sin `bitsandbytes` y sin offload seguro, o sin GGUF).

## Checklist de aceptación (local)

- ✅ `pip install -e .` y `pip install -e .[api]`
- ✅ `pytest -q` (sin descargas)
- ✅ `python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_120b_like`
- ✅ `python -m c3rnt2 bench --mock --profile rtx4080_16gb_120b_like --max-new-tokens 16`
- ✅ Gating bloquea promoción si no cumple `bench_thresholds` y deja motivo en `data/logs/`
