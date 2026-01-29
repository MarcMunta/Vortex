# VORTEX-X (C3 + RNT-2 + LAVA + BAD)

Este repo implementa un prototipo modular para una “IA 120B-like” local combinando:
- **C3**: pesos paginados y comprimidos con caché coherente.
- **VORTEX-Tok**: tokenización reversible de tasa variable (macro/patch/ESC).
- **V-Blocks**: LocalMixer + SSM + LAVA Memory + GatedMLP.
- **BAD**: Blockwise Adaptive Decoding (draft+verify).
- **Agente**: herramientas, memoria persistente, auto-entrenamiento y auto-mejora segura.

## Instalación
```bash
python -m venv .venv
. .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -e .
```

## CLI (VORTEX-X)
Comandos principales via `python -m c3rnt2`:
```bash
python -m c3rnt2 tokenizer-train
python -m c3rnt2 eval
python -m c3rnt2 chat
python -m c3rnt2 agent-demo
python -m c3rnt2 self-train --once
python -m c3rnt2 self-improve
python -m c3rnt2 apply-patch --diff data/selfimprove/runs/<id>/proposed.diff --approve
```

## Tokenizador VORTEX-Tok
Entrenamiento MVP (patch codebook + macro codebook opcional):
```bash
python -m c3rnt2 tokenizer-train --corpus data/corpora
```

## VORTEX-X Core
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
Generaci?n r?pida (BAD decode + stats):
```bash
python scripts/bench_generate.py --profile dev_small
```
Salida reciente en `data/bench/latest.txt`.

## Configuraci?n
`config/settings.yaml` define perfiles (`dev_small`, `core_only`, `c3_paged`, `agent`) con par?metros para VORTEX, BAD y self-train.

Flags ?tiles:
- `core.mtp_k`: activa Multi-Token Prediction (MTP).
- `core.compile_step` / `core.compile_local_mixer_step`: compila rutas cr?ticas (experimental).
- `vortex_model.lava_clusters` / `vortex_model.lava_cluster_top`: ANN coarse->fine en LAVA.
- `vortex_model.lava_read_every` / `vortex_model.lava_write_every` / `vortex_model.lava_write_on_surprise`: duty-cycling LAVA.
- `bad.entropy_top_k`: top-k para entrop?a aproximada.
- `bad.penalty_window`: ventana de repetici?n.
- `bad.top_p_min_k` / `bad.top_p_max_k`: l?mites top-k adaptativo para top_p.
- `decode.exact_copy_mode` / `decode.escape_restrict`: control de ESC/bytes.
- `depth_gating.*`: router de profundidad (min/max + hysteresis).
- `c3.pinned_memory` / `c3.prefetch_depth`: transferencia as?ncrona.

## Notas
- Este MVP prioriza arquitectura, métricas y exactitud. Rendimiento se optimiza en fases siguientes.
- El repo está diseñado para enchufar un backend HF más grande en el futuro.
