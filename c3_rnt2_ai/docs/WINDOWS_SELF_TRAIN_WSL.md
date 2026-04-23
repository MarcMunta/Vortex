# Windows self-train HF via WSL2 (modo legado / compatibilidad)

Este repo sigue soportando entrenar LoRA/QLoRA (HF/PEFT) desde Windows usando **WSL2** para evitar picos de VRAM y problemas de compatibilidad.

Ruta principal actual:
- serving diario: `docker compose up -d sglang-runtime vortex-api`
- training principal: `rtx4080_16gb_programming_train_docker`

Usa este documento solo si necesitas mantener el flujo legado `rtx4080_16gb_programming_train_wsl`.

## Requisitos

- Windows 11/10 con WSL2 habilitado.
- Una distro instalada (ej. Ubuntu).
- En WSL: Python 3 + venv + dependencias `hf/train`.

## 1) Instalar WSL2 (PowerShell)

```powershell
wsl --install -d Ubuntu
```

Reinicia si te lo pide.

## 2) Preparar el entorno Python dentro de WSL (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Ve al repo montado desde Windows (ajusta la ruta):

```bash
cd /mnt/c/Users/<YOU>/Vortex/c3_rnt2_ai

python3 -m venv .venv_wsl
source .venv_wsl/bin/activate

python -m pip install -U pip
python -m pip install -e ".[hf,train]"
```

Nota: si tu entrenamiento usa CUDA en WSL, asegúrate de tener WSL GPU habilitado y drivers al día.

## 3) Configurar el perfil (settings manifest)

El perfil recomendado para training local de programacion es `rtx4080_16gb_programming_train_wsl`.

Tambien puedes reutilizar `rtx4080_16gb_120b_like`, pero la ruta soportada para entrenamiento local de programacion ya viene con:

- `server.train_strategy: wsl_subprocess_unload`
- `server.wsl_python: python`
- `server.wsl_workdir: "/mnt/d/Vortex/c3_rnt2_ai"`
- `hf_train.model_name: "google/gemma-4-E4B-it"`
- `hf_train.max_seq_len: 768`
- `hf_train.micro_batch_size: 1`
- `hf_train.grad_accum_steps: 8`
- `hf_train.max_steps: 12`

Para que el subprocess en WSL ejecute el repo correcto, configura `server.wsl_workdir` a la ruta WSL del repo:

```yaml
server:
  wsl_workdir: "/mnt/c/Users/<YOU>/Vortex/c3_rnt2_ai"
```

## 4) Validación (Windows, sin descargas)

```powershell
cd c3_rnt2_ai
python -m c3rnt2 doctor --deep --mock --profile rtx4080_16gb_programming_train_wsl
python -m c3rnt2 train-once --profile rtx4080_16gb_programming_train_wsl
```

## 5) Serve + self-train (Windows -> WSL)

Desde Windows, el loop de self-train descargará/unload el modelo de inferencia y lanzará el entrenamiento dentro de WSL:

```powershell
cd c3_rnt2_ai
python -m c3rnt2 serve-self-train --profile rtx4080_16gb_programming_train_wsl --host 0.0.0.0 --port 8000
```

Si WSL no está disponible, `doctor --deep` y el entrenamiento fallan **fail-closed** con un error claro.

## 6) Entrenar expertos HF por dominio (mock / real)

Mock (no descarga pesos, crea estructura + manifest):

```powershell
python -m c3rnt2 train-hf-experts --profile rtx4080_16gb_120b_like --data data\\corpora --output data\\experts_hf --domains code,docs --mock
```

Real: ejecuta el mismo comando dentro de WSL (o asegúrate de que el entorno HF esté disponible en Windows).

## 7) Promoción manual (120B-like)

Para promover un experto HF desde quarantine a registry, el perfil `rtx4080_16gb_120b_like` requiere un archivo de aprobación:

- `data/APPROVE_PROMOTION`

La función `c3rnt2.continuous.promotion.promote_hf_expert()` aplica `bench_thresholds` y escribe el motivo en:

- `data/logs/promotions.jsonl`
