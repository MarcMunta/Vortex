# Vortex backend

Backend local para Llama 2 chat, agente, skills y RAG de soporte.

## Perfil unico

`rtx4080_16gb_llama2_7b_q4_local`

Modelo:

`data/models/gguf/llama-2-7b-chat.Q4_K_M.gguf`

## Checks

```powershell
python -m vortex doctor --profile rtx4080_16gb_llama2_7b_q4_local
python -m pytest tests\test_support_chatbot.py tests\test_settings_normalization.py
python -m vortex support-chatbot --question "no teng internat" --show-chunks
```

## Agent tool training

El modo agente usa una gramatica JSON en llama.cpp para que el modelo solo pueda devolver acciones de herramienta validas.

Dataset SFT:

```powershell
python -m vortex build-agent-tool-dataset --dataset-out config/datasets/agent_tool_use_sft.jsonl
```

Entrenamiento LoRA/QLoRA:

```powershell
python -m vortex train-agent-tools --profile rtx4080_16gb_llama2_7b_q4_local --steps 80 --local-files-only
```

El adaptador queda en `data/registry/hf_train`. La promocion es automatica por defecto si pasan los gates de eval/bench; usa `--manual-promotion` para revisar antes de activar.

Detalles de practica y entrega: ver `..\README.md`.
