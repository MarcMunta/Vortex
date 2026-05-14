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

Detalles de practica y entrega: ver `..\README.md`.
