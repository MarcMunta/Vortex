# Gemma 4 HF Chat Code Outputs

## Verify Gemma HF

Use:

```powershell
curl http://127.0.0.1:8000/v1/status
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/v1/models
```

Expected:

- `engine_kind: hf`
- `active_model: google/gemma-4-E4B-it`
- `active_profile: rtx4080_16gb_programming_gemma4_local`
- no Qwen/SGLang fallback
- no `:30000` URL for HF internal runtime

## Avoid Qwen/SGLang Legacy

Default profile is `rtx4080_16gb_programming_gemma4_local`.
Qwen/SGLang remains manual via Docker profiles `manual` or `qwen-sglang`.
Do not start `sglang-runtime` unless explicitly testing legacy external runtime.

## Token Contract

Frontend sends `max_tokens` on every chat request:

- normal chat: `2048`
- code: `3072`
- complete Flutter/Dart/code: `4096`

Backend applies:

- default max: `2048`
- code max: `3072`
- hard cap: `4096`
- VRAM floor/ceil: `256`/`4096`

Responses include `max_tokens_requested`, `max_tokens_effective`, `backend`, `model`, and `finish_reason`.

## Code Mode

Frontend detects code intent from terms like `codigo`, `implementa`, `crea`, `Flutter`, `Dart`, `widget`, `login`, `completo`, `sin cortar`, `RenderFlex`, and `overflow`.

For Flutter/Dart, prompt envelope requires:

- complete compilable Dart
- ```dart fences
- imports
- full widgets/classes
- `Form`, `GlobalKey<FormState>`, `TextFormField`, validators, controllers, `dispose`
- no `print(password)`
- validation steps: `flutter analyze`, `flutter test`

## Sources, RAG, Web

Simple code generation disables:

- `include_sources`
- `web_ingest`
- `rag_mode`
- `grounding`

Sources/RAG/web only activate when internet is enabled and the prompt asks for sources, docs, official docs, or search.

## Truncated Responses

Frontend flags truncation when:

- backend sends `finish_reason: length`
- Markdown code fence is unclosed
- Dart braces/parens/brackets are unbalanced
- response ends on incomplete Flutter fragments like `controller:`, `children: [`, `validator:`, `onPressed:`, `=>`, `.`, or `,`

UI shows `La respuesta parece cortada.` and button `Continuar respuesta`.
Button sends:

```text
Continua exactamente desde donde lo dejaste. No repitas el codigo anterior. Cierra cualquier bloque de codigo abierto.
```

## Flutter Long-Output Test

```powershell
curl -X POST http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{
    "model": "google/gemma-4-E4B-it",
    "stream": false,
    "include_sources": false,
    "web_ingest": false,
    "max_tokens": 4096,
    "messages": [
      {
        "role": "user",
        "content": "Crea un login básico en Flutter. Dame el código completo y compilable en Dart, con Form, validadores, controllers, dispose, estado loading y ScaffoldMessenger. No lo cortes."
      }
    ]
  }'
```

Expected: complete `dart` block with `main`, `MaterialApp`, login screen, `_formKey`, controllers, validators, `dispose`, safe submit flow, no sources.

## RTX 4080 16 GB

Recommended local Gemma 4 E4B limits:

- default chat: `2048`
- code: `3072`
- complete code: `4096`
- VRAM floor: `256`
- VRAM ceil: `4096`
- safety margin: `768 MB`
