# TODO
- [x] gpu_decompress triton hook + metrics (CPU decompress + H2D overlap)
- [ ] KV 2-bit real (per-channel) y layout optimizado
- [ ] Speculative decoding + draft model
- [~] Integracion TensorRT-LLM opcional (backend stub)
- [x] Router multiobjetivo con logs de mem_cost + estabilidad
- [x] Expert packs entrenables (LoRA/QLoRA) por dominio.

## Estado del push (automatizado)

- **Backup:**: rama `backup/main-before-force-20260129T203153Z` creada y empujada.
- **Push forzado:**: `git push --force origin main` ejecutado (resultado: "Everything up-to-date").
- **Verificacion:**: `origin/main` y `HEAD` apuntan a `1dbe718f4c53a446531673330e677fa6bd8b48f6`.

-- Nota: los cambios locales sin commitear se guardaron en un stash `pre-force-push-20260129T203153Z` antes de la operacion.
