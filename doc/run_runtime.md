# Running The Runtime

Create or refresh the local game catalog:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

Committed config variants live under `src/face_of_agi/runtime/configs/` for
OpenAI, Ollama, and vLLM. They all use the active role set:

- `models.agent`
- `models.change`
- `models.compacter`
- `models.updater.agent`

Do not run live OpenAI, Ollama, or vLLM calls from automated tests. Provider
adapters should be tested with fake clients.

State memory is SQLite-backed. Databases created by older runtime shapes are
not migrated; reset disposable DB files before running this branch when schema
validation reports an obsolete table.
