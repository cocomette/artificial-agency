# Running The Runtime

Create or refresh the local game catalog:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

Run the starter config:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --config src/face_of_agi/runtime/configs/starter_loop.yaml
```

Committed v1 runtime configs live under `src/face_of_agi/runtime/configs/`.
The active role set is vLLM-only:

- `models.agent`
- `models.change`
- `models.memory`
- `models.world`
- `models.goal`
- `models.reward_judge`
- optional `models.shared_vlm`

`models.historizer` and `models.updater` are removed keys and fail config
loading. Use the `vllm/` configs for real runs; legacy-named config folders are
kept only where tests still exercise config loading.

Do not run live model calls from automated tests. Provider adapters should be
tested with fake clients.

State memory is SQLite-backed. Databases created by older runtime shapes are
not migrated; reset disposable DB files before running this branch when schema
validation reports an obsolete table.
