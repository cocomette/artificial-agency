# Running The Runtime

The active runtime is the Kaggle-first Transformers online learner. Runtime
YAML uses `agent:` and rejects legacy `models:` configs.

Create or refresh the local game catalog:

```bash
uv run --no-dev python -m face_of_agi.runtime.shell --list-games
```

Run the local shell harness:

```bash
uv run --group dev python -m face_of_agi.runtime.shell \
  --config src/face_of_agi/runtime/configs/starter_loop.yaml \
  --database runs/memory.sqlite
```

Run the Kaggle entrypoint locally only when the Kaggle gateway sidecar is
available:

```bash
python -m face_of_agi.runtime.kaggle \
  --config src/face_of_agi/runtime/configs/kaggle_transformers.yaml \
  --database-dir /kaggle/working/runs
```

State memory is SQLite-backed. Databases created by older runtime shapes are
not migrated; reset disposable DB files before running this branch when schema
validation reports an obsolete table.

Automated tests use fake backbones and do not load real model weights:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```
