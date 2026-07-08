# End-To-End Checks

The active E2E path is the Kaggle submission notebook or local shell with real
bundled Transformers weights. Automated CI does not run these checks.

Local real-weight smoke run:

```bash
uv run --group dev python -m face_of_agi.runtime.shell \
  --config src/face_of_agi/runtime/configs/starter_loop.yaml \
  --database runs/memory.sqlite
```

Kaggle notebook build smoke:

```bash
cd kaggle
make notebook
```

Run the full model-free suite before any real-weight E2E:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```
