# Test Suite

Use the test suite for model-free regression coverage. It should be fast,
deterministic, and safe to run without hosted credentials, Ollama, or local
model weights.

## Setup

From the repo root:

```bash
uv sync --group test --no-dev
```

The full development environment also includes the test dependency group:

```bash
uv sync --group dev
```

## Run

Run the regression suite used by pull request checks:

```bash
uv run --locked --group test --no-dev python -m pytest -q
```

Run a focused test file while developing:

```bash
uv run --locked --group test --no-dev python -m pytest tests/suites/test_runtime_smoke.py -q
```

GitHub Actions runs the full model-free command automatically for pull
requests. This gate uses only the lightweight `test` dependency group and does
not call OpenAI, Ollama, local model backends, or manual E2E runners in
`tests/e2e/`.
