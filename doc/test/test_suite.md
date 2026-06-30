# Test Suite

Use the test suite for model-free regression coverage. It should be fast,
deterministic, and safe to run without hosted credentials or a live vLLM
server.

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
not call external model APIs or a live vLLM server.

The focused coverage includes `ObservationText` serialization, vLLM request
assembly with text-only payloads, runtime config rejection of removed
backends, ACTION6 ARC-grid coordinates, SQLite smoke coverage, and dashboard
config validation.
