# Tests

- `suites/`: model-free pytest regression suites for the current runtime.
- `e2e/`: manual runners that may call live model or environment paths.
- `fixtures/`: shared test and E2E inputs.

Run the current regression suite from the repo root:

```bash
uv run pytest tests/suites -q
```
