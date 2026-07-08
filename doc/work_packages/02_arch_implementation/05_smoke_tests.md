# Step 05: Smoke Tests

## Objective

Verify that the new boundaries are usable without real ARC environments or
model backends.

## Implementation

- Add pytest smoke tests with fake environment and fake agent classes.
- Verify contract imports.
- Verify SQLite table initialization and generic record persistence.
- Verify reset-only runtime orchestration.
- Confirm `env.step` is not called.

## Dependencies

Depends on Steps 01 through 04.

## Acceptance Check

- `uv run pytest` passes.
