# Step 04: Cleanup and Tests

## Objective

Remove stale code comments and verify the refactor.

## Implementation

- Remove work-package references from code comments and docstrings.
- Update tests to use the new model registry field names.
- Run import checks and pytest.
- Commit the refactor as a checkpoint.

## Dependencies

Depends on Steps 01 through 03.

## Acceptance Check

- `rg "Work Package|work package" src tests` returns no matches.
- `uv run --locked pytest` passes.
