# Step 01: Dependencies and Docs

## Objective

Add the dependencies and user-facing setup commands needed for the first real
architecture implementation.

## Implementation

- Add `arc-agi` as a runtime dependency.
- Add `pytest` in a dev dependency group.
- Update `uv.lock`.
- Update README with:
  - `uv sync --group dev`
  - `uv run python -c "import face_of_agi"`
  - `uv run pytest`

## Parallelism

Can run in parallel with Step 02 after this work package exists.

## Acceptance Check

- `pyproject.toml` includes runtime and dev dependencies.
- README documents the new check commands.
