# Step 04: Docs and Checks

## Objective

Document the skeleton and verify that it imports.

## Implementation

- Update `README.md` with the package purpose.
- Document the existing `uv` workflow for setup and import checking.
- Run a minimal import check:

```bash
uv run python -c "import face_of_agi"
```

- Commit the work package and skeleton files after checks pass.

## Dependencies

Depends on Steps 01, 02, and 03.

## Acceptance Check

- README reflects the new skeleton.
- Import check passes.
- A git commit captures the skeleton work without staging unrelated changes.
