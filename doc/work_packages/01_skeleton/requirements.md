# Work Package 01: Skeleton Framework Requirements

## Goal

Create the first `face_of_agi` framework skeleton from the architecture docs.
The result should provide project structure and empty module shells only.

## Source References

- `doc/architecture/arch.md`
- `doc/architecture/techstack.md`
- `doc/project/arc-agi-3_concept.md`
- `doc/project/arc-agi-3_technicals.md`
- `doc/prompts/description.md`

## Requirements

- Use `src/face_of_agi` as the Python package namespace.
- Preserve the architecture split: environment adapter, orchestration, model layer, state memory `M`, experimental memory `E`, context documents, tool routing, updater, and runtime loop.
- Keep every component provider-neutral and backend-neutral.
- Do not choose a VLM, image generator, local runtime, custom network, database schema, or ARC-AGI Toolkit integration detail.
- Do not define detailed method signatures or data contracts yet.
- Keep code minimal, readable, and documented with short docstrings.
- Update README with the package purpose and the `uv` commands needed to install/check the skeleton.
- Run a minimal import check after implementation.
- Commit only the work package and skeleton changes.

## Acceptance Criteria

- `src/face_of_agi` exists and imports successfully.
- `pyproject.toml` packages `src/face_of_agi`.
- The module layout visibly maps to the architecture docs.
- Work package step files exist and can be delegated independently.
- No concrete runtime behavior is implemented.
