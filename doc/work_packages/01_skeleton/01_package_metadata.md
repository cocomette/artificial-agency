# Step 01: Package Metadata

## Objective

Move the project metadata away from the old Ollama notebook starter and toward
the `face_of_agi` package skeleton.

## Implementation

- Update `pyproject.toml` project name to `face-of-agi`.
- Update the project description to describe the ARC-AGI-3 framework skeleton.
- Point Hatch packaging at `src/face_of_agi`.
- Keep the existing Python version and dependencies unchanged unless packaging fails.

## Parallelism

This step can run in parallel with Step 02 because metadata and directory
creation do not overlap.

## Acceptance Check

- `pyproject.toml` no longer packages `src/ollama`.
- Packaging configuration targets `src/face_of_agi`.
