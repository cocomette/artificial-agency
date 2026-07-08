# Step 01: Model Subpackages

## Objective

Create model-owned folders for the two tool models, the X agent model, and the
updater model.

## Implementation

- Add `src/face_of_agi/models/tools/world`.
- Add `src/face_of_agi/models/tools/goal`.
- Add `src/face_of_agi/models/orchestrator_agent`.
- Add `src/face_of_agi/models/updater`.
- Add `contracts.py`, `config.py`, `adapter.py`, and `__init__.py` to each.

## Parallelism

Can be implemented before or alongside Step 02.

## Acceptance Check

- Every model role package exposes its contract, config, and adapter names.
