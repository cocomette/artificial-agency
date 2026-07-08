# Work Package 03: Model Layout Requirements

## Goal

Move model-specific boundaries out of the global contracts file and into the
model package layout. The deterministic orchestration layer should continue to
coordinate runtime flow, memory, and database boundaries, while model folders
own their own contracts, configs, and adapters.

## Requirements

- Add model subpackages for:
  - world model tool
  - goal model tool
  - orchestrator agent model X
  - updater model P
- Each model subpackage must contain:
  - `contracts.py`
  - `config.py`
  - `adapter.py`
  - `__init__.py`
- Keep global `contracts.py` for shared runtime data only.
- Remove model role Protocols from global `contracts.py`.
- Keep model adapters generic enough to support Ollama first without hard-coding Ollama into the interface.
- Keep the `orchestration` package as deterministic logic only.
- Remove work-package references from code comments and class docstrings.
- Keep smoke tests passing.

## Acceptance Criteria

- Model role contracts import from the corresponding model subpackages.
- `ModelRegistry` wires `world_tool`, `goal_tool`, `orchestrator_agent`, and `updater`.
- Runtime uses `orchestrator_agent`, not a generic global `AgentModel`.
- No code comments mention work packages.
- `uv run --locked pytest` passes.
