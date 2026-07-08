# Step 03: Registry and Runtime Wiring

## Objective

Update deterministic runtime wiring to use the model package layout.

## Implementation

- Update `ModelRegistry` to use `world_tool`, `goal_tool`,
  `orchestrator_agent`, and `updater`.
- Update the deterministic orchestrator to call `require_orchestrator_agent`.
- Keep environment reset-only behavior unchanged.
- Keep the `orchestration` package free of model adapter implementation details.

## Dependencies

Depends on Step 02.

## Acceptance Check

- Runtime smoke test still proves reset-only orchestration with an injected X
  agent model.
