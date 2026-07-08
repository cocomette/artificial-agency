# Step 04: Runtime Boundaries

## Objective

Wire the contracts, memory, model registry, environment adapter, and
orchestrator into a reset-only runtime flow.

## Implementation

- Runtime accepts multiple game ids and injected environment adapters.
- Orchestrator runs one reset-only decision flow for each requested game.
- The flow stores the initial observation in state memory.
- The flow calls the agent role and stores the returned trace.
- The flow does not call `env.step`.
- The flow does not call the updater yet.

## Dependencies

Depends on Steps 02 and 03.

## Acceptance Check

- Runtime can process a fake game through reset and trace persistence.
