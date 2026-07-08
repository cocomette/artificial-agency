# Orchestration Inputs

Orchestration receives these inputs from assembled dependencies and from each
game step.

## Startup Inputs

- `RuntimeConfig`: run id, optional database path, and selected game ids.
- `EnvironmentConfig`: ARC game selection and environment-local settings.
- `EnvironmentAdapter`: selected ARC-AGI integration boundary.
- `StateMemory`: persistent memory domain `M`.
- `ExperimentalMemory`: rolling experiment buffer `E`.
- `ModelRegistry`: registered model role implementations.
- Initial `ContextDocuments`: role context documents for `S`, `G`, and `X`.

## Per-Step Inputs

- Current `Observation` from the environment module.
- Current `EnvironmentInfo`, including lifecycle state and available actions.
- Current action space.
- Persistent memory records from `M`, including current and past real states.
- Rolling experimental memory records from `E`, for debug/trace inspection.
- Model outputs returned by active `X`, `S`, and `P` roles.

## Model Context Inputs

Orchestration feeds current world context into updater `P`, which summarizes
relevant information into the agent context for later Agent `X` turns. Agent
`X` receives only its composed agent context, observations, action history, and
action space. World receives resolved observations and actions through its
model-role boundary. Goal model contracts remain present but dormant in normal
runtime.
