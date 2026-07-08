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
- Initial empty v1 ledger and current frame bundle.

## Per-Step Inputs

- Current `Observation` from the environment module.
- Current `EnvironmentInfo`, including lifecycle state and available actions.
- Current action space.
- Persistent memory records from `M`, including current and past real states.
- Rolling experimental memory records from `E`, for debug/trace inspection.
- Model outputs returned by active Agent X, Change Summary, Memory, World,
  Goal, Interest, and Reward Judge roles.

## Model Context Inputs

Orchestration feeds original first/current frames and sanitized
action/change/reward ledger rows into Memory. Goal reads Memory, with an
additional reward-only Goal call before Memory regeneration. Agent X reads
Memory, Goal, current frame, recent reward-bearing action history, valid
actions, and World/Interest candidate tables for the final selection stage.
World and Interest also receive recent action history. Reward Judge receives
the executed World prediction and observed Change Summary.
