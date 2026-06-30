# Orchestration Inputs

Orchestration receives these inputs from assembled dependencies and from each
game step.

## Startup Inputs

- `RuntimeConfig`: run id, optional database path, and selected game ids.
- `EnvironmentConfig`: ARC game selection and environment-local settings.
- `EnvironmentAdapter`: selected ARC-AGI integration boundary.
- `StateMemory`: persistent memory domain `M`.
- `ExperimentalMemory`: rolling experiment frame buffer `E`.
- `ModelRegistry`: registered model role implementations.
- Initial `ContextDocuments`: role context documents for `S`, `G`, and `X`.

## Per-Step Inputs

- Current `Observation` from the environment module.
- Current `EnvironmentInfo`, including lifecycle state and available actions.
- Current action space.
- Persistent memory records from `M`, including current and past real states.
- Rolling experimental memory records from `E`, including prior tool outputs.
- Model outputs returned by `X`, `S`, `G`, and `P`.

## Agent Tool Inputs

When `X` asks to call a tool, orchestration receives a `ToolCall` containing:

- tool name: `world` or `goal`
- `ObservationRef` pointing to state or experimental memory
- candidate `ActionSpec` for world calls only

The reference may identify a current state, a past real state, or a prior
experimental prediction. Orchestration resolves the reference before calling
the tool model.

Tool calls do not carry inline frames as model inputs. The frame or prediction
used by `S` or `G` must be the exact object resolved from `M` or `E` by
orchestration.
