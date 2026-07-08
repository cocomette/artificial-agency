# Model Outputs

## Agent `X`

The orchestrator agent returns:

- one final `ActionSpec`
- `AgentTrace` with reasoning summary and metadata

The shared Agent X loop treats provider tool calls and final structured action
output as alternate results of the same model step: tool calls continue the
loop, while final structured output with no tool calls ends it.

Provider-specific responses are normalized inside the Agent X model layer
before orchestration sees the final `DecisionResult`.

World and goal predictions are consumed by updater/persistence and influence
later X decisions through maintained context.

## World Prediction Model `S`

The world model returns:

- `predicted_description` for the next visual state
- source observation reference
- candidate action
- optional explanation and metadata
- metadata for persistence with the frame turn in `M`

## Goal Prediction Model `G`

The goal model returns:

- `predicted_description` for the goal-relevant visual state
- source observation reference
- optional explanation and metadata
- metadata for persistence with the frame turn in `M`

## Updater `P`

The updater returns revised context documents for the orchestration-selected
task. During frame/game-loop updates, role-specific updater tasks return
game-specific `L` for world, goal, and agent roles. At end-of-run, one shared
general updater task returns game-agnostic `K` through one invocation per
role.
Orchestration applies these outputs to its live `ContextDocuments` working
state, then persists the resulting authoritative contexts into `M`.

## Output Rule

Model outputs are data. Orchestration decides whether they are temporary `E`
artifacts, committed `M` history, active agent context, reusable references, or
updater input.
