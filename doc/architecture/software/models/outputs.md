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

World predictions are consumed by updater/persistence and influence later X
decisions through maintained context. Goal prediction outputs remain part of
the dormant goal contract, but normal runtime leaves them unset.

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

This is a dormant direct-adapter contract in the normal runtime.

## Updater `P`

The updater returns revised context documents for the orchestration-selected
task. During frame/game-loop updates, role-specific updater tasks return
game-specific `L` for world and agent roles. At end-of-run, one shared
general updater task returns game-agnostic `K` through one invocation for world
and one for agent. The goal updater output contract remains available for
direct calls but is not used by normal runtime.
The world game-context updater's provider output is structured as an
`updated_context` map with `world_understanding` plus every action-glossary
key. The updater adapter serializes that map into the world game-context string
before orchestration stores it, so world model input contracts stay unchanged.
The agent game-context updater's provider output is structured as an
`updated_context` map with `goals`, `game_mechanics`, `policy`, `history`, and
`extras`. The updater adapter serializes that map into the agent game-context
string before orchestration stores it, so Agent X still receives its composed
agent context through the existing decision input.
Orchestration applies these outputs to its live `ContextDocuments` working
state, then persists the resulting authoritative contexts into `M`.

## Output Rule

Model outputs are data. Orchestration decides whether they are temporary `E`
artifacts, committed `M` history, active agent context, reusable references, or
updater input.
