# Model Outputs

## Agent `X`

The orchestrator agent returns:

- one final `ActionSpec`
- `AgentTrace` with reasoning summary, tool calls, tool results, and metadata

The agent can request tool calls during deliberation, but orchestration handles
the actual routing, memory reference resolution, and persistence.

Provider-specific responses are normalized inside the Agent X model layer
before orchestration sees the final `DecisionResult`.

Predicted frames can be included in the active context returned to `X`, but the
reusable identity of a prediction is the persisted reference id assigned by
orchestration.

## World Tool `S`

The world tool returns:

- predicted next observation
- source observation reference
- candidate action
- optional explanation and metadata
- result reference id after orchestration stores it in `E`

## Goal Tool `G`

The goal tool returns:

- goal-relevant prediction or desired observation
- source observation reference
- optional explanation and metadata
- result reference id after orchestration stores it in `E`

## Updater `P`

The updater returns revised game-specific context documents for world, goal,
and agent roles. Orchestration applies these outputs to its live
`ContextDocuments` working state, then persists the resulting authoritative
contexts into `M`.

## Output Rule

Model outputs are data. Orchestration decides whether they are temporary `E`
artifacts, committed `M` history, active agent context, reusable references, or
updater input.
