# Model Outputs

## Agent `X`

The orchestrator agent returns:

- one final `ActionSpec`
- `AgentTrace` with reasoning summary, optional tool call/result records, and
  metadata

The current runtime exposes no real tools to `X`, so normal traces contain no
tool results. Provider-specific responses are normalized inside the Agent X
model layer before orchestration sees the final `DecisionResult`.

## Change Summary

The change model returns:

- concise transition summary text
- structured change fields
- cropped changed-cell count
- provider metadata

Orchestration uses this output to build compact action history and updater
input.

## Agent Context Historizer

The historizer returns a structured summary of recent agent context evolution
over the fields `goals`, `game_mechanics`, `policy`, `history`, and `extras`.

## Updater `P`

The updater returns revised context documents for the orchestration-selected
task.

- agent game updater replaces `RoleContext.game`
- agent general updater replaces `RoleContext.general`

Orchestration applies these outputs to its live `ContextDocuments` working
state, then persists the resulting authoritative contexts into `M`.

## Output Rule

Model outputs are data. Orchestration decides whether they are committed `M`
history, active agent context, action history evidence, or updater input.
