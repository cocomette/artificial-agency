# Shared Contracts

Key active contracts in `face_of_agi.contracts`:

- `Observation` and `ObservationRef`: frame payloads and stable memory refs.
- `ActionSpec`: provider-neutral action values.
- `FrameControlMode` and `FrameTurnContext`: frame-turn model context.
- `ToolCall` and `ToolResult`: generic Agent X tool plumbing, with arbitrary
  `ToolResult.output`.
- `AgentTrace` and `DecisionResult`: frame decision output. In the active loop
  this wraps updater-selected actions; Agent X is dormant.
- `RoleContext` and `ContextDocuments`: agent context documents.
- `UpdaterFrameTransitionInput`: transition evidence for updater/persistence.
- `MStateRecord`: persisted state memory row.
- `TurnMetrics`: timing/progress metrics for a turn.

Contracts should stay provider-neutral and serializable through the memory
helpers.
