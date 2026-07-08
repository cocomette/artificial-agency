# Shared Contracts

Shared contracts define provider-neutral data passed between orchestration,
environment, memory, and model roles.

Current key contracts include:

- `ActionSpec`
- `Observation` and `ObservationRef`
- `FrameControlMode` and `FrameTurnContext`
- `AgentTrace` and `DecisionResult`
- `ActionHistoryEntry`
- `TurnMetrics`
- `RoleContext` and `ContextDocuments`
- `MStateRecord` and `EExperimentRecord`
- `GameRunResult`

Role-specific model contracts live under `src/face_of_agi/models/**`.
