# Game Loop Interfaces

The game loop exchanges current shared contracts:

- `FrameTurnContext`
- `DecisionResult`
- `AgentTrace`
- `UpdaterFrameTransitionInput`
- `ActionHistoryEntry`
- `AgentContextHistorySummary`
- `GameMemoryDocument`
- `TurnMetrics`

Removed post-decision prediction packets are not part of the active branch.
