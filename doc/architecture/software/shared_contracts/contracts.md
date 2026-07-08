# Shared Contracts

Key active contracts in `face_of_agi.contracts`:

- `Observation` and `ObservationRef`: frame payloads and stable memory refs.
- `ActionSpec`: provider-neutral action values.
- `FrameControlMode` and `FrameTurnContext`: frame-turn model context.
- `ToolCall` and `ToolResult`: generic Agent X tool plumbing, with arbitrary
  `ToolResult.output`.
- `AgentTrace` and `DecisionResult`: Agent X decision output.
- `AgentCandidateAction`: candidate rows for the two-stage Agent flow.
- `MemoryDocument`, `WorldPrediction`, `GoalPrediction`, and
  `RewardJudgeScore`: v1 role outputs.
- `ActionHistoryEntry`: prompt-facing recent action evidence, including
  concise reward and proxy learning-progress feedback when available.
- `TurnLedgerEntry` and `TurnReward`: per-turn evidence plus separated
  prediction accuracy, proxy learning-progress, Goal delta, resource cost, and
  total reward accounting.
- `MStateRecord`: persisted state memory row.
- `TurnMetrics`: timing/progress metrics plus aggregate model token usage for
  a turn.

Contracts should stay provider-neutral and serializable through the memory
helpers.
