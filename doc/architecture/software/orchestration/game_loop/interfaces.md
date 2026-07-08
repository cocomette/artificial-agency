# Game Loop Interfaces

Important shared contracts:

- `FrameTurnSnapshot`: immutable in-loop snapshot for one frame turn.
- `FrameTurnContext`: provider-neutral model input context derived from the
  snapshot.
- `AgentCandidateAction`: one candidate in the two-stage action pipeline.
- `WorldPrediction`: predicted transition text for one candidate action.
- `RewardJudgeScore`: text-comparison score for executed World prediction.
- `TurnReward`: separated prediction accuracy, optional delayed LP, Goal
  delta, progress bonus, resource cost, weights, and total reward.
- `MemoryDocument`: regenerated free-form run memory.
- `GoalPrediction`: structured goal and remaining-step estimate.
- `TurnLedgerEntry`: durable per-turn action/change/reward evidence.
- `LoRAUpdateRecord`: persisted online update attempt state.
- `MStateRecord`: durable state row with agent context, trace, metrics, and
  metadata.

The game loop passes typed objects between action modules. Persistence converts
them to JSON only at the memory boundary.
