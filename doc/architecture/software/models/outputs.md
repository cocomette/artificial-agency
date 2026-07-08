# Model Outputs

## Agent X

Agent X returns candidate proposals and final decisions:

- `AgentCandidateAction` rows for proposed coordinate actions
- a final `DecisionResult` whose action must match one candidate
- `AgentTrace` metadata for debugging and model-input inspection

## Change Summary

The change summary role returns `ChangeSummaryResult`:

- `summary`
- `changed_pixel_percent`
- `change_detected`
- metadata

## Memory

Memory returns `MemoryDocument`:

- free-form `text`
- optional metadata

## World

World returns `WorldPrediction`:

- candidate `AgentCandidateAction`
- predicted change-summary-style text
- confidence
- metadata

## Goal

Goal returns `GoalPrediction`:

- `goal`
- `subgoals`
- `steps_remaining`
- `confidence`
- metadata

## Reward Judge

Reward Judge returns `RewardJudgeScore`:

- `score`
- `notes`
- `error_tags`
- metadata
