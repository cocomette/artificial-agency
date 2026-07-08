# Shared Contracts

## Core Runtime Contracts

- `ActionSpec`: one ARC action plus optional action data.
- `Observation`: normalized frame or frame bundle returned by the environment.
- `EnvironmentInfo`: lifecycle state, available actions, and metadata.
- `RuntimeConfig`: run-level configuration.
- `GameRunResult`: summarized run output.

## Memory Contracts

- `ObservationRef`: reference used for observed state/update surfaces.
- `MemoryDomain`: either `state` or `experimental`.

## Tool And Agent Contracts

- `PredictionCall`: orchestration-owned prediction request shape.
- `DescriptionPrediction`: structured prediction payload; a top-level array of areas
  with `bbox_2d` coordinate arrays in `[x0, y0, x1, y1]` order and a concise
  `description`.
- `PredictionResult`: output from description prediction roles;
  it carries `predicted_description` and source metadata.
- `ToolCall` / `ToolResult`: compatibility aliases for the Agent X tool-loop
  and existing E schema.
- `AgentTrace`: complete trace of an agent decision step.
- `ActionHistoryEntry`: compact prompt-facing record of one prior frame-turn
  action and whether that turn was controllable.
- `AgentToolRuntime`: controlled per-turn extension point exposed to `X`.
- `DecisionResult`: final action plus trace.
- `PostDecisionPredictions`: description predictions produced by orchestration
  for updater evidence. Normal runtime sets the world prediction and leaves the
  dormant goal prediction unset.

## Context And Update Contracts

- `RoleContext`: game-agnostic plus game-specific text context for one role.
- `ContextDocuments`: current role contexts for world, goal, and agent.
- `TurnMetrics`: frame-turn action cost, load-adjusted decision duration, and
  score/progress metadata assembled by orchestration for persistence and update
  boundaries.
- `cumulative_score`: cumulative environment progress metadata when available.
- `agent_context_word_count`: compactness feedback for Agent X's prompt updater;
  lower values are better when strategy quality is preserved.
- `WorldGameContextUpdateInput`: active updater input for world game-context
  updates. Its prompt updater surface is the previous role context, committed
  world post-decision description prediction, selected real action or synthetic
  `NONE` action, and attached current observation frame. The dormant
  `GoalGameContextUpdateInput` contract remains available for direct adapter
  calls but is not used by the normal runtime loop.
  World game updater provider output returns `updated_context` as a complete
  `world_understanding` plus action-effect map, which the adapter serializes
  into the world game-context string.
- `AgentGameContextUpdateInput`: updater input for agent game-context updates,
  including the previous agent context, previous and current observed frames,
  compact action history, timing, and score progress metadata. Full
  `AgentTrace` is persisted separately and is not copied into the agent updater
  prompt. Agent game updater provider output returns `updated_context` as a
  complete `goals`, `game_mechanics`, `policy`, `history`, and `extras` map,
  which the adapter serializes into the agent game-context string.
- `GeneralKnowledgeUpdateInput`: shared general updater input for one role's
  end-of-run `K` update.

## Boundary Rule

Contracts define data shape. Modules still own behavior:

- environment owns ARC communication
- orchestration owns execution and persistence coordination
- models own provider-neutral role implementations
- memory owns SQLite primitives
- updates own post-step context update behavior
