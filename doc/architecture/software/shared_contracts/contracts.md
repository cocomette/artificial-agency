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

- `PredictionCall`: orchestration-owned world/goal prediction request shape.
- `DescriptionPrediction`: S/G prediction payload; a top-level array of areas
  with `bbox_2d` coordinate arrays in `[x0, y0, x1, y1]` order and a concise
  `description`.
- `PredictionResult`: output from world/goal prediction roles;
  it carries `predicted_description` and source metadata.
- `ToolCall` / `ToolResult`: compatibility aliases for the Agent X tool-loop
  and existing E schema.
- `AgentTrace`: complete trace of an agent decision step.
- `ActionHistoryEntry`: compact prompt-facing record of one prior frame-turn
  action and whether that turn was controllable.
- `AgentToolRuntime`: controlled per-turn extension point exposed to `X`.
- `DecisionResult`: final action plus trace.
- `PostDecisionPredictions`: world and goal description predictions produced
  by orchestration for updater evidence.

## Context And Update Contracts

- `RoleContext`: game-agnostic plus game-specific text context for one role.
- `ContextDocuments`: current role contexts for world, goal, and agent.
- `TurnMetrics`: frame-turn action cost, load-adjusted decision duration, and
  score/progress metadata assembled by orchestration for persistence and update
  boundaries.
- `score_delta`: raw environment progress metadata when available.
- `WorldGameContextUpdateInput` and `GoalGameContextUpdateInput`: updater
  inputs for world or goal game-context updates. Their prompt updater surface is
  the previous role context, the committed role-specific post-decision
  description prediction, and attached previous/current observation frames; the
  world updater also receives the selected real action or synthetic `NONE`
  action.
- `AgentGameContextUpdateInput`: updater input for agent game-context updates,
  including the previous agent context, previous and current observed frames,
  live `AgentTrace`, committed description predictions, timing, and score
  progress metadata.
- `GeneralKnowledgeUpdateInput`: shared general updater input for one role's
  end-of-run `K` update.

## Boundary Rule

Contracts define data shape. Modules still own behavior:

- environment owns ARC communication
- orchestration owns execution and persistence coordination
- models own provider-neutral role implementations
- memory owns SQLite primitives
- updates own post-step context update behavior
