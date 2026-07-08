# Shared Contracts

## Core Runtime Contracts

- `ActionSpec`: one ARC action plus optional action data.
- `Observation`: normalized frame or frame bundle returned by the environment.
- `EnvironmentInfo`: lifecycle state, available actions, and metadata.
- `RuntimeConfig`: run-level configuration.
- `GameRunResult`: summarized run output.

## Memory Contracts

- `ObservationRef`: reference to a real state record in `M` or an experimental
  prediction record in `E`.
- `MemoryRecord`: generic persisted record.
- `MemoryDomain`: either `state` or `experimental`.

## Tool And Agent Contracts

- `ToolCall`: request from `X` to call `S` or `G`; world calls include an
  action, goal calls do not.
- `ToolResult`: output from a world or goal tool call; it becomes reusable
  only after orchestration persists it and returns a reference id.
- `AgentTrace`: complete trace of an agent decision step.
- `AgentToolRuntime`: controlled per-turn tool boundary exposed to `X`.
- `DecisionResult`: final action plus trace.
- `PostDecisionPredictions`: committed world and goal predictions produced by
  orchestration after `X` chooses a real action.

## Context And Update Contracts

- `RoleContext`: game-agnostic plus game-specific text context for one role.
- `ContextDocuments`: current role contexts for world, goal, and agent.
- `RewardUpdateQuantities`: structured update signals for updater `P`.
- `ToolContextUpdateInput`: updater input for world or goal context updates,
  including the previous role context, transition references, committed
  post-decision predictions, matching live tool results, and update quantities.
- `AgentContextUpdateInput`: updater input for agent context updates,
  including the previous agent context, live `AgentTrace`, transition
  references, committed post-decision predictions, and update quantities.

## Boundary Rule

Contracts define data shape. Modules still own behavior:

- environment owns ARC communication
- orchestration owns execution and persistence coordination
- models own provider-neutral role implementations
- memory owns SQLite primitives
- updates own post-step context update behavior
