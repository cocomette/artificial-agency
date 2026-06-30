# Shared Contracts

## Core Runtime Contracts

- `ActionSpec`: one ARC action plus optional action data.
- `Observation`: normalized frame or frame bundle returned by the environment.
- `EnvironmentInfo`: lifecycle state, available actions, and metadata.
- `RuntimeConfig`: run-level configuration.
- `GameRunResult`: summarized run output.

## Memory Contracts

- `ObservationRef`: reference to a real state record in `M` or an experimental
  record in `E`.
- `MemoryRecord`: generic persisted record.
- `MemoryDomain`: either `state` or `experimental`.

## Agent Contracts

- `ToolCall`: provider-neutral request shape retained for future tools. The
  current runtime exposes no real tools.
- `ToolResult`: provider-neutral tool output shape retained for future tools.
- `AgentTrace`: complete trace of an agent decision step.
- `ActionHistoryEntry`: compact record of one prior frame-turn decision
  exposed to later X prompts.
- `AgentToolRuntime`: controlled per-turn tool boundary exposed to `X`; current
  configs expose an empty tool list.
- `DecisionResult`: final action plus trace.

## Context And Update Contracts

- `RoleContext`: game-agnostic plus game-specific text context for one role.
- `ContextDocuments`: current agent context document.
- `TurnMetrics`: frame-turn cost, trace timing, and score/progress metrics.
- `UpdaterFrameTransitionInput`: observed transition, trace, action, metrics,
  and action-history evidence for updater `P`.
- `AgentGameContextUpdateInput`: updater input for agent game-context updates,
  including previous/current observations, live `AgentTrace`, action history,
  context history, and update quantities.
- `GeneralKnowledgeUpdateInput`: agent general updater input for end-of-run
  `K^X` updates.

## Boundary Rule

Contracts define data shape. Modules still own behavior:

- environment owns ARC communication
- orchestration owns execution and persistence coordination
- models own provider-neutral role implementations
- memory owns SQLite primitives
- updates own post-step context update behavior
