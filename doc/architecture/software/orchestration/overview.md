# Orchestration Overview

Orchestration is the central control layer for the runtime. It owns the game
loop and is the only module that coordinates environment calls, model calls,
memory reads and writes, updater calls, and action submission.

The orchestrator agent `X` is a model role. The orchestration layer is the
deterministic software controller that calls `X` and handles every side effect
around it.

The top-level `Orchestrator` acts as a facade and dependency coordinator.
Concrete workflows live in focused sub-orchestration components. The current
ARC game loop is implemented as `GameLoopStateMachine` in
`src/face_of_agi/orchestration/game_loop/state_machine.py`.

The game loop defines frame-bundle unrolling, synthetic `NONE` decisions during
non-controllable animation frames, and real environment action submission only
on controllable final frames.

## Responsibilities

At a high level, orchestration:

- starts and advances one ARC-AGI game run
- receives observations and metadata from the environment module
- composes current agent context from global `K^X` and selected-game `L^X`
- calls Agent `X` only on controllable final frames
- exposes an `AgentToolRuntime` with no available tools in the current runtime
- receives the final action and trace from `X`, or synthesizes `NONE` and an
  orchestration-owned trace for animation frames
- sends final-frame real actions to the environment module
- resolves the actual next frame for transition evidence
- calls the change summary model for compact transition text
- calls the agent context historizer when configured
- runs updater `P` after each observed frame transition
- applies updater-returned agent context to live working context documents
- persists frame transitions, traces, metrics, action history, and context into
  `M`

## Boundary

Orchestration depends on typed interfaces from the environment, memory, models,
updates, and shared contracts modules. It does not depend on a specific model
backend or provider-specific response format.

The true main loop lives here. Runtime can assemble and invoke orchestration,
but it should not grow independent game-step logic.
