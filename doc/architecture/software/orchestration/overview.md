# Orchestration Overview

Orchestration is the central control layer for the target runtime. It owns the
game loop and is the only module that coordinates environment calls, model
calls, memory reads and writes, updater calls, tool routing, and action
submission.

The orchestrator agent `X` is a model role. The orchestration layer is the
deterministic software controller that calls `X` and handles every side effect
around it.

The top-level `Orchestrator` acts as a facade and dependency coordinator.
Concrete workflows should live in focused sub-orchestration components. The
current ARC game loop is implemented as `GameLoopStateMachine` in
`src/face_of_agi/orchestration/game_loop/state_machine.py`.

The target game-loop state machine is documented in
[`game_loop/overview.md`](game_loop/overview.md). It defines frame-bundle
unrolling, synthetic `NONE` decisions during non-controllable animation frames,
and real environment action submission only on controllable final frames.

## Target Responsibilities

At a high level, orchestration:

- starts and advances one ARC-AGI game run
- receives observations and metadata from the environment module
- composes current role contexts from global `K` and selected-game `L`
- calls the orchestrator agent `X` only on controllable final frames
- resolves observation and world description prediction references from `M` and `E`
- receives the final action and trace from `X`, or synthesizes `NONE` and an
  orchestration-owned trace for animation frames
- runs world predictions after each frame decision, including animation-unroll
  frames with synthetic `NONE`
- sends final-frame real actions to the environment module
- runs updater `P` after each observed frame transition
- applies updater-returned contexts to the live working context documents
- persists frame transitions, traces, predictions, and current contexts into `M`
- prunes rolling experimental memory when it exceeds the configured turn window

## Boundary

Orchestration should depend on typed interfaces from the environment, memory,
models, updates, and shared contracts modules. It should not depend on a
specific model backend or on provider-specific response formats.

When Agent X tools are introduced, orchestration should resolve their memory
references, call the configured tool boundary, and store temporary results in
rolling `E`.

The true main loop now lives here. Runtime can continue to assemble and invoke
orchestration, but it should not grow independent game-step logic.
