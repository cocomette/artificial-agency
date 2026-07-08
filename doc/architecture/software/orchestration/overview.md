# Orchestration Overview

Orchestration is the central control layer for the target runtime. It owns the
game loop and is the only module that coordinates environment calls, model
calls, tool routing, memory reads and writes, updater calls, and action
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
- composes current role contexts
- calls the orchestrator agent `X`
- exposes world `S` and goal `G` models as tools to `X`
- passes `X` a per-turn `AgentToolRuntime` for controlled tool requests
- resolves observation and prediction references from `M` and `E`
- stores tool output frames in rolling `E` and exposes their references back to
  `X`
- receives the final action and trace from `X`
- sends the final action to the environment module
- runs updater `P` after each real environment step
- applies updater-returned contexts to the live working context documents
- persists real transitions, traces, and current contexts into `M`
- prunes rolling experimental memory when it exceeds the configured turn window

## Boundary

Orchestration should depend on typed interfaces from the environment, memory,
models, updates, and shared contracts modules. It should not depend on a
specific model backend or on provider-specific response formats.

The orchestrator agent can ask to reuse prior memory records by id. The
orchestration layer resolves those references, calls the requested tool with
the referenced observation or prediction, and stores the new result back into
rolling `E` unless it is later committed into `M`.

The true main loop now lives here. Runtime can continue to assemble and invoke
orchestration, but it should not grow independent game-step logic.
