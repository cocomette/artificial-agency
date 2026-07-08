# Orchestration Overview

Orchestration is the central control layer for the current runtime. It owns the
ARC game loop and is the only module that coordinates environment calls,
model-role calls, updater calls, memory reads/writes, and action submission.

The top-level `Orchestrator` is a facade and dependency coordinator. The frame
loop is implemented by `GameLoopStateMachine` in
`src/face_of_agi/orchestration/game_loop/state_machine.py`.

## Responsibilities

At a high level, orchestration:

- starts and advances one ARC-AGI game run
- receives observations and metadata from the environment module
- composes the active agent context
- calls Agent X only on controllable final frames
- synthesizes `NONE` decisions for animation-unroll frames
- replays known-state simulation rows after Agent X when a prior same-run
  frame/action transition is known
- validates final actions before submitting them to ARC
- summarizes observed transition changes
- updates same-run game memory after controllable real-action turns
- summarizes prior agent context history when enough history exists
- applies updater P to the live agent context
- persists frame transitions, traces, metrics, memory metadata, and contexts
  into `M`
- prunes rolling experimental memory when configured

## Fallback Boundary

The game loop catches only model/provider/output-validation failures for model
roles:

- Agent X falls back to a deterministic legal action.
- Change summary falls back to deterministic visual-difference evidence.
- Historizer falls back to `not_available`.
- Game memory keeps the previous document.
- Updater keeps the previous context.

Programming errors, invalid orchestration state, and non-model exceptions still
raise.

## Known-State Simulation

Known-state simulation is a runtime-only orchestration path. After Agent X
selects an action on a controllable frame, orchestration may match the current
frame hash plus action against prior same-run M rows. A match replays the
historical successor frame and action-history evidence without submitting an
ARC action and without calling the change-summary model. The normal downstream
game-memory, historizer, updater, persistence, and frame-turn debug paths still
run for the simulated row.

When simulation exits, orchestration submits catch-up actions to ARC so the live
environment reaches the simulated endpoint before normal execution resumes. The
simulation path does not change model prompts or model input contracts.

## Boundary

Orchestration depends on typed interfaces from environment, memory, models,
updates, and shared contracts. It should not depend on a specific provider or
provider response shape.
