# Software Architecture Overview

This folder describes the target software architecture for the ARC-AGI-3
agent runtime. It refines the higher-level direction from
[`../system_architecture.md`](../system_architecture.md) and the concrete stack
choices from [`../techstack.md`](../techstack.md).

The architecture is modular, but not peer-to-peer. The orchestration layer is
the middle man for the running program. It owns the main execution loop,
communicates with the ARC-AGI environment through the environment adapter,
calls the orchestrator agent, runs world and goal model predictions for the
current frame flow, maintains their learned contexts, and reads and writes
SQLite-backed memory.

## Module Map

- [`orchestration`](orchestration/overview.md): central runtime controller.
- [`environment`](environment/overview.md): thin ARC-AGI integration boundary.
- [`models`](models/overview.md): provider-neutral model role modules.
- [`memory`](memory/overview.md): SQLite-backed state and experimental memory.
- [`runtime`](runtime/overview.md): startup, config loading, and assembly.
- [`updates`](updates/overview.md): post-step context update behavior.
- [`shared_contracts`](shared_contracts/overview.md): typed cross-module data.
- [`config.md`](config.md): runtime YAML configuration reference.
- [`diagrams.md`](diagrams.md): high-level and sequence diagrams.

## Ownership Rule

The orchestration layer is the only module allowed to coordinate cross-module
side effects during a game step.

That means:

- environment frames flow into orchestration before any model sees them
- world and goal contexts flow into Agent X and updater inputs through
  orchestration
- world and goal prediction outputs return to orchestration before being stored
- updater context outputs return to orchestration before becoming active
- SQLite reads and writes are coordinated by orchestration, not by model adapters
- only orchestration submits the final action to the ARC-AGI environment
- the main loop is owned by orchestration

`M` is the durable source of truth for committed run state. During a turn,
orchestration may hold live Python objects for the current observations,
trace, committed prediction results, and role contexts. Those objects are the
in-turn working state owned by orchestration; they are not a separate memory
domain. When the turn boundary is reached, orchestration writes the
authoritative result back to `M`.

The runtime module may start the program and assemble dependencies, but it
should not become a second controller for the game loop.

## Runtime Shape

At each frame turn, orchestration:

1. reads the current observation and action space from the environment module
2. loads or hydrates the relevant persistent memory `M` and rolling
   experimental memory `E`
3. composes live working contexts for the world, goal, and orchestrator agent
   roles
4. either synthesizes `NONE` for animation-unroll frames or calls the
   orchestrator agent `X` on controllable final frames
5. receives one final frame action from `X` or the synthetic animation decision
6. runs S/G predictions for updater evidence
7. submits that action to the ARC-AGI environment only on controllable final
   frames; animation-unroll frames use synthetic `NONE`
8. invokes updater role `P` with the live transition, trace, S/G predictions,
   current contexts, transition timing, and score/progress metadata
9. applies updater-returned context documents to the live working contexts
10. persists the frame transition, predictions, trace, transition timing,
   score/progress metadata, and current context documents into `M`
11. clears or expires `E` for the completed step
