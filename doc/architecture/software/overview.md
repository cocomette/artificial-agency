# Software Architecture Overview

This folder describes the current software architecture for the ARC-AGI-3
agent runtime. It refines the higher-level direction from
[`../system_architecture.md`](../system_architecture.md) and the concrete stack
choices from [`../techstack.md`](../techstack.md).

The architecture is modular, but not peer-to-peer. The orchestration layer is
the middle man for the running program. It owns the main execution loop,
communicates with the ARC-AGI environment through the environment adapter,
calls model roles, and reads and writes SQLite-backed memory.

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
- model roles receive typed inputs composed by orchestration
- updater context outputs return to orchestration before becoming active
- SQLite reads and writes are coordinated by orchestration, not by model adapters
- only orchestration submits final actions to the ARC-AGI environment
- the main loop is owned by orchestration

`M` is the durable source of truth for committed run state. During a turn,
orchestration may hold live Python objects for current observations, traces,
transition summaries, and role contexts. Those objects are the in-turn working
state owned by orchestration; they are not a separate memory domain. When the
turn boundary is reached, orchestration writes the authoritative result back
to `M`.

The runtime module may start the program and assemble dependencies, but it
should not become a second controller for the game loop.

## Runtime Shape

At each frame turn, orchestration:

1. reads the current observation and action space from the environment module
2. loads or prewrites the current frame state in persistent memory `M`
3. composes live working context for the orchestrator agent role
4. either synthesizes `NONE` for animation-unroll frames or calls Agent `X` on
   controllable final frames
5. receives one final frame action from `X` or the synthetic animation decision
6. submits that action to ARC only on controllable final frames
7. resolves the observed next frame
8. calls the change summary model on the observed transition
9. summarizes recent agent context history when a historizer is configured
10. invokes updater `P` with the live transition, trace, action history, and
    update quantities
11. applies updater-returned context documents to live working context
12. persists the frame transition, trace, metrics, action history entry, and
    current context into `M`
13. clears per-turn transient state and advances to the next frame

The current runtime exposes no real world or goal model providers. Agent tool
contracts remain provider-neutral, but the configured tool list is empty.
