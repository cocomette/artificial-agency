# Software Architecture Overview

This folder describes the target software architecture for the ARC-AGI-3
agent runtime. It refines the higher-level direction from
[`../system_architecture.md`](../system_architecture.md) and the concrete stack
choices from [`../techstack.md`](../techstack.md).

The architecture is modular, but not peer-to-peer. The orchestration layer is
the middle man for the running program. It owns the main execution loop,
communicates with the ARC-AGI environment through the environment adapter,
calls the orchestrator agent, routes world and goal model calls as tools, and
reads and writes SQLite-backed memory.

## Module Map

- [`orchestration`](orchestration/overview.md): central runtime controller.
- [`environment`](environment/overview.md): thin ARC-AGI integration boundary.
- [`models`](models/overview.md): provider-neutral model role modules.
- [`memory`](memory/overview.md): SQLite-backed state and experimental memory.
- [`runtime`](runtime/overview.md): startup, config loading, and assembly.
- [`updates`](updates/overview.md): post-step context update behavior.
- [`shared_contracts`](shared_contracts/overview.md): typed cross-module data.
- [`diagrams.md`](diagrams.md): high-level and sequence diagrams.

## Ownership Rule

The orchestration layer is the only module allowed to coordinate cross-module
side effects during a game step.

That means:

- environment frames flow into orchestration before any model sees them
- the orchestrator agent receives tool access through orchestration
- world and goal tool outputs return to orchestration before being stored
- updater context outputs return to orchestration before becoming active
- SQLite reads and writes are coordinated by orchestration, not by model adapters
- only orchestration submits the final action to the ARC-AGI environment
- the main loop is owned by orchestration

`M` is the durable source of truth for committed run state. During a turn,
orchestration may hold live Python objects for the current observations,
trace, tool results, and role contexts. Those objects are the in-turn working
state owned by orchestration; they are not a separate memory domain. When the
turn boundary is reached, orchestration writes the authoritative result back
to `M`.

The orchestrator agent can reason over memory by reference. It does not need to
carry full experimental paths in context. Instead, it can request tool calls
against `ObservationRef` values that point to current or past records in
persistent memory `M`, or to temporary predictions already stored in
experimental memory `E`.

During the experiment loop, every input frame given to `S` or `G` must be
resolved from memory. Even if `X` has predicted frames visible in its active
context, a new tool call passes a memory reference, not an inline frame.
Orchestration resolves that reference to the exact persisted frame or
prediction before calling the tool.

The runtime module may start the program and assemble dependencies, but it
should not become a second controller for the game loop.

## Runtime Shape

At each real environment step, orchestration:

1. reads the current observation and action space from the environment module
2. loads or hydrates the relevant persistent memory `M` and rolling
   experimental memory `E`
3. composes live working contexts for the world, goal, and orchestrator agent
   roles
4. calls the orchestrator agent `X`
5. routes any `S` or `G` tool calls requested by `X`, resolving references
   from `M` or `E`
6. stores tool outputs in `E` and returns referenceable results to `X`
7. receives one final action from `X`
8. runs committed post-decision S/G predictions for updater input
9. submits that action to the ARC-AGI environment
10. invokes the updater role `P` with the live transition, trace, committed
   predictions, current contexts, and update quantities
11. applies updater-returned context documents to the live working contexts
12. persists the real transition, predictions, trace, update quantities, and
   current context documents into `M`
13. clears or expires `E` for the completed step
