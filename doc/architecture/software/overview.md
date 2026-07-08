# Software Architecture Overview

This folder describes the active software architecture for the current
FACE-OF-AGI runtime branch.

The architecture is modular, but orchestration is the middle man. It owns the
main execution loop, communicates with ARC through the environment adapter,
calls model roles, applies updater output, and coordinates SQLite-backed
memory.

## Module Map

- [`orchestration`](orchestration/overview.md): central runtime controller.
- [`environment`](environment/overview.md): thin ARC-AGI integration boundary.
- [`models`](models/overview.md): provider-neutral model role modules.
- [`memory`](memory/overview.md): SQLite-backed state and experimental memory.
- [`runtime`](runtime/overview.md): startup, config loading, and assembly.
- [`updates`](updates/overview.md): updater behavior.
- [`shared_contracts`](shared_contracts/overview.md): typed cross-module data.
- [`config.md`](config.md): runtime YAML configuration reference.

## Active Runtime Roles

The current branch wires:

- Agent X decision role.
- Transition change-summary role.
- Agent-context historizer role.
- Same-run game memory role.
- Updater P with agent game and general update tasks.

World and goal tool modules are not part of the active runtime in this branch.

## Ownership Rule

The orchestration layer is the only module allowed to coordinate cross-module
side effects during a game step.

That means:

- environment frames flow into orchestration before any model sees them
- Agent X receives only orchestration-built inputs
- change summary, game memory, historizer, and updater calls return to
  orchestration before persistence
- SQLite reads and writes are coordinated by orchestration
- only orchestration submits final actions to ARC
- the main loop is owned by orchestration

The runtime module may start the program and assemble dependencies, but it
should not become a second controller for the game loop.
