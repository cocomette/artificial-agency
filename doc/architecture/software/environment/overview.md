# Environment Overview

The environment module is the only direct integration boundary with the
ARC-AGI framework. It wraps ARC toolkit objects and exposes observations,
actions, lifecycle state, and metadata to orchestration.

The environment module should stay thin. It should not call models, write
memory, run updates, or own the main loop.

## Target Shape

The environment module exposes:

- game selection
- `reset()`
- `step(action)`
- current action space
- current environment info
- normalized `Observation` objects
- raw ARC metadata when useful

The ARC-AGI framework remains the source of truth for real transitions. The
software architecture should not hard-code game-specific action meanings or
runtime action policy at this boundary.
