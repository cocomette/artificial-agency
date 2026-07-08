# Memory Domains

## Persistent State Memory `M`

`M` is the committed record of real environment interaction and the source
store for orchestration-owned prediction and replay work.

It stores:

- one frame-after-frame state row in `m_states`; the current row is prewritten
  before `X` acts and completed after the turn resolves
- game and run identifiers
- current real observation
- chosen action
- current world, goal, and agent context documents
- agent trace
- world and goal predictions
- transition timing
- score/progress metadata
- frame metadata

`M` is durable and should support inspection, replay, recovery, and
inspection that starts from current or past real states.

At normal run completion, the runtime keeps only the latest `m_states` row per
game. Failed runs keep their rows for debugging.

During an active turn, orchestration may hold current observations, traces,
committed prediction results, and role contexts as live Python objects. This is
the in-turn working state used to pass data between model roles at the right
time. It is not a third memory domain; the committed source of truth remains
`M`.

## Experimental Memory `E`

`E` is a rolling memory buffer reserved for experimental tool-output
descriptions.

It stores:

- future tool-produced output descriptions
- source `m_states.id` values for the real frame used by a future tool
- future tool call metadata
- tool result metadata, explanations, and candidate actions when present

`E` does not copy input frames. Future tool inputs should point to `M`;
`E` ids are not callable tool inputs.

`E` is pruned as a rolling turn buffer. The starter runtime defaults to keeping
the latest 2 frame turns per run and game.

## Separation Rule

`M` stores what actually happened and what was committed into the run history.
`E` stores imagined or temporary artifacts. Orchestration resolves callable
tool sources from `M`; keeping `E` non-callable prevents future tools from
confusing observed facts with simulations.

Models do not read or write either memory domain directly. Orchestration
resolves references, passes live objects into model calls, applies model
outputs to the working state, and commits authoritative results back to `M`.
