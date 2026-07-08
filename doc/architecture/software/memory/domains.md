# Memory Domains

## Persistent State Memory `M`

`M` is the committed record of real environment interaction.

It stores:

- one complete frame-after-frame state row in `m_states`
- game and run identifiers
- current real observation
- chosen action
- current world, goal, and agent context documents
- agent trace
- committed post-decision world and goal predictions
- frame metadata

`M` is durable and should support inspection, replay, recovery, and
experiments that start from past real states.

At normal run completion, the runtime keeps only the latest `m_states` row per
game. Failed runs keep their rows for debugging.

During an active turn, orchestration may hold current observations, traces,
tool results, and role contexts as live Python objects. This is the in-turn
working state used to pass data between model roles at the right time. It is
not a third memory domain; the committed source of truth remains `M`.

## Experimental Memory `E`

`E` is a rolling memory buffer for experimental tool-output frames. It is also
the reference store for experimental tree paths. The orchestrator agent can ask
orchestration to run `S` or `G` from a prior prediction id rather than carrying
the whole imagined path in context.

It stores:

- tool-produced output observations
- source memory references for the real or experimental input frame
- world and goal tool call metadata
- tool result metadata, explanations, and candidate actions when present

`E` does not copy input frames. Tool inputs point to `M` or to an older `E`
output through `ObservationRef`.

`E` is pruned as a rolling turn buffer. The starter runtime defaults to keeping
the latest 2 frame turns per run and game.

## Separation Rule

`M` stores what actually happened and what was committed into the run history.
`E` stores imagined or temporary artifacts. Both can be read by orchestration
to resolve references, but they must stay separate so the agent does not
confuse observed facts with simulations.

Models do not read or write either memory domain directly. Orchestration
resolves references, passes live objects into model calls, applies model
outputs to the working state, and commits authoritative results back to `M`.
