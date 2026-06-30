# Memory Domains

## Persistent State Memory `M`

`M` is the committed record of real environment interaction.

It stores:

- one complete frame-after-frame state row in `m_states`
- game and run identifiers
- current real observation
- chosen action or synthetic `NONE`
- current agent context document
- agent trace
- turn metrics, action history evidence, and frame metadata

`M` is durable and supports inspection, replay, recovery, and context hydration.

At normal run completion, the runtime keeps only the latest `m_states` row per
game. Failed runs keep their rows for debugging.

During an active turn, orchestration may hold current observations, traces,
transition summaries, and role contexts as live Python objects. This is the
in-turn working state used to pass data between model roles at the right time.
It is not a third memory domain; the committed source of truth remains `M`.

## Experimental Memory `E`

`E` is a rolling memory buffer reserved for experimental tool-output records.
The current vLLM-only runtime exposes no real model tools and normally writes
no `E` rows.

The domain remains separate so future tool outputs can be referenced without
mixing imagined artifacts into committed state history.

## Separation Rule

`M` stores what actually happened and what was committed into the run history.
`E` stores imagined or temporary artifacts when tools are configured. Both can
be read by orchestration to resolve references, but they must stay separate so
the agent does not confuse observed facts with simulations.

Models do not read or write either memory domain directly. Orchestration
resolves references, passes live objects into model calls, applies model
outputs to the working state, and commits authoritative results back to `M`.
