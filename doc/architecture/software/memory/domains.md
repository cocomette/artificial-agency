# Memory Domains

## `M` State Memory

`M` is the durable source of truth for completed frame turns. Each complete
row contains:

- game/run identifiers and frame indexes
- current observation payload
- chosen action
- agent context
- decision trace
- turn metrics
- metadata

Source rows may be prewritten before the frame decision is resolved and
completed after the observed transition is resolved.

## `E` Experimental Memory

`E` stores generic tool invocations and outputs for the dormant Agent X adapter
path. It is pruned by run/game/turn buffer settings and is not part of learned
context hydration.
