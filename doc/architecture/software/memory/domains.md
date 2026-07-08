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
- metadata, including the SHA-256 `current_frame_hash` for the frame after the
  same ARC-edge crop used by the change-summary input
- per-turn compacter context metadata when the compacter runs

Source rows may be prewritten before the frame decision is resolved and
completed after the observed transition is resolved.
The completed row preserves the prewritten frame hash and crop edges. Same-run
repeated-state lookups scan prior complete rows for that cropped-frame hash and
return rows that also have a persisted updater strategy snapshot.
Completed-level compacter summaries are stored in
`compacter_level_summaries` and carry only the latest solved level's compact
strategy summary forward as previous-level context.
Known-state simulation persists synthetic M rows with `simulated: true` in
metadata. Those rows remain available to agent-facing same-state history, but
simulation transition-edge lookup excludes them so replay edges always come from
real environment observations.

## `E` Experimental Memory

`E` stores generic tool invocations and outputs for the dormant Agent X adapter
path. It is pruned by run/game/turn buffer settings and is not part of learned
context hydration.
