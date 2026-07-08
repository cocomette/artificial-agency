# Persistence Rules

## Read And Write Ownership

Orchestration coordinates all reads and writes to SQLite-backed memory.

Model adapters return data to orchestration. They do not write directly to
or read directly from `M`, `E`, or SQLite.

## Reference Reads

The orchestrator agent may request a tool call against a memory reference.
Orchestration resolves that reference from:

- `M`, for current or past real observations and committed records
- `E`, for rolling tool-output predictions and experimental tree nodes

This lets the agent build branching experiments without keeping the whole path
in its context window.

## Tool Input Invariant

All experiment-loop tool inputs are memory references. `X` may see predicted
frames in its active context, but when it calls `S` or `G`, the source frame is
identified by reference id and resolved by orchestration from persisted memory.
No inline, unpersisted frame should be accepted as a tool input.

## Temporary Writes

Tool outputs generated during deliberation are written to `E` first. The input
side is stored only as an `ObservationRef`; the output side stores the
tool-produced observation/frame directly in `E`. This lets the orchestrator
agent chain world and goal tool calls against intermediate results by
reference id.

The persistence step happens before the result is returned as reusable context.
This guarantees later tool calls branch from the exact persisted E output,
not from a lossy or altered copy in the agent context.

`E` is a rolling buffer. Rows remain available across turns until they fall out
of the configured latest-turn window.

## Committed Writes

After the final real action is applied, orchestration writes the committed
transition to `M`. The current concrete M persistence mechanism is the
`m_states` table: one row per frame turn with the complete frame state,
selected action, current context documents, agent trace, and metadata.

Committed records include:

- current observation
- final action
- next observation
- full agent trace
- committed post-decision world and goal predictions
- selected tool artifacts
- update quantities
- updated contexts

At the start of a game, orchestration reads the latest `m_states` row for that
game and hydrates the starting context documents when one exists. An empty
database is a valid first-run state.

At normal run completion, runtime prunes `m_states` to keep only the newest row
per game. Manual cleanup through `--clean-db` clears memory table rows,
including E and legacy generic rows.

## Promotion From `E` To `M`

Some `E` artifacts may be copied or summarized into `M` as part of the trace.
The original operational boundary remains: `E` is rolling experimental memory,
while `M` is the durable history.
