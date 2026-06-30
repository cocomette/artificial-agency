# Persistence Rules

## Read And Write Ownership

Orchestration coordinates all reads and writes to SQLite-backed memory.

Model adapters return data to orchestration. They do not write directly to or
read directly from `M`, `E`, or SQLite.

## Reference Reads

The current runtime uses `M` references for observed frame turns and context
hydration. `E` remains available for future tool-output records but is not
written by the default vLLM-only path.

## Committed Writes

After each frame decision, orchestration writes the committed transition to
`M`. The current concrete M persistence mechanism is the `m_states` table: one
row per frame turn with the complete frame state, selected action, current
agent context, agent trace, turn metrics, and metadata.

Committed records include:

- current observation
- final action or synthetic `NONE`
- actual next observation reference when available
- full agent trace
- transition summary and cropped changed-cell evidence
- turn metrics
- updated agent context

At the start of a game, orchestration can read learned agent context from the
newest `m_states` rows when `use_learned_contexts` is enabled. An empty
database is a valid first-run state.

At normal run completion, runtime prunes `m_states` to keep only the newest row
per game. Manual cleanup through `--clean-db` clears memory table rows,
including E and legacy generic rows.

## Experimental Writes

Future tool outputs should be written to `E` before a reference is returned to
Agent `X`. The current vLLM-only runtime exposes no tools, so this path is
normally inactive.
