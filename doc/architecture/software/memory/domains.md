# Memory Domains

## State Memory `M`

`M` is the durable SQLite-backed state memory. Current rows store frame-turn
observations, chosen actions, Agent X traces, turn metrics, agent context, and
game-memory metadata.

## Experimental Memory `E`

`E` is a rolling SQLite-backed table for temporary experiment records. The
current active runtime keeps the table available for orchestration features,
but Agent X world/goal tool use is not wired in this branch.
