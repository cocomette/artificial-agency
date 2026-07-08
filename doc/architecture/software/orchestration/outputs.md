# Orchestration Outputs

The game loop returns `GameRunResult` and persists detailed frame-turn state to
SQLite.

Persisted state includes:

- current observation
- chosen action
- Agent X trace
- turn metrics
- agent context after updater output
- game-memory document metadata

Debug sinks may also capture sanitized model inputs and provider requests.
