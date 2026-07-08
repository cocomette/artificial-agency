# Persistence Rules

- Orchestration writes source `M` rows before Agent X acts when state memory is
  enabled.
- Orchestration completes the same row after the transition, change summary,
  judge score, reward, Memory, Goal, and metrics are available.
- Candidate predictions, judge scores, rewards, Goal predictions, turn ledger
  rows, and model-input debug records are written to explicit v1 tables.
  Reward metadata stores resource-cost components, and `TurnMetrics` stores
  aggregate model prompt, completion, and total token counts for the turn.
- Normal runs may prune `M` to the latest row per game unless
  `debug_keep_all_m_states` is enabled.
- Game-over reset appends a reset ledger marker for Memory regeneration; it
  does not clear the run ledger or replace the original first frame.
- The current SQLite schema is intentionally not migrated from older run DBs;
  reset disposable DBs when schema validation fails.
