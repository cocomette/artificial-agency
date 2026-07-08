# Persistence Rules

- Orchestration writes source `M` rows before the frame decision is resolved
  when state memory is enabled.
- Orchestration completes the same row after the transition, change summary,
  updater output, and metrics are available.
- Learned context hydration combines the latest agent general context with the
  latest selected-game agent game context.
- Completed-level compacter summaries are stored in
  `compacter_level_summaries` with the completed level number, source M-state
  ids, `previous_actions_summary`, and `previous_strategy_summary`.
- Normal runs may prune `M` to the latest row per game unless
  `debug_keep_all_m_states` is enabled.
- The current SQLite schema is intentionally not migrated from older run DBs;
  reset disposable DBs when schema validation fails.
