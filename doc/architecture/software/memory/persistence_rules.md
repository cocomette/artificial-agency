# Persistence Rules

- Orchestration owns all memory writes.
- Model adapters return provider-neutral outputs; orchestration decides how to
  persist them.
- `M` rows are prewritten before Agent X acts and completed after transition
  processing.
- Completed rows include action, trace, metrics, agent context, and game-memory
  metadata.
- Frame-turn source rows include durable current-frame hash metadata when the
  frame can be visually hashed. The hash uses the ARC-grid crop derived from
  the change-summary model image crop, and persists the crop edges beside the
  hash. Known-state simulation uses those hashes to match prior same-run
  transitions.
- Persisted ACTION6 payloads keep the submitted ARC-grid coordinates and the
  ARC-grid target value derived from that final coordinate. Model-visible
  bounding boxes are runtime-only and are not persisted in memory.
- Simulated rows are completed M rows marked with simulation metadata. Catch-up
  metadata is merged into the exit or last simulated row after ARC catch-up
  actions are submitted.
- Normal runs prune `M` to the latest state per game unless
  `debug_keep_all_m_states` is enabled.
