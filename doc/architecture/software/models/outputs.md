# Model Outputs

## Agent X

Agent X is dormant in the current runtime game loop. If the adapter path is
re-enabled, it returns a `DecisionResult` with:

- final `ActionSpec`
- `AgentTrace` containing frame refs, reasoning summary, optional tool calls,
  optional generic tool results, and metadata

## Change Summary

The change summary role returns `ChangeSummaryResult`:

- `summary`
- `change_detected`
- metadata

## Compacter

The compacter role returns the latest world description, special-event memory,
per-allowed-action effect summaries, `previous_actions_summary`, and
`previous_strategy_summary`. Orchestration stores per-turn compacter output in
M metadata and stores the final compact summaries from a solved-level
transition in `compacter_level_summaries` for the next level.

## Updater P

Updater P returns `current_strategy` and required `next_actions`.
`next_actions` is an exact-length array of action objects
matching `updater_actions_window`.
ACTION6 updater action objects include a required `target` description,
crop-relative normalized `bbox`, and `target_rgb_color`; the updater adapter
converts that targeting description into execution coordinates in `data`.
Orchestration persists those summaries as the agent game context and stores the
compacter output separately in frame-turn metadata. Actions are submitted in
order on controllable frames without an Agent X revision step.
