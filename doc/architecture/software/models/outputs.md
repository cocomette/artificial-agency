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

## World Model

The world-model role returns the latest world description, special-event
memory, and per-allowed-action effect summaries.

## Historizer

The historizer role returns `probing_evolution`, `policy_evolution`, and
`updater_mode`.
Orchestration combines this with the fresh world-model output as
`AgentContextHistorySummary` for updater input.

## Level Summary

The level-summary role returns `solution_method`, a compact reusable method
summary for the next level of the same game. Orchestration stores it in
`level_solution_summaries`.

## Updater P

Updater P returns `RoleContext` replacements. Agent probing returns
`probing_strategy` and required `next_actions`; agent policy returns
`policy_strategy` and required `next_actions`. `next_actions` is an exact-length
array of action objects matching the active mode's configured action window.
ACTION6 updater action objects include a required `target` description,
crop-relative normalized `bbox`, and `target_rgb_color`; the updater adapter
converts that targeting description into execution coordinates in `data`.
Orchestration persists those summaries as the agent game context and stores the
world-model output separately in frame-turn metadata. Actions are submitted in
order on controllable frames without an Agent X revision step. The general task
replaces the general segment and preserves the game segment.
