# Update Outputs

Updater `P` returns revised context documents for the task selected by
orchestration.

## Output Contexts

- During the frame/game loop, the agent updater returns `current_strategy` and
  `next_actions`.
- The compacter call returns top-level `world_description`, `special_events`,
  per-action effects as `action_effects`, `previous_actions_summary`, and
  `previous_strategy_summary`.

## Persistence Rule

Updater outputs go back to orchestration. Orchestration applies them to the
live working `ContextDocuments`, persists the resulting updater summaries into
`M`, and uses them when composing the next model calls. The compacter output is
persisted separately in frame-turn metadata.

Game-specific `L` contexts are selected from the latest state for the current
game before model calls.

The updater does not own a separate memory store and does not write directly
to SQLite.

Turn metrics are persisted by orchestration with the committed frame-turn
state. Updater backends do not compute or mutate those runtime facts.

## Scope Rule

Game-specific contexts may be updated during a game. Updater backends do not
choose persistence timing themselves.
