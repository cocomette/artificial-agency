# Update Outputs

Updater `P` returns revised context documents for the task selected by
orchestration.

## Output Contexts

- During the frame/game loop, role-specific updater tasks return
  `L^S_i,t+1`, `L^G_i,t+1`, and `L^X_i,t+1`.
- The historizer call returns top-level `probing_evolution`,
  `policy_evolution`, and `updater_mode`.
- The world-model call returns top-level `world_description`,
  `special_events`, and per-action effects as `action_effects`.
- The agent probing updater provider returns top-level
  `probing_strategy` and `next_actions`.
- The agent policy updater provider returns top-level
  `policy_strategy` and `next_actions`.
- At end-of-run, the shared general updater task returns `K^S`, `K^G`, and
  `K^X` through three role-specific invocations.

## Persistence Rule

Updater outputs go back to orchestration. Orchestration applies them to the
live working `ContextDocuments`, persists the resulting updater summaries into
`M`, and uses them when composing the next model calls. The world-model output
and the historizer strategy evolution are persisted separately in frame-turn
metadata.

Game-specific `L` contexts are selected from the latest state for the current
game. Game-agnostic `K` contexts are selected from the latest persisted state
across all games, then recombined with the current game's `L` before model
calls.

The updater does not own a separate memory store and does not write directly
to SQLite.

Turn metrics are persisted by orchestration with the committed frame-turn
state. Updater backends do not compute or mutate those runtime facts.

## Scope Rule

Game-specific contexts may be updated during a game. Game-agnostic contexts
`K^m` are updated only after finishing a game. Updater backends do not choose
`K` versus `L` timing themselves.
