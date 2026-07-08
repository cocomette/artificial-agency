# Model Roles

## Agent X

Agent X is dormant in the current runtime game loop. Its adapter code still
exists, but runtime bootstrap registers `orchestrator_agent=None`; controllable
actions are selected by updater P and wrapped by orchestration as decision
traces.

## Change Summary

The change summary role receives the previous observation, current observation,
chosen action, action glossary, and optional post-action animation frame bundle.
It may receive previous change-summary element names and descriptions as stable
identity hints, but not previous mutations. The attached frames remain the
source of truth for the current visible change. It returns an
`elements` array plus a `change_detected` boolean indicating whether any visible
difference was detected across the attached image set. Each element has
`element_name`, `element_description`, and `element_mutation`; unchanged visible
elements keep an empty mutation. Orchestration stores the element list in action
history and derives the prompt-facing `change_summary` text as one bullet per
element, rendering the name, description, and either the mutation or an explicit
no-detected-changes note. The derived summary is stored in action history
alongside cumulative `completed_levels` and controllable `action_count`. Its attached frames are
cropped by `input_image_crop_arc_grid_edges`, defaulting to 4 source ARC-grid
cells per edge. Animation bundles keep the ordered frame array after exact
consecutive duplicate filtering, then resize all attached frames to fit the
change model's `animation_frame_budget_coefficient` configured-frame-area
budget. The coefficient defaults to `2` and values below `2` are clamped to
`2`. Every change-summary prompt includes deterministic same-color component
facts for each attached frame.

## Compacter

The compacter role runs before the agent updater on every controllable turn,
including known-state simulation turns. It receives the previous compacter
context from state metadata plus action history, strategy history, allowed
actions, and the attached current frame for the latest transition. Previous
frames and animation bundles are not attached to compacter provider calls;
their transition evidence remains available through action history. The action
history includes cumulative `completed_levels` and controllable
`action_count`. On a level-completion turn, orchestration attaches the latest
retained solved-level frame so the compacter can still describe the level that
was just completed. The attached current frame is cropped by
`input_image_crop_arc_grid_edges`, defaulting to 4 source ARC-grid cells per
edge. It also receives deterministic current-frame component rows in the same
cropped coordinate space, capped by compacter `max_nb_components`; each row
uses a one-word rendered color name.

The compacter returns `world_description`, `special_events`, `action_effects`,
`previous_actions_summary`, and `previous_strategy_summary`. The action and
strategy summaries are compact rolling summaries for the current level. On a
level-completion boundary, orchestration stores those summary fields in
`compacter_level_summaries`; updater P receives the stored
`previous_strategy_summary` as the minimal previous-level strategy summary.

## Updater P

Updater P has one active task: `agent`. It updates `current_strategy`, then
chooses the next action chain after controllable transitions.

The agent updater receives allowed actions, previous strategy summary, previous
actions summary, world model context from the compacter, current raw
action-history and strategy-history windows, and only the previous turn's
`current_strategy`. It also receives deterministic current-frame
component rows containing one-word
rendered color names, counts, and normalized boxes in the updater crop
coordinate space, capped by updater `max_nb_components`. Its attached current frame is cropped by
`input_image_crop_arc_grid_edges`, and ACTION6 outputs carry a target
description, cropped normalized target bounding box, and target RGB color. The
updater adapter deterministically selects the closest-color pixel inside that
box, preferring pixels closest to the box center among equal color matches,
then maps that point back to the full ARC grid and attaches the targeted ARC
cell value. The raw bbox remains transient runtime matching metadata and is not
persisted. `next_actions` is an ordered array whose length must equal
`updater_actions_window`. The updater receives the fresh compacter output, but
does not persist or rewrite it.

During known-state simulation, change summary and environment steps are skipped
while replaying known historical transitions. The compacter still runs after
each replayed transition is appended so updater P receives a fresh compact
view of the growing simulated action and strategy history. The updater receives
the fresh compacter output, the previous turn's `current_strategy`, and current
simulated action history.
Simulation
currently assumes
`updater_actions_window == 1`. Historical `ACTION6` edges match current
`ACTION6` outputs deterministically when the persisted historical target cell
value equals the current target cell value and the current cropped-normalized
bbox wraps the historical submitted coordinate in the same full ARC-grid
coordinate space.

`next_actions` is queued by orchestration. One queued action is submitted on
each controllable frame after previous-to-current transition modeling. The game
loop does not call Agent X to revise those actions.
