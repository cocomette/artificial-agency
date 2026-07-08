# Model Inputs

## Agent X

Agent X is dormant in the current runtime game loop. If the adapter path is
re-enabled, it receives:

- composed agent context
- current observation frame
- allowed current actions
- allowed-action glossary for prompt semantics
- optional recent action history
- optional generic tool runtime

## Change Summary

Change summary receives previous observation, current observation, chosen
action, glossary actions, and previous change-summary element names plus
descriptions. It does not receive previous change-summary mutations. Normal
transitions attach previous/current images plus prompt text. Post-action
animation bundles attach the ordered frame array after exact consecutive
duplicate filtering, with the first frame as previous, the last frame as
current, and any middle frames as animation evidence. Its attached frames are
cropped with
`input_image_crop_arc_grid_edges` before resizing; the default crop removes 4
cells per edge in the source ARC 64x64 grid. Submitted ACTION6 coordinates in
the change-summary action context are normalized from 0 to 1000 relative to
the cropped first frame, and the selected ACTION6 target description is
included when available. Normal transitions resize to `input_image_size`; animation
bundles resize every attached frame so the full bundle uses at most
`animation_frame_budget_coefficient` configured-frame areas. The coefficient
defaults to `2` and values below `2` are clamped to `2`. Every change-summary
prompt includes deterministic same-color connected-component facts for each
attached frame. The component rows are extracted from the cropped frame and
use one-word rendered color names, counts, and normalized boxes, capped by
change-summary `max_nb_components`.

## Compacter

The compacter call receives the previous compacter context, bounded action
history, strategy history, allowed actions, the current observation frame, and
metadata identifying the run, game, and source state. It also receives
deterministic current-frame components capped by compacter `max_nb_components`.
It does not receive the previous frame or post-action animation frames.
Animation evidence is preserved in action history, and the attached current
frame is cropped with `input_image_crop_arc_grid_edges` before resizing to
`input_image_size`.

On level completion, orchestration stores the compacter
`previous_actions_summary` and `previous_strategy_summary` as the solved-level
summary. The next updater call receives the compact previous strategy summary
as the previous-level strategy context plus a new-level reset notice.

## Updater P

The agent game updater receives:

- current observation after the transition, cropped with
  `input_image_crop_arc_grid_edges` before provider encoding
- deterministic current-frame components capped by updater `max_nb_components`;
  these rows list same-color connected components with one-word rendered color
  names, counts, and normalized boxes in the updater crop coordinate space
- allowed actions and allowed-action glossary
- previous strategy summary
- previous actions summary
- world model context containing `world_description`, `special_events`, and
  per-action `action_effects`
- current raw updater strategy-history buffer
- previous current strategy containing only the `current_strategy` emitted by
  the previous updater turn

The updater returns `current_strategy` and `next_actions`.
ACTION6 updater outputs include a required `target` description, `bbox`
coordinates normalized relative to the cropped attached frame, and
`target_rgb_color`. The updater adapter targets the pixel inside `bbox` whose
color is closest to `target_rgb_color`, breaking ties by distance to the bbox
center, and maps that pixel back to the full ARC grid before environment
submission.
