# Updater Inputs

## Agent Game Context

`AgentGameContextUpdateInput` contains:

- current observation
- deterministic current-frame components capped by updater `max_nb_components`
- allowed actions
- glossary actions
- recent action history with the latest transition, including cumulative
  `completed_levels` and controllable `action_count`
- previous strategy summary
- previous actions summary
- world model context containing `world_description`, `special_events`,
  and per-action `action_effects`
- previous turns strategy numbered oldest to newest, containing recent
  same-run `current_strategy` snapshots plus the current previous game context
  as the newest `[latest]` item

Change-summary inputs receive previous/current observation frames for normal
transitions, attached in that order. They also receive previous change-summary
element names and descriptions as stable identity hints, but not previous
mutations. Direct frame evidence is the only visual source of truth for what
changed.
The change-summary output is an `elements` array with stable element names,
element descriptions, and chronological element mutations; orchestration derives
the action-history `change_summary` string from those fields.
Post-action animation bundles attach the ordered frame array after exact
consecutive duplicate filtering to change summary; those bundle images are
cropped first and then resized so the full bundle fits within the change model's
`animation_frame_budget_coefficient` configured-frame-area budget. The
coefficient defaults to `2` and values below `2` are clamped to `2`. The
prepared frame sequence is sent as image inputs, and each change-summary prompt
includes deterministic same-color component facts for the attached frames.
Compacter and agent game-updater inputs attach only the current observation
frame. Change-summary, compacter, and agent game-updater images are cropped by
`input_image_crop_arc_grid_edges` before provider encoding; the default crop is
4 source ARC-grid cells per edge.
ACTION6 updater outputs are interpreted in the cropped image coordinate space
and mapped back to the full ARC grid.
