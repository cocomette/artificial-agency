# Updater Inputs

## Agent Game Context

`AgentGameContextUpdateInput` contains:

- previous agent context containing both current `probing_strategy` and
  `policy_strategy`
- current observation
- allowed actions
- glossary actions
- recent action history with the latest transition, including the updater mode
  that selected controllable actions, cumulative `completed_levels`, and
  controllable `action_count`
- fresh world-model context containing `world_description`, `special_events`,
  and per-action `action_effects`
- historizer strategy evolution containing `probing_evolution`,
  `policy_evolution`, and `strategy_summary`

Change-summary inputs receive previous/current observation frames for normal
transitions, attached in that order, plus the previous world-model context from
state metadata. Empty previous world-model context is passed as an empty string
on first-run/no-memory turns. They also receive the previous change-summary
element list, excluding the previous `change_detected` flag; first-run and
post-reset turns pass an empty list. The world context helps interpret what
changed, but it is not ground truth and must not replace direct frame evidence.
The change-summary output is an `elements` array with stable element names,
element descriptions, and chronological element mutations; orchestration derives
the action-history `change_summary` string from those fields.
Post-action animation bundles attach the ordered frame array after exact
consecutive duplicate filtering to change summary; those bundle images are
cropped first and then resized so the full bundle fits within the change model's
`animation_frame_budget_coefficient` configured-frame-area budget. The
coefficient defaults to `2` and values below `2` are clamped to `2`. Optional
change-summary Gaussian blur and Gaussian noise are added only to the prepared
image copies sent to the change-summary provider, independently per attached
frame, after changed-pixel counts and bundle filtering have already used the
clean frames. Blur runs before noise.
When `activate_diff_mask` is enabled, change summary inserts one binary
black/white changed-pixel mask between each consecutive prepared observation
frame before provider encoding.
For vLLM change-summary configs, `frame_input_mode: video` sends the prepared
ordered frame array as one OpenAI-compatible pre-extracted video frame sequence
rather than separate image items.
World-model and agent game-updater inputs attach only the current observation
frame. Change-summary,
world-model, historizer, and agent game-updater images are cropped by
`input_image_crop_arc_grid_edges` before provider encoding; the default crop is
4 source ARC-grid cells per edge.
ACTION6 updater outputs are interpreted in the cropped image coordinate space
and mapped back to the full ARC grid. The historizer receives the attached
current observation frame plus the fresh world-model output and strategy
history.

## Agent General Context

`GeneralKnowledgeUpdateInput` contains the previous agent context and run-level
metadata such as run id, game id, stop reason, step count, completed levels,
final state, and persisted state ids.
