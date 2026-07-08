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
action, glossary actions, and previous change-summary elements. Normal
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
defaults to `2` and values below `2` are clamped to `2`. When
`activate_bounding_boxes` is enabled, the adapter computes raw changed-pixel
masks for each consecutive cropped-frame pair and accumulates those raw pixels
over the frame sequence. The first frame receives the first computed cumulative
mask, and each later frame receives the cumulative mask after its incoming
transition. The adapter resizes the clean frames, then dilates a temporary copy
of each frame's cumulative mask with `dilation_bounding_boxes`, extracts the
mask edge pixels, scales them to the final model-image size, and draws magenta
edges with `width_bounding_boxes` final-image pixels. It appends the
magenta-outline instruction fragment only when `activate_bounding_boxes` is
enabled. Dilation is only a drawing step and is not added back into the
cumulative mask. If `gaussian_blur_kernel_size` is greater than `1`, the
adapter applies a Gaussian blur pass to every prepared change-summary frame
copy; the kernel size is an odd final-image pixel width, with radius derived as
`(kernel_size - 1) / 6`. If `gaussian_noise_deviation` is greater than zero,
the adapter then adds independent zero-centered RGB Gaussian noise with that
standard deviation to every prepared change-summary frame immediately before
the provider call. Blur and noise are only applied to those model-visible image
copies; they do not affect stored observations, changed-pixel counts, frame
deduplication, or inputs to other model roles. If `activate_diff_mask` is set,
the adapter inserts binary black/white changed-pixel masks between consecutive
prepared observation frames. vLLM change-summary configs can
set `frame_input_mode` to `video`, which sends the prepared ordered frame array
as one OpenAI-compatible `video_url` pre-extracted frame sequence with
`media_io_kwargs` temporal metadata instead of separate `image_url` items.

## World Model

The world-model call receives the previous world-model context, bounded action
history, allowed actions, the current observation frame, and metadata
identifying the run, game, and source state. It does not receive the previous
frame or post-action animation frames. Animation evidence is preserved in
action history, and the attached current frame is cropped with
`input_image_crop_arc_grid_edges` before resizing to `input_image_size`.

## Historizer

The historizer call receives the fresh world-model output from the current
turn, including special events, plus a same-run ordered list of recent updater
strategy snapshots containing `probing_strategy` and `policy_strategy`, bounded
action history, and allowed actions. It is text-only and does not attach image
inputs.

## Level Summary

The level-summary call receives the completed level number and the ordered
same-run updater strategy snapshots from the M rows that belong to the solved
level interval.

## Updater P

Agent probing and policy game updates receive:

- previous agent context containing both current `probing_strategy` and
  `policy_strategy`
- current observation after the transition, cropped with
  `input_image_crop_arc_grid_edges` before provider encoding
- allowed actions and allowed-action glossary
- bounded action history including the latest summarized transition,
  cumulative `completed_levels`, and controllable `action_count`
- world-model context containing `world_description`, `special_events`, and
  per-action `action_effects`
- historizer `probing_evolution` and `policy_evolution`
- optional `same_past_state_detections` from prior same-run M rows with the
  exact same current-frame hash after the change-summary ARC-edge crop,
  including both strategy fields and the historizer evolution fields stored for
  that turn
- optional latest same-run previous-level `solution_method`

Agent probing returns only `probing_strategy`. Agent policy
returns only `policy_strategy`. Both receive the fresh world-model
context, both previous game summaries, and the historizer evolution summaries.
Agent general updates receive the previous agent context and run-level summary
metadata at the end of a run.
ACTION6 updater outputs include a required `target` description, `bbox`
coordinates normalized relative to the cropped attached frame, and
`target_rgb_color`. The updater adapter targets the pixel inside `bbox` whose
color is closest to `target_rgb_color`, breaking ties by distance to the bbox
center, and maps that pixel back to the full ARC grid before environment
submission. Prompt-facing action history renders the target description rather
than coordinates.
